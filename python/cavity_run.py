#!/usr/bin/env python3
"""Run one reproducible molecular pilot only on GREEN_workstation."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from cavity_prepare import prepare
from cavity_analyze import analyze
from cavity_build_manifest import core_digest, CORE_FILES


def provenance(directory, executable):
    source=Path(__file__).resolve().parent.parent
    binary_sha=hashlib.sha256(executable.read_bytes()).hexdigest()
    digest=core_digest(source)
    build=json.loads((executable.parent.parent/"build_manifest.json").read_text())
    if build["binary_sha256"]!=binary_sha or build["core_source_sha256_lf"]!=digest:
        raise RuntimeError("Executable/build manifest/current core source do not match")
    commit=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    return {"binary_sha256":binary_sha,
            "input_sha256":hashlib.sha256((directory/"input.h5").read_bytes()).hexdigest(),
            "core_source_sha256_lf":digest,"core_source_files":CORE_FILES, "source_commit":commit,
            "build_manifest":build,
            "hostname":socket.gethostname(), "executable":str(executable)}


def _run(spec_path, directory, beta=1000., methods=("HF","GW"), restart_unconverged=False,
        energy_threshold=1e-8, native_threads=4, force_restart=False, mixing_type="SIGMA_MIXING",
        mixing_weight=.5,number_tolerance=1e-12,const_density=True):
    if socket.gethostname().split(".")[0] != "kadanoff":
        raise RuntimeError("Scientific calculations must run on GREEN_workstation")
    if native_threads < 1 or not 0 < energy_threshold <= 1e-8:
        raise ValueError("Use positive thread count and an energy tolerance at most 1e-8")
    if mixing_type not in ("SIGMA_MIXING","DIIS"):
        raise ValueError("Choose SIGMA_MIXING or DIIS")
    if not 0 < mixing_weight <= 1:
        raise ValueError("Mixing weight must lie in (0, 1]")
    if not 0 < number_tolerance <= 1e-9:
        raise ValueError("Number-search tolerance must lie in (0, 1e-9]")
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    spec=json.loads(Path(spec_path).read_text())
    if not const_density and spec.get("fix_spin",False):
        raise ValueError("Fixed spin populations require chemical-potential searches")
    if not (directory/"input.h5").exists():
        with (directory/"prepare.log").open("w") as log:
            # Keep PySCF's C-level and Python-level verbose output in the case log.
            command=[os.environ.get("PYTHON","python"),str(Path(__file__).with_name("cavity_prepare.py")),
                     str(Path(spec_path).resolve()),str(directory)]
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    else:
        existing=json.loads((directory/"manifest.json").read_text())
        if any(existing.get(k)!=v for k,v in spec.items()):
            raise RuntimeError("Existing input does not match the requested Hamiltonian")
    executable=Path.home()/"green/install-qed/cavity-general-density/bin/mbpt.exe"
    grid=Path.home()/"green/install/share/ir/1e5.h5"
    results={}
    for method in methods:
        if method not in ("HF", "GW"):
            raise ValueError("Only implemented QED HF/GW methods can be requested")
        result=f"{method.lower()}.h5"
        restart=False
        if (directory/result).exists():
            value=analyze(directory,result)
            if value["accepted_numerically"] and value["native_energy_residual"]<energy_threshold and not force_restart:
                saved=directory/f"{method.lower()}_analysis.json"
                if saved.exists():
                    value={**json.loads(saved.read_text()), **value}
                stamp_file=directory/f"{method.lower()}_provenance.json"
                if stamp_file.exists():
                    stamp=json.loads(stamp_file.read_text())
                    value.update(provenance=stamp,beta_hartree_inverse=stamp["beta_hartree_inverse"],
                                 command=stamp["command"])
                if value.get("beta_hartree_inverse",beta)!=beta:
                    raise RuntimeError("Cached result has a different temperature")
                old_constant=value.get("provenance",{}).get("const_density",True)
                if old_constant!=const_density:
                    raise RuntimeError("Cached ensemble differs; request an explicit restart")
                results[method]=value
                saved.write_text(json.dumps(value,indent=2)+"\n")
                continue
            if not (restart_unconverged or force_restart):
                raise RuntimeError(f"Existing {method} run is unconverged; inspect before restarting")
            restart=True
        if not const_density and not restart:
            raise ValueError("Fixed chemical potential requires a previously initialized checkpoint")
        command=[str(executable),"--scf_type",method,"--kernel","CPU",
                 "--BETA",str(beta),"--grid_file",str(grid),"--itermax","120",
                 "--E_thr",str(energy_threshold),"--tolerance",str(number_tolerance),
                 "--const_density",str(const_density).lower(),
                 "--mixing_type",mixing_type,"--mixing_weight",str(mixing_weight),
                 "--results_file",result,"--diis_file",f"{method.lower()}_diis.h5"]
        if restart:
            command += ["--restart","true","--diis_restart","false"]
        # A finite molecule has one Coulomb set; no periodic Ewald correction set.
        command += ["--dfintegral_file","df_hf_int"]
        stamp=provenance(directory,executable)
        stamp.update(beta_hartree_inverse=beta, command=command, native_threads=native_threads,
                     const_density=const_density,
                     ensemble="target mean particle number" if const_density else "fixed checkpoint chemical potential; particle number checked after convergence")
        # Fedora's default libopenblas is SINGLE_THREADED on this workstation.
        # Preload its installed LP64 pthread variant for this child process only.
        threaded_blas=Path("/usr/lib64/libopenblasp.so.0")
        environment=dict(os.environ,OPENBLAS_NUM_THREADS=str(native_threads),
                         OMP_NUM_THREADS="1",MKL_NUM_THREADS=str(native_threads))
        if threaded_blas.exists():
            environment["LD_PRELOAD"]=str(threaded_blas)+(" "+environment["LD_PRELOAD"]
                                                       if environment.get("LD_PRELOAD") else "")
            stamp["blas_runtime"]={"path":str(threaded_blas.resolve()),
                "sha256":hashlib.sha256(threaded_blas.read_bytes()).hexdigest()}
        stamp_path=directory/f"{method.lower()}_provenance.json"
        if restart and stamp_path.exists():
            old=json.loads(stamp_path.read_text())
            if old.get("input_sha256") != stamp["input_sha256"]:
                raise RuntimeError("Input changed since the calculation checkpoint")
            stamp["previous_run_provenance"]=old
        stamp_path.write_text(json.dumps(stamp,indent=2)+"\n")
        # Integral generation above retains the caller's conservative settings.
        started=time.time()
        with (directory/f"{method.lower()}.log").open("a" if restart else "w") as log:
            log.write("\nCAVITY RUN START "+json.dumps(command)+"\n")
            log.flush()
            subprocess.run(command,cwd=directory,env=environment,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        value=analyze(directory,result)
        value.update(wall_seconds=time.time()-started,beta_hartree_inverse=beta,
                     command=command,provenance=stamp,
                     method="QED_HF" if method=="HF" else "QED_GW_joint")
        latest=(directory/f"{method.lower()}.log").read_text().rsplit("CAVITY RUN START",1)[-1]
        value["native_converged"]="Simulation Converged" in latest
        (directory/f"{method.lower()}_analysis.json").write_text(json.dumps(value,indent=2)+"\n")
        if not value["accepted_numerically"]:
            raise RuntimeError(f"{method} failed the numerical acceptance checks")
        results[method]=value
    (directory/"case_results.json").write_text(json.dumps(results,indent=2)+"\n")
    return results


@contextmanager
def case_guard(directory):
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/"native-case.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Also detect a legacy runner that was started before lock support.
        for process in Path("/proc").iterdir():
            if not process.name.isdigit(): continue
            try:
                command=(process/"cmdline").read_bytes().split(b"\0")
                if b"/green/install-qed/cavity-general" in command[0] and command[0].endswith(b"/bin/mbpt.exe") and \
                   (process/"cwd").resolve()==directory:
                    raise RuntimeError(f"Case already has native writer PID {process.name}")
            except (FileNotFoundError,PermissionError,ProcessLookupError): pass
        yield directory


def run(spec_path, directory, beta=1000., methods=("HF","GW"), restart_unconverged=False,
        energy_threshold=1e-8, native_threads=4, force_restart=False, mixing_type="SIGMA_MIXING",
        mixing_weight=.5,number_tolerance=1e-12,const_density=True):
    with case_guard(directory) as directory:
        return _run(spec_path,directory,beta,methods,restart_unconverged,energy_threshold,
                    native_threads,force_restart,mixing_type,mixing_weight,number_tolerance,const_density)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("directory")
    parser.add_argument("--beta",type=float,default=1000.)
    parser.add_argument("--methods",nargs="+",default=["HF","GW"])
    parser.add_argument("--restart-unconverged",action="store_true")
    parser.add_argument("--energy-threshold",type=float,default=1e-8)
    parser.add_argument("--native-threads",type=int,default=4)
    parser.add_argument("--force-restart",action="store_true")
    parser.add_argument("--mixing-type",choices=["SIGMA_MIXING","DIIS"],default="SIGMA_MIXING")
    parser.add_argument("--mixing-weight",type=float,default=.5)
    parser.add_argument("--number-tolerance",type=float,default=1e-12)
    parser.add_argument("--fixed-chemical-potential",action="store_true",
                        help="Freeze the initialized checkpoint mu; requires restart and subsequent particle-number acceptance")
    args=parser.parse_args()
    print(json.dumps(run(args.spec,args.directory,args.beta,args.methods,
                         args.restart_unconverged,args.energy_threshold,args.native_threads,args.force_restart,
                         args.mixing_type,args.mixing_weight,args.number_tolerance,
                         not args.fixed_chemical_potential),indent=2))
