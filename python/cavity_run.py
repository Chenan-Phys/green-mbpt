#!/usr/bin/env python3
"""Run one reproducible molecular pilot only on GREEN_workstation."""
import argparse
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


def provenance(directory, executable):
    source=Path(__file__).resolve().parent.parent
    files=["src/dyson.cpp", "src/hf_solver.cpp", "src/gw_cpu_kernel.cpp",
           "src/green/mbpt/cavity.h", "src/green/mbpt/kernels.h"]
    digest=hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update((source/name).read_bytes().replace(b"\r\n",b"\n"))
    commit=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    return {"binary_sha256":hashlib.sha256(executable.read_bytes()).hexdigest(),
            "input_sha256":hashlib.sha256((directory/"input.h5").read_bytes()).hexdigest(),
            "core_source_sha256_lf":digest.hexdigest(), "source_commit":commit,
            "hostname":socket.gethostname(), "executable":str(executable)}


def run(spec_path, directory, beta=1000., methods=("HF","GW"), restart_unconverged=False,
        energy_threshold=1e-8, native_threads=4, force_restart=False, mixing_type="SIGMA_MIXING"):
    if socket.gethostname().split(".")[0] != "kadanoff":
        raise RuntimeError("Scientific calculations must run on GREEN_workstation")
    if native_threads < 1 or not 0 < energy_threshold <= 1e-8:
        raise ValueError("Use positive thread count and an energy tolerance at most 1e-8")
    if mixing_type not in ("SIGMA_MIXING","DIIS"):
        raise ValueError("Choose SIGMA_MIXING or DIIS")
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    spec=json.loads(Path(spec_path).read_text())
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
    executable=Path.home()/"green/install-qed/cavity-general/bin/mbpt.exe"
    grid=Path.home()/"green/install/share/ir/1e5.h5"
    results={}
    for method in methods:
        if method not in ("HF", "GW"):
            raise ValueError("Only implemented QED HF/GW methods can be requested")
        result=f"{method.lower()}.h5"
        restart=False
        if (directory/result).exists():
            value=analyze(directory,result)
            if value["accepted_numerically"] and not force_restart:
                saved=directory/f"{method.lower()}_analysis.json"
                if saved.exists():
                    value={**json.loads(saved.read_text()), **value}
                if value.get("beta_hartree_inverse",beta)!=beta:
                    raise RuntimeError("Cached result has a different temperature")
                results[method]=value
                continue
            if not (restart_unconverged or force_restart):
                raise RuntimeError(f"Existing {method} run is unconverged; inspect before restarting")
            restart=True
        command=[str(executable),"--scf_type",method,"--kernel","CPU",
                 "--BETA",str(beta),"--grid_file",str(grid),"--itermax","120",
                 "--E_thr",str(energy_threshold),"--mixing_type",mixing_type,"--mixing_weight","0.5",
                 "--results_file",result,"--diis_file",f"{method.lower()}_diis.h5"]
        if restart:
            command += ["--restart","true","--diis_restart","false"]
        # A finite molecule has one Coulomb set; no periodic Ewald correction set.
        command += ["--dfintegral_file","df_hf_int"]
        stamp=provenance(directory,executable)
        stamp.update(beta_hartree_inverse=beta, command=command, native_threads=native_threads)
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
            subprocess.run(command,cwd=directory,env=environment,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        value=analyze(directory,result)
        value.update(wall_seconds=time.time()-started,beta_hartree_inverse=beta,
                     command=command,provenance=stamp,
                     method="QED_HF" if method=="HF" else "QED_GW_joint")
        value["native_converged"]="Simulation Converged" in (directory/f"{method.lower()}.log").read_text()
        (directory/f"{method.lower()}_analysis.json").write_text(json.dumps(value,indent=2)+"\n")
        if not value["accepted_numerically"]:
            raise RuntimeError(f"{method} failed the numerical acceptance checks")
        results[method]=value
    (directory/"case_results.json").write_text(json.dumps(results,indent=2)+"\n")
    return results


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
    args=parser.parse_args()
    print(json.dumps(run(args.spec,args.directory,args.beta,args.methods,
                         args.restart_unconverged,args.energy_threshold,args.native_threads,args.force_restart,
                         args.mixing_type),indent=2))
