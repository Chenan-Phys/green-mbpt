#!/usr/bin/env python3
"""Run one reproducible molecular pilot only on GREEN_workstation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from cavity_prepare import prepare
from cavity_analyze import analyze


def run(spec_path, directory, beta=1000., methods=("HF","GW")):
    if socket.gethostname().split(".")[0] != "kadanoff":
        raise RuntimeError("Scientific calculations must run on GREEN_workstation")
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
        result=f"{method.lower()}.h5"
        if (directory/result).exists():
            value=analyze(directory,result)
            if not value["accepted_numerically"]:
                raise RuntimeError(f"Existing {method} run is unconverged; inspect before restarting")
            results[method]=value
            continue
        command=[str(executable),"--scf_type",method,"--kernel","CPU",
                 "--BETA",str(beta),"--grid_file",str(grid),"--itermax","120",
                 "--E_thr","1e-10","--mixing_type","SIGMA_MIXING","--mixing_weight","0.5",
                 "--results_file",result,"--diis_file",f"{method.lower()}_diis.h5"]
        # A finite molecule has one Coulomb set; no periodic Ewald correction set.
        command += ["--dfintegral_file","df_hf_int"]
        started=time.time()
        with (directory/f"{method.lower()}.log").open("w") as log:
            subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,check=True)
        value=analyze(directory,result)
        value.update(wall_seconds=time.time()-started,beta_hartree_inverse=beta,
                     command=command,method="QED_HF" if method=="HF" else "QED_GW_joint")
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
    args=parser.parse_args()
    print(json.dumps(run(args.spec,args.directory,args.beta,args.methods),indent=2))
