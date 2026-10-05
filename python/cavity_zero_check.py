#!/usr/bin/env python3
"""Compare zero-coupling joint GW with the installed stable executable."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import h5py
import numpy as np
from cavity_analyze import complex_array

root=Path(sys.argv[1]).resolve()
if socket.gethostname().split(".")[0]!="kadanoff":
    raise RuntimeError("Run on GREEN_workstation")
source=root/"zero"
target=root/"legacy"; target.mkdir(exist_ok=True)
if not (target/"input.h5").exists():
    shutil.copy2(source/"input.h5",target/"input.h5")
    with h5py.File(target/"input.h5","a") as f:
        del f["QED"]
command=[str(Path.home()/"green/install/bin/mbpt.exe"),"--scf_type","GW","--kernel","CPU",
         "--BETA","1000","--grid_file",str(Path.home()/"green/install/share/ir/1e5.h5"),
         "--itermax","120","--E_thr","1e-10","--mixing_type","SIGMA_MIXING",
         "--mixing_weight","0.5","--dfintegral_file",str(source/"df_hf_int"),
         "--dfintegral_hf_file",str(source/"df_hf_int")]
if not (target/"sim.h5").exists():
    with (target/"gw.log").open("w") as log:
        subprocess.run(command,cwd=target,stdout=log,stderr=subprocess.STDOUT,check=True)
def read(path):
    with h5py.File(path) as f:
        it=max(int(k[4:]) for k in f if k.startswith("iter") and k[4:].isdigit())
        group=f[f"iter{it}"]
        return float(group["Energy_HF"][()])+float(group["Energy_2b"][()]),complex_array(group["G_tau/data"])[-1]
e,g=read(source/"gw.h5"); er,gr=read(target/"sim.h5")
result={"energy_error_hartree":abs(e-er),"density_matrix_error":float(np.linalg.norm(g-gr)),"command":command}
(root/"zero_coupling_check.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
if result["energy_error_hartree"]>1e-9 or result["density_matrix_error"]>1e-8:
    raise AssertionError("Zero-coupling recovery failed")
