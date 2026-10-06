#!/usr/bin/env python3
"""Compare a transferred coupling initial guess against an independent cold solution."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from cavity_analyze import analyze,complex_array
from cavity_run import run
from cavity_thermal_hf import clone_reference
from cavity_warm_start import seed

root=Path(sys.argv[1]).resolve()
cold=root/"H2_charge+1_lambda0"
source=root/"H2_charge+1_lambda0.005"
target=root/"warm-zero"
if not target.exists():
    clone_reference(cold,target)
    seed(source,target)
warm=run(root/"specs/H2_charge+1_lambda0.json",target,methods=("GW",),
         restart_unconverged=True,energy_threshold=1e-10,number_tolerance=1e-12)["GW"]
reference=analyze(cold,"gw.h5")
with h5py.File(target/"gw.h5") as f,h5py.File(cold/"gw.h5") as c:
    wi=max(int(k[4:]) for k in f if k.startswith("iter") and k[4:].isdigit())
    ci=max(int(k[4:]) for k in c if k.startswith("iter") and k[4:].isdigit())
    error=float(np.linalg.norm(complex_array(f[f"iter{wi}/G_tau/data"][-1])-
                               complex_array(c[f"iter{ci}/G_tau/data"][-1])))
    native=float(f[f"iter{wi}/QED/Density_residual_frobenius"][()])
out={"energy_difference_hartree":abs(warm["energy_hartree"]-reference["energy_hartree"]),
     "density_endpoint_difference":error,"native_python_density_residual_difference":
     abs(native-warm["density_residual_frobenius_electrons"]),"warm":warm,"cold":reference}
if out["energy_difference_hartree"]>1e-9 or error>1e-7 or out["native_python_density_residual_difference"]>1e-12:
    raise AssertionError("Warm-start/density-criterion validation failed")
(root/"warm_start_validation.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if not isinstance(v,dict)},indent=2))
