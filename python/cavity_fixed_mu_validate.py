#!/usr/bin/env python3
"""Check initialized fixed-mu GW restart against a converged neutral H2 control."""
import argparse
import json
from pathlib import Path
import numpy as np
from cavity_analyze import analyze
from cavity_thermal_hf import clone_reference
from cavity_warm_start import seed
from cavity_run import run


def validate(source,target,spec):
    source=Path(source); target=Path(target)
    old=analyze(source,"gw.h5")
    if not old["accepted_numerically"]: raise RuntimeError("Source did not converge")
    clone_reference(source,target)
    seed(source,target)
    new=run(spec,target,methods=("GW",),restart_unconverged=True,force_restart=True,
            const_density=False,native_threads=4)["GW"]
    difference=float(np.linalg.norm(np.load(source/"density_ao.npy")-np.load(target/"density_ao.npy")))
    result={"source":str(source),"target":str(target),"energy_difference_hartree":new["energy_hartree"]-old["energy_hartree"],
            "ao_density_difference_frobenius":difference,"result":new,
            "passed":bool(new["accepted_numerically"] and abs(new["energy_hartree"]-old["energy_hartree"])<1e-8 and difference<1e-6)}
    (target/"fixed_mu_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    if not result["passed"]: raise RuntimeError("Fixed-mu validation failed")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("source"); parser.add_argument("target"); parser.add_argument("spec")
    args=parser.parse_args(); print(json.dumps(validate(args.source,args.target,args.spec),indent=2))
