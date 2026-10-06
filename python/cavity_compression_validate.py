#!/usr/bin/env python3
"""Verify lossless storage plus native restart of a fresh accepted H2 copy."""
import argparse
import json
from pathlib import Path
import shutil
import h5py
import numpy as np
from cavity_thermal_hf import clone_reference
from cavity_lossless_compress import compress
from cavity_run import run
from cavity_analyze import analyze


def validate(source,target,spec):
    source=Path(source); target=Path(target)
    old=analyze(source,"gw.h5")
    clone_reference(source,target)
    for name in ("gw.h5","gw_provenance.json"): shutil.copy2(source/name,target/name)
    # Small files can grow because chunk metadata outweighs compressed data.
    # This disposable copied case must still exercise a genuine gzip restart.
    storage=compress(target/"gw.h5",replace_original=True,allow_size_increase=True)
    if storage["status"]!="compressed_verified": raise RuntimeError("Compression was not exercised")
    with h5py.File(target/"gw.h5") as handle:
        filters=[]
        handle.visititems(lambda name,obj:filters.append(obj.compression)
            if isinstance(obj,h5py.Dataset) and obj.size>256 else None)
        if not filters or not all(value=="gzip" for value in filters):
            raise RuntimeError("Native checkpoint lacks the requested gzip filters")
    check=analyze(target,"gw.h5")
    if check!=old: raise RuntimeError("Compressed diagnostics changed")
    new=run(spec,target,methods=("GW",),force_restart=True,restart_unconverged=True,native_threads=4)["GW"]
    difference=float(np.linalg.norm(np.load(source/"density_ao.npy")-np.load(target/"density_ao.npy")))
    result={"storage":storage,"verified_gzip_dataset_count":len(filters),
            "energy_difference_hartree":new["energy_hartree"]-old["energy_hartree"],
            "density_difference_frobenius":difference,"result":new,
            "passed":bool(new["accepted_numerically"] and abs(new["energy_hartree"]-old["energy_hartree"])<1e-8 and difference<1e-6)}
    (target/"compression_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    if not result["passed"]: raise RuntimeError("Native compressed restart failed")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("source"); parser.add_argument("target"); parser.add_argument("spec")
    args=parser.parse_args(); print(json.dumps(validate(args.source,args.target,args.spec),indent=2))
