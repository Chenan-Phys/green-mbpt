#!/usr/bin/env python3
"""Create a fresh equivalent input after bounded Hermitian roundoff cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import h5py
import numpy as np
from cavity_thermal_hf import clone_reference
from cavity_run import case_guard
from cavity_lowdin import hermitian_roundoff,packed_like
from cavity_analyze import complex_array


def prepare(source,target):
    source=Path(source); target=Path(target)
    with case_guard(source): clone_reference(source,target)
    diagnostics={}
    with h5py.File(target/"input.h5","a") as inp:
        for name in ("QED/dipole","QED/second_moment","HF/H-k","HF/Fock-k"):
            ds=inp[name]; value=complex_array(ds) if name.startswith("HF/") else np.asarray(ds)
            clean=hermitian_roundoff(value)
            diagnostics[name]={"original_antihermitian_norm":float(np.linalg.norm(value-np.swapaxes(value.conj(),-1,-2))),
                               "roundoff_correction_norm":float(np.linalg.norm(clean-value)),"operator_norm":float(np.linalg.norm(value))}
            ds[...]=packed_like(clean,ds) if name.startswith("HF/") else clean.real
    manifest=json.loads((target/"manifest.json").read_text())
    manifest.update(input_sha256=hashlib.sha256((target/"input.h5").read_bytes()).hexdigest(),
                    hermitian_roundoff_cleanup=diagnostics,
                    hermitian_roundoff_source=str(source))
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (target/"hermitian_roundoff.json").write_text(json.dumps(diagnostics,indent=2)+"\n")
    return diagnostics


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("source"); parser.add_argument("target")
    args=parser.parse_args(); print(json.dumps(prepare(args.source,args.target),indent=2))
