#!/usr/bin/env python3
"""Prepare a fresh native input with an accepted independent PF HF starting Fock."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import h5py
import numpy as np
from cavity_run import case_guard


def prepare(source,target):
    source=Path(source).resolve(); target=Path(target).resolve()
    root=Path("/data/cwei/green_runs").resolve()
    if not source.is_relative_to(root) or not target.is_relative_to(root):
        raise RuntimeError("Initial-guess preparation must stay in task data")
    with case_guard(source):
        reference=json.loads((source/"thermal_hf_reference.json").read_text())
        if not reference["accepted"]: raise RuntimeError("Independent HF reference is not accepted")
        source_sha=hashlib.sha256((source/"input.h5").read_bytes()).hexdigest()
        if reference.get("input_sha256",source_sha)!=source_sha:
            raise RuntimeError("Reference Hamiltonian changed")
        target.mkdir(parents=True,exist_ok=False)
        for name in ("input.h5","molecule.json","manifest.json"):
            shutil.copy2(source/name,target/name)
        for name in ("cderi_mol.h5","df_hf_int","cderi_mol_native.h5"):
            if not (source/name).exists(): continue
            (target/name).symlink_to((source/name).resolve(),target_is_directory=(source/name).is_dir())
        fock=np.load(source/"thermal_hf_fock.npy")
        if fock.ndim==2: fock=fock[None]
        with h5py.File(target/"input.h5","a") as handle:
            field=handle["HF/Fock-k"]
            if field.dtype.kind=="c": value=fock[:,None].astype(field.dtype)
            else:
                value=np.zeros(field.shape,dtype=field.dtype)
                value[...,0]=fock[:,None]
            if value.shape!=field.shape: raise RuntimeError("Starting Fock dimensions do not match")
            field[...]=value
        manifest=json.loads((target/"manifest.json").read_text())
        manifest.update(input_sha256=hashlib.sha256((target/"input.h5").read_bytes()).hexdigest(),
                        starting_fock_reference=reference,starting_fock_source=str(source),
                        starting_fock_interpretation="initial self-energy only; physical H/S/QED/DF operators are unchanged")
        (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return str(target)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("source"); parser.add_argument("target")
    args=parser.parse_args(); print(prepare(args.source,args.target))
