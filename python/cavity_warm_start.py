#!/usr/bin/env python3
"""Seed a new coupling point from a converged same-system GW solution.

The source is preserved. Only an initial guess is transferred; the target
Hamiltonian is loaded from its own input and must converge independently.
"""
import json
from pathlib import Path
import h5py
from cavity_analyze import analyze
from cavity_run import case_guard


def seed(source,target):
    source=Path(source).resolve(); target=Path(target).resolve()
    root=Path("/data/cwei/green_runs").resolve()
    if not source.is_relative_to(root) or not target.is_relative_to(root):
        raise RuntimeError("Initial guesses must stay in task data")
    with case_guard(source),case_guard(target):
        if (target/"gw.h5").exists(): raise RuntimeError("Target GW checkpoint already exists")
        if not (source/"gw.h5").exists(): return {"seeded":False,"reason":"source GW not available"}
        value=analyze(source,"gw.h5")
        if not value["accepted_numerically"]:
            return {"seeded":False,"reason":"source GW did not pass acceptance"}
        left=json.loads((source/"manifest.json").read_text())
        right=json.loads((target/"manifest.json").read_text())
        keys=("atoms","basis","auxbasis","ecp","charge","spin","fix_spin","omega_ev",
              "origin_angstrom","polarization","nao","nelectron","native_basis")
        if any(left.get(k)!=right.get(k) for k in keys):
            raise RuntimeError("Coupling warm start requires the same system, basis, cavity and spin sector")
        temporary=target/"gw_initial_guess.h5"
        if temporary.exists(): raise RuntimeError("Inspect existing initial-guess file first")
        try:
            with h5py.File(source/"gw.h5","r") as old,h5py.File(temporary,"w") as new:
                iterations=sorted(int(k[4:]) for k in old if k.startswith("iter") and k[4:].isdigit())
                keep={f"iter{i}" for i in iterations[-2:]}
                for key in old:
                    if not (key.startswith("iter") and key[4:].isdigit()) or key in keep:
                        old.copy(key,new)
                for key,value in old.attrs.items(): new.attrs[key]=value
            temporary.rename(target/"gw.h5")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        provenance=json.loads((source/"gw_provenance.json").read_text())
        record={"seeded":True,"source":str(source),"source_lambda_au":left["lambda_au"],
                "target_lambda_au":right["lambda_au"],"source_input_sha256":left["input_sha256"],
                "interpretation":"initial guess only; target must converge under its own Hamiltonian"}
        provenance.update(input_sha256=right["input_sha256"],initial_guess=record)
        (target/"gw_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
        return record
