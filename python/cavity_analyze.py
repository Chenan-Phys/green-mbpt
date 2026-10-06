#!/usr/bin/env python3
"""Read native molecular QED outputs and enforce a minimal pilot acceptance check."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.linalg import eigh


def complex_array(dataset):
    a = np.asarray(dataset)
    if a.dtype.kind == "c":
        return a
    if a.shape[-1] == 2:
        return a[..., 0] + 1j*a[..., 1]
    return a.astype(complex)


def analyze(directory, result="sim.h5"):
    directory = Path(directory)
    spec = json.loads((directory/"manifest.json").read_text())
    with h5py.File(directory/result) as f, h5py.File(directory/"input.h5") as inp:
        iteration = max(int(k[4:]) for k in f if k.startswith("iter") and k[4:].isdigit())
        last = f[f"iter{iteration}"]
        g = complex_array(last["G_tau/data"])
        ns = g.shape[1]
        dm_spin = -g[-1, :, 0]
        dm = dm_spin.sum(axis=0)*(2 if ns == 1 else 1)
        overlap = complex_array(inp["HF/S-k"])[0, 0]
        ne = np.trace(dm@overlap).real
        ev, vectors = eigh(overlap)
        if ev[0] <= 0:
            raise ValueError("Nonpositive overlap")
        root = (vectors*np.sqrt(ev))@vectors.conj().T
        orth_dm = root@dm@root
        d = np.asarray(inp["QED/dipole"])
        energy = float(last["Energy_HF"][()])+float(last["Energy_2b"][()])
        out = {"iteration": iteration, "energy_hartree": energy, "nelectron": ne,
               "expected_nelectron": spec["nelectron"],
               "spin_electrons": ([float(np.trace(dm_spin[0]@overlap).real)]*2 if ns==1 else
                                  [float(np.trace(x@overlap).real) for x in dm_spin]),
               "dipole_projected_au": float(inp["QED/nuclear_dipole"][()])+np.trace(d@dm).real,
               "population_by_atom": [], "omega_ev": spec.get("omega_ev", 2.0),
               "lambda_au": spec.get("lambda_au", 0.0)}
        expected_spin=sorted([(spec["nelectron"]+spec.get("spin",0))/2,
                              (spec["nelectron"]-spec.get("spin",0))/2])
        out["spin_sector_error"]=float(np.max(np.abs(np.array(sorted(out["spin_electrons"]))-expected_spin)))
        out["accepted_for_requested_spin_sector"]=out["spin_sector_error"]<1e-5
        ao_atoms = np.asarray(inp["QED/ao_atom_index"])
        out["population_by_atom"] = [float(np.diag(orth_dm)[ao_atoms==i].real.sum())
                                      for i in range(len(spec["atoms"]))]
        for key in ("Photon_variance", "Photon_energy_correction", "Coherent_b", "Bosonic_residual"):
            if f"QED/{key}" in last:
                out[key] = float(last[f"QED/{key}"][()])
        if iteration > 1:
            prev = f[f"iter{iteration-1}"]
            ep = float(prev["Energy_HF"][()])+float(prev["Energy_2b"][()])
            out["last_energy_change_hartree"] = abs(energy-ep)
            out["native_energy_residual"] = sum(abs(float(last[k][()])-float(prev[k][()]))
                for k in ("Energy_1b", "Energy_HF", "Energy_2b"))
        out["accepted_numerically"] = (abs(ne-spec["nelectron"]) < 1e-6 and
             out.get("native_energy_residual", float("inf")) < 1e-8 and
             out.get("Bosonic_residual", 0.0) < 1e-8)
    np.save(directory/"density_ao.npy", dm)
    (directory/"analysis.json").write_text(json.dumps(out, indent=2)+"\n")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--result", default="sim.h5")
    args = p.parse_args()
    print(json.dumps(analyze(args.directory,args.result), indent=2))
