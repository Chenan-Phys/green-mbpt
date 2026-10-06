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
        end = complex_array(last["G_tau/data"][-1])
        ns = end.shape[0]
        dm_spin = -end[:, 0]
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
        for key in ("Photon_variance", "Photon_energy_correction", "Coherent_b", "Bosonic_residual", "Density_residual_frobenius"):
            if f"QED/{key}" in last:
                out[key] = float(last[f"QED/{key}"][()])
        for key in ("Mu_spin","Target_spin_electrons"):
            if f"QED/{key}" in last:
                out[key]=np.asarray(last[f"QED/{key}"]).tolist()
        if iteration > 1:
            prev = f[f"iter{iteration-1}"]
            ep = float(prev["Energy_HF"][()])+float(prev["Energy_2b"][()])
            out["last_energy_change_hartree"] = abs(energy-ep)
            out["native_energy_residual"] = sum(abs(float(last[k][()])-float(prev[k][()]))
                for k in ("Energy_1b", "Energy_HF", "Energy_2b"))
            old_spin=-complex_array(prev["G_tau/data"][-1])[:,0]
            weight=2 if ns==1 else 1
            out["density_residual_frobenius_electrons"]=float(np.sqrt(sum(
                np.linalg.norm(root@(weight*(x-y))@root)**2 for x,y in zip(dm_spin,old_spin))))
        requested_threshold=1e-8
        method=Path(result).stem
        stamp_path=directory/f"{method}_provenance.json"
        if stamp_path.exists():
            command=json.loads(stamp_path.read_text()).get("command",[])
            if "--E_thr" in command:
                requested_threshold=min(1e-8,float(command[command.index("--E_thr")+1]))
        out["requested_energy_threshold_hartree"]=requested_threshold
        out["accepted_numerically"] = (abs(ne-spec["nelectron"]) < 1e-6 and
             out.get("native_energy_residual", float("inf")) < requested_threshold and
             out.get("density_residual_frobenius_electrons", float("inf")) < 1e-6 and
             out.get("Bosonic_residual", 0.0) < 1e-8)
        if spec.get("fix_spin",False):
            out["accepted_numerically"] &= out["accepted_for_requested_spin_sector"]
        out["native_basis"]=spec.get("native_basis","ao")
        ao_dm=dm
        if "QED/native_to_ao" in inp:
            x=np.asarray(inp["QED/native_to_ao"])
            ao_dm=x@dm@x.conj().T
            np.save(directory/"density_native.npy",dm)
    np.save(directory/"density_ao.npy", ao_dm)
    # Both HF and GW have a static HF energy part. Check it independently;
    # energy/density changes alone can stagnate after DIIS extrapolation.
    from cavity_hf_reference import hf_expectation
    out["hf_functional_energy_hartree"]=hf_expectation(directory,result)
    with h5py.File(directory/result) as f:
        reported_hf=float(f[f"iter{iteration}/Energy_HF"][()])
    out["hf_functional_consistency_error_hartree"]=abs(reported_hf-
                                                     out["hf_functional_energy_hartree"])
    out["accepted_numerically"] &= out["hf_functional_consistency_error_hartree"]<1e-7
    (directory/"analysis.json").write_text(json.dumps(out, indent=2)+"\n")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--result", default="sim.h5")
    args = p.parse_args()
    print(json.dumps(analyze(args.directory,args.result), indent=2))
