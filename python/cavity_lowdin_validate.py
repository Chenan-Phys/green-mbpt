#!/usr/bin/env python3
"""Verify full-basis operator/factor conversion using cold native HF/GW runs."""
import json
from pathlib import Path
import sys
import numpy as np
from cavity_analyze import analyze
from cavity_lowdin import transform
from cavity_run import run
from cavity_thermal_hf import solve
from cavity_hf_warm_start import prepare as hf_seed


def main(source,target):
    source=Path(source).resolve(); target=Path(target).resolve()
    if not target.exists(): transform(source,target)
    spec=target.parent/(target.name+"_spec.json")
    spec.write_text((source/"manifest.json").read_text())
    # The runner compares requested fields with the converted manifest. Its
    # provenance-only fields differ, so request just the physical specification.
    data=json.loads(spec.read_text())
    keys=("system","atoms","basis","auxbasis","ecp","charge","spin","fix_spin",
          "omega_ev","lambda_au","origin_angstrom","polarization")
    spec.write_text(json.dumps({k:data[k] for k in keys if k in data},indent=2)+"\n")
    result=run(spec,target,energy_threshold=1e-10,restart_unconverged=True,number_tolerance=1e-12)
    out={"source":str(source),"target":str(target),"checks":{}}
    for method in ("HF","GW"):
        old=analyze(source,method.lower()+".h5")
        # analyze writes the physical AO density in either representation.
        reference=np.load(source/"density_ao.npy")
        new=analyze(target,method.lower()+".h5")
        error=float(np.linalg.norm(np.load(target/"density_ao.npy")-reference))
        difference=abs(new["energy_hartree"]-old["energy_hartree"])
        out["checks"][method]={"energy_difference_hartree":difference,"ao_density_difference":error,
                               "converted":result[method],"original":old}
        if difference>2e-9 or error>2e-7: raise AssertionError("Basis equivalence failed")
    thermal=solve(target)
    if not thermal["accepted"] or abs(thermal["energy_hartree"]-result["HF"]["energy_hartree"])>1e-8:
        raise AssertionError("Transformed independent HF reference failed")
    warm=target.parent/(target.name+"-hf-start")
    if not warm.exists(): hf_seed(target,warm)
    seeded=run(spec,warm,methods=("HF",),energy_threshold=1e-10,number_tolerance=1e-12,
               restart_unconverged=True)["HF"]
    error=abs(seeded["energy_hartree"]-result["HF"]["energy_hartree"])
    if error>1e-9: raise AssertionError("Independent HF starting Fock changed the solution")
    out.update(thermal_reference=thermal,warm_hf_energy_difference_hartree=error)
    (target.parent/(target.name+"_basis_validation.json")).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"checks":{k:{a:b for a,b in v.items() if not isinstance(b,dict)}
                                      for k,v in out["checks"].items()},
                      "warm_hf_energy_difference_hartree":error},indent=2))


if __name__=="__main__": main(sys.argv[1],sys.argv[2])
