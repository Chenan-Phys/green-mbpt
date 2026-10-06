#!/usr/bin/env python3
"""Check the unrestricted cavity energy and charge bookkeeping against finite CI."""
import json
from pathlib import Path
import sys
import numpy as np
from cavity_run import run
from cavity_validate import exact_reference, hf_expectation


def main(root):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    specs=root/"specs"; specs.mkdir(exist_ok=True)
    table={}
    for charge in (1,-1):
        for coupling in (0.,.005):
            label=f"H2_charge{charge:+d}_lambda{coupling:g}"
            spec={"system":"H2 charged validation", "atoms":[["H",0,0,-.37],["H",0,0,.37]],
                  "basis":"sto-3g","charge":charge,"spin":1,"omega_ev":2.,
                  "lambda_au":coupling,"origin_angstrom":[0,0,0],"polarization":[0,0,1],"fix_spin":True}
            path=specs/f"{label}.json"; path.write_text(json.dumps(spec,indent=2)+"\n")
            directory=root/label
            measurements=run(path,directory,energy_threshold=1e-10)
            measurements["independent_hf_energy"]=hf_expectation(directory)
            measurements["hf_expectation_error"]=abs(measurements["HF"]["energy_hartree"]-
                                                     measurements["independent_hf_energy"])
            measurements["exact_cut8"]=exact_reference(directory,coupling,8)
            measurements["exact_cut16"]=exact_reference(directory,coupling,16)
            ne=2-charge
            expected=sorted([(ne+1)/2,(ne-1)/2])
            for method in ("HF","GW"):
                if not np.allclose(sorted(measurements[method]["spin_electrons"]),expected,atol=1e-5,rtol=0):
                    raise AssertionError(f"Unexpected spin occupation: {label} {method}")
            if measurements["hf_expectation_error"]>1e-7:
                raise AssertionError("Unrestricted energy estimator mismatch")
            if abs(measurements["exact_cut8"]-measurements["exact_cut16"])>1e-9:
                raise AssertionError("Photon cutoff unconverged")
            table[label]=measurements
            (root/"charged_validation.json").write_text(json.dumps(table,indent=2)+"\n")
    print(json.dumps({k:{"HF_error":v["hf_expectation_error"],
                        "GW_spin_electrons":v["GW"]["spin_electrons"]} for k,v in table.items()},indent=2))


if __name__=="__main__":
    main(sys.argv[1])
