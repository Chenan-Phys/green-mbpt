#!/usr/bin/env python3
"""Check the connected photon coordinate convention against independent finite CI."""
import json
from pathlib import Path
import socket
import sys
import h5py
from cavity_analyze import analyze
from cavity_validate import exact_reference

if socket.gethostname().split(".")[0]!="kadanoff": raise RuntimeError("Workstation only")
directory=Path(sys.argv[1]).resolve()
with h5py.File(directory/"input.h5") as f: coupling=float(f["QED/lambda_au"][()])
a=exact_reference(directory,coupling,8,observables=True)
b=exact_reference(directory,coupling,16,observables=True)
if max(abs(a[k]-b[k]) for k in a)>1e-9: raise AssertionError("Photon cutoff not converged")
gw=analyze(directory,"gw.h5")
out={"exact_cut8":a,"exact_cut16":b,"GW_connected_photon_q_variance":gw["Photon_variance"],
     "q_variance_difference":gw["Photon_variance"]-b["connected_photon_q_variance"],
     "interpretation":"GW approximation versus unfitted finite-CI reference; the GW energy-correction field is E_photon + E_bilinear/2, not photon occupation energy"}
if b["connected_photon_q_variance"]<=1 or gw["Photon_variance"]<=1:
    raise AssertionError("Unexpected coordinate-variance sign in this weak-coupling H2 benchmark")
(directory.parent/"photon_variance_validation.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
