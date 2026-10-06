#!/usr/bin/env python3
"""Plot measured pilot diagnostics; no unpublished calculation is inferred."""
import argparse
import json
from pathlib import Path
import socket
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(); parser.add_argument("full"); parser.add_argument("minimal"); parser.add_argument("output")
parser.add_argument("--matched-aux")
args=parser.parse_args()
if socket.gethostname().split(".")[0]!="kadanoff": raise RuntimeError("Workstation only")
full=json.loads(Path(args.full).read_text())["cases"]
minimal=json.loads(Path(args.minimal).read_text())["cases"]
energy=[]; charge=[]
entries=[(full,"R4"),(full,"R8"),(minimal,"R4")]
labels=["def2-SVP/default\n4 Å","def2-SVP/default\n8 Å","STO-3G/Weigend\n4 Å"]
colors=["#266b86","#6399a4","#b2783a"]
if args.matched_aux:
    entries.append((json.loads(Path(args.matched_aux).read_text())["cases"],"R4"))
    labels.append("STO-3G/default\n4 Å")
    colors.append("#865946")
for cases,label in entries:
    zero=cases[f"{label}_lambda0"]["methods"]["HF"]
    plus=cases[f"{label}_lambda0.005"]["methods"]["HF"]
    energy.append((plus["energy_hartree"]-zero["energy_hartree"])*27211.386245988)
    charge.append((sum(plus["population_by_atom"][14:])-sum(zero["population_by_atom"][14:]))*1e5)
fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
axes[0].bar(labels,energy,color=colors)
axes[0].set_ylabel("Cavity HF total-energy shift (meV)")
for i,value in enumerate(energy): axes[0].text(i,value+.35,f"{value:.3f}",ha="center")
axes[0].set_ylim(0,28)
axes[1].bar(labels,charge,color=colors)
axes[1].axhline(0,color="0.3",linewidth=.7)
axes[1].set_ylabel("Acceptor Lowdin response (10⁻⁵ electron)")
for i,value in enumerate(charge):
    label="unresolved" if abs(value)<1e-4 else f"{value:.3g}"
    axes[1].text(i,value-.10,label,ha="center",va="top")
axes[1].set_ylim(-2.3,.2)
fig.suptitle("TTF–TCNE HF diagnostics: λ = 0.005 a.u., ω = 2 eV, β = 1000 Ha⁻¹",fontsize=12)
for ax in axes:
    ax.spines[["top","right"]].set_visible(False)
output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(output,dpi=180)
fig.savefig(output.with_suffix(".svg"))
print(output)
