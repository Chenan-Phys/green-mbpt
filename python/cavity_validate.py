#!/usr/bin/env python3
"""Executable PF validation: native recovery, independent HF and polaritonic FCI."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from pyscf import ao2mo, fci, gto, scf
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh
from cavity_run import run
from cavity_analyze import complex_array


def exact_reference(directory, coupling, cutoff):
    mol=gto.loads((directory/"molecule.json").read_text())
    with h5py.File(directory/"input.h5") as f:
        s=complex_array(f["HF/S-k"])[0,0].real
        d=np.asarray(f["QED/dipole"])
        r2=np.asarray(f["QED/second_moment"])
        omega=float(f["QED/omega_hartree"][()])
    values,vectors=eigh(s)
    x=(vectors/np.sqrt(values))@vectors.T
    h=x.T@scf.RHF(mol).get_hcore()@x
    eri=ao2mo.kernel(mol,x,compact=False).reshape((mol.nao_nr(),)*4)
    norb=mol.nao_nr()
    nelec=mol.nelec
    na=math.comb(norb,nelec[0]); nb=math.comb(norb,nelec[1]); dim=na*nb
    h2=fci.direct_spin1.absorb_h1e(h,eri,norb,nelec,.5)
    d=x.T@d@x
    correction=x.T@r2@x-d@d
    he=np.empty((dim,dim)); q=np.empty((dim,dim)); c=np.empty((dim,dim))
    for i in range(dim):
        vector=np.eye(dim)[:,i].reshape(na,nb)
        he[:,i]=fci.direct_spin1.contract_2e(h2,vector,norb,nelec).ravel()
        q[:,i]=fci.direct_spin1.contract_1e(d,vector,norb,nelec).ravel()
        c[:,i]=fci.direct_spin1.contract_1e(correction,vector,norb,nelec).ravel()
    e0,psi=eigh(he,subset_by_index=[0,0])
    # A fixed exact coherent shift reduces cutoff error, without changing eigenvalues.
    q=q-np.eye(dim)*float(psi[:,0]@q@psi[:,0])
    he=he+np.eye(dim)*mol.energy_nuc()+.5*coupling**2*(q@q+c)
    b=diags(np.sqrt(np.arange(1,cutoff)),1,shape=(cutoff,cutoff))
    ham=kron(csr_matrix(he),eye(cutoff))+kron(eye(dim),diags(omega*np.arange(cutoff)))
    ham+=kron(csr_matrix(-np.sqrt(omega/2)*coupling*q),b+b.T)
    return float(eigsh(ham,k=1,which="SA",tol=1e-12,return_eigenvectors=False)[0])


def hf_expectation(directory):
    mol=gto.loads((directory/"molecule.json").read_text())
    mf=(scf.UHF(mol) if mol.spin else scf.RHF(mol)).density_fit()
    mf.with_df._cderi=str(directory/"cderi_mol.h5")
    with h5py.File(directory/"hf.h5") as f, h5py.File(directory/"input.h5") as inp:
        iteration=max(int(k[4:]) for k in f if k.startswith("iter") and k[4:].isdigit())
        g=complex_array(f[f"iter{iteration}/G_tau/data"])
        dm_spin=-g[-1,:,0].real
        dm=2*dm_spin[0] if g.shape[1]==1 else dm_spin.sum(axis=0)
        d=np.asarray(inp["QED/dipole"]); r2=np.asarray(inp["QED/second_moment"])
        coupling=float(inp["QED/lambda_au"][()])
    if mol.spin:
        j,k=mf.get_jk(dm=dm_spin)
        electronic=np.trace((mf.get_hcore()+.5*j.sum(axis=0))@dm)
        electronic-=.5*sum(np.trace(ks@ds) for ks,ds in zip(k,dm_spin))
        exchange=sum(np.trace(d@ds@d@ds) for ds in dm_spin)
    else:
        j,k=mf.get_jk(dm=dm)
        electronic=np.einsum("ij,ji->",mf.get_hcore()+.5*j-.25*k,dm)
        exchange=.5*np.trace(d@dm@d@dm)
    electronic+=mol.energy_nuc()
    dse=.5*coupling**2*(np.trace(r2@dm)-exchange)
    return float(electronic+dse)


def main(root):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    specs=root/"specs"; specs.mkdir(exist_ok=True)
    table={}
    for label,coupling,origin in [("zero",0.,[0,0,0]),("plus",.005,[0,0,0]),
                                  ("minus",-.005,[0,0,0]),("origin",.005,[0,0,1])]:
        spec={"system":"H2 validation","atoms":[["H",0,0,-.37],["H",0,0,.37]],
              "basis":"sto-3g","charge":0,"spin":0,"omega_ev":2.,
              "lambda_au":coupling,"origin_angstrom":origin,"polarization":[0,0,1]}
        path=specs/f"{label}.json"; path.write_text(json.dumps(spec,indent=2)+"\n")
        directory=root/label
        results=run(path,directory,energy_threshold=1e-10)
        results["independent_hf_energy"]=hf_expectation(directory)
        results["hf_expectation_error"]=abs(results["HF"]["energy_hartree"]-results["independent_hf_energy"])
        results["exact_cut8"]=exact_reference(directory,coupling,8)
        results["exact_cut16"]=exact_reference(directory,coupling,16)
        table[label]=results
        (root/"validation_results.json").write_text(json.dumps(table,indent=2)+"\n")
    for label in table:
        if table[label]["hf_expectation_error"]>1e-7:
            raise AssertionError(f"HF expectation mismatch: {label}")
        if abs(table[label]["exact_cut8"]-table[label]["exact_cut16"])>1e-9:
            raise AssertionError(f"Photon cutoff unconverged: {label}")
    for method in ("HF","GW"):
        parity=abs(table["plus"][method]["energy_hartree"]-table["minus"][method]["energy_hartree"])
        table[f"{method}_parity_error"]=parity
        if parity>1e-9:
            raise AssertionError("Coupling parity failed")
        table[f"{method}_origin_energy_change"]=table["origin"][method]["energy_hartree"]-table["plus"][method]["energy_hartree"]
    table["GW_cavity_shift"]=table["plus"]["GW"]["energy_hartree"]-table["zero"]["GW"]["energy_hartree"]
    table["exact_cavity_shift"]=table["plus"]["exact_cut16"]-table["zero"]["exact_cut16"]
    (root/"validation_results.json").write_text(json.dumps(table,indent=2)+"\n")
    print(json.dumps({k:v for k,v in table.items() if not isinstance(v,dict)},indent=2))


if __name__=="__main__":
    main(sys.argv[1])
