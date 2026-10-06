#!/usr/bin/env python3
"""Check a gapped UHF reference's orbital stability with the PF response term."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import shutil
import h5py
import numpy as np
from scipy.linalg import eigh
from pyscf import gto,scf
from pyscf.scf import stability,_response_functions
from cavity_analyze import complex_array
from cavity_run import case_guard


def check(directory):
    if socket.gethostname().split(".")[0]!="kadanoff": raise RuntimeError("Workstation only")
    with case_guard(directory) as directory:
        reference=json.loads((directory/"thermal_hf_reference.json").read_text())
        if not reference["accepted"] or abs(reference["entropy_kb"])>1e-8:
            raise ValueError("Requires an accepted, gapped integer-occupation HF reference")
        if reference["input_sha256"]!=hashlib.sha256((directory/"input.h5").read_bytes()).hexdigest():
            raise ValueError("Reference input changed")
        mol=gto.loads((directory/"molecule.json").read_text())
        if not mol.spin: raise ValueError("This diagnostic currently checks unrestricted references")
        with h5py.File(directory/"input.h5") as inp:
            h=complex_array(inp["HF/H-k"])[0,0].real
            s=complex_array(inp["HF/S-k"])[0,0].real
            d=np.asarray(inp["QED/dipole"]); r2=np.asarray(inp["QED/second_moment"])
            coupling=float(inp["QED/lambda_au"][()])
        mf=scf.UHF(mol).density_fit()
        mf.with_df._cderi=str(directory/("cderi_mol_native.h5" if (directory/"cderi_mol_native.h5").exists() else "cderi_mol.h5"))
        mf.get_hcore=lambda mol=None:h+.5*coupling**2*r2
        mf.get_ovlp=lambda mol=None:s
        original_veff=mf.get_veff
        mf.get_veff=lambda mol=None,dm=None,*args,**kwargs:original_veff(mol,np.asarray(dm))-coupling**2*np.matmul(d,np.matmul(dm,d))
        original_response=mf.gen_response
        def gen_response(*args,**kwargs):
            response=original_response(*args,**kwargs)
            return lambda dm1:response(dm1)-coupling**2*np.matmul(d,np.matmul(dm1,d))
        mf.gen_response=gen_response
        pairs=[eigh(f,s) for f in np.load(directory/"thermal_hf_fock.npy")]
        mf.mo_energy=np.asarray([p[0] for p in pairs]); mf.mo_coeff=np.asarray([p[1] for p in pairs])
        mf.mo_occ=np.zeros_like(mf.mo_energy)
        for spin,n in enumerate(mol.nelec): mf.mo_occ[spin,:n]=1
        with (directory/"thermal_hf_stability.log").open("w") as log:
            mf.stdout=log; mf.verbose=4
            dimension=sum(n*(mol.nao_nr()-n) for n in mol.nelec)
            if dimension:
                mo,stable=stability.uhf_internal(mf,with_symmetry=False,return_status=True,nroots=min(3,dimension),tol=1e-8)
            else:
                mo,stable=mf.mo_coeff,True
                log.write("No internal occupied-virtual rotations in this finite orbital space.\n")
        out={"internal_orbital_stable":bool(stable),"negative_eigenvalue_threshold_hartree":-1e-5,
             "reference":reference,"interpretation":"gapped determinant UHF internal orbital Hessian with PF exchange response; not spin purity or a global-minimum test"}
        if not stable:
            np.save(directory/"unstable_mode_initial_density.npy",mf.make_rdm1(mo,mf.mo_occ))
            out["initial_guess_file"]="unstable_mode_initial_density.npy"
        (directory/"thermal_hf_stability.json").write_text(json.dumps(out,indent=2)+"\n")
        return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("directory")
    parser.add_argument("--relax-to")
    args=parser.parse_args()
    value=check(args.directory)
    if args.relax_to:
        if value["internal_orbital_stable"]: raise ValueError("Reference has no detected unstable internal mode")
        from cavity_thermal_hf import clone_reference,solve
        clone_reference(args.directory,args.relax_to)
        shutil.copy2(Path(args.directory)/"unstable_mode_initial_density.npy",Path(args.relax_to)/"thermal_hf_density.npy")
        value["relaxed_reference"]=solve(args.relax_to)
        if value["relaxed_reference"]["accepted"]:
            value["relaxed_stability"]=check(args.relax_to)
        (Path(args.relax_to)/"instability_relaxation.json").write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps(value,indent=2))
