#!/usr/bin/env python3
"""Independent finite-temperature coherent PF HF with the same saved DF factors.

This checks the native HF fixed point, rather than merely re-evaluating its energy.
Mean spin populations are constrained; this is a grand-canonical thermal HF state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import shutil
import h5py
import numpy as np
from pyscf import gto, scf
from pyscf.scf import addons
from pyscf.scf import _response_functions
from scipy.linalg import eigh
from cavity_analyze import complex_array, analyze
from cavity_run import case_guard


def _solve(directory, beta=1000.,newton_polish=False,gradient_tolerance=1e-7):
    if socket.gethostname().split(".")[0] != "kadanoff":
        raise RuntimeError("Scientific calculations must run on GREEN_workstation")
    directory=Path(directory).resolve()
    if not 0<gradient_tolerance<=1e-6:
        raise ValueError("Reference gradient tolerance must be at most 1e-6 Ha")
    mol=gto.loads((directory/"molecule.json").read_text())
    with h5py.File(directory/"input.h5") as inp:
        d=np.asarray(inp["QED/dipole"])
        r2=np.asarray(inp["QED/second_moment"])
        coupling=float(inp["QED/lambda_au"][()])
        h=complex_array(inp["HF/H-k"])[0,0].real
        overlap=complex_array(inp["HF/S-k"])[0,0].real
    mf=(scf.UHF(mol) if mol.spin else scf.RHF(mol)).density_fit()
    mf.with_df._cderi=str(directory/("cderi_mol_native.h5" if (directory/"cderi_mol_native.h5").exists() else "cderi_mol.h5"))
    original_veff=mf.get_veff
    def get_veff(mol=None, dm=None, *args, **kwargs):
        if dm is None: dm=mf.make_rdm1()
        dm=np.asarray(dm)
        # Build the full Coulomb/exchange potential. An incremental update
        # with vhf_last would also retain the previous custom PF correction.
        value=original_veff(mol,dm,hermi=kwargs.get("hermi",1))
        correction=np.asarray([d@x@d for x in dm]) if mf.istype("UHF") else .5*d@dm@d
        return value-coupling**2*correction
    mf.get_veff=get_veff
    mf.get_hcore=lambda mol=None: h+.5*coupling**2*r2
    mf.get_ovlp=lambda mol=None: overlap
    mf=addons.smearing_(mf,sigma=1/beta,method="fermi",fix_spin=bool(mol.spin))
    mf.conv_tol=1e-11
    mf.conv_tol_grad=gradient_tolerance
    mf.max_cycle=40 if newton_polish else 400
    mf.diis_space=12
    dm0=None
    native_spin=None
    seed_path=directory/"thermal_hf_density.npy"
    seed_sha=hashlib.sha256(seed_path.read_bytes()).hexdigest() if seed_path.exists() else None
    if seed_path.exists(): dm0=np.load(seed_path)
    if (directory/"hf.h5").exists():
        with h5py.File(directory/"hf.h5") as native:
            it=max(int(k[4:]) for k in native if k.startswith("iter") and k[4:].isdigit())
            native_spin=-complex_array(native[f"iter{it}/G_tau/data"][-1])[:,0].real
            dm0=native_spin if mol.spin else 2*native_spin[0]
    if dm0 is None:
        with h5py.File(directory/"input.h5") as inp:
            starting_fock=complex_array(inp["HF/Fock-k"])[:,0].real
        pairs=[eigh(f,overlap) for f in starting_fock]
        if mol.spin:
            eps=np.asarray([p[0] for p in pairs]); coefficients=np.asarray([p[1] for p in pairs])
        else: eps,coefficients=pairs[0]
        dm0=mf.make_rdm1(coefficients,mf.get_occ(eps,coefficients))
    # Bypass PySCF's one-electron shortcut, which omits the PF iteration.
    energy=scf.hf.SCF.scf(mf,dm0=dm0)
    newton_used=False
    if not mf.converged and newton_polish:
        if abs(mf.entropy)>1e-8:
            raise RuntimeError("Newton preconditioning currently requires a gapped integer-occupation HF state")
        mf=mf.undo_smearing()
        original_response=mf.gen_response
        def gen_response(*args,**kwargs):
            response=original_response(*args,**kwargs)
            def apply(dm1):
                correction=np.matmul(d,np.matmul(dm1,d))
                return response(dm1)-coupling**2*(correction if mol.spin else .5*correction)
            return apply
        mf.gen_response=gen_response
        mf=mf.newton()
        mf.max_cycle=100
        mf.kernel(mf.mo_coeff,mf.mo_occ)
        newton_used=True
        mf=addons.smearing_(mf.undo_soscf(),sigma=1/beta,method="fermi",fix_spin=bool(mol.spin))
        mf.max_cycle=100
        energy=scf.hf.SCF.scf(mf,dm0=mf.make_rdm1())
    dm=mf.make_rdm1()
    spin_dm=dm if mol.spin else np.asarray([dm/2,dm/2])
    values,vectors=eigh(overlap)
    root=(vectors*np.sqrt(values))@vectors.T
    inverse_root=(vectors/np.sqrt(values))@vectors.T
    raw_fock=mf.get_hcore()+mf.get_veff(dm=dm)
    fs=raw_fock if mol.spin else np.asarray([raw_fock,raw_fock])
    raw_commutator=max(np.linalg.norm(f@x@overlap-overlap@x@f) for f,x in zip(fs,spin_dm))
    commutator=max(np.linalg.norm(inverse_root@(f@x@overlap-overlap@x@f)@inverse_root) for f,x in zip(fs,spin_dm))
    out={"converged":bool(mf.converged),"energy_hartree":float(energy),"newton_preconditioning":newton_used,
         "initial_density_sha256":seed_sha,
         "input_sha256":hashlib.sha256((directory/"input.h5").read_bytes()).hexdigest(),
         "free_energy_hartree":float(mf.e_free),"entropy_kb":float(mf.entropy),
         "beta_hartree_inverse":beta,"spin_electrons":[float(np.trace(x@overlap)) for x in spin_dm],
         "fock_density_commutator_norm":float(commutator),
         "raw_matrix_commutator_norm":float(raw_commutator),
         "commutator_metric":"orthonormal representation, Frobenius norm in Ha",
         "requested_orbital_gradient_tolerance_hartree":gradient_tolerance,
         "native_basis":json.loads((directory/"manifest.json").read_text()).get("native_basis","ao"),
         "interpretation":"independent same-DF thermal PF HF; internal energy and Helmholtz functional differ"}
    if mol.spin: out["spin_squared_hf"]=float(mf.spin_square()[0])
    if native_spin is not None:
        target=native_spin if mol.spin else np.asarray([native_spin[0],native_spin[0]])
        out["native_density_difference_frobenius_electrons"]=float(np.sqrt(sum(
            np.linalg.norm(root@(x-y)@root)**2 for x,y in zip(spin_dm,target))))
        native=analyze(directory,"hf.h5")
        out["native_energy_difference_hartree"]=float(energy-native["energy_hartree"])
        out["native_accepted_numerically"]=native["accepted_numerically"]
    expected=np.asarray(mol.nelec,dtype=float)
    out["accepted"]=bool(mf.converged and commutator<1e-6 and
                         np.max(np.abs(np.asarray(out["spin_electrons"])-expected))<1e-6)
    np.save(directory/"thermal_hf_density.npy",dm)
    np.save(directory/"thermal_hf_fock.npy",raw_fock)
    (directory/"thermal_hf_reference.json").write_text(json.dumps(out,indent=2)+"\n")
    return out


def solve(directory,beta=1000.,newton_polish=False,gradient_tolerance=1e-7):
    with case_guard(directory) as directory:
        return _solve(directory,beta,newton_polish,gradient_tolerance)


def clone_reference(source,target):
    if socket.gethostname().split(".")[0]!="kadanoff": raise RuntimeError("Workstation only")
    source=Path(source).resolve(); target=Path(target).resolve()
    root=Path("/data/cwei/green_runs").resolve()
    if not source.is_relative_to(root) or not target.is_relative_to(root):
        raise RuntimeError("Reference must remain inside task data")
    target.mkdir(parents=True,exist_ok=False)
    for name in ("input.h5","molecule.json","manifest.json"):
        shutil.copy2(source/name,target/name)
    for name in ("cderi_mol.h5","df_hf_int","cderi_mol_native.h5"):
        if not (source/name).exists(): continue
        (target/name).symlink_to((source/name).resolve(),target_is_directory=(source/name).is_dir())
    if (source/"thermal_hf_density.npy").exists():
        shutil.copy2(source/"thermal_hf_density.npy",target/"thermal_hf_density.npy")
    (target/"clone_provenance.json").write_text(json.dumps({"source":str(source),
        "purpose":"independent reference; immutable Hamiltonian/integrals reused; optional unconverged density is only an initial guess"},indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--beta",type=float,default=1000.)
    parser.add_argument("--newton-polish",action="store_true")
    parser.add_argument("--gradient-tolerance",type=float,default=1e-7)
    parser.add_argument("--clone-from")
    args=parser.parse_args()
    if args.clone_from: clone_reference(args.clone_from,args.directory)
    print(json.dumps(solve(args.directory,args.beta,args.newton_polish,args.gradient_tolerance),indent=2))
