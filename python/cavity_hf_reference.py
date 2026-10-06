"""Independent same-DF coherent HF energy evaluated at a native density."""
from pathlib import Path
import socket
import h5py
import numpy as np
from pyscf import gto, scf


def _complex(dataset):
    value=np.asarray(dataset)
    return value if value.dtype.kind=="c" else value[...,0]+1j*value[...,1]


def hf_expectation(directory,result="hf.h5"):
    if socket.gethostname().split(".")[0]!="kadanoff":
        raise RuntimeError("Scientific reference calculations must run on GREEN_workstation")
    directory=Path(directory)
    mol=gto.loads((directory/"molecule.json").read_text())
    mf=(scf.UHF(mol) if mol.spin else scf.RHF(mol)).density_fit()
    mf.with_df._cderi=str(directory/"cderi_mol.h5")
    with h5py.File(directory/result) as f, h5py.File(directory/"input.h5") as inp:
        iteration=max(int(k[4:]) for k in f if k.startswith("iter") and k[4:].isdigit())
        end=_complex(f[f"iter{iteration}/G_tau/data"][-1])
        dm_spin=-end[:,0].real
        native_spin=dm_spin.copy()
        native_dm=2*native_spin[0] if end.shape[0]==1 else native_spin.sum(axis=0)
        if "QED/native_to_ao" in inp:
            x=np.asarray(inp["QED/native_to_ao"])
            dm_spin=np.matmul(x,np.matmul(dm_spin,x.T))
        dm=2*dm_spin[0] if end.shape[0]==1 else dm_spin.sum(axis=0)
        d=np.asarray(inp["QED/dipole"]); r2=np.asarray(inp["QED/second_moment"])
        coupling=float(inp["QED/lambda_au"][()])
    if mol.spin:
        j,k=mf.get_jk(dm=dm_spin)
        electronic=np.trace((mf.get_hcore()+.5*j.sum(axis=0))@dm)
        electronic-=.5*sum(np.trace(ks@ds) for ks,ds in zip(k,dm_spin))
        exchange=sum(np.trace(d@ds@d@ds) for ds in native_spin)
    else:
        j,k=mf.get_jk(dm=dm)
        electronic=np.einsum("ij,ji->",mf.get_hcore()+.5*j-.25*k,dm)
        exchange=.5*np.trace(d@native_dm@d@native_dm)
    electronic+=mol.energy_nuc()
    return float(electronic+.5*coupling**2*(np.trace(r2@native_dm)-exchange))
