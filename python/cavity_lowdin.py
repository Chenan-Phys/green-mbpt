#!/usr/bin/env python3
"""Make a fresh, equivalent full Lowdin representation on the workstation.

No basis function or auxiliary function is removed. The original AO factors
remain available to an independent HF-energy oracle and for real-space density.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import h5py
import numpy as np
from pyscf import lib
from scipy.linalg import eigh
from cavity_analyze import complex_array
from cavity_run import case_guard


def packed_like(value, dataset):
    if dataset.dtype.kind == "c": return value.astype(dataset.dtype)
    out=np.empty(dataset.shape,dtype=dataset.dtype)
    out[...,0]=value.real
    out[...,1]=value.imag
    return out


def hermitian_roundoff(value):
    """Remove transformation roundoff only; refuse appreciable asymmetry."""
    adjoint=np.swapaxes(value.conj(),-1,-2)
    if np.linalg.norm(value-adjoint)>1e-10*max(1.,np.linalg.norm(value)):
        raise ValueError("Operator has more than Hermitian transformation roundoff")
    return .5*(value+adjoint)


def transform(source,target):
    if socket.gethostname().split(".")[0]!="kadanoff": raise RuntimeError("Workstation only")
    source=Path(source).resolve(); target=Path(target).resolve()
    root=Path("/data/cwei/green_runs").resolve()
    if not source.is_relative_to(root) or not target.is_relative_to(root):
        raise RuntimeError("Basis conversion must stay in task data")
    with case_guard(source):
        spec=json.loads((source/"manifest.json").read_text())
        if spec.get("native_basis","ao")!="ao": raise ValueError("Source must use the original AO basis")
        with h5py.File(source/"input.h5") as inp:
            overlap=complex_array(inp["HF/S-k"])[0,0]
        if np.max(np.abs(overlap.imag))>1e-12:
            raise NotImplementedError("This exporter conversion requires real scalar molecular operators")
        values,vectors=eigh(overlap)
        if values[0]<=0: raise ValueError("Nonpositive overlap; no truncation is allowed here")
        x=(vectors/np.sqrt(values))@vectors.conj().T
        n=len(values)
        error=float(np.linalg.norm(x.conj().T@overlap@x-np.eye(n)))
        if error>1e-8: raise RuntimeError("Lowdin orthogonality check failed")
        target.mkdir(parents=True,exist_ok=False)
        for name in ("input.h5","molecule.json"):
            shutil.copy2(source/name,target/name)
        (target/"cderi_mol.h5").symlink_to((source/"cderi_mol.h5").resolve())
        with h5py.File(target/"input.h5","a") as inp:
            q=inp["QED"]
            q["native_to_ao"]=x.real
            q["original_overlap"]=overlap.real
            for name in ("dipole","second_moment"):
                original=np.asarray(q[name])
                q["original_"+name]=original
                q[name][...]=hermitian_roundoff(x.conj().T@original@x).real
            for name in ("HF/H-k","HF/Fock-k","HF/S-k"):
                ds=inp[name]
                value=complex_array(ds)
                value=np.matmul(x.conj().T,np.matmul(value,x))
                value=hermitian_roundoff(value)
                if name=="HF/S-k": value[...]=np.eye(n)
                ds[...]=packed_like(value,ds)
        # Native VQ uses flattened real/imaginary storage on its last axis.
        factors=target/"df_hf_int"; factors.mkdir()
        for original in sorted((source/"df_hf_int").glob("*.h5")):
            if not original.name.startswith("VQ_"):
                shutil.copy2(original,factors/original.name)
                continue
            shutil.copy2(original,factors/original.name)
            with h5py.File(factors/original.name,"a") as handle:
                for key in handle:
                    ds=handle[key]
                    if not isinstance(ds,h5py.Dataset): continue
                    if ds.shape[-2:]!=(n,2*n) or ds.ndim!=4:
                        raise ValueError(f"Unsupported native DF layout: {original.name}/{key} {ds.shape}")
                    for k in range(ds.shape[0]):
                        for start in range(0,ds.shape[1],32):
                            stop=min(start+32,ds.shape[1])
                            raw=np.asarray(ds[k,start:stop]).reshape(-1,n,n,2)
                            value=raw[...,0]+1j*raw[...,1]
                            value=np.matmul(x.conj().T,np.matmul(value,x))
                            value=hermitian_roundoff(value)
                            raw[...,0]=value.real; raw[...,1]=value.imag
                            ds[k,start:stop]=raw.reshape(stop-start,n,2*n)
        # A separately transformed PySCF cache permits thermal HF in this same
        # representation; the original AO cache is never changed.
        with h5py.File(source/"cderi_mol.h5") as old,h5py.File(target/"cderi_mol_native.h5","w") as new:
            for key,value in old.attrs.items(): new.attrs[key]=value
            def copy(name,obj):
                if isinstance(obj,h5py.Group):
                    group=new.require_group(name)
                    for key,value in obj.attrs.items(): group.attrs[key]=value
                elif obj.ndim==2 and obj.shape[1]==n*(n+1)//2:
                    ds=new.create_dataset(name,shape=obj.shape,dtype=obj.dtype)
                    for start in range(0,obj.shape[0],32):
                        stop=min(start+32,obj.shape[0])
                        v=lib.unpack_tril(np.asarray(obj[start:stop]))
                        v=np.matmul(x.conj().T,np.matmul(v,x))
                        v=hermitian_roundoff(v)
                        if np.max(np.abs(v.imag))>1e-10: raise ValueError("Expected real scalar molecular factors")
                        ds[start:stop]=lib.pack_tril(v.real)
                    for key,value in obj.attrs.items(): ds.attrs[key]=value
                else: old.copy(obj,new,name=name)
            old.visititems(copy)
        spec.update(native_basis="lowdin",native_to_ao_interpretation="full symmetric S^-1/2; no discarded functions",
                    basis_conversion_source=str(source),basis_conversion_source_input_sha256=spec["input_sha256"],
                    original_overlap_condition_number=float(values[-1]/values[0]),
                    basis_orthogonality_error=error,
                    input_sha256=hashlib.sha256((target/"input.h5").read_bytes()).hexdigest())
        (target/"manifest.json").write_text(json.dumps(spec,indent=2)+"\n")
        if (source/"hf.h5").exists():
            with h5py.File(source/"hf.h5") as checkpoint:
                it=max(int(k[4:]) for k in checkpoint if k.startswith("iter") and k[4:].isdigit())
                spin=-complex_array(checkpoint[f"iter{it}/G_tau/data"][-1])[:,0].real
            root_s=(vectors*np.sqrt(values))@vectors.conj().T
            spin=np.matmul(root_s,np.matmul(spin,root_s.conj().T)).real
            density=spin if len(spin)==2 else 2*spin[0]
            np.save(target/"thermal_hf_density.npy",density)
            (target/"initial_density_provenance.json").write_text(json.dumps({"source":str(source),
                "source_iteration":it,"interpretation":"density initial guess only; acceptance is not inherited"},indent=2)+"\n")
        return {"target":str(target),"overlap_condition_number":spec["original_overlap_condition_number"],
                "orthogonality_error":error}


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("source"); parser.add_argument("target")
    args=parser.parse_args(); print(json.dumps(transform(args.source,args.target),indent=2))
