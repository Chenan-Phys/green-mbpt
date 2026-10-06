#!/usr/bin/env python3
"""Losslessly compress dormant task checkpoints, preserving every logical object."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import h5py
import numpy as np
from cavity_run import case_guard


def logical_digest(path):
    digest=hashlib.sha256(); count=0
    def token(value): digest.update(json.dumps(value,sort_keys=True).encode()+b"\0")
    def data(value):
        array=np.asarray(value)
        if array.dtype.kind=="O":
            strings=[]
            for item in array.flat:
                if isinstance(item,bytes): strings.append({"bytes":item.hex()})
                elif isinstance(item,str): strings.append({"text":item})
                else: raise TypeError("Only primitive or string objects can be verified")
            token(strings)
        else: digest.update(array.tobytes(order="C"))
    with h5py.File(path,"r") as handle:
        def visit(name,obj):
            nonlocal count
            count+=1
            token([name,"dataset" if isinstance(obj,h5py.Dataset) else "group"])
            for key in sorted(obj.attrs):
                attribute=obj.attrs.get_id(key)
                if h5py.check_dtype(ref=attribute.dtype): raise TypeError("Reference attributes are unsupported")
                token([key,str(attribute.dtype),attribute.shape]); data(obj.attrs[key])
            if isinstance(obj,h5py.Group):
                for key in obj:
                    if not isinstance(obj.get(key,getlink=True),h5py.HardLink):
                        raise TypeError("Checkpoint soft/external links are unsupported")
            else:
                if h5py.check_dtype(ref=obj.dtype): raise TypeError("Reference datasets are unsupported")
                token([str(obj.dtype),obj.shape,obj.maxshape])
                if obj.shape is None: return
                if not obj.ndim: data(obj[()]); return
                row_bytes=max(1,int(np.prod(obj.shape[1:]))*obj.dtype.itemsize)
                block=max(1,(8*1024**2)//row_bytes)
                for start in range(0,obj.shape[0],block): data(obj[start:start+block])
        visit("/",handle); handle.visititems(visit)
    return digest.hexdigest(),count


def compress(path,replace_original=False,allow_size_increase=False):
    if socket.gethostname().split(".")[0]!="kadanoff": raise RuntimeError("Workstation only")
    original=Path(path).absolute()
    if original.is_symlink(): raise ValueError("Do not replace symlinked checkpoints")
    path=original.resolve()
    permitted=[Path("/data/cwei/green_runs/qed-validation")]
    if path.name not in ("hf.h5","gw.h5") or not any(path.is_relative_to(root) for root in permitted):
        raise ValueError("Only this task's native result checkpoints may be compressed")
    with case_guard(path.parent):
        with h5py.File(path.parent/"input.h5") as inp:
            if "QED/schema" not in inp: raise ValueError("Requires a QED case input")
        with h5py.File(path) as handle:
            arrays=[]
            handle.visititems(lambda name,obj:arrays.append(obj.compression) if isinstance(obj,h5py.Dataset) and obj.size>256 else None)
            if arrays and all(value=="gzip" for value in arrays):
                return {"path":str(path),"status":"already_compressed"}
        before=path.stat()
        if shutil.disk_usage(path.parent).free<60*1024**3+int(1.1*before.st_size):
            raise RuntimeError("Insufficient temporary space while preserving the 60 GiB floor")
        temporary=path.with_name(path.stem+".lossless-gzip.h5")
        if temporary.exists(): raise FileExistsError("Inspect the existing temporary copy first")
        record={"path":str(path),"started_utc":datetime.now(timezone.utc).isoformat(),
                "before_bytes":before.st_size,"status":"verifying_original"}
        # Reject unsupported objects before creating a replacement, then hash
        # every dataset value and attribute. The digest ignores storage layout.
        expected,count=logical_digest(path)
        command=[shutil.which("h5repack") or "h5repack","-f","SHUF","-f","GZIP=1",str(path),str(temporary)]
        try:
            subprocess.run(command,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            actual,new_count=logical_digest(temporary)
            if actual!=expected or count!=new_count:
                raise RuntimeError("Logical checkpoint identity changed; original is preserved")
            after_original=path.stat()
            if (after_original.st_size,after_original.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):
                raise RuntimeError("Original checkpoint changed during verification")
            after=temporary.stat().st_size
            if after>=before.st_size and not allow_size_increase:
                temporary.unlink()
                return {"path":str(path),"status":"no_size_reduction","logical_sha256":expected}
            if not replace_original:
                record.update(status="verified_copy_original_retained",verified_copy=str(temporary),
                    after_bytes=after,logical_sha256=actual,logical_objects_retained=count,command=command,
                    finished_utc=datetime.now(timezone.utc).isoformat())
                path.with_name(path.stem+"_lossless_storage.json").write_text(json.dumps(record,indent=2)+"\n")
                return record
            # Same-directory atomic replacement follows full bitwise value and
            # attribute verification; no iteration or scientific data is removed.
            temporary.replace(path)
            record.update(status="compressed_verified",after_bytes=after,
                          reclaimed_bytes=before.st_size-after,logical_sha256=actual,
                          logical_objects_retained=count,command=command,
                          finished_utc=datetime.now(timezone.utc).isoformat())
            path.with_name(path.stem+"_lossless_storage.json").write_text(json.dumps(record,indent=2)+"\n")
            return record
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("checkpoint",nargs="+")
    parser.add_argument("--replace-original",action="store_true",
        help="Replace verified originals only with explicit authorization for the named remote files; default keeps the originals")
    args=parser.parse_args()
    for path in args.checkpoint: print(json.dumps(compress(path,args.replace_original),indent=2),flush=True)
