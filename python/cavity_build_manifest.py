#!/usr/bin/env python3
"""Record the executable and edited core sources immediately after a separate build."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess

CORE_FILES=["src/dyson.cpp","src/hf_solver.cpp","src/gw_cpu_kernel.cpp",
            "src/green/mbpt/cavity.h","src/green/mbpt/kernels.h","src/green/mbpt/dyson.h"]


def core_digest(source):
    digest=hashlib.sha256()
    for name in CORE_FILES:
        digest.update(name.encode())
        digest.update((source/name).read_bytes().replace(b"\r\n",b"\n"))
    return digest.hexdigest()


if __name__=="__main__":
    if socket.gethostname().split(".")[0]!="kadanoff": raise RuntimeError("Workstation only")
    parser=argparse.ArgumentParser(); parser.add_argument("prefix"); args=parser.parse_args()
    source=Path(__file__).resolve().parent.parent; prefix=Path(args.prefix).resolve()
    result={"stable_base":"d593a9c970194dea5855ca38e71f9cd6ac550cbf",
            "core_source_files":CORE_FILES,"core_source_sha256_lf":core_digest(source),
            "binary_sha256":hashlib.sha256((prefix/"bin/mbpt.exe").read_bytes()).hexdigest(),
            "source_commit":subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip(),
            "source_working_tree_changed":bool(subprocess.check_output(
                ["git","-C",str(source),"status","--porcelain"],text=True).strip())}
    (prefix/"build_manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
