#!/usr/bin/env python3
"""Pause a task-owned runner/native process without interrupting checkpoint writes."""
import argparse
import os
from pathlib import Path
import signal
import socket
import time

parser=argparse.ArgumentParser()
parser.add_argument("pid",type=int)
parser.add_argument("--directory")
args=parser.parse_args()
if socket.gethostname().split(".")[0]!="kadanoff": raise RuntimeError("Workstation only")
proc=Path("/proc")/str(args.pid)
if not proc.exists(): raise SystemExit(0)
command=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode()
if args.directory:
    expected=Path(args.directory).resolve()
    root=Path("/data/cwei/green_runs").resolve()
    if not expected.is_relative_to(root): raise RuntimeError("Case is outside task data root")
    if str((proc/"cwd").resolve())!=str(expected): raise RuntimeError("PID belongs to another case")
    if "green/install-qed/cavity-general/bin/mbpt.exe" not in command: raise RuntimeError("Unexpected executable")
else:
    allowed=[str(Path.home()/"green/cavity-applications")+"/",str(Path.home()/"green/qed-research/python/cavity_run.py")]
    if not any(x in command for x in allowed): raise RuntimeError("Unexpected runner")
for _ in range(200):
    try:
        paths=[]
        for fd in (proc/"fd").iterdir():
            try: paths.append(str(fd.resolve()))
            except FileNotFoundError: pass
    except FileNotFoundError: break
    if not any(Path(x).name in ("hf.h5","gw.h5","hf_diis.h5","gw_diis.h5") for x in paths):
        os.kill(args.pid,signal.SIGTERM)
        print("Stopped owned process outside checkpoint write",args.pid)
        break
    time.sleep(.05)
else: raise RuntimeError("Checkpoint stayed open; process preserved")
