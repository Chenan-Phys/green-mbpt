#!/usr/bin/env python3
"""Serialize bounded workstation GW pilots and retain status after each case."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import socket
import time
from cavity_run import run
from cavity_collect import collect


def active_native():
    active=[]
    executable=str(Path.home()/"green/install-qed/cavity-general/bin/mbpt.exe").encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            command=(proc/"cmdline").read_bytes()
            if executable in command:
                arguments=command.split(b"\0")
                method=arguments[arguments.index(b"--scf_type")+1]
                manifest=(proc/"cwd").resolve()/"manifest.json"
                nao=json.loads(manifest.read_text()).get("nao",1000) if manifest.exists() else 1000
                if method==b"GW" or nao>=256:
                    active.append(int(proc.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError): pass
    return active


def main(campaign_path):
    if socket.gethostname().split(".")[0]!="kadanoff": raise RuntimeError("Workstation only")
    campaign=json.loads(Path(campaign_path).read_text())
    status_path=Path(campaign["status_file"])
    lock=status_path.parent/"native-gw.lock"
    status_path.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        status={"started_utc":datetime.now(timezone.utc).isoformat(),"cases":{}}
        for entry in campaign["cases"]:
            name=entry["name"]
            status["cases"][name]={"status":"waiting_for_resources"}
            status_path.write_text(json.dumps(status,indent=2)+"\n")
            # HF can momentarily consume much more memory than its idle RSS.
            # Wait for other GW/large HF jobs; smaller HF jobs fit the memory margin.
            while True:
                memory={line.split(':')[0]:int(line.split()[1]) for line in
                        Path('/proc/meminfo').read_text().splitlines() if ':' in line}
                if not active_native() and memory['MemAvailable']>entry.get('required_memory_gib',12)*1024**2:
                    break
                time.sleep(10)
            status["cases"][name]={"status":"running","started_utc":datetime.now(timezone.utc).isoformat()}
            status_path.write_text(json.dumps(status,indent=2)+"\n")
            try:
                result=run(entry["spec"],entry["directory"],beta=entry.get("beta",1000.),methods=("GW",),
                           restart_unconverged=True,native_threads=entry.get("threads",8),mixing_type="DIIS")
                status["cases"][name].update(status="completed",result=result)
            except Exception as error:
                status["cases"][name].update(status="failed",error=str(error))
            status["cases"][name]["finished_utc"]=datetime.now(timezone.utc).isoformat()
            status_path.write_text(json.dumps(status,indent=2)+"\n")
            root=Path(entry["directory"]).parent
            (root.parent/(root.name+"_compact_results.json")).write_text(json.dumps(collect(root),indent=2)+"\n")
        status["finished_utc"]=datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("campaign")
    main(parser.parse_args().campaign)
