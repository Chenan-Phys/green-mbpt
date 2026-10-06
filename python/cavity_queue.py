#!/usr/bin/env python3
"""Serialize bounded workstation GW pilots and retain status after each case."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import socket
import time
import h5py
from cavity_run import run
from cavity_collect import collect
from cavity_warm_start import seed


def active_native():
    active=[]
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            command=(proc/"cmdline").read_bytes()
            arguments=command.split(b"\0")
            if b"/green/install-qed/cavity-general" in arguments[0] and arguments[0].endswith(b"/bin/mbpt.exe"):
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
    grid=Path.home()/"green/install/share/ir/1e5.h5"
    with h5py.File(grid) as handle: ntau=len(handle["fermi/xgrid"])+2
    status_path.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        status={"started_utc":datetime.now(timezone.utc).isoformat(),"cases":{}}
        for entry in campaign["cases"]:
            name=entry["name"]
            manifest=Path(entry["directory"])/"manifest.json"
            if manifest.exists(): specification=json.loads(manifest.read_text())
            else: specification=json.loads(Path(entry["spec"]).read_text())
            # Reserve the worst bounded history growth for this job, plus the
            # free-space floor. Linear mixing still stores G and Sigma_tau.
            if "nao" not in specification:
                raise RuntimeError("Prepare campaign inputs before resource assessment")
            ns=2 if specification.get("spin",0) else 1
            history_bytes=int(1.25*120*2*ntau*ns*specification["nao"]**2*16)+2*1024**3
            required_disk=int(entry.get("required_disk_gib",60)*1024**3)+history_bytes
            status["cases"][name]={"status":"waiting_for_resources",
                "disk_preflight":{"estimated_bounded_growth_bytes":history_bytes,
                                  "required_free_bytes":required_disk,"tau_points":ntau}}
            status_path.write_text(json.dumps(status,indent=2)+"\n")
            # HF can momentarily consume much more memory than its idle RSS.
            # Wait for other GW/large HF jobs; smaller HF jobs fit the memory margin.
            while True:
                memory={line.split(':')[0]:int(line.split()[1]) for line in
                        Path('/proc/meminfo').read_text().splitlines() if ':' in line}
                disk=shutil.disk_usage(status_path.parent).free
                if disk<required_disk:
                    status["cases"][name].update(status="insufficient_disk_space",free_bytes=disk)
                    status_path.write_text(json.dumps(status,indent=2)+"\n")
                    raise RuntimeError("Disk preflight failed; no next GW case was started")
                if not active_native() and memory['MemAvailable']>entry.get('required_memory_gib',12)*1024**2:
                    break
                time.sleep(10)
            status["cases"][name].update(status="running",started_utc=datetime.now(timezone.utc).isoformat())
            status_path.write_text(json.dumps(status,indent=2)+"\n")
            try:
                if entry.get("warm_start") and not (Path(entry["directory"])/"input.h5").exists():
                    from cavity_prepare import prepare
                    prepare(json.loads(Path(entry["spec"]).read_text()),entry["directory"])
                if entry.get("warm_start") and not (Path(entry["directory"])/"gw.h5").exists():
                    status["cases"][name]["initial_guess"]=seed(entry["warm_start"],entry["directory"])
                result=run(entry["spec"],entry["directory"],beta=entry.get("beta",1000.),methods=("GW",),
                           restart_unconverged=True,native_threads=entry.get("threads",8),
                           mixing_type=entry.get("mixing_type","SIGMA_MIXING"),
                           mixing_weight=entry.get("mixing_weight",.35),
                           number_tolerance=entry.get("number_tolerance",1e-13),
                           const_density=entry.get("const_density",True))
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
