#!/usr/bin/env python3
"""Export compact, clearly labeled evidence from workstation application runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
from cavity_analyze import analyze


def collect(root):
    if socket.gethostname().split(".")[0]!="kadanoff":
        raise RuntimeError("Collect scientific outputs on GREEN_workstation")
    root=Path(root).resolve()
    running={}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            command=(proc/"cmdline").read_bytes()
            arguments=command.split(b"\0")
            if b"/green/install-qed/cavity-general" in arguments[0] and arguments[0].endswith(b"/bin/mbpt.exe"):
                method=arguments[arguments.index(b"--scf_type")+1].decode()
                running[(str((proc/"cwd").resolve()),method)]=int(proc.name)
        except (FileNotFoundError,PermissionError,ProcessLookupError): pass
    cases={}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory/"manifest.json").exists(): continue
        item={"manifest":json.loads((directory/"manifest.json").read_text()),"methods":{}}
        for method in ("HF","GW"):
            saved=directory/f"{method.lower()}_analysis.json"
            log=directory/f"{method.lower()}.log"
            progress={"status":"not_started"}
            if log.exists():
                content=log.read_text(errors="replace")
                content=content.rsplit("CAVITY RUN START",1)[-1]
                starts=[line for line in content.splitlines() if "Starting iteration" in line]
                progress.update(last_iteration_message=starts[-1] if starts else "initializing")
                convergence=content.rfind("Simulation Converged")
                limit=content.rfind("Reached Maximum number")
                progress["status"]=("converged" if convergence>limit else
                    "iteration_limit" if limit>=0 else "unfinished")
            if (str(directory),method) in running:
                progress.update(status="running",pid=running[(str(directory),method)])
            if saved.exists():
                value=json.loads(saved.read_text())
                if progress["status"]=="converged":
                    # Refresh diagnostics, including individual spin occupations.
                    value.update(analyze(directory,f"{method.lower()}.h5"))
                value["progress"]=progress
                item["methods"][method]=value
            elif log.exists():
                item["methods"][method]={"progress":progress}
        cases[directory.name]=item
    return {"collected_utc":datetime.now(timezone.utc).isoformat(),"run_root":str(root),
            "stable_base":"d593a9c970194dea5855ca38e71f9cd6ac550cbf", "cases":cases}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root"); parser.add_argument("output")
    args=parser.parse_args()
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(collect(args.root),indent=2)+"\n")
    print(output)
