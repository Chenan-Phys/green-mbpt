#!/usr/bin/env bash
set -euo pipefail
source "$HOME/green/env.sh"
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYSCF_MAX_MEMORY=2000 PYSCF_TMPDIR=/data/cwei/pyscf_tmp TMPDIR=/data/cwei/pyscf_tmp
solver="$HOME/green/qed-research/python/cavity_run.py"
case "$1" in
  ttf-tcne)
    root=/data/cwei/green_runs/qed-ttf-tcne/pilot
    python "$solver" "$root/specs/R4_lambda0.005.json" "$root/R4_lambda0.005" --methods HF --force-restart
    python "$HOME/green/cavity-applications/ttf-tcne/run_pilot.py" "$root" --methods HF --restart-unconverged
    ;;
  na20)
    python "$HOME/green/cavity-applications/na20/run_pilot.py" /data/cwei/green_runs/qed-na20/pilot --methods HF --restart-unconverged
    ;;
  *) echo "Choose ttf-tcne or na20" >&2; exit 2 ;;
esac
