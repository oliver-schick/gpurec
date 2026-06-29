#!/bin/bash -l
# ======================================================================
# AXgrp AU: per-root AU for the EXISTING clade-grouped gpurec refits
# (AXgrp_DTL_br1_O_<root>.rates.txt.json, 15 roots already fitted). The doc's
# "clade -> Eury AU 1.00" was computed from AleRax's FIXED cached per_fam_likelihoods,
# NOT from these gpurec refits -- this computes the AU from the refits themselves.
#
# MODE=eval (default, --array=0-14): forward-eval per-family logL of each AXgrp
#   sidecar -> perfam_<root>.json (needs A100).
# MODE=au (single task): au_branchwise.py over the perfam_*.json -> au_branchwise.json.
#   FORCED CPU (CUDA_VISIBLE_DEVICES=) -- the AU lstsq hits CUSOLVER_STATUS_INTERNAL_ERROR
#   on V100; the bootstrap is a tiny CPU job anyway.
#
# eval:  sbatch -p short-a100 --gres=gpu:a100:1 -c 8 --mem=48G -t 00:50:00 --array=0-14 \
#          --export=ALL,GPUREC_DIR=...,SIDEBASE=.../undine,OUTDIR=.../undine/AXAU slurm/axgrp_au_saion.sbatch
# au:    sbatch -p short-a100 --gres=gpu:a100:1 -t 00:20:00 --dependency=afterok:<ARRAY> \
#          --export=ALL,GPUREC_DIR=...,SIDEBASE=...,OUTDIR=.../undine/AXAU,MODE=au slurm/axgrp_au_saion.sbatch
# ======================================================================
#SBATCH --job-name=AXgrpAU
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${SIDEBASE:?}"; : "${OUTDIR:?}"; : "${MODE:=eval}"; : "${PY_MODULE:=python/3.11.11}"
ROOTS=(Alti AMD Asgard Cluster2 DPANN Eury HalobacThermopl Kor MHH Micra5 MicraDia TACA TackA TAC UndineClu2)
cd "$GPUREC_DIR"
export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"
module load "$PY_MODULE" 2>/dev/null || echo "WARN module load $PY_MODULE"
if module load gcc/14.2.0 2>/dev/null; then
  GCC14LIB=/apps/unit/SzollosiU/gcc/14.2.0/lib64
  [ -d "$GCC14LIB" ] && export LD_LIBRARY_PATH="$GCC14LIB:${LD_LIBRARY_PATH:-}"
fi
for SO in .torch_ext/preprocess_cpp/preprocess_cpp.so .torch_ext/recon_sampler/recon_sampler.so; do [ -f "$SO" ] && touch "$SO"; done
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3.11"
# Per-task node-local Triton cache (avoid the shared-NFS stale-file-handle race).
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$OUTDIR"

if [ "$MODE" = "au" ]; then
  echo "=== AXgrp au_branchwise (CPU) $(date) ==="
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. "$PY" experiments/au_branchwise.py "$OUTDIR" 2>&1
  exit $?
fi

# MODE=eval : one root per array task -> perfam_<root>.json
ROOT="${ROOTS[$SLURM_ARRAY_TASK_ID]}"
SIDE="$SIDEBASE/AXgrp_DTL_br1_O_${ROOT}.rates.txt.json"
PERFAM="$OUTDIR/perfam_${ROOT}.json"
echo "=== AXgrp eval root=$ROOT $(hostname) $(date) ==="
[ -s "$SIDE" ] || { echo "MISSING $SIDE"; exit 1; }
[ -s "$PERFAM" ] || PYTHONPATH=. "$PY" experiments/eval_perfam_undine.py --root "$ROOT" \
  --sidecar "$SIDE" --fm-mode e-only --out "$PERFAM" 2>&1 | tail -4 || { echo "[$ROOT] EVAL FAILED"; exit 1; }
echo "=== [$ROOT] DONE $(date) ==="
