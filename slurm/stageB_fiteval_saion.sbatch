#!/bin/bash -l
# ======================================================================
# STAGE B per-root, PHASE 1 (fit + per-family logL). One array task per root:
#   fit free branchwise DTL with origination PINNED to that root's rfx p^O
#   (rooting_rfx/rfx_<root>.npz, the no-dump profile), then forward-eval the
#   per-family logL -> perfam_<root>.json for the BRANCHWISE AU test.
# Tests whether branchwise rates bring Eury into the root confidence set.
#
# sbatch -p short-a100 --gres=gpu:a100:1 -c 8 --mem=80G -t 00:55:00 --array=0-7 \
#   --export=ALL,GPUREC_DIR=...,RFXDIR=.../rooting_rfx,OUTDIR=.../stageB_au \
#   slurm/stageB_fiteval_saion.sbatch
# ======================================================================
#SBATCH --job-name=sBfit
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${RFXDIR:?}"; : "${OUTDIR:?}"; : "${PY_MODULE:=python/3.11.11}"
: "${STEPS:=250}"; : "${FBS:=500}"; : "${SIGMA:=1.0}"
ROOTS=(Alti AMD Asgard Cluster2 DPANN Eury HalobacThermopl Kor MHH Micra5 MicraDia TACA TackA TAC UndineClu2)
ROOT="${ROOTS[$SLURM_ARRAY_TASK_ID]}"
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
mkdir -p "$OUTDIR"
PONPZ="$RFXDIR/rfx_${ROOT}.npz"
RATES="$OUTDIR/stageB_${ROOT}.rates"; PERFAM="$OUTDIR/perfam_${ROOT}.json"
echo "=== STAGE B fit+eval root=$ROOT on $(hostname) $(date)  PONPZ=$PONPZ ==="
[ -s "$PONPZ" ] || { echo "MISSING $PONPZ"; exit 1; }

if [ ! -s "$RATES.json" ]; then
  PYTHONPATH=. "$PY" experiments/run_undine_branchwise.py --root "$ROOT" --out "$RATES" \
    --origination fixed --origination-fixed-po-npz "$PONPZ" \
    --prior brownian --brownian-sigma "$SIGMA" --fm-mode e-only --optimizer lbfgs \
    --steps "$STEPS" --family-batch-size "$FBS" 2>&1 | tail -6 || { echo "[$ROOT] FIT FAILED"; exit 1; }
fi

if [ ! -s "$PERFAM" ]; then
  PYTHONPATH=. "$PY" experiments/eval_perfam_undine.py --root "$ROOT" --sidecar "$RATES.json" \
    --fm-mode e-only --out "$PERFAM" 2>&1 | tail -3 || { echo "[$ROOT] EVAL FAILED"; exit 1; }
fi
echo "=== [$ROOT] DONE ($(date)) ==="
