#!/bin/bash -l
# ======================================================================
# STAGE B FULL (BOTH branch- and family-heterogeneous), MAP-partition (option 1).
# Three modes via $MODE:
#   partition : rfx_<root>.npz gamma -> per-component comp_<j>.ale_list + init_<j>.json
#   component : (array) fit branchwise theta_k on comp_<j> (init_<j>, p^O pinned) + sample genome
#   combine   : sum per-component genomes -> both-heterogeneous ancestral genome
# Family het = the partition (each family in its best rate component); branch het = theta_k[S,3].
#
# partition:  sbatch -p short-a100 --gres=gpu:a100:1 -c 4 --mem=32G -t 00:15:00 \
#   --export=ALL,GPUREC_DIR=...,RFX=.../rfx_Eury.npz,OUTDIR=.../stageB_full_Eury,MODE=partition ...
# component:  sbatch --dependency=afterok:<part> --array=0-19 ... MODE=component ROOT=Eury PONPZ=...
# combine:    sbatch --dependency=afterok:<comp> -c 4 ... MODE=combine
# ======================================================================
#SBATCH --job-name=sBfull
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${OUTDIR:?}"; : "${MODE:?}"; : "${PY_MODULE:=python/3.11.11}"
: "${ROOT:=Eury}"; : "${STEPS:=250}"; : "${FBS:=500}"; : "${SIGMA:=1.0}"; : "${NSAMP:=200}"; : "${MINFAM:=50}"
: "${STOCH:=0}"
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

if [ "$MODE" = "partition" ]; then
  : "${RFX:?}"
  EXTRA=""; [ "$STOCH" = "1" ] && EXTRA="--stochastic --seed ${SEED:-0}"
  echo "=== STAGE B FULL partition root=$ROOT $(hostname) $(date) ==="
  PYTHONPATH=. "$PY" experiments/stage_b_partition.py --rfx-npz "$RFX" --out-dir "$OUTDIR" \
    --min-fam "$MINFAM" $EXTRA 2>&1
  exit $?
fi

if [ "$MODE" = "combine" ]; then
  echo "=== STAGE B FULL combine $(hostname) $(date) ==="
  PYTHONPATH=. "$PY" experiments/stage_b_combine.py --dir "$OUTDIR" \
    --label "StageB-FULL-${ROOT}$([ "$STOCH" = "1" ] && echo "-SEM")" 2>&1
  exit $?
fi

# --- MODE=component (array): one rate-component per task ---
: "${PONPZ:?}"
J=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
LIST="$OUTDIR/comp_${J}.ale_list"; INIT="$OUTDIR/init_${J}.json"
RATES="$OUTDIR/comp_${J}.rates"; PNC="$OUTDIR/comp_${J}_pnc"
if [ ! -s "$LIST" ]; then echo "=== comp $J absent -> skip ==="; exit 0; fi
echo "=== STAGE B FULL component $J root=$ROOT ($(wc -l < "$LIST") families) $(hostname) $(date) ==="

if [ ! -s "$RATES.json" ]; then
  PYTHONPATH=. "$PY" experiments/run_undine_branchwise.py --root "$ROOT" --out "$RATES" \
    --origination fixed --origination-fixed-po-npz "$PONPZ" --init-from-rates "$INIT" \
    --ale-list "$LIST" --prior brownian --brownian-sigma "$SIGMA" --fm-mode e-only \
    --optimizer lbfgs --steps "$STEPS" --family-batch-size "$FBS" 2>&1 | tail -6 \
    || { echo "[comp $J] FIT FAILED"; exit 1; }
fi
# per-component per-family logL (for the both-het AU / ranking)
PERFAM="$OUTDIR/perfam_${J}.json"
if [ ! -s "$PERFAM" ]; then
  PYTHONPATH=. "$PY" experiments/eval_perfam_undine.py --root "$ROOT" --sidecar "$RATES.json" \
    --ale-list "$LIST" --fm-mode e-only --out "$PERFAM" 2>&1 | tail -3 \
    || { echo "[comp $J] EVAL FAILED"; exit 1; }
fi
# genome sampling (optional: GENOME=0 skips it -- the rooting runs only need per-family logL)
if [ "${GENOME:-1}" = "1" ] && [ ! -s "$PNC.npz" ]; then
  PYTHONPATH=. "$PY" experiments/per_node_copies.py --root "$ROOT" --sidecar "$RATES.json" \
    --ale-list "$LIST" --engine cpp --n-samples "$NSAMP" --fm-mode e-only --out "$PNC" 2>&1 | tail -4 \
    || { echo "[comp $J] PNC FAILED"; exit 1; }
fi
echo "=== comp $J DONE ($(date)) ==="
