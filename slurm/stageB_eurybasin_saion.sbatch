#!/bin/bash -l
# ======================================================================
# EM both-het warm-started OFF THE EURY BASIN, at the Eury root only:
#   per-component branchwise fit, init from the paper's DTL_br2 (uniform-clade,
#   Eury-basin) per-branch rates, with UNIFORM origination (the Eury-basin O).
#   Reuses the existing Eury partition (bhroot/Eury/comp_*.ale_list). Tests
#   whether family-heterogeneity UNDER uniform O de-bloats DPANN while staying
#   Eury-consistent (vs the single uniform-clade model's DPANN ~1453, and the
#   rfx-shared-pO both-het's 1121 which roots TackA).
#
# setup:     sbatch -p short-a100 ... MODE=setup    (build DTL_br2 init + copy comp lists)
# component: sbatch --dependency=afterok:<setup> --array=0-19 ... MODE=component
# combine:   sbatch --dependency=afterany:<comp> ... MODE=combine
# ======================================================================
#SBATCH --job-name=eurybas
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${OUTDIR:?}"; : "${MODE:?}"; : "${PARTDIR:?}"; : "${ROOT:=Eury}"
: "${PY_MODULE:=python/3.11.11}"; : "${STEPS:=250}"; : "${FBS:=500}"; : "${SIGMA:=1.0}"; : "${NSAMP:=200}"
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
INIT="$OUTDIR/init_dtlbr2.json"

if [ "$MODE" = "setup" ]; then
  echo "=== eurybasin setup: DTL_br2 init + copy comp lists $(date) ==="
  PYTHONPATH=. "$PY" experiments/alerax_rate_init.py --root "$ROOT" --alerax-model DTL_br2 --out "$INIT" || exit 1
  cp "$PARTDIR"/comp_*.ale_list "$OUTDIR"/ && echo "copied $(ls "$OUTDIR"/comp_*.ale_list | wc -l) component lists"
  exit 0
fi

if [ "$MODE" = "combine" ]; then
  echo "=== eurybasin combine $(date) ==="
  PYTHONPATH=. "$PY" experiments/stage_b_combine.py --dir "$OUTDIR" --label "StageB-EURYBASIN-uniformO-Eury" 2>&1
  exit $?
fi

# MODE=component: per-component branchwise fit (uniform O, DTL_br2 init) + genome
J=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
LIST="$OUTDIR/comp_${J}.ale_list"; RATES="$OUTDIR/comp_${J}.rates"; PNC="$OUTDIR/comp_${J}_pnc"
[ -s "$LIST" ] || { echo "comp $J absent -> skip"; exit 0; }
echo "=== eurybasin component $J ($(wc -l < "$LIST") fam) $(date) ==="
if [ ! -s "$RATES.json" ]; then
  PYTHONPATH=. "$PY" experiments/run_undine_branchwise.py --root "$ROOT" --out "$RATES" \
    --origination uniform --init-from-rates "$INIT" --ale-list "$LIST" \
    --prior brownian --brownian-sigma "$SIGMA" --fm-mode e-only --optimizer lbfgs \
    --steps "$STEPS" --family-batch-size "$FBS" 2>&1 | tail -6 || { echo "[comp $J] FIT FAILED"; exit 1; }
fi
if [ ! -s "$PNC.npz" ]; then
  PYTHONPATH=. "$PY" experiments/per_node_copies.py --root "$ROOT" --sidecar "$RATES.json" \
    --ale-list "$LIST" --engine cpp --n-samples "$NSAMP" --fm-mode e-only --out "$PNC" 2>&1 | tail -4 || { echo "[comp $J] PNC FAILED"; exit 1; }
fi
echo "=== comp $J DONE ($(date)) ==="
