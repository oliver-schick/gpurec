#!/bin/bash -l
# ======================================================================
# Eury-basin both-het CROSS-ROOT, FITTED-O variant: per-component branchwise fit
# warm-started from DTL_br1_O (the deep fitted-O Eury basin) -- rates init from
# DTL_br1_O, origination FIXED at DTL_br1_O's structured O. Reuses the bhroot
# partition. Tests whether the family-het EM, in the deep fitted-O basin, roots
# Eury (vs the rfx-shallow both-het which buried Eury at #13 = EM noise).
# NO --requeue (wall-timeout would otherwise loop).
#
# setup:     MODE=setup PARTDIR=.../bhroot/<R> OUTDIR=.../bhroot_fittedO/<R>
# component: MODE=component --array=0-19
# ======================================================================
#SBATCH --job-name=bhFitO
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${OUTDIR:?}"; : "${MODE:?}"; : "${PARTDIR:?}"; : "${ROOT:?}"
: "${PY_MODULE:=python/3.11.11}"; : "${STEPS:=250}"; : "${FBS:=500}"; : "${SIGMA:=1.0}"
AXBASE="/work/SzollosiU/gergely-szollosi/williams_run/undine/alerax_ref/3_Reconciliation/This_study/5_reconcilation models"
cd "$GPUREC_DIR"
export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"
module load "$PY_MODULE" 2>/dev/null || echo "WARN module load $PY_MODULE"
if module load gcc/14.2.0 2>/dev/null; then
  GCC14LIB=/apps/unit/SzollosiU/gcc/14.2.0/lib64; [ -d "$GCC14LIB" ] && export LD_LIBRARY_PATH="$GCC14LIB:${LD_LIBRARY_PATH:-}"
fi
for SO in .torch_ext/preprocess_cpp/preprocess_cpp.so .torch_ext/recon_sampler/recon_sampler.so; do [ -f "$SO" ] && touch "$SO"; done
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3.11"
mkdir -p "$OUTDIR"; INIT="$OUTDIR/init_dtlbr1o.json"
MDIR="$AXBASE/DTL_br1_O/Undine_C60_${ROOT}root2_OR"
MP="$MDIR/model_parameters/model_parameters.txt"
ATREE="$MDIR/species_trees/starting_species_tree.newick"
[ -f "$ATREE" ] || ATREE="/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study/4_species_tree/Undine_C60_${ROOT}root_short_name.nw"

if [ "$MODE" = "setup" ]; then
  echo "=== bhFitO setup root=$ROOT $(date) ==="
  PYTHONPATH=. "$PY" experiments/alerax_rate_init.py --root "$ROOT" --alerax-model DTL_br1_O --out "$INIT" || exit 1
  cp "$PARTDIR"/comp_*.ale_list "$OUTDIR"/ && echo "copied $(ls "$OUTDIR"/comp_*.ale_list|wc -l) comp lists"
  exit 0
fi

# MODE=component
J=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
LIST="$OUTDIR/comp_${J}.ale_list"; RATES="$OUTDIR/comp_${J}.rates"; PERFAM="$OUTDIR/perfam_${J}.json"
[ -s "$LIST" ] || { echo "comp $J absent -> skip"; exit 0; }
echo "=== bhFitO component $J root=$ROOT ($(wc -l < "$LIST") fam) $(date) ==="
if [ ! -s "$RATES.json" ]; then
  PYTHONPATH=. "$PY" experiments/run_undine_branchwise.py --root "$ROOT" --out "$RATES" \
    --origination fixed --origination-fixed-from "$MP" --origination-fixed-atree "$ATREE" \
    --init-from-rates "$INIT" --ale-list "$LIST" --prior brownian --brownian-sigma "$SIGMA" \
    --fm-mode e-only --optimizer lbfgs --steps "$STEPS" --family-batch-size "$FBS" 2>&1 | tail -6 \
    || { echo "[comp $J] FIT FAILED"; exit 1; }
fi
if [ ! -s "$PERFAM" ]; then
  PYTHONPATH=. "$PY" experiments/eval_perfam_undine.py --root "$ROOT" --sidecar "$RATES.json" \
    --ale-list "$LIST" --fm-mode e-only --out "$PERFAM" 2>&1 | tail -3 || { echo "[comp $J] EVAL FAILED"; exit 1; }
fi
echo "=== comp $J DONE ($(date)) ==="
