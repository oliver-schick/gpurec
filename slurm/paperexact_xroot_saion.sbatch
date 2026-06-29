#!/bin/bash -l
# ======================================================================
# paperexact: refit the PAPER'S EXACT model (4_Ancestral 24-class clade DTL +
# 4-class structured O), ONE fit per root, warm-started from the Eury basin, at all
# 15 candidate roots -- the comparable-model control for the paperML free-O DPANN
# result. Per root R:
#   (1) construct a per-root clade-groups dir (make_paper_root_model.py): the paper
#       24 DTL + 4 O classes mapped onto R's tree by bipartition (root branch -> the
#       paper O=0.165 origination-mass class, symmetric across candidate roots).
#   (2) refit with --clade-groups <dirR> --init-from-alerax (seed at the paper /
#       Eury-basin rates) --origination optimize (O optimized WITHIN the 4 classes)
#       --prior none (faithful to the paper's unregularized clade fit).
#   (3) forward-eval per-family logL -> perfam_<root>.json for the AU test.
# Rooting statistic = prior-free data_log_likelihood_ln in each sidecar.
#
# MODE=fit (default, --array=0-14). MODE=au (single, forced CPU) -> au_branchwise.
#
# fit: sbatch -p largegpu --gres=gpu:a100:1 -c 8 --mem=64G -t 02:00:00 --array=0-14 \
#        --export=ALL,GPUREC_DIR=...,OUTDIR=.../undine/paperexact,\
#PAPER_DIR=.../undine/paper_ancestral/4_Ancestral_reconstruction/Euryarchaeota_root,\
#DATA_DIR=/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study \
#        slurm/paperexact_xroot_saion.sbatch
# au:  sbatch -p short-a100 --gres=gpu:a100:1 -t 00:20:00 --dependency=afterok:<ARRAY> \
#        --export=ALL,GPUREC_DIR=...,OUTDIR=.../undine/paperexact,MODE=au slurm/paperexact_xroot_saion.sbatch
# ======================================================================
#SBATCH --job-name=paperexact
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${OUTDIR:?}"; : "${MODE:=fit}"; : "${PY_MODULE:=python/3.11.11}"
: "${STEPS:=250}"; : "${FBS:=500}"
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
  echo "=== paperexact au_branchwise (CPU) $(date) ==="
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. "$PY" experiments/au_branchwise.py "$OUTDIR" 2>&1
  exit $?
fi

: "${PAPER_DIR:?}"; : "${DATA_DIR:?}"
ROOT="${ROOTS[$SLURM_ARRAY_TASK_ID]}"
MODELDIR="$OUTDIR/paper_root_models/$ROOT"
FITTREE="$DATA_DIR/4_species_tree/Undine_C60_${ROOT}root_short_name.nw"
RATES="$OUTDIR/fitpaper_${ROOT}.rates.txt"
PERFAM="$OUTDIR/perfam_${ROOT}.json"
echo "=== paperexact root=$ROOT $(hostname) $(date) ==="

# (1) construct the per-root paper-exact clade-groups dir
if [ ! -s "$MODELDIR/model_parameters/model_parameters.txt" ]; then
  PYTHONPATH=. "$PY" experiments/make_paper_root_model.py --root "$ROOT" \
    --paper-tree "$PAPER_DIR/species_trees/starting_species_tree.newick" \
    --paper-mp "$PAPER_DIR/model_parameters/model_parameters.txt" \
    --fit-tree "$FITTREE" --out-dir "$MODELDIR" || { echo "[$ROOT] CONSTRUCT FAILED"; exit 1; }
fi

# (2) refit the paper's exact clade model, warm-started from the Eury basin
if [ ! -s "$RATES.json" ]; then
  PYTHONPATH=. "$PY" experiments/run_undine_branchwise.py --root "$ROOT" --out "$RATES" \
    --clade-groups "$MODELDIR" --init-from-alerax --origination optimize \
    --prior none --fm-mode e-only --optimizer lbfgs --steps "$STEPS" \
    --family-batch-size "$FBS" --dtype float64 2>&1 | tail -12 \
    || { echo "[$ROOT] FIT FAILED"; exit 1; }
fi

# (3) per-family logL at the fitted rates -> AU input
[ -s "$PERFAM" ] || PYTHONPATH=. "$PY" experiments/eval_perfam_undine.py --root "$ROOT" \
  --sidecar "$RATES.json" --fm-mode e-only --out "$PERFAM" 2>&1 | tail -4
echo "=== [$ROOT] DONE $(date) ==="
