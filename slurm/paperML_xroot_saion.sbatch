#!/bin/bash -l
# ======================================================================
# paperML: free-fit ALL rates (per-branch D/L/T + free per-branch O), PURE ML,
# warm-started from the PAPER's 4_Ancestral reconstruction rates, at all 15 roots.
# Per root: (1) build init from the paper rates, (2) eval logL AT the paper rates
# (unfit), (3) free-fit all rates (--origination optimize, --prior none), (4) eval
# per-fam logL at the fitted rates for the AU rooting test. Tests:
#   - are the paper rates ML?  [Eury: logL_ML - logL@paper, and does it reach the
#     FULLbasin Eury optimum -1,631,537?]
#   - is Eury the ML root across all 15?  [AU on the fitted per-fam logL]
# Matches the FULLbasin protocol (lbfgs, 250 steps, fm e-only, prior none); only the
# warm start differs (paper 4_Ancestral instead of per-root DTL_br1_O) -> independent
# confirmation. NO --requeue (wall-timeout must not loop).
#
# fit:  MODE=fit  --array=0-14   (one root per task)
# au:   MODE=au   (afterok, single task; collects + au_rank)
# ======================================================================
#SBATCH --job-name=paperML
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${OUTDIR:?}"; : "${MODE:=fit}"
: "${PY_MODULE:=python/3.11.11}"; : "${STEPS:=250}"; : "${FBS:=400}"
ROOTS=(Eury HalobacThermopl TackA MHH Kor TACA DPANN TAC Asgard MicraDia AMD UndineClu2 Alti Cluster2 Micra5)
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
# Per-task node-local Triton cache: concurrent array tasks racing on the shared
# NFS ~/.triton cache caused "OSError: [Errno 116] Stale file handle" during JIT
# kernel compile (killed roots 1/2/4/5). Isolate per (job,task) on node-local /tmp.
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$OUTDIR"

if [ "$MODE" = "au" ]; then
  echo "=== paperML AU + summary $(date) ==="
  PYTHONPATH=. "$PY" experiments/paperML_au.py --dir "$OUTDIR" 2>&1
  exit $?
fi

# MODE=fit : one root per array task
R="${ROOTS[$SLURM_ARRAY_TASK_ID]}"
INIT="$OUTDIR/init_paper_${R}.json"
EVALINIT="$OUTDIR/eval_paperinit_${R}.json"
RATES="$OUTDIR/fitML_${R}.rates.txt"
EVALFIT="$OUTDIR/eval_fitML_${R}.json"
echo "=== paperML root=$R  $(date) ==="

# (1) warm-start init from the paper 4_Ancestral rates (theta + structured O)
[ -s "$INIT" ] || PYTHONPATH=. "$PY" experiments/paper_rate_init.py --root "$R" --out "$INIT" || { echo "[$R] INIT FAILED"; exit 1; }

# (2) logL AT the paper rates, UNFIT (the ML-gap reference)
[ -s "$EVALINIT" ] || PYTHONPATH=. "$PY" experiments/eval_perfam_undine.py \
    --root "$R" --sidecar "$INIT" --fm-mode e-only --out "$EVALINIT" 2>&1 | tail -2

# (3) free-fit ALL rates (per-branch DTL + free O), pure ML, from the paper rates
if [ ! -s "$RATES.json" ]; then
  PYTHONPATH=. "$PY" experiments/run_undine_branchwise.py --root "$R" --out "$RATES" \
    --origination optimize --init-from-rates "$INIT" \
    --prior none --fm-mode e-only --optimizer lbfgs --steps "$STEPS" --family-batch-size "$FBS" 2>&1 | tail -10 \
    || { echo "[$R] FIT FAILED"; exit 1; }
fi

# (4) per-fam logL at the FITTED rates (for the AU rooting test)
[ -s "$EVALFIT" ] || PYTHONPATH=. "$PY" experiments/eval_perfam_undine.py \
    --root "$R" --sidecar "$RATES.json" --fm-mode e-only --out "$EVALFIT" 2>&1 | tail -2
echo "=== $R DONE $(date) ==="
