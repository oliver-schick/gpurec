#!/bin/bash -l
# ======================================================================
# Generic big-tree (Undine C60) branch-wise fit. Free per-branch D/L/T, warm-started
# + brownian smoothness; the origination treatment is passed via EXTRA_ARGS so one
# wrapper drives many parallel configs (uniform-O, origination-floor alpha grid, ...).
#
# sbatch -p largegpu --gres=gpu:a100:1 -c 16 --mem=80G -t 10:00:00 \
#   --export=ALL,GPUREC_DIR=...,ROOT=Eury,OUT=...,WARM=...,SIGMA=1.0,STEPS=250,\
#            EXTRA_ARGS="--origination optimize --origination-floor 0.10" \
#   slurm/undine_fit_saion.sbatch
# ======================================================================
#SBATCH --job-name=undine_fit
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --requeue
set -euo pipefail

: "${GPUREC_DIR:?must be exported}"
: "${ROOT:=Eury}"
: "${OUT:?must be exported}"
: "${WARM:?must be exported}"           # warm-start theta(+omega) sidecar
: "${SIGMA:=1.0}"
: "${STEPS:=250}"
: "${FBS:=500}"
: "${EXTRA_ARGS:=}"                     # origination-specific args (word-split, unquoted)
: "${PY_MODULE:=python/3.11.11}"

echo "=== undine_fit_saion.sbatch on $(hostname) at $(date) ==="
echo "    job=${SLURM_JOB_ID:-NA}  EXTRA_ARGS=[$EXTRA_ARGS]  OUT=$OUT"

cd "$GPUREC_DIR"
export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"

module load "$PY_MODULE" 2>/dev/null || echo "WARN: module load $PY_MODULE failed"
if module load gcc/14.2.0 2>/dev/null; then
  GCC14LIB=/apps/unit/SzollosiU/gcc/14.2.0/lib64
  [ -d "$GCC14LIB" ] && export LD_LIBRARY_PATH="$GCC14LIB:${LD_LIBRARY_PATH:-}"
elif module load gcc/11.2.1 2>/dev/null; then
  GCCLIB="$(dirname "$(gcc -print-file-name=libstdc++.so.6 2>/dev/null)")"
  [ -d "$GCCLIB" ] && export LD_LIBRARY_PATH="$GCCLIB:${LD_LIBRARY_PATH:-}"
fi
SO=".torch_ext/preprocess_cpp/preprocess_cpp.so"
[ -f "$SO" ] && touch "$SO" && echo "=== touched $SO ==="

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "WARN: nvidia-smi unavailable"

if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3.11"; fi
echo "=== python: $PY ($($PY --version 2>&1)) ==="
"$PY" - <<'PYEOF' || { echo "ERROR: torch/CUDA check failed"; exit 1; }
import sys, torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else 2)
PYEOF

mkdir -p "$(dirname "$OUT")"
# EXTRA_ARGS is intentionally UNQUOTED (word-split into argv).
echo "=== CMD: $PY experiments/run_undine_branchwise.py --root $ROOT --out $OUT (+common +[$EXTRA_ARGS]) ==="
PYTHONPATH=. "$PY" experiments/run_undine_branchwise.py \
  --root "$ROOT" --out "$OUT" \
  --fm-mode e-only --optimizer lbfgs --steps "$STEPS" \
  --family-batch-size "$FBS" \
  --prior brownian --brownian-sigma "$SIGMA" \
  --init-from-rates "$WARM" \
  $EXTRA_ARGS 2>&1
echo "=== DONE undine_fit ($(date)) -> $OUT ==="
