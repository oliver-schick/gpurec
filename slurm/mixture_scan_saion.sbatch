#!/bin/bash -l
# ======================================================================
# Regime-mixture SCAN, fanned out on short-a100 (32 A100s). Array job: each task
# computes one (D,L,T) regime's forward (root_pi[F,S] + extinction E[S]) and saves
# a regime npz. A dependent mixem job then runs the numpy EM over all regimes.
# Tests whether the data wants a core/accessory split + de-concentrates origination.
#
# sbatch -p short-a100 --gres=gpu:a100:1 -c 8 --mem=48G -t 00:55:00 --array=0-49 \
#   --export=ALL,GPUREC_DIR=...,ROOT=Eury,OUTDIR=... slurm/mixture_scan_saion.sbatch
# (jobs <1h on short-a100 are NON-PREEMPTIBLE.)
# ======================================================================
#SBATCH --job-name=mixscan
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --requeue
set -euo pipefail
: "${GPUREC_DIR:?must be exported}"
: "${ROOT:=Eury}"
: "${OUTDIR:?must be exported}"
: "${PY_MODULE:=python/3.11.11}"

cd "$GPUREC_DIR"
export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"
module load "$PY_MODULE" 2>/dev/null || echo "WARN module load $PY_MODULE"
if module load gcc/14.2.0 2>/dev/null; then
  GCC14LIB=/apps/unit/SzollosiU/gcc/14.2.0/lib64
  [ -d "$GCC14LIB" ] && export LD_LIBRARY_PATH="$GCC14LIB:${LD_LIBRARY_PATH:-}"
fi
SO=.torch_ext/preprocess_cpp/preprocess_cpp.so; [ -f "$SO" ] && touch "$SO"
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3.11"; fi
echo "=== mixscan regime ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date) ==="
PYTHONPATH=. "$PY" experiments/em_origination.py --roots "$ROOT" --fm-mode e-only \
  --save-forward-idx "$SLURM_ARRAY_TASK_ID" --out-dir "$OUTDIR" 2>&1
echo "=== done regime ${SLURM_ARRAY_TASK_ID} ($(date)) ==="
