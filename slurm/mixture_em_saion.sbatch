#!/bin/bash -l
# ======================================================================
# Run the regime-mixture EM over the regime_*.npz produced by mixture_scan
# (numpy; needs the venv python for numpy, runs on a compute node). Prints the
# mixture-vs-single gain, active regimes, and marginal origination report.
#
# sbatch -p short-a100 --gres=gpu:a100:1 -c 8 --mem=64G -t 00:30:00 \
#   --dependency=afterok:<arrayjobid> \
#   --export=ALL,GPUREC_DIR=...,OUTDIR=... slurm/mixture_em_saion.sbatch
# ======================================================================
#SBATCH --job-name=mixem
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --requeue
set -euo pipefail
: "${GPUREC_DIR:?must be exported}"
: "${OUTDIR:?must be exported}"
: "${PY_MODULE:=python/3.11.11}"

cd "$GPUREC_DIR"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"
module load "$PY_MODULE" 2>/dev/null || echo "WARN module load $PY_MODULE"
if module load gcc/14.2.0 2>/dev/null; then
  GCC14LIB=/apps/unit/SzollosiU/gcc/14.2.0/lib64
  [ -d "$GCC14LIB" ] && export LD_LIBRARY_PATH="$GCC14LIB:${LD_LIBRARY_PATH:-}"
fi
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3.11"; fi
echo "=== mixem over $OUTDIR on $(hostname) $(date) ==="
echo "regimes present: $(ls "$OUTDIR"/regime_*.npz 2>/dev/null | wc -l)"
PYTHONPATH=. "$PY" experiments/em_origination.py --mixture-npz "$OUTDIR" --em-iters 400 2>&1
echo "=== mixem done ($(date)) ==="
