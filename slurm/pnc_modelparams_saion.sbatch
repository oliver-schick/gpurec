#!/bin/bash -l
#SBATCH --job-name=pnc_mp
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${ROOT:=Eury}"; : "${MP:?}"; : "${ATREE:?}"; : "${OUT:?}"; : "${NSAMP:=500}"; : "${PY_MODULE:=python/3.11.11}"
cd "$GPUREC_DIR"; export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"
module load "$PY_MODULE" 2>/dev/null || true
if module load gcc/14.2.0 2>/dev/null; then GCC14LIB=/apps/unit/SzollosiU/gcc/14.2.0/lib64; [ -d "$GCC14LIB" ] && export LD_LIBRARY_PATH="$GCC14LIB:${LD_LIBRARY_PATH:-}"; fi
for SO in .torch_ext/preprocess_cpp/preprocess_cpp.so .torch_ext/recon_sampler/recon_sampler.so; do [ -f "$SO" ] && touch "$SO"; done
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3.11"
echo "=== validate gpurec @ paper reconstruction rates: $MP ==="
PYTHONPATH=. "$PY" experiments/per_node_copies.py --root "$ROOT" --model-params "$MP" --atree "$ATREE" \
  --engine cpp --n-samples "$NSAMP" --out "$OUT" 2>&1 | tail -25
