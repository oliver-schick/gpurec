#!/bin/bash -l
#SBATCH --job-name=rootrfx
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${OUTDIR:?}"; : "${PY_MODULE:=python/3.11.11}"
ROOTS=(Alti AMD Asgard Cluster2 DPANN Eury HalobacThermopl Kor MHH Micra5 MicraDia TACA TackA TAC UndineClu2)
R="${ROOTS[$SLURM_ARRAY_TASK_ID]}"
cd "$GPUREC_DIR"; export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"
module load "$PY_MODULE" 2>/dev/null || true
if module load gcc/14.2.0 2>/dev/null; then GCC14LIB=/apps/unit/SzollosiU/gcc/14.2.0/lib64; [ -d "$GCC14LIB" ] && export LD_LIBRARY_PATH="$GCC14LIB:${LD_LIBRARY_PATH:-}"; fi
[ -f .torch_ext/preprocess_cpp/preprocess_cpp.so ] && touch .torch_ext/preprocess_cpp/preprocess_cpp.so
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3.11"
echo "=== rooting_rfx root=$R on $(hostname) $(date) ==="
PYTHONPATH=. "$PY" experiments/em_origination.py --rooting-rfx --roots "$R" --fm-mode e-only --out-dir "$OUTDIR" 2>&1 | tail -20
