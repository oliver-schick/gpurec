#!/bin/bash -l
# ======================================================================
# Sample per-node genomes/events (per_node_copies) from a LIST of gpurec rate
# sidecars -- e.g. the origination-floor sweep -- to read copies/orig/T_in/T_out
# at each node (root / 511 / DPANN). Uses the EXACT p^O the fit wrote
# (origination_prob, so floors are honoured). Writes <OUT_DIR>/pnc_<label>.*
#
# sbatch -p largegpu --gres=gpu:a100:1 -c 16 --mem=64G -t 06:00:00 \
#   --export=ALL,GPUREC_DIR=...,OUT_DIR=...,ROOT=Eury,NSAMP=500,\
#     SIDECARS="uniform=/p/uniform.json a0.10=/p/a0.10.json ..." \
#   slurm/pnc_sidecars_saion.sbatch
# ======================================================================
#SBATCH --job-name=pnc_sidecars
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --requeue
set -euo pipefail

: "${GPUREC_DIR:?must be exported}"
: "${OUT_DIR:?must be exported}"
: "${ROOT:=Eury}"
: "${NSAMP:=500}"
: "${SIDECARS:?must be exported (space-sep label=path pairs)}"
: "${PY_MODULE:=python/3.11.11}"

echo "=== pnc_sidecars_saion.sbatch on $(hostname) at $(date) ==="
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
for SO in .torch_ext/preprocess_cpp/preprocess_cpp.so .torch_ext/recon_sampler/recon_sampler.so; do
  [ -f "$SO" ] && touch "$SO"
done
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3.11"; fi
PYTHONPATH=. "$PY" -c "from gpurec.core.sampler_cpp import _ext; _ext(); print('recon_sampler OK')" \
  || { echo "ERROR: recon_sampler build failed"; exit 1; }
touch .torch_ext/recon_sampler/recon_sampler.so 2>/dev/null || true

mkdir -p "$OUT_DIR"
for pair in $SIDECARS; do
  label="${pair%%=*}"; sc="${pair#*=}"
  out="$OUT_DIR/pnc_${label}"
  echo "=== [$label] per_node_copies --sidecar $sc -> $out ==="
  PYTHONPATH=. "$PY" experiments/per_node_copies.py --root "$ROOT" --sidecar "$sc" \
    --engine cpp --n-samples "$NSAMP" --out "$out" 2>&1 | tail -20
done
echo "=== ALL DONE pnc_sidecars ($(date)) ==="
