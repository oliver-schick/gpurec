#!/bin/bash -l
# ======================================================================
# Per-branch event tally (O/D/T/L/S) on the big tree (Undine C60) via the
# C++ backtrack reconciliation sampler, for the paper-vs-FULLbasin event
# comparison (DPANN transfer-inflation diagnosis). Drives the committed
# experiments/per_node_copies.py at THREE rate sets:
#   1. paper DTL_br2   (BIC-best, pure rich-clade DTL, no O column)
#   2. paper DTL_br1_O (DTL + 4-class {DPANN,Eury,TackA,rest} origination)
#   3. FULLbasin       (gpurec free-per-branch fit sidecar)  [SIDECAR env]
# Writes <OUT>.perspecies_events.tsv (all S branches) + .per_clade.json + .npz.
#
# sbatch -p largegpu --gres=gpu:a100:1 -c 16 --mem=64G -t 06:00:00 \
#   --export=ALL,GPUREC_DIR=...,OUT_DIR=...,ROOT=Eury,SIDECAR=...,NSAMP=500 \
#   slurm/pnc_events_saion.sbatch
# ======================================================================
#SBATCH --job-name=pnc_events
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --requeue
set -euo pipefail

: "${GPUREC_DIR:?must be exported}"      # gpurec checkout (branch cpp-rust-free)
: "${OUT_DIR:?must be exported}"         # output directory for the three runs
: "${ROOT:=Eury}"
: "${SIDECAR:?must be exported}"         # FULLbasin gpurec rates sidecar JSON
: "${NSAMP:=500}"
: "${PY_MODULE:=python/3.11.11}"

echo "=== pnc_events_saion.sbatch on $(hostname) at $(date) ==="
echo "    job=${SLURM_JOB_ID:-NA} part=${SLURM_JOB_PARTITION:-NA} gres=${SLURM_JOB_GRES:-NA}"

cd "$GPUREC_DIR"
export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"

module load "$PY_MODULE" 2>/dev/null || echo "WARN: module load $PY_MODULE failed"
if module load gcc/11.2.1 2>/dev/null; then
  GCCLIB="$(dirname "$(gcc -print-file-name=libstdc++.so.6 2>/dev/null)")"
  [ -d "$GCCLIB" ] && export LD_LIBRARY_PATH="$GCCLIB:${LD_LIBRARY_PATH:-}"
fi

# Skip JIT rebuilds: touch both cached .so newer than the (reset-bumped) sources.
for SO in .torch_ext/preprocess_cpp/preprocess_cpp.so .torch_ext/recon_sampler/recon_sampler.so; do
  if [ -f "$SO" ]; then touch "$SO"; echo "=== touched cached $SO ==="
  else echo "WARN: no cached $SO; first import may JIT-build (needs gcc>=9)."; fi
done

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "WARN: nvidia-smi unavailable"

if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3.11"; fi
echo "=== python: $PY ($($PY --version 2>&1)) ==="
"$PY" - <<'PYEOF' || { echo "ERROR: torch/CUDA check failed"; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", ok)
sys.exit(0 if ok else 2)
PYEOF

mkdir -p "$OUT_DIR"

run() {  # $1=label  $2..=extra args
  local label="$1"; shift
  local out="$OUT_DIR/pnc_${label}"
  echo "=== [$label] CMD: $PY experiments/per_node_copies.py --root $ROOT $* --engine cpp --n-samples $NSAMP --out $out ==="
  PYTHONPATH=. "$PY" experiments/per_node_copies.py --root "$ROOT" "$@" \
      --engine cpp --n-samples "$NSAMP" --out "$out" 2>&1
  echo "=== [$label] done -> $out.perspecies_events.tsv ==="
}

# 1+2 validate the tally against the paper's own per-species event counts;
# 3 is the FULLbasin comparison run.
run "paper_DTL_br2"  --alerax-model DTL_br2
run "paper_DTL_br1_O" --alerax-model DTL_br1_O
run "FULLbasin"      --sidecar "$SIDECAR"

echo "=== ALL DONE pnc_events ($(date)) ==="
