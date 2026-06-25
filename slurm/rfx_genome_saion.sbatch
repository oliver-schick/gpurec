#!/bin/bash -l
# ======================================================================
# gamma-weighted ancestral GENOME under the per-family RANDOM-EFFECTS model.
# One array task per MIX_GRID regime (0..49). Inactive regimes (mixing weight
# < THRESH) skip instantly; each active regime samples reconciliations for ALL
# families at its OWN (D,L,T) + the SHARED p^O (from rfx_<root>.npz), writing
# per-family presence/copies [F,S]. A dependent combine job gamma-weights them.
#
# sbatch -p short-a100 --gres=gpu:a100:1 -c 8 --mem=80G -t 00:55:00 --array=0-49%16 \
#   --export=ALL,GPUREC_DIR=...,ROOT=Eury,RFX=.../rfx_Eury.npz,OUTDIR=.../rfx_genome \
#   slurm/rfx_genome_saion.sbatch
# then the combine (afterok):
#   sbatch --dependency=afterok:<arrayjobid> -p short-a100 ... RGMODE=combine ... (same script)
# ======================================================================
#SBATCH --job-name=rfxgen
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${ROOT:=Eury}"; : "${RFX:?}"; : "${OUTDIR:?}"
: "${PY_MODULE:=python/3.11.11}"; : "${NSAMP:=200}"; : "${THRESH:=0.004}"; : "${RGMODE:=sample}"

cd "$GPUREC_DIR"
export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"
module load "$PY_MODULE" 2>/dev/null || echo "WARN module load $PY_MODULE"
if module load gcc/14.2.0 2>/dev/null; then
  GCC14LIB=/apps/unit/SzollosiU/gcc/14.2.0/lib64
  [ -d "$GCC14LIB" ] && export LD_LIBRARY_PATH="$GCC14LIB:${LD_LIBRARY_PATH:-}"
fi
for SO in .torch_ext/preprocess_cpp/preprocess_cpp.so .torch_ext/recon_sampler/recon_sampler.so; do [ -f "$SO" ] && touch "$SO"; done
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3.11"; fi
mkdir -p "$OUTDIR"

# ---- combine mode (dependent job): gamma-weight every sampled regime ----
if [ "$RGMODE" = "combine" ]; then
  echo "=== rfx genome COMBINE over $OUTDIR on $(hostname) $(date) ==="
  PYTHONPATH=. "$PY" experiments/rfx_genome.py --mode combine --rfx-npz "$RFX" \
    --pattern "$OUTDIR/pnc_rfx_regime_*.npz" --out "$OUTDIR/rfx_genome_${ROOT}" 2>&1
  echo "=== combine done ($(date)) ==="
  exit 0
fi

# ---- sample mode (array): one regime per task ----
K="${SLURM_ARRAY_TASK_ID}"
SC="$OUTDIR/sc_regime_${K}.json"
PNC="$OUTDIR/pnc_rfx_regime_${K}"
echo "=== [regime $K] rfx genome on $(hostname) $(date) ==="

# RESUME: if this regime already produced per-family copies, skip.
if [ -s "$PNC.npz" ]; then echo "=== [regime $K] already sampled -> skip ==="; exit 0; fi

PYTHONPATH=. "$PY" experiments/rfx_genome.py --mode sidecar --rfx-npz "$RFX" \
  --regime-idx "$K" --weight-thresh "$THRESH" --out-sidecar "$SC"
rc=$?
if [ "$rc" -eq 2 ]; then echo "=== [regime $K] inactive -> skip ==="; exit 0; fi
[ "$rc" -ne 0 ] && { echo "[regime $K] SIDECAR FAILED"; exit 1; }

PYTHONPATH=. "$PY" experiments/per_node_copies.py --root "$ROOT" --sidecar "$SC" \
  --engine cpp --n-samples "$NSAMP" --fm-mode e-only --out "$PNC" 2>&1 | tail -6 \
  || { echo "[regime $K] PNC FAILED"; exit 1; }
echo "=== [regime $K] DONE ($(date)) ==="
