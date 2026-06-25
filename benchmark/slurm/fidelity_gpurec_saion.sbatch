#!/bin/bash -l
# ======================================================================
# CORRECTNESS / same-optimum fidelity check on Saion (A100), forward-only.
# Sets gpurec's per-branch rates + origination to AleRax's OWN fitted values
# (from the dataset's "reconciliation models/branch wise/<root>/") and compares
# per-family log-likelihood family-by-family to AleRax's per_fam_likelihoods.txt.
# If the two likelihood functions agree, Pearson r ~ 1 (handout: ~0.999998).
# Submitted by bin/70_check_fidelity.sh. ASCII only.
# ======================================================================
#SBATCH --job-name=gpurec_fidelity
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --requeue
set -euo pipefail

: "${GPUREC_DIR:?must be exported}"
: "${WILLIAMS_DIR:?must be exported}"   # has ccps/, rooted_phylogeny/, fraction_missing, reconciliation models/
: "${ROOT:?must be exported}"
: "${OUT:?must be exported}"            # fidelity report (json/txt) path
: "${RUSE_OUT:?must be exported}"
: "${FM_MODE:=e-only}"
: "${MIN_SPECIES:=1}"                   # fidelity: evaluate ALL families
: "${GMODE:=global}"                    # global | specieswise (which AleRax reference to check)
: "${PY_MODULE:=python/3.11.11}"

echo "=== fidelity_gpurec_saion.sbatch on $(hostname) at $(date) ==="
echo "    job=${SLURM_JOB_ID:-NA} part=${SLURM_JOB_PARTITION:-NA} root=$ROOT"
cd "$GPUREC_DIR"
export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"   # load the prebuilt .so (bin/15), no compute-node rebuild
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"   # ninja on PATH if a rebuild is ever needed
module load "$PY_MODULE" 2>/dev/null || echo "WARN: module load $PY_MODULE failed"
if module load gcc/11.2.1 2>/dev/null; then  # recent libstdc++ for the gcc-built .so (safety net)
  GCCLIB="$(dirname "$(gcc -print-file-name=libstdc++.so.6 2>/dev/null)")"
  [ -d "$GCCLIB" ] && export LD_LIBRARY_PATH="$GCCLIB:${LD_LIBRARY_PATH:-}"
fi

SO=".torch_ext/preprocess_cpp/preprocess_cpp.so"
[ -f "$SO" ] && { touch "$SO"; echo "touched cached $SO"; } || echo "WARN: no cached .so"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version --format=csv 2>/dev/null || echo "WARN: nvidia-smi unavailable"
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3.11"; fi
"$PY" - <<'PYEOF' || { echo "ERROR: torch/CUDA check failed (not an A100?)"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE"); sys.exit(2)
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
PYEOF

mkdir -p "$(dirname "$OUT")" "$(dirname "$RUSE_OUT")"

# Pick the correctness tool by mode:
#  global      -> eval_at_alerax_rates.py   (gpurec @ AleRax's GLOBAL D,L,T)
#  specieswise -> eval_branchwise_perfam.py (gpurec @ AleRax's per-branch rates+O)
REFB="$WILLIAMS_DIR/reconciliation models"
if [ "$GMODE" = "global" ]; then
  MP="$REFB/global/$ROOT/model_parameters/model_parameters.txt"
  PF="$REFB/global/$ROOT/per_fam_likelihoods.txt"
  [ -f "$MP" ] && [ -f "$PF" ] || { echo "ERROR: missing global reference: $MP / $PF (run bin/70 to extract it)"; exit 1; }
  # global = every branch identical; take the first data line's D,L,T (cols 2,3,4).
  read -r DR LR TR <<EOF2
$(awk 'NF>=4 && $2 ~ /^[0-9.eE+-]+$/ {print $2, $3, $4; exit}' "$MP")
EOF2
  [ -n "${DR:-}" ] || { echo "ERROR: could not parse global rates from $MP"; exit 1; }
  echo "=== AleRax global rates (D,L,T) = $DR,$LR,$TR ==="
  ARGS=( experiments/eval_at_alerax_rates.py
         --root "$ROOT" --data-dir "$WILLIAMS_DIR"
         --rates "$DR,$LR,$TR" --rate-order DLT --alerax-perfam "$PF"
         --fm-mode "$FM_MODE" --min-species "$MIN_SPECIES" --out "$OUT" )
else
  ARGS=( experiments/eval_branchwise_perfam.py
         --root "$ROOT" --data-dir "$WILLIAMS_DIR"
         --fm-mode "$FM_MODE" --min-species "$MIN_SPECIES" --out "$OUT" )
fi

echo "=== CMD: $PY ${ARGS[*]} ==="
START=$(date +%s)
if command -v ruse >/dev/null 2>&1 || module load ruse 2>/dev/null; then
  PYTHONPATH=. ruse --stdout "$PY" "${ARGS[@]}" | tee "$RUSE_OUT"
else
  PYTHONPATH=. "$PY" "${ARGS[@]}" 2>&1 | tee "$RUSE_OUT"
fi
echo "=== DONE fidelity (wall=$(( $(date +%s) - START ))s) -> $OUT ; Pearson r in the log above ==="
