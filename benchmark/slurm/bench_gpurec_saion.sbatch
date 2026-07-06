#!/bin/bash -l
# ======================================================================
# gpurec branch-wise DTL rate estimation on Saion (A100), timed with ruse.
# Submitted by bin/40_run_gpurec.sh. Resource flags (-p,--gres,-c,--mem,-t)
# come on the sbatch command line; run params via --export. ASCII only.
# Drives the committed experiments/run_williams_branchwise.py.
# ======================================================================
#SBATCH --job-name=gpurec_bench
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --requeue
set -euo pipefail

: "${GPUREC_DIR:?must be exported}"      # gpurec checkout (branch cpp-rust-free)
: "${DATASET:=williams}"                 # williams (data-dir/root) | davin (explicit paths)
: "${OUT_RATES:?must be exported}"       # rates output file
: "${RUSE_OUT:?must be exported}"        # ruse summary file
# williams inputs:
: "${WILLIAMS_DIR:=}"                    # dir with ccps/ rooted_phylogeny/ fraction_missing
: "${ROOT:=DPANN}"
# davin (explicit) inputs:
: "${DS_TREE:=}"                         # rooted species tree (newick)
: "${DS_ALE_GLOB:=}"                     # glob for .ale, e.g. /…/ALEs/*.ale
: "${DS_FM:=}"                           # whitespace fraction_missing file
: "${FAMILIES:=0}"                       # 0 = all families, else first N (pilot)
: "${STEPS:=200}"
: "${DTYPE:=float64}"
: "${MIN_SPECIES:=4}"
: "${FM_MODE:=e-only}"     # AleRax-matching fraction-missing mode (docs/oliver-handoff.md)
: "${GMODE:=global}"       # global (one shared D,L,T) | specieswise (per-branch)
: "${FAMILY_BATCH_SIZE:=auto}"   # auto = driver sizes it from free GPU mem + family clade counts; or 0 (all) / int
: "${MEM_FACTOR:=}"              # auto-batch: override --mem-factor (empty = driver default); set after a probe
: "${MEM_SAFETY:=}"              # auto-batch: override --mem-safety (empty = driver default)
: "${PY_MODULE:=python/3.11.11}"

echo "=== bench_gpurec_saion.sbatch on $(hostname) at $(date) ==="
echo "    job=${SLURM_JOB_ID:-NA} part=${SLURM_JOB_PARTITION:-NA} gres=${SLURM_JOB_GRES:-NA}"

cd "$GPUREC_DIR"

# Cache the JIT-built C++ ext in the repo's .torch_ext (on shared /work, built
# once by bin/15_setup_gpurec.sh) so this compute job LOADS it instead of
# rebuilding (a rebuild needs gcc>=9, often absent on compute nodes).
export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"   # ninja on PATH if a rebuild is ever needed

# Admin recommendation: use the env module python, not the system one.
module load "$PY_MODULE" 2>/dev/null || echo "WARN: module load $PY_MODULE failed"
# Best-effort: ensure a recent libstdc++ for the gcc-built preprocess_cpp.so
# (largegpu usually has a new enough system one; this is a harmless safety net).
if module load gcc/11.2.1 2>/dev/null; then
  GCCLIB="$(dirname "$(gcc -print-file-name=libstdc++.so.6 2>/dev/null)")"
  [ -d "$GCCLIB" ] && export LD_LIBRARY_PATH="$GCCLIB:${LD_LIBRARY_PATH:-}"
fi

# CRITICAL (project memory reference-saion-deploy-loop): a git reset --hard
# bumps the mtime of core/cpp/*.cpp so torch's JIT thinks the cached ext is
# stale and rebuilds preprocess_cpp -- which fails on compute nodes lacking
# devtoolset-11. If a prebuilt .so exists, touch it newer than the sources
# so torch LOADS the cache instead of rebuilding.
SO=".torch_ext/preprocess_cpp/preprocess_cpp.so"
if [ -f "$SO" ]; then
  touch "$SO"
  echo "=== touched cached $SO (skip JIT rebuild) ==="
else
  echo "WARN: no cached $SO; first import may JIT-build (needs gcc>=9 / devtoolset-11)."
fi

# Record the GPU/driver/CUDA the job actually landed on (Saion GPU nodes are
# heterogeneous: largegpu=A100 with a recent driver, the gpu partition mixes
# usable V100s and too-old P100s). We pin -p largegpu --gres=gpu:a100:1, so
# this should always be an A100; log it + hard-fail if CUDA is not usable
# (catches a stray landing on an old-driver node instead of silently using CPU).
echo "=== nvidia-smi (driver / CUDA) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "WARN: nvidia-smi unavailable"

# Pick the interpreter: prefer the repo venv (self-contained on largegpu),
# else the module python.
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3.11"; fi
echo "=== python: $PY ($($PY --version 2>&1)) ==="
"$PY" - <<'PYEOF' || { echo "ERROR: torch import / CUDA check failed -- wrong node (not an A100)?"; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| torch-cuda", torch.version.cuda, "| available", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
else:
    print("CUDA NOT AVAILABLE -- this node's driver is too old for this torch build.")
    sys.exit(2)
PYEOF

mkdir -p "$(dirname "$OUT_RATES")" "$(dirname "$RUSE_OUT")"

# Input args differ by dataset: williams uses the data-dir/root layout;
# davin passes an explicit tree + .ale glob (quoted so the shell does NOT expand
# the glob -- the driver globs it internally) + the converted fraction_missing.
if [ "$DATASET" = davin ]; then
  IN_ARGS=( --species-tree "$DS_TREE" --ale-glob "$DS_ALE_GLOB" --fraction-missing "$DS_FM" )
else
  IN_ARGS=( --root "$ROOT" --data-dir "$WILLIAMS_DIR" )
fi
ARGS=( experiments/bench_gpurec_fit.py --mode "$GMODE" "${IN_ARGS[@]}"
       --families "$FAMILIES" --steps "$STEPS" --dtype "$DTYPE"
       --min-species "$MIN_SPECIES" --fm-mode "$FM_MODE"
       --family-batch-size "$FAMILY_BATCH_SIZE" )
[ -n "$MEM_FACTOR" ] && ARGS+=( --mem-factor "$MEM_FACTOR" )
[ -n "$MEM_SAFETY" ] && ARGS+=( --mem-safety "$MEM_SAFETY" )
ARGS+=( --out "$OUT_RATES" )

echo "=== CMD: $PY ${ARGS[*]} ==="
START=$(date +%s)
if command -v ruse >/dev/null 2>&1 || module load ruse 2>/dev/null; then
  PYTHONPATH=. ruse --stdout "$PY" "${ARGS[@]}" | tee "$RUSE_OUT"
else
  PYTHONPATH=. "$PY" "${ARGS[@]}" 2>&1 | tee "$RUSE_OUT"
fi
END=$(date +%s)
ELAPSED=$(( END - START ))

# Detect failure reliably: ruse/tee swallow python's exit code, so check the
# driver's success artifact (it writes OUT_RATES only after a clean optimize).
# Without this, a CUDA crash mid-run would still print a misleading "DONE".
if [ ! -f "$OUT_RATES" ]; then
  echo "=== gpurec FAILED: no rates written ($OUT_RATES). See the traceback above. ==="
  echo "    (common at scale: CUDA illegal access from an un-batched forward over many"
  echo "     big families -> lower FAMILY_BATCH_SIZE; or a real OOM -> lower it further.)"
  exit 1
fi

# The runner already writes <OUT_RATES>.json with elapsed_s + n_families_kept;
# add a top-level timing record for the collector.
TIMING="$(dirname "$OUT_RATES")/timing_$(basename "$OUT_RATES" .txt).json"
cat > "$TIMING" <<EOF
{
  "tool": "gpurec",
  "cluster": "saion",
  "host": "$(hostname)",
  "job_id": "${SLURM_JOB_ID:-NA}",
  "partition": "${SLURM_JOB_PARTITION:-NA}",
  "gres": "${SLURM_JOB_GRES:-NA}",
  "mode": "$GMODE",
  "root": "$ROOT",
  "families_arg": $FAMILIES,
  "steps": $STEPS,
  "dtype": "$DTYPE",
  "fm_mode": "$FM_MODE",
  "wall_seconds": $ELAPSED,
  "rates_sidecar": "${OUT_RATES}.json",
  "ruse_file": "$RUSE_OUT"
}
EOF
echo "=== DONE gpurec wall=${ELAPSED}s  timing -> $TIMING ==="
