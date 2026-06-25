#!/bin/bash -l
# ======================================================================
# AleRax DTL rate estimation on Deigo (CPU/MPI), timed with ruse.
# Submitted by bin/30_run_alerax.sh. Resource flags (-p,-n,-t,-C,--mem)
# are passed on the sbatch command line; run parameters come in via
# --export. ASCII only.
# ======================================================================
#SBATCH --job-name=alerax_bench
#SBATCH --cpus-per-task=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
set -euo pipefail

: "${ALERAX_DIR:?must be exported}"        # AleRax source/build tree on Deigo
: "${FAMILIES_FILE:?must be exported}"     # families.txt
: "${SP_TREE:?must be exported}"           # rooted species tree (newick)
: "${OUTDIR:?must be exported}"            # AleRax -p output prefix dir
: "${RUSE_OUT:?must be exported}"          # ruse summary file path
: "${REC_MODEL:=UndatedDTL}"
: "${RATE_FLAG:=--per-species-rates}"
: "${MODULES:=gcc/11.2.1 cmake/3.31.2 openmpi.gcc/5.0.3}"
: "${SEED:=42}"
: "${FRACTION_MISSING:=}"   # path to the dataset's fraction_missing (matches gpurec --fm-mode e-only)

echo "=== alerax_deigo.sbatch on $(hostname) at $(date) ==="
echo "    job=$SLURM_JOB_ID ntasks=${SLURM_NTASKS:-?} part=${SLURM_JOB_PARTITION:-?}"

# shellcheck disable=SC1090
module purge 2>/dev/null || true
for m in $MODULES; do module load "$m" 2>/dev/null || echo "WARN: module load $m failed"; done
module load ruse 2>/dev/null || echo "WARN: no ruse module (will time with /usr/bin/time)"
module list 2>&1 || true

# ---- Build AleRax once (cached) --------------------------------------
BIN="$ALERAX_DIR/build/bin/alerax"
if [ ! -x "$BIN" ]; then
  echo "=== building AleRax in $ALERAX_DIR (no cached binary) ==="
  cd "$ALERAX_DIR"
  mkdir -p build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j "${SLURM_CPUS_ON_NODE:-16}" alerax
fi
[ -x "$BIN" ] || { echo "ERROR: AleRax binary still missing at $BIN"; exit 1; }
echo "=== AleRax binary: $BIN ==="

# ---- Run, timed ------------------------------------------------------
mkdir -p "$(dirname "$RUSE_OUT")"
rm -rf "$OUTDIR"          # AleRax wants a fresh prefix dir
mkdir -p "$OUTDIR"

# AleRax flags:
#   -f families.txt   -s species.nwk   -p outdir
#   --gene-tree-samples 0     : evaluate the CCPs as-is (no resampling)
#   --species-tree-search SKIP: FIX the species tree (estimate rates only)
#   --rec-model UndatedDTL    : the DTL model (matches gpurec)
#   $RATE_FLAG                : --per-species-rates (branch-wise) etc.
ALERAX_ARGS=( -f "$FAMILIES_FILE" -s "$SP_TREE" -p "$OUTDIR"
              --gene-tree-samples 0 --species-tree-search SKIP
              --rec-model "$REC_MODEL" --seed "$SEED" )
# shellcheck disable=SC2206  # intentional word-split: RATE_FLAG is e.g. "--model-parametrization GLOBAL"
[ -n "$RATE_FLAG" ] && ALERAX_ARGS+=( $RATE_FLAG )
# Match gpurec's fraction-missing modeling (gpurec --fm-mode e-only).
if [ -n "$FRACTION_MISSING" ] && [ -f "$FRACTION_MISSING" ]; then
  ALERAX_ARGS+=( --fraction-missing-file "$FRACTION_MISSING" )
  echo "=== using fraction-missing: $FRACTION_MISSING ==="
else
  echo "=== WARN: no fraction-missing file ($FRACTION_MISSING) -- AleRax will NOT model missing data, gpurec (e-only) will -> NOT apples-to-apples ==="
fi

echo "=== CMD: srun --mpi=pmix $BIN ${ALERAX_ARGS[*]} ==="
START=$(date +%s)
if command -v ruse >/dev/null 2>&1; then
  # ruse wraps the launcher; for MPI we time the whole srun.
  ruse --stdout srun --mpi=pmix "$BIN" "${ALERAX_ARGS[@]}" | tee "$RUSE_OUT"
else
  /usr/bin/time -v srun --mpi=pmix "$BIN" "${ALERAX_ARGS[@]}" 2> "$RUSE_OUT"
fi
END=$(date +%s)
ELAPSED=$(( END - START ))

# ---- Emit a machine-readable timing record --------------------------
TIMING="${OUTDIR%/}/timing.json"
# Use AleRax's ACTUAL post-trim family count (matches gpurec's kept count), not
# the raw families.txt line count; fall back to the file count if not found.
NFAM=$(grep -oE 'Remaining families:[[:space:]]*[0-9]+' "${OUTDIR%/}/alerax.log" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
[ -z "$NFAM" ] && NFAM=$(grep -c '^gene_tree' "$FAMILIES_FILE" 2>/dev/null || echo 0)
cat > "$TIMING" <<EOF
{
  "tool": "alerax",
  "cluster": "deigo",
  "host": "$(hostname)",
  "job_id": "${SLURM_JOB_ID:-NA}",
  "partition": "${SLURM_JOB_PARTITION:-NA}",
  "ntasks": ${SLURM_NTASKS:-0},
  "mode": "${RATE_FLAG:-global}",
  "rec_model": "$REC_MODEL",
  "n_families": $NFAM,
  "wall_seconds": $ELAPSED,
  "ruse_file": "$RUSE_OUT"
}
EOF
echo "=== DONE alerax wall=${ELAPSED}s  timing -> $TIMING ==="
