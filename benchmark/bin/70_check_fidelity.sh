#!/usr/bin/env bash
# CORRECTNESS check: does gpurec compute the SAME likelihood as AleRax at
# AleRax's OWN fitted rates? Forward-only (no optimization); correlates per-family
# logL with AleRax's per_fam_likelihoods.txt. Expected Pearson r ~ 1 (~0.999998).
#   MODE=global      -> eval_at_alerax_rates.py   (AleRax's GLOBAL D,L,T)
#   MODE=specieswise -> eval_branchwise_perfam.py (AleRax's per-branch rates + O)
#
# Self-contained: uses the dataset's SHIPPED AleRax reference (does NOT need our
# own AleRax run). Extracts just the ~3 MB global/<ROOT> + branch wise/<ROOT>
# refs from the shared /bucket tarball (the whole reconciliation-models dir is
# 14 GB -- never extract it all).
#
# usage: bin/70_check_fidelity.sh
source "$(cd "$(dirname "$0")/.." && pwd)/lib.sh"
start_log 70_check_fidelity
require_not_placeholder "$WILLIAMS_DIR_SAION"

REF_BASE="reconciliation models"
case "$MODE" in
  global)       REF_SUB="$REF_BASE/global/$ROOT" ;;
  specieswise)  REF_SUB="$REF_BASE/branch wise/$ROOT" ;;
  *) die "fidelity check supports MODE=global|specieswise (got '$MODE')" ;;
esac
TARPFX="3_Reconciliation/Williams_et_al_2017"
TARBALL="${SAION_WILLIAMS_TARBALL:-$SHARED_TARBALL}"

# 1. Ensure the (small) AleRax reference for this mode+root is on Saion.
if rsh "$SAION_SSH" "[ -f '$WILLIAMS_DIR_SAION/$REF_SUB/per_fam_likelihoods.txt' ]"; then
  log "AleRax reference already present: $WILLIAMS_DIR_SAION/$REF_SUB"
else
  [ -n "$TARBALL" ] || die "no tarball to extract reference from (set SHARED_TARBALL or SAION_WILLIAMS_TARBALL)"
  rsh "$SAION_SSH" "[ -f '$TARBALL' ]" || die "tarball not found on Saion: $TARBALL (run bin/10_stage.sh first)"
  log "Extracting AleRax reference (global + branch wise) for root=$ROOT from $TARBALL (~3 MB; NOT the 14 GB whole dir)"
  rsh "$SAION_SSH" "tar --warning=no-unknown-keyword --exclude='._*' --exclude='.DS_Store' -xzf '$TARBALL' \
      --strip-components=2 -C '$WILLIAMS_DIR_SAION' \
      '$TARPFX/$REF_BASE/global/$ROOT' \
      '$TARPFX/$REF_BASE/branch wise/$ROOT'" \
    || die "reference extraction failed (does the tarball contain '$REF_SUB'?)"
  rsh "$SAION_SSH" "[ -f '$WILLIAMS_DIR_SAION/$REF_SUB/per_fam_likelihoods.txt' ]" \
    || die "extracted, but per_fam_likelihoods.txt missing under $REF_SUB"
  log "  reference extracted."
fi

# 2. Submit the forward-only fidelity job on Saion.
# Stage the fidelity wrapper too (so this works even if 10_stage predates it).
KIT="$(cd "$(dirname "$0")/.." && pwd)"
push "$SAION_SSH" "$KIT/slurm/fidelity_gpurec_saion.sbatch" "$SAION_GPUREC_DIR/slurm"
OUTDIR="$SAION_BENCH_DIR/out/fidelity_${MODE}_${ROOT}"
OUT="$OUTDIR/fidelity.json"
RUSE="$OUTDIR/ruse.txt"
log "Submitting gpurec fidelity check (root=$ROOT) on Saion: $SAION_PARTITION / $SAION_GRES"
JOB=$(rsh "$SAION_SSH" "
  cd '$SAION_GPUREC_DIR'; mkdir -p '$OUTDIR'
  sbatch --parsable \
    -p '$SAION_PARTITION' --gres='$SAION_GRES' -c '$SAION_CPUS' --mem='$SAION_MEM' -t '$SAION_TIME' \
    --export=ALL,GPUREC_DIR='$SAION_GPUREC_DIR',WILLIAMS_DIR='$WILLIAMS_DIR_SAION',ROOT='$ROOT',OUT='$OUT',RUSE_OUT='$RUSE',FM_MODE='$GPUREC_FM_MODE',GMODE='$MODE',MIN_SPECIES='1',PY_MODULE='$SAION_PYTHON_MODULE' \
    '$SAION_GPUREC_DIR/slurm/fidelity_gpurec_saion.sbatch'
")
[ -n "$JOB" ] || die "sbatch did not return a job id"
mkdir -p "$RESULTS_DIR"; echo "fidelity - $JOB" >> "$RESULTS_DIR/jobids.txt"
log "Fidelity job submitted: $JOB"
log "  When done, the log (and $OUT) report: matched families, TOTAL logL gpurec vs AleRax, and Pearson r (~1 = correct)."
log "  Pull it with bin/50_collect.sh; the r value prints in the job's .out / ruse.txt."