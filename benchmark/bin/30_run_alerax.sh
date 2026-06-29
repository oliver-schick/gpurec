#!/usr/bin/env bash
# Submit the AleRax timed run on Deigo for a phase (pilot|full).
# usage: bin/30_run_alerax.sh pilot|full
source "$(cd "$(dirname "$0")/.." && pwd)/lib.sh"
PHASE="${1:?usage: 30_run_alerax.sh pilot|full}"
start_log "30_run_alerax_${PHASE}"
case "$PHASE" in pilot) N="$PILOT_N";; full) N="$FULL_N";; *) die "phase must be pilot|full";; esac

TAG="$(run_tag "$PHASE" "$N")"
FAM="$DEIGO_BENCH_DIR/families_${PHASE}.txt"
SP="$DS_TREE_DEIGO"                 # rooted_phylogeny/<ROOT> (williams) or ReferenceTree.nwk (davin)
FM_RAW="$DS_FM_DEIGO"               # whitespace 'species fraction' (davin already converted at stage)
FM="$DEIGO_BENCH_DIR/fraction_missing_${DS_TAG}.filtered"
OUTDIR="$DEIGO_BENCH_DIR/out/alerax_${TAG}"
RUSE="$OUTDIR/ruse.txt"

rsh "$DEIGO_SSH" "[ -s '$FAM' ]" || die "missing $FAM -- run bin/20_prepare_families.sh $PHASE first"

# AleRax assert(false)s on any fraction_missing species absent from the tree --
# a no-op in Release builds, so it instead silently corrupts _fm[0]. Filter the
# file to exactly the tree's leaf set so AleRax's fraction-missing matches gpurec's
# (which skips extras). Harmless when the file already matches (e.g. Davin: 1007=1007).
log "Filtering fraction_missing to '$DS_TAG' tree leaves -> $FM"
rsh "$DEIGO_SSH" "
  grep -oE '[(,][A-Za-z0-9_.]+:' '$SP' | sed 's/^[(,]//;s/:\$//' | sort -u > '$DEIGO_BENCH_DIR/.tree_leaves_${DS_TAG}'
  awk 'NR==FNR{k[\$1]=1;next} (\$1 in k)' '$DEIGO_BENCH_DIR/.tree_leaves_${DS_TAG}' '$FM_RAW' > '$FM'
  echo \"  fraction_missing filtered to \$(wc -l < '$FM') tree species (raw \$(wc -l < '$FM_RAW'))\"
" || die "failed to filter fraction_missing"

# Re-deploy the wrapper so edits always reach the cluster (avoids stale-sbatch bugs).
KIT="$(cd "$(dirname "$0")/.." && pwd)"
push "$DEIGO_SSH" "$KIT/slurm/alerax_deigo.sbatch" "$DEIGO_BENCH_DIR/slurm"

log "Submitting AleRax ($PHASE, mode=$MODE) on Deigo: $DEIGO_NTASKS ranks, part=$DEIGO_PARTITION"
JOB=$(rsh "$DEIGO_SSH" "
  cd '$DEIGO_BENCH_DIR'
  sbatch --parsable \
    -p '$DEIGO_PARTITION' -C '$DEIGO_CONSTRAINT' \
    -n '$DEIGO_NTASKS' --mem-per-cpu='$DEIGO_MEM_PER_CPU' -t '$DEIGO_TIME' \
    --export=ALL,ALERAX_DIR='$DEIGO_ALERAX_DIR',FAMILIES_FILE='$FAM',SP_TREE='$SP',OUTDIR='$OUTDIR',RUSE_OUT='$RUSE',REC_MODEL='$REC_MODEL',RATE_FLAG='$ALERAX_RATE_FLAG',MODULES='$DEIGO_MODULES',SEED='$SEED',FRACTION_MISSING='$FM' \
    '$DEIGO_BENCH_DIR/slurm/alerax_deigo.sbatch'
")
[ -n "$JOB" ] || die "sbatch did not return a job id"
echo "alerax $PHASE $JOB" >> "$RESULTS_DIR/jobids.txt"
mkdir -p "$RESULTS_DIR"
log "AleRax submitted: job $JOB  (output dir $OUTDIR)"
log "  watch: ssh $DEIGO_SSH squeue --me ; tail -f the %x_%j.out in $DEIGO_BENCH_DIR"
