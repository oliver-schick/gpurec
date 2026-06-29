#!/usr/bin/env bash
# Submit the gpurec timed run on Saion for a phase (pilot|full).
# usage: bin/40_run_gpurec.sh pilot|full
source "$(cd "$(dirname "$0")/.." && pwd)/lib.sh"
PHASE="${1:?usage: 40_run_gpurec.sh pilot|full}"
start_log "40_run_gpurec_${PHASE}"
case "$PHASE" in pilot) N="$PILOT_N";; full) N="$FULL_N";; *) die "phase must be pilot|full";; esac
case "$MODE" in
  global|specieswise) ;;  # both handled by bench_gpurec_fit.py
  *) log "WARN: MODE=$MODE not wired on the gpurec side (only global|specieswise)";;
esac

TAG="$(run_tag "$PHASE" "$N")"
OUTDIR="$SAION_BENCH_DIR/out/gpurec_${TAG}"
OUT_RATES="$OUTDIR/rates.txt"
RUSE="$OUTDIR/ruse.txt"

# Re-deploy the wrapper + the benchmark driver so edits always reach the cluster.
# bench_gpurec_fit.py goes into experiments/ (untracked -> survives reset --hard)
# so it can import run_williams_branchwise's helpers.
KIT="$(cd "$(dirname "$0")/.." && pwd)"
push "$SAION_SSH" "$KIT/slurm/bench_gpurec_saion.sbatch" "$SAION_GPUREC_DIR/slurm"
push "$SAION_SSH" "$KIT/bench_gpurec_fit.py" "$SAION_GPUREC_DIR/experiments"

log "Submitting gpurec ($PHASE, dataset=$DATASET, tag=$DS_TAG) on Saion: part=$SAION_PARTITION gres=$SAION_GRES"
JOB=$(rsh "$SAION_SSH" "
  cd '$SAION_GPUREC_DIR'
  mkdir -p '$OUTDIR'
  sbatch --parsable \
    -p '$SAION_PARTITION' --gres='$SAION_GRES' -c '$SAION_CPUS' --mem='$SAION_MEM' -t '$SAION_TIME' \
    --export=ALL,GPUREC_DIR='$SAION_GPUREC_DIR',DATASET='$DATASET',WILLIAMS_DIR='$WILLIAMS_DIR_SAION',ROOT='$ROOT',DS_TREE='$DS_TREE_SAION',DS_ALE_GLOB='$DS_ALE_DIR_SAION/*.ale',DS_FM='$DS_FM_SAION',OUT_RATES='$OUT_RATES',RUSE_OUT='$RUSE',FAMILIES='$N',STEPS='$GPUREC_STEPS',DTYPE='$GPUREC_DTYPE',MIN_SPECIES='$GPUREC_MIN_SPECIES',FM_MODE='$GPUREC_FM_MODE',GMODE='$MODE',FAMILY_BATCH_SIZE='$GPUREC_FAMILY_BATCH_SIZE',PY_MODULE='$SAION_PYTHON_MODULE' \
    '$SAION_GPUREC_DIR/slurm/bench_gpurec_saion.sbatch'
")
[ -n "$JOB" ] || die "sbatch did not return a job id"
mkdir -p "$RESULTS_DIR"
echo "gpurec $PHASE $JOB" >> "$RESULTS_DIR/jobids.txt"
log "gpurec submitted: job $JOB  (output dir $OUTDIR)"
log "  watch: ssh $SAION_SSH squeue --me"
