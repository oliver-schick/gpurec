#!/usr/bin/env bash
# Pull timing records, ruse summaries, rate files and logs from both
# clusters into results/. Safe to run repeatedly (rsync deltas).
source "$(cd "$(dirname "$0")/.." && pwd)/lib.sh"
start_log 50_collect
mkdir -p "$RESULTS_DIR"

log "Pulling AleRax (Deigo) outputs -> $RESULTS_DIR/deigo"
pull "$DEIGO_SSH" "$DEIGO_BENCH_DIR/out/" "$RESULTS_DIR/deigo" || log "  (nothing yet on deigo)"
# Slurm job logs land at the TOP level of these dirs (cwd at submit). Grab just
# those .out files -- exclude '*/' so rsync does NOT recurse the whole checkout.
rsync -az -e "ssh -o BatchMode=yes" --exclude='*/' --include='alerax_*.out' --exclude='*' \
  "${DEIGO_SSH}:${DEIGO_BENCH_DIR}/" "$RESULTS_DIR/deigo/logs/" 2>/dev/null || true

log "Pulling gpurec (Saion) outputs -> $RESULTS_DIR/saion"
pull "$SAION_SSH" "$SAION_BENCH_DIR/out/" "$RESULTS_DIR/saion" || log "  (nothing yet on saion)"
rsync -az -e "ssh -o BatchMode=yes" --exclude='*/' --include='gpurec_*.out' --exclude='*' \
  "${SAION_SSH}:${SAION_GPUREC_DIR}/" "$RESULTS_DIR/saion/logs/" 2>/dev/null || true

log "COLLECT complete. Build the comparison with: python3 bin/60_report.py"
