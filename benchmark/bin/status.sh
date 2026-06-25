#!/usr/bin/env bash
# Quick status of the benchmark jobs on both clusters. Read-only; safe to run
# anytime (and as often as you like). Shows:
#   1. live queue (squeue) on Saion + Deigo
#   2. sacct state/elapsed for every job this kit submitted (works post-finish)
#   3. which output artifacts exist yet (= that run finished)
# Add --logs to also tail the most recent job .out on each cluster.
#
# usage: bin/status.sh [--logs]
source "$(cd "$(dirname "$0")/.." && pwd)/lib.sh"
start_log status          # also saved to results/logs/status.log (overwritten each run) so it can be shared
WANT_LOGS="${1:-}"

QFMT='%.10i %.16j %.9P %.8T %.10M %.5D %R'   # jobid name part state time nodes reason

echo "================ LIVE QUEUE (squeue --me) ================"
echo "-- Saion ($SAION_SSH) --"
rsh "$SAION_SSH" "squeue --me -o '$QFMT'" 2>/dev/null || echo "  (squeue unavailable)"
echo "-- Deigo ($DEIGO_SSH) --"
rsh "$DEIGO_SSH" "squeue --me -o '$QFMT'" 2>/dev/null || echo "  (squeue unavailable)"

# ---- sacct for the jobs THIS kit submitted (state survives after completion) -
if [ -f "$RESULTS_DIR/jobids.txt" ]; then
  saion_ids=""; deigo_ids=""
  while read -r tool _phase jid; do
    [ -z "${jid:-}" ] && continue
    case "$tool" in
      alerax) deigo_ids="${deigo_ids}${jid}," ;;
      *)      saion_ids="${saion_ids}${jid}," ;;   # gpurec, fidelity
    esac
  done < "$RESULTS_DIR/jobids.txt"
  SFMT="JobID,JobName%18,State,Elapsed,Start,End"
  echo
  echo "================ RECORDED JOBS (sacct) ================"
  if [ -n "$saion_ids" ]; then
    echo "-- Saion --"
    rsh "$SAION_SSH" "sacct -X -j '${saion_ids%,}' --format=$SFMT" 2>/dev/null || echo "  (sacct unavailable)"
  fi
  if [ -n "$deigo_ids" ]; then
    echo "-- Deigo --"
    rsh "$DEIGO_SSH" "sacct -X -j '${deigo_ids%,}' --format=$SFMT" 2>/dev/null || echo "  (sacct unavailable)"
  fi
else
  echo; echo "(no $RESULTS_DIR/jobids.txt yet -- nothing submitted by the kit)"
fi

# ---- finished outputs (presence = that run produced results) -----------------
echo
echo "================ OUTPUTS PRESENT (= finished) ================"
echo "-- Saion (gpurec / fidelity) --"
rsh "$SAION_SSH" "find '$SAION_BENCH_DIR/out' \( -name 'timing*.json' -o -name 'fidelity.json' \) -printf '  %p\n' 2>/dev/null | sort" 2>/dev/null || echo "  (none yet)"
echo "-- Deigo (alerax) --"
rsh "$DEIGO_SSH" "find '$DEIGO_BENCH_DIR/out' -name 'timing.json' -printf '  %p\n' 2>/dev/null | sort" 2>/dev/null || echo "  (none yet)"

# ---- optional: tail the latest job log on each cluster -----------------------
if [ "$WANT_LOGS" = "--logs" ]; then
  echo
  echo "================ LATEST LOG TAILS (--logs) ================"
  echo "-- Saion: newest gpurec_*.out --"
  rsh "$SAION_SSH" "f=\$(ls -t '$SAION_GPUREC_DIR'/gpurec_*.out 2>/dev/null | head -1); [ -n \"\$f\" ] && { echo \"\$f:\"; tail -n 15 \"\$f\"; } || echo '  (no gpurec .out yet)'" 2>/dev/null || true
  echo "-- Deigo: newest alerax_*.out --"
  rsh "$DEIGO_SSH" "f=\$(ls -t '$DEIGO_BENCH_DIR'/alerax_*.out 2>/dev/null | head -1); [ -n \"\$f\" ] && { echo \"\$f:\"; tail -n 15 \"\$f\"; } || echo '  (no alerax .out yet)'" 2>/dev/null || true
fi
echo
echo "States: PD=pending R=running CG=completing CD=completed F=failed TO=timeout."
echo "When the jobs you care about show CD + outputs present -> bin/50_collect.sh then bin/60_report.py."
