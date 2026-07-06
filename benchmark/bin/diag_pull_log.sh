#!/usr/bin/env bash
# Pull a cluster job's log to a local text file (which the assistant can read via
# the shared /work mount). Grabs the section from the gpurec [auto-batch] line
# through the end (the traceback), plus the raw tail as a fallback.
#   usage: bin/diag_pull_log.sh [JOBID] [saion|deigo]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.sh"

JOB="${1:-4655026}"          # default: the crashed gpurec probe
WHERE="${2:-saion}"

if [ "$WHERE" = saion ]; then
  SSH="$SAION_SSH"; LOG="$SAION_GPUREC_DIR/gpurec_bench_${JOB}.out"
else
  SSH="$DEIGO_SSH"; LOG="$DEIGO_BENCH_DIR/alerax_bench_${JOB}.out"
fi

mkdir -p "$RESULTS_DIR/diag"
DEST="$RESULTS_DIR/diag/${WHERE}_${JOB}.txt"
echo "Pulling $SSH:$LOG -> $DEST"
{
  echo "===== [auto-batch] ONWARD ====="
  ssh -o BatchMode=yes "$SSH" "sed -n '/\[auto-batch\]/,\$p' '$LOG' 2>/dev/null | head -200" || true
  echo
  echo "===== LAST 200 LINES (full traceback fallback) ====="
  ssh -o BatchMode=yes "$SSH" "tail -200 '$LOG' 2>/dev/null" || true
} > "$DEST"
echo "wrote $DEST ($(wc -l < "$DEST") lines)"
