#!/usr/bin/env bash
# Shared helpers for the benchmark kit. Sourced by every bin/ script.
# ASCII only (cluster logs choke on non-ASCII).
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_HERE}/config.sh"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Tee all stdout+stderr of the calling script to results/logs/<name>.log so
# the run is captured to a file (readable later / pasteable). Call once near
# the top of a bin/ script: start_log <name>
start_log() {
  local name="$1" dir="${RESULTS_DIR}/logs"
  mkdir -p "$dir"
  local f="${dir}/${name}.log"
  exec > >(tee "$f") 2>&1
  printf '=== %s @ %s ===\n' "$name" "$(date '+%Y-%m-%d %H:%M:%S')"
  printf '(logging to %s)\n' "$f"
}

# Run a command on a cluster over SSH. Filters the harmless saion warnings.
# usage: rsh <ssh-alias> "<remote command>"
rsh() {
  local host="$1"; shift
  ssh -o BatchMode=yes "$host" "$@" 2> >(grep -ivE "forwarding|bind on|bind \[|channel" >&2)
}

# rsync a local path to a cluster path over the SSH alias.
# usage: push <ssh-alias> <local> <remote>
push() {
  local host="$1" src="$2" dst="$3"
  rsh "$host" "mkdir -p '$dst'"
  rsync -az --info=stats1 -e "ssh -o BatchMode=yes" "$src" "${host}:${dst}/"
}

# Pull a remote path back to a local dir.
# usage: pull <ssh-alias> <remote> <localdir>
pull() {
  local host="$1" src="$2" dstdir="$3"
  mkdir -p "$dstdir"
  rsync -az -e "ssh -o BatchMode=yes" "${host}:${src}" "$dstdir/"
}

# Tag for output files: <mode>_<root>_<phase>_<N>
run_tag() {
  local phase="$1" n="$2"
  printf '%s_%s_%s_%s' "$MODE" "$ROOT" "$phase" "$n"
}

require_not_placeholder() {
  case "$1" in
    *"<YourUnitU>"*|*"<your-id>"*)
      die "config.sh still has placeholders in: $1
  -> Edit config.sh and set SAION_GPUREC_DIR / *_BENCH_DIR / WILLIAMS_DIR_* to your real /work paths." ;;
  esac
}
