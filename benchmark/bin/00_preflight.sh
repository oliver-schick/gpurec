#!/usr/bin/env bash
# Preflight: comprehensive, READ-ONLY check of what exists vs what the
# benchmark needs, on BOTH clusters and locally. Creates nothing and
# submits nothing -- it prints an [OK]/[NEED] line per item and a final
# "ACTIONS NEEDED" list. Run bin/10_stage.sh to create dirs + copy data.
source "$(cd "$(dirname "$0")/.." && pwd)/lib.sh"
start_log 00_preflight

NEED=()   # collected action items
mark()  { printf '  [%s] %s\n' "$1" "$2"; }     # $1=OK|NEED|warn  $2=text
need()  { NEED+=("$1"); }

require_not_placeholder "$SAION_GPUREC_DIR"
require_not_placeholder "$WILLIAMS_DIR_SAION"
require_not_placeholder "$DEIGO_BENCH_DIR"

# nearest-existing-ancestor writability probe (no mkdir).
WRITABLE_PROBE='d="__D__"; while [ ! -e "$d" ]; do d=$(dirname "$d"); done; [ -w "$d" ] && echo "writable($d)" || echo "NOTwritable($d)"'
probe_writable() { rsh "$1" "${WRITABLE_PROBE/__D__/$2}" 2>/dev/null; }

# does a path exist on a host?
rexists() { rsh "$1" "[ -e '$2' ] && echo yes || echo no" 2>/dev/null; }

echo "============================================================"
echo " LOCAL (this computer)"
echo "============================================================"
mark OK "DATA_SOURCE=$DATA_SOURCE  (cluster = fetch on each cluster; local = upload from LaCie)"
command -v ssh >/dev/null 2>&1 && mark OK "local ssh present" || { mark NEED "local ssh MISSING"; need "install ssh on this machine"; }
if [ "$DATA_SOURCE" = local ]; then
  command -v rsync >/dev/null 2>&1 && mark OK "local rsync present" || { mark NEED "local rsync MISSING"; need "install rsync (DATA_SOURCE=local)"; }
  if [ -d "$LOCAL_WILLIAMS_DIR/ccps" ]; then
    n=$(find "$LOCAL_WILLIAMS_DIR/ccps" -maxdepth 1 -name '*.ale' ! -name '._*' | wc -l)
    mark OK "local Williams ccps ($n .ale) at $LOCAL_WILLIAMS_DIR"
  else mark NEED "local Williams data missing: $LOCAL_WILLIAMS_DIR"; need "fix LOCAL_WILLIAMS_DIR (DATA_SOURCE=local source)"; fi
  [ -d "$LOCAL_ALERAX_DIR/src" ] && mark OK "local AleRax source at $LOCAL_ALERAX_DIR" \
    || { mark NEED "local AleRax missing: $LOCAL_ALERAX_DIR"; need "fix LOCAL_ALERAX_DIR (DATA_SOURCE=local)"; }
else
  mark OK "cluster-side fetch: Williams from Zenodo, gpurec/AleRax via git clone (no local copy needed)"
fi

echo
echo "============================================================"
echo " SHARED STORAGE (download the 4.2 GB tarball once)"
echo "============================================================"
if [ "$DATA_SOURCE" = cluster ] && [ -n "$SHARED_TARBALL" ]; then
  mark OK "shared tarball: $SHARED_TARBALL (download host '$SHARED_TARBALL_DLHOST')"
  echo "  dir writability on '$SHARED_TARBALL_DLHOST': $(probe_writable "$SHARED_TARBALL_DLHOST" "$SHARED_TARBALL")"
  [ "$(rexists "$SHARED_TARBALL_DLHOST" "$SHARED_TARBALL")" = yes ] && mark OK "tarball already downloaded (will reuse)"
  sdir="$(dirname "$SHARED_TARBALL")"
  for h in "$SAION_SSH" "$DEIGO_SSH"; do
    if rsh "$h" 'true' 2>/dev/null; then
      [ "$(rexists "$h" "$sdir")" = yes ] && mark OK "shared dir visible from '$h'" \
        || { mark NEED "shared dir NOT visible from '$h': $sdir"; need "'$h': $sdir absent -- if the FS is not truly shared, set SHARED_TARBALL='' (download per-cluster) or a path both see"; }
    fi
  done
else
  mark OK "per-cluster download (SHARED_TARBALL empty, or DATA_SOURCE=local)"
fi

echo
echo "============================================================"
echo " SAION  (alias '$SAION_SSH')  -- gpurec / A100"
echo "============================================================"
if ! rsh "$SAION_SSH" 'true' 2>/dev/null; then
  mark NEED "cannot ssh '$SAION_SSH'"; need "fix SAION_SSH or connect to OIST network/VPN (skipping Saion checks)"
else
  mark OK "ssh '$SAION_SSH' -> $(rsh "$SAION_SSH" hostname 2>/dev/null)"
  rsh "$SAION_SSH" 'command -v sbatch >/dev/null' && mark OK "sbatch present" || { mark NEED "no sbatch on Saion"; need "Saion: Slurm not found"; }
  # partition
  if rsh "$SAION_SSH" "sinfo -h -p '$SAION_PARTITION' -o %P >/dev/null 2>&1 && [ -n \"\$(sinfo -h -p '$SAION_PARTITION' -o %P 2>/dev/null)\" ]"; then
    mark OK "partition '$SAION_PARTITION' exists ($(rsh "$SAION_SSH" "sinfo -h -p '$SAION_PARTITION' -o '%a %D nodes'" 2>/dev/null | head -1))"
  else mark NEED "partition '$SAION_PARTITION' not visible to you"; need "Saion: check SAION_PARTITION (sinfo -s) / your assoc"; fi
  # python module + ruse
  rsh "$SAION_SSH" "bash -lc 'module avail $SAION_PYTHON_MODULE' 2>&1 | grep -qi python" \
    && mark OK "module $SAION_PYTHON_MODULE available" || { mark warn "module $SAION_PYTHON_MODULE not found by 'module avail'"; need "Saion: confirm SAION_PYTHON_MODULE name"; }
  rsh "$SAION_SSH" "bash -lc 'module avail ruse' 2>&1 | grep -qi ruse" \
    && mark OK "ruse module available" || mark warn "no ruse module (wrapper falls back to /usr/bin/time)"
  # gpurec checkout
  if [ "$(rexists "$SAION_SSH" "$SAION_GPUREC_DIR/.git")" = yes ]; then
    info=$(rsh "$SAION_SSH" "cd '$SAION_GPUREC_DIR' && echo \"origin=\$(git config --get remote.$GPUREC_REMOTE.url) head=\$(git rev-parse --abbrev-ref HEAD 2>/dev/null)\"" 2>/dev/null)
    mark OK "gpurec checkout present ($info)"
    rsh "$SAION_SSH" "cd '$SAION_GPUREC_DIR' && git ls-remote --exit-code $GPUREC_REMOTE $GPUREC_BRANCH >/dev/null 2>&1" \
      && mark OK "remote branch $GPUREC_REMOTE/$GPUREC_BRANCH reachable" \
      || { mark NEED "cannot reach $GPUREC_REMOTE/$GPUREC_BRANCH"; need "Saion: verify gpurec remote/branch + git auth"; }
    [ "$(rexists "$SAION_SSH" "$SAION_GPUREC_DIR/.venv/bin/python")" = yes ] \
      && mark OK "gpurec .venv present" \
      || { mark warn ".venv not found in gpurec dir (sbatch falls back to module python3.11; needs torch+triton+gpurec installed)"; need "Saion: ensure a Python env with torch+triton+gpurec is importable"; }
    [ "$(rexists "$SAION_SSH" "$SAION_GPUREC_DIR/.torch_ext/preprocess_cpp/preprocess_cpp.so")" = yes ] \
      && mark OK "cached preprocess_cpp.so present (no JIT rebuild)" \
      || mark warn "no cached .so -- first gpurec import JIT-builds (needs a devtoolset-11 node)"
  elif [ "$GPUREC_AUTOCLONE" = 1 ]; then
    mark warn "gpurec checkout missing -- bin/10_stage.sh will clone $GPUREC_CLONE_URL"
  else
    mark NEED "gpurec checkout MISSING at $SAION_GPUREC_DIR (GPUREC_AUTOCLONE=0)"
    need "Saion: git clone your fork to $SAION_GPUREC_DIR (branch $GPUREC_BRANCH)"
  fi
  rsh "$SAION_SSH" "command -v curl >/dev/null" && mark OK "curl present (for Zenodo download)" \
    || { [ "$DATA_SOURCE" = cluster ] && { mark NEED "no curl on Saion login node"; need "Saion: need curl for Zenodo download (or set DATA_SOURCE=local)"; }; }
  # Williams data on Saion
  if [ "$(rexists "$SAION_SSH" "$WILLIAMS_DIR_SAION/ccps")" = yes ]; then
    n=$(rsh "$SAION_SSH" "find '$WILLIAMS_DIR_SAION/ccps' -maxdepth 1 -name '*.ale' ! -name '._*' | wc -l" 2>/dev/null)
    mark OK "Williams ccps present ($n .ale)"
  else mark NEED "Williams data MISSING on Saion: $WILLIAMS_DIR_SAION/ccps"; need "Saion: bin/10_stage.sh will copy Williams here (~3.3 GB)"; fi
  [ "$(rexists "$SAION_SSH" "$WILLIAMS_DIR_SAION/rooted_phylogeny/$ROOT")" = yes ] \
    && mark OK "species tree rooted_phylogeny/$ROOT present" || mark NEED "species tree rooted_phylogeny/$ROOT missing (staged with data)"
  # bench/out dir writability
  ws="$(probe_writable "$SAION_SSH" "$SAION_BENCH_DIR")"
  echo "  bench dir: $ws"
  case "$ws" in NOTwritable*) need "Saion: $SAION_BENCH_DIR not creatable ($ws) -- fix SAION_BENCH_DIR";; esac
fi

echo
echo "============================================================"
echo " DEIGO  (alias '$DEIGO_SSH')  -- AleRax / CPU-MPI"
echo "============================================================"
if ! rsh "$DEIGO_SSH" 'true' 2>/dev/null; then
  mark NEED "cannot ssh '$DEIGO_SSH'"; need "fix DEIGO_SSH or connect to OIST network/VPN (skipping Deigo checks)"
else
  mark OK "ssh '$DEIGO_SSH' -> $(rsh "$DEIGO_SSH" hostname 2>/dev/null)"
  rsh "$DEIGO_SSH" 'command -v sbatch >/dev/null' && mark OK "sbatch present" || { mark NEED "no sbatch on Deigo"; need "Deigo: Slurm not found"; }
  if rsh "$DEIGO_SSH" "[ -n \"\$(sinfo -h -p '$DEIGO_PARTITION' -o %P 2>/dev/null)\" ]"; then
    mark OK "partition '$DEIGO_PARTITION' exists"
  else mark NEED "partition '$DEIGO_PARTITION' not visible"; need "Deigo: check DEIGO_PARTITION"; fi
  # modules: actually try to LOAD each exact module name (catches e.g.
  # 'openmpi' vs 'openmpi.gcc/5.0.3'); a substring match would false-pass.
  echo "  test-loading DEIGO_MODULES ($DEIGO_MODULES) + ruse ..."
  for m in $DEIGO_MODULES ruse; do
    if rsh "$DEIGO_SSH" "bash -lc 'module load $m' 2>/dev/null"; then
      mark OK "module '$m' loads"
    else
      mark NEED "module '$m' fails to load on Deigo"
      need "Deigo: fix DEIGO_MODULES name '$m' (ssh $DEIGO_SSH 'module avail ${m%%/*}')"
    fi
  done
  # Williams data on Deigo
  if [ "$(rexists "$DEIGO_SSH" "$WILLIAMS_DIR_DEIGO/ccps")" = yes ]; then
    n=$(rsh "$DEIGO_SSH" "find '$WILLIAMS_DIR_DEIGO/ccps' -maxdepth 1 -name '*.ale' ! -name '._*' | wc -l" 2>/dev/null)
    mark OK "Williams ccps present ($n .ale)"
  else mark NEED "Williams data MISSING on Deigo: $WILLIAMS_DIR_DEIGO/ccps"; need "Deigo: bin/10_stage.sh will copy Williams here (~3.3 GB)"; fi
  # AleRax
  if [ "$(rexists "$DEIGO_SSH" "$DEIGO_ALERAX_DIR/build/bin/alerax")" = yes ]; then
    mark OK "AleRax already built on Deigo"
  elif [ "$(rexists "$DEIGO_SSH" "$DEIGO_ALERAX_DIR/src")" = yes ]; then
    mark warn "AleRax source staged but not built (the sbatch builds it on first run)"
  else mark warn "AleRax not staged -- bin/10_stage.sh clones it ($ALERAX_CLONE_URL), sbatch builds it"; fi
  rsh "$DEIGO_SSH" "command -v curl >/dev/null" && mark OK "curl present (for Zenodo download)" \
    || { [ "$DATA_SOURCE" = cluster ] && { mark NEED "no curl on Deigo login node"; need "Deigo: need curl for Zenodo download (or DATA_SOURCE=local)"; }; }
  w="$(probe_writable "$DEIGO_SSH" "$DEIGO_BENCH_DIR")"
  echo "  bench dir: $w"
  case "$w" in NOTwritable*) need "Deigo: $DEIGO_BENCH_DIR not creatable ($w) -- set DEIGO_BENCH_DIR/WILLIAMS_DIR_DEIGO to your writable Deigo path (likely under /flash or /bucket, not /work)";; esac
fi

echo
echo "============================================================"
if [ "${#NEED[@]}" -eq 0 ]; then
  echo " PREFLIGHT: all green. Proceed to bin/10_stage.sh."
else
  echo " ACTIONS NEEDED (${#NEED[@]}):"
  for x in "${NEED[@]}"; do echo "   - $x"; done
  echo
  echo " Most of these are resolved by bin/10_stage.sh (creates dirs, copies"
  echo " Williams to both clusters, deploys gpurec, stages AleRax). Items about"
  echo " ssh/modules/the gpurec clone you must fix by hand first."
fi
echo "============================================================"
