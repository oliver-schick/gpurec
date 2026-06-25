#!/usr/bin/env bash
# Create directories and stage inputs onto BOTH clusters.
#   DATA_SOURCE=cluster (default): each cluster fetches for itself --
#     Williams from Zenodo (curl + selective tar extract), gpurec/AleRax via
#     git clone. No local copy needed; nothing large is uploaded.
#   DATA_SOURCE=local: upload Williams + AleRax from the LOCAL_* paths instead.
# Idempotent: existing checkouts/data are reused; re-running is cheap.
source "$(cd "$(dirname "$0")/.." && pwd)/lib.sh"
start_log 10_stage
KIT="$(cd "$(dirname "$0")/.." && pwd)"
require_not_placeholder "$SAION_GPUREC_DIR"
require_not_placeholder "$DEIGO_BENCH_DIR"
command -v ssh >/dev/null || die "ssh not found on this machine"
[ "$DATA_SOURCE" = local ] && { command -v rsync >/dev/null || die "rsync needed for DATA_SOURCE=local"; }

WILLIAMS_ITEMS=(ccps rooted_phylogeny fraction_missing)
mk_dirs() { local host="$1"; shift; rsh "$host" "mkdir -p $(printf '%q ' "$@")"; }

# Download the 4.2 GB tarball ONCE to the shared path (visible from both
# clusters), so neither cluster re-downloads it. No-op if already there or if
# SHARED_TARBALL/cluster mode is off.
ensure_shared_tarball() {
  [ "$DATA_SOURCE" = local ] && return
  [ -z "$SHARED_TARBALL" ] && return
  local host="$SHARED_TARBALL_DLHOST"
  if rsh "$host" "[ -s '$SHARED_TARBALL' ]"; then
    log "Shared tarball already present: $SHARED_TARBALL ($(rsh "$host" "du -h '$SHARED_TARBALL' 2>/dev/null | cut -f1"))"
    return
  fi
  log "Downloading 3_Reconciliation.tar.gz ONCE on '$host' -> $SHARED_TARBALL (4.2 GB; shared by both clusters)"
  rsh "$host" "
    set -e
    command -v curl >/dev/null || { echo 'no curl on $host'; exit 1; }
    mkdir -p '$(dirname "$SHARED_TARBALL")'
    curl -fL --retry 3 -o '$SHARED_TARBALL.part' '$ZENODO_TARBALL_URL'
    mv '$SHARED_TARBALL.part' '$SHARED_TARBALL'
    echo '  downloaded:' \$(du -h '$SHARED_TARBALL' | cut -f1)
  " || die "shared tarball download failed on '$host' (check $SHARED_TARBALL is writable + login-node internet)"
}

# Fetch Williams data onto <host>:<dest> (the .../Williams_et_al_2017 dir).
fetch_williams() {
  local host="$1" dest="$2" tarball="$3"   # tarball = optional pre-existing path on host
  if rsh "$host" "[ -d '$dest/ccps' ] && [ -n \"\$(find '$dest/ccps' -maxdepth 1 -name '*.ale' ! -name '._*' -print -quit)\" ]"; then
    log "  [$host] Williams already present at $dest (skip; rm ccps/ to force)"
    return
  fi
  rsh "$host" "mkdir -p '$dest'"

  if [ "$DATA_SOURCE" = local ]; then
    [ -d "$LOCAL_WILLIAMS_DIR/ccps" ] || die "DATA_SOURCE=local but local Williams missing: $LOCAL_WILLIAMS_DIR"
    log "  [$host] uploading Williams from $LOCAL_WILLIAMS_DIR (~3.3 GB) ..."
    local srcs=(); for it in "${WILLIAMS_ITEMS[@]}"; do srcs+=("$LOCAL_WILLIAMS_DIR/$it"); done
    rsync -az --info=progress2 --exclude='._*' --exclude='.DS_Store' \
      -e "ssh -o BatchMode=yes" "${srcs[@]}" "${host}:${dest}/"
  else
    # cluster-side: use an existing tarball if given, else download from Zenodo.
    rsh "$host" "command -v curl >/dev/null" || die "[$host] curl not found (needed for Zenodo download)"
    local tb
    if [ -n "$tarball" ] && rsh "$host" "[ -f '$tarball' ]"; then
      tb="$tarball"; log "  [$host] using existing tarball $tb"
    else
      tb="$(dirname "$dest")/3_Reconciliation.tar.gz"
      log "  [$host] downloading 3_Reconciliation.tar.gz from Zenodo (4.2 GB, one time) ..."
      rsh "$host" "
        set -e
        if [ -s '$tb' ]; then echo '  tarball already downloaded ('\$(du -h '$tb' | cut -f1)')';
        else curl -fL --retry 3 -o '$tb.part' '$ZENODO_TARBALL_URL' && mv '$tb.part' '$tb'; fi
      " || die "[$host] Zenodo download failed (login-node internet? proxy?)"
    fi
    log "  [$host] extracting Williams_et_al_2017 subset -> $dest"
    # --warning=no-unknown-keyword silences the macOS xattr header noise.
    rsh "$host" "tar --warning=no-unknown-keyword --exclude='._*' --exclude='.DS_Store' -xzf '$tb' \
        --strip-components=$WILLIAMS_TAR_STRIP -C '$dest' $WILLIAMS_TAR_MEMBERS"
  fi
  local n; n=$(rsh "$host" "find '$dest/ccps' -maxdepth 1 -name '*.ale' ! -name '._*' | wc -l" 2>/dev/null)
  log "  [$host] Williams ready: $n .ale at $dest"
}

# ---------------- Shared tarball (download once) ----------------------
ensure_shared_tarball

# ---------------- Saion: dirs + gpurec + data ------------------------
log "Saion: creating directories"
mk_dirs "$SAION_SSH" "$SAION_BENCH_DIR" "$SAION_BENCH_DIR/out" "$WILLIAMS_DIR_SAION"

if rsh "$SAION_SSH" "[ -d '$SAION_GPUREC_DIR/.git' ]"; then
  : # checkout exists; deploy below
elif [ "$GPUREC_AUTOCLONE" = 1 ]; then
  log "Saion: cloning gpurec from $GPUREC_CLONE_URL -> $SAION_GPUREC_DIR"
  rsh "$SAION_SSH" "mkdir -p '$(dirname "$SAION_GPUREC_DIR")' && git clone '$GPUREC_CLONE_URL' '$SAION_GPUREC_DIR'" \
    || die "Saion: gpurec clone failed (check URL / git auth on the cluster)"
else
  die "Saion: no gpurec checkout at $SAION_GPUREC_DIR and GPUREC_AUTOCLONE=0"
fi

log "Saion: deploying gpurec ($GPUREC_BRANCH)"
rsh "$SAION_SSH" "
  set -e
  cd '$SAION_GPUREC_DIR'
  git config remote.$GPUREC_REMOTE.fetch '+refs/heads/*:refs/remotes/$GPUREC_REMOTE/*'
  git fetch $GPUREC_REMOTE
  git reset --hard $GPUREC_REMOTE/$GPUREC_BRANCH
  echo '  HEAD now:' \$(git log --oneline -1)
  mkdir -p slurm
  if [ -f .torch_ext/preprocess_cpp/preprocess_cpp.so ]; then
     touch .torch_ext/preprocess_cpp/preprocess_cpp.so; echo '  touched cached .so (skip JIT rebuild)'
  else echo '  WARN: no cached .so -- first run JIT-builds (needs devtoolset-11 node)'; fi
"
log "Saion: staging gpurec sbatch wrappers + benchmark driver (untracked; survive reset --hard)"
push "$SAION_SSH" "$KIT/slurm/bench_gpurec_saion.sbatch" "$SAION_GPUREC_DIR/slurm"
push "$SAION_SSH" "$KIT/slurm/fidelity_gpurec_saion.sbatch" "$SAION_GPUREC_DIR/slurm"
push "$SAION_SSH" "$KIT/bench_gpurec_fit.py" "$SAION_GPUREC_DIR/experiments"

log "Saion: ensuring Williams data"
fetch_williams "$SAION_SSH" "$WILLIAMS_DIR_SAION" "${SAION_WILLIAMS_TARBALL:-}"

# ---------------- Deigo: dirs + AleRax + data ------------------------
log "Deigo: creating directories"
mk_dirs "$DEIGO_SSH" "$DEIGO_BENCH_DIR" "$DEIGO_BENCH_DIR/out" "$DEIGO_BENCH_DIR/slurm" "$WILLIAMS_DIR_DEIGO"

if rsh "$DEIGO_SSH" "[ -d '$DEIGO_ALERAX_DIR/src' ]"; then
  log "Deigo: AleRax source already present at $DEIGO_ALERAX_DIR"
elif [ "$DATA_SOURCE" = local ]; then
  log "Deigo: uploading AleRax source from $LOCAL_ALERAX_DIR"
  [ -d "$LOCAL_ALERAX_DIR/src" ] || die "local AleRax missing: $LOCAL_ALERAX_DIR"
  rsh "$DEIGO_SSH" "mkdir -p '$DEIGO_ALERAX_DIR'"
  rsync -az --delete --exclude='build/' --exclude='.git/' \
    -e "ssh -o BatchMode=yes" "$LOCAL_ALERAX_DIR/" "${DEIGO_SSH}:${DEIGO_ALERAX_DIR}/"
elif [ "$ALERAX_AUTOCLONE" = 1 ]; then
  log "Deigo: cloning AleRax (recursive) from $ALERAX_CLONE_URL"
  rsh "$DEIGO_SSH" "mkdir -p '$(dirname "$DEIGO_ALERAX_DIR")' && git clone --recursive '$ALERAX_CLONE_URL' '$DEIGO_ALERAX_DIR'" \
    || die "Deigo: AleRax clone failed (recursive submodules; check git/network)"
else
  die "Deigo: no AleRax at $DEIGO_ALERAX_DIR and ALERAX_AUTOCLONE=0"
fi

log "Deigo: staging alerax sbatch wrapper"
push "$DEIGO_SSH" "$KIT/slurm/alerax_deigo.sbatch" "$DEIGO_BENCH_DIR/slurm"

log "Deigo: ensuring Williams data"
fetch_williams "$DEIGO_SSH" "$WILLIAMS_DIR_DEIGO" "${DEIGO_WILLIAMS_TARBALL:-}"

log "STAGE complete. Re-run bin/00_preflight.sh to confirm all green, then bin/20_prepare_families.sh pilot"
