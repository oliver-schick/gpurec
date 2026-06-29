#!/usr/bin/env bash
# Build the AleRax families.txt ON Deigo for a phase (pilot|full), using the
# SAME first-N sorted .ale files gpurec uses (sorted glob of ccps/*.ale,
# minus macOS ._* stubs). No mapping lines: AleRax auto-maps gene->species by
# the prefix-before-first-'_' rule, identical to gpurec.
#
# usage: bin/20_prepare_families.sh pilot|full
source "$(cd "$(dirname "$0")/.." && pwd)/lib.sh"
PHASE="${1:?usage: 20_prepare_families.sh pilot|full}"
start_log "20_prepare_${PHASE}"

case "$PHASE" in
  pilot) N="$PILOT_N" ;;
  full)  N="$FULL_N" ;;
  *) die "phase must be pilot|full" ;;
esac

FAM="$DEIGO_BENCH_DIR/families_${PHASE}.txt"
ALE_DIR="$DS_ALE_DIR_DEIGO"     # ccps/ (williams) or ALEs/ (davin)
log "Deigo: building $FAM  (N=$N ; 0=all) from $ALE_DIR  [dataset=$DATASET]"

# Remote generator. find (not ls *.ale) to dodge ARG_MAX at thousands of files.
# Family name = filename up to the first dot (works for both datasets: williams
# 'fix_1000_c60_lg_combined.ale' and davin 'COG0001_0.faa...ale'). No mapping
# lines: AleRax auto-maps gene->species by prefix-before-first-'_', as gpurec does.
rsh "$DEIGO_SSH" "
  set -e
  ALE='$ALE_DIR'
  N=$N
  OUT='$FAM'
  mkdir -p '$DEIGO_BENCH_DIR'
  [ -d \"\$ALE\" ] || { echo \"ERROR: no .ale dir on deigo: \$ALE (run 10_stage.sh)\"; exit 1; }
  LIST=\$(find \"\$ALE\" -maxdepth 1 -name '*.ale' ! -name '._*' | sort)
  if [ \"\$N\" -gt 0 ]; then LIST=\$(printf '%s\n' \"\$LIST\" | head -n \"\$N\"); fi
  cnt=\$(printf '%s\n' \"\$LIST\" | sed '/^\$/d' | wc -l)
  {
    echo '[FAMILIES]'
    printf '%s\n' \"\$LIST\" | sed '/^\$/d' | while read -r f; do
      name=\$(basename \"\$f\" | cut -d. -f1)
      echo \"- \$name\"
      echo \"gene_tree = \$f\"
    done
  } > \"\$OUT\"
  echo \"  wrote \$OUT with \$cnt families\"
"
log "PREPARE($PHASE) complete. Next: bin/30_run_alerax.sh $PHASE  and  bin/40_run_gpurec.sh $PHASE"
