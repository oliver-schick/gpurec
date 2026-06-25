#!/bin/bash -l
# ======================================================================
# STAGE B per-root, PHASE 2 (genome): sample the ancestral genome at each
# root's fitted branchwise sidecar (from phase 1) and read root/511/DPANN vs
# the paper -> the "reasonable copies" deliverable, per root. Run after phase 1.
#
# sbatch --dependency=afterok:<phase1> -p short-a100 --gres=gpu:a100:1 -c 8 --mem=80G \
#   -t 00:55:00 --array=0-7 --export=ALL,GPUREC_DIR=...,OUTDIR=.../stageB_au \
#   slurm/stageB_genome_saion.sbatch
# ======================================================================
#SBATCH --job-name=sBgen
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${OUTDIR:?}"; : "${PY_MODULE:=python/3.11.11}"; : "${NSAMP:=200}"
ROOTS=(Alti AMD Asgard Cluster2 DPANN Eury HalobacThermopl Kor MHH Micra5 MicraDia TACA TackA TAC UndineClu2)
ROOT="${ROOTS[$SLURM_ARRAY_TASK_ID]}"
cd "$GPUREC_DIR"
export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"
module load "$PY_MODULE" 2>/dev/null || echo "WARN module load $PY_MODULE"
if module load gcc/14.2.0 2>/dev/null; then
  GCC14LIB=/apps/unit/SzollosiU/gcc/14.2.0/lib64
  [ -d "$GCC14LIB" ] && export LD_LIBRARY_PATH="$GCC14LIB:${LD_LIBRARY_PATH:-}"
fi
for SO in .torch_ext/preprocess_cpp/preprocess_cpp.so .torch_ext/recon_sampler/recon_sampler.so; do [ -f "$SO" ] && touch "$SO"; done
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3.11"
RATES="$OUTDIR/stageB_${ROOT}.rates"; PNC="$OUTDIR/stageB_${ROOT}_pnc"
[ -s "$RATES.json" ] || { echo "[$ROOT] no sidecar $RATES.json -- phase 1 missing"; exit 1; }
echo "=== STAGE B genome root=$ROOT on $(hostname) $(date) ==="
if [ ! -s "$PNC.perspecies_events.tsv" ]; then
  PYTHONPATH=. "$PY" experiments/per_node_copies.py --root "$ROOT" --sidecar "$RATES.json" \
    --engine cpp --n-samples "$NSAMP" --fm-mode e-only --out "$PNC" 2>&1 | tail -4 || { echo "[$ROOT] PNC FAILED"; exit 1; }
fi
"$PY" - "$PNC.perspecies_events.tsv" "$ROOT" <<'PYEOF'
import sys
rows={}
with open(sys.argv[1]) as fh:
    hd=fh.readline().rstrip().split("\t"); ci={c:i for i,c in enumerate(hd)}
    for ln in fh:
        p=ln.rstrip().split("\t"); rows[int(p[ci["node"]])]=p
def g(n,c):
    return float(rows[n][ci[c]]) if n in rows else float("nan")
# node ids are root-tree-specific; 512/511/510 hold for Eury-rooted numbering. Report by label too.
lab={}
for n,p in rows.items():
    L=p[ci["label"]]
    if L in ("DPANN","root") or n in (510,511,512): lab[n]=L or str(n)
print(f"STAGEB_GENOME root={sys.argv[2]}: "
      + "  ".join(f"{(p[ci['label']] or 'n'+str(n))}={g(n,'presence'):.0f}"
                  for n,p in sorted(rows.items()) if (p[ci['label']] in ('DPANN','root') or n in (510,511,512))))
PYEOF
echo "=== [$ROOT] genome DONE ($(date)) ==="
