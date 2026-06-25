#!/bin/bash -l
# ======================================================================
# STAGE B: free per-branch DTL fit with origination PINNED to the random-
# effects shared p^O (rfx_<root>.npz log_pO; 511=0, root 0.22 -- the no-dump
# profile). Isolates the branchwise-rate (esp. loss) channel: the 511 dump is
# structurally prevented (O fixed), so per-branch loss can only TRIM genomes.
# Then sample the ancestral genome and read root/511/DPANN vs the paper.
#
# sbatch -p short-a100 --gres=gpu:a100:1 -c 8 --mem=80G -t 00:55:00 \
#   --export=ALL,GPUREC_DIR=...,ROOT=Eury,PONPZ=.../rfx_Eury_stageB.npz,OUTDIR=.../stageB,WARM=... \
#   slurm/stageB_saion.sbatch
# ======================================================================
#SBATCH --job-name=stageB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${ROOT:=Eury}"; : "${PONPZ:?}"; : "${OUTDIR:?}"; : "${PY_MODULE:=python/3.11.11}"
: "${STEPS:=250}"; : "${NSAMP:=200}"; : "${FBS:=500}"; : "${SIGMA:=1.0}"
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
mkdir -p "$OUTDIR"
RATES="$OUTDIR/stageB_${ROOT}.rates"; PNC="$OUTDIR/stageB_${ROOT}_pnc"
WARMARG=""; [ -n "${WARM:-}" ] && [ -s "${WARM:-}" ] && WARMARG="--init-from-rates ${WARM}"

echo "=== STAGE B fit (free branchwise DTL, O fixed @ rfx p^O) root=$ROOT $(hostname) $(date) ==="
echo "    PONPZ=$PONPZ  WARM=${WARM:-none}  steps=$STEPS sigma=$SIGMA"
if [ ! -s "$RATES.json" ]; then
  PYTHONPATH=. "$PY" experiments/run_undine_branchwise.py --root "$ROOT" --out "$RATES" \
    --origination fixed --origination-fixed-po-npz "$PONPZ" \
    --prior brownian --brownian-sigma "$SIGMA" --fm-mode e-only --optimizer lbfgs \
    --steps "$STEPS" --family-batch-size "$FBS" $WARMARG 2>&1 | tail -8 || { echo "FIT FAILED"; exit 1; }
fi

echo "=== STAGE B sample genomes ==="
if [ ! -s "$PNC.perspecies_events.tsv" ]; then
  PYTHONPATH=. "$PY" experiments/per_node_copies.py --root "$ROOT" --sidecar "$RATES.json" \
    --engine cpp --n-samples "$NSAMP" --fm-mode e-only --out "$PNC" 2>&1 | tail -4 || { echo "PNC FAILED"; exit 1; }
fi

"$PY" - "$PNC.perspecies_events.tsv" <<'PYEOF'
import sys
rows={}
with open(sys.argv[1]) as fh:
    hd=fh.readline().rstrip().split("\t"); ci={c:i for i,c in enumerate(hd)}
    for ln in fh:
        p=ln.rstrip().split("\t"); rows[int(p[ci["node"]])]=p
def g(n,c):
    return float(rows[n][ci[c]]) if n in rows else float("nan")
print("STAGEB_GENOME  node: presence / copies   (paper presence) [rfx presence]")
for n,lab,pp,rfx in [(512,"LACA(root)",1331,1562),(511,"node511",1368,1384),(510,"DPANN",937,1057)]:
    print(f"  {lab:12s} node{n}: pres={g(n,'presence'):.0f}  cop={g(n,'copies'):.0f}   paper={pp}  rfx={rfx}")
PYEOF
echo "=== STAGE B DONE ($(date)) ==="
