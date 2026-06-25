#!/bin/bash -l
# ======================================================================
# "root fraction + uniform + branchwise" for Eury: branchwise DTL + a 2-CLASS
# origination (root gets its own level, all other branches share one uniform
# level -- both fit). Warm-started from the Stage-A rate centroid. Then sample
# the ancestral genome and report LACA/511/DPANN + the fitted root O fraction.
#
# sbatch -p short-a100 --gres=gpu:a100:1 -c 8 --mem=80G -t 00:55:00 \
#   --export=ALL,GPUREC_DIR=...,RFX=.../rfx_Eury.npz,OUTDIR=.../rootclass_Eury slurm/rootclass_saion.sbatch
# ======================================================================
#SBATCH --job-name=rootcls
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${RFX:?}"; : "${OUTDIR:?}"; : "${ROOT:=Eury}"; : "${PY_MODULE:=python/3.11.11}"
: "${STEPS:=250}"; : "${FBS:=500}"; : "${SIGMA:=1.0}"; : "${NSAMP:=200}"
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
INIT="$OUTDIR/init.json"; RATES="$OUTDIR/rootclass.rates"; PNC="$OUTDIR/rootclass_pnc"
echo "=== rootclass (uniform + root-fraction + branchwise) root=$ROOT $(hostname) $(date) ==="

# Stage-A warm start
[ -s "$INIT" ] || PYTHONPATH=. "$PY" experiments/rfx_init_rates.py --rfx-npz "$RFX" --out "$INIT" \
  || { echo "INIT FAILED"; exit 1; }

# fit: branchwise DTL + 2-class origination (root level + uniform rest)
if [ ! -s "$RATES.json" ]; then
  PYTHONPATH=. "$PY" experiments/run_undine_branchwise.py --root "$ROOT" --out "$RATES" \
    --origination optimize --origination-root-class --init-from-rates "$INIT" \
    --prior brownian --brownian-sigma "$SIGMA" --fm-mode e-only --optimizer lbfgs \
    --steps "$STEPS" --family-batch-size "$FBS" 2>&1 | tail -8 || { echo "FIT FAILED"; exit 1; }
fi

# sample genome
if [ ! -s "$PNC.perspecies_events.tsv" ]; then
  PYTHONPATH=. "$PY" experiments/per_node_copies.py --root "$ROOT" --sidecar "$RATES.json" \
    --engine cpp --n-samples "$NSAMP" --fm-mode e-only --out "$PNC" 2>&1 | tail -4 || { echo "PNC FAILED"; exit 1; }
fi

"$PY" - "$PNC.perspecies_events.tsv" "$RATES.json" <<'PYEOF'
import sys, json
rows={}
with open(sys.argv[1]) as fh:
    hd=fh.readline().rstrip().split("\t"); ci={c:i for i,c in enumerate(hd)}
    for ln in fh:
        p=ln.rstrip().split("\t"); rows[int(p[ci["node"]])]=p
def g(n,c): return float(rows[n][ci[c]]) if n in rows else float("nan")
sc=json.load(open(sys.argv[2])); od=sc.get("origination",{})
po=od.get("origination_prob"); rb=512
rootfrac=f"{po[rb]:.4f}" if po else "?"
print("ROOTCLASS_GENOME (uniform + root-fraction + branchwise, Eury) -- copies")
print(f"  root O fraction (fitted p^O at LACA) = {rootfrac}")
for n,lab,pp in [(512,"LACA(root)",1456),(511,"node511",1493),(510,"DPANN",1066)]:
    print(f"  {lab:12s} node{n}: presence={g(n,'presence'):.0f}  copies={g(n,'copies'):.0f}   paper_copies={pp}")
PYEOF
echo "=== rootclass DONE ($(date)) ==="
