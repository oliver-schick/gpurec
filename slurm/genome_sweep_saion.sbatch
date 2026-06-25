#!/bin/bash -l
# ======================================================================
# 32-config GENOME sweep on short-a100 (one GPU/config). Each task: a full
# per-branch undine fit with a different principled config (origination
# treatment x loss/rate regularization), THEN sample ancestral genomes and
# read the internal node sizes (root / 511 / DPANN / deep clades) + the
# prior-free data logL. Target: which non-arbitrary settings deflate the
# high-in-the-tree genomes (no ~2k at DPANN), and what does the data support?
#
# sbatch -p short-a100 --gres=gpu:a100:1 -c 8 --mem=72G -t 00:58:00 --array=0-31 \
#   --export=ALL,GPUREC_DIR=...,ROOT=Eury,WARM=...,OUTDIR=... slurm/genome_sweep_saion.sbatch
# (each task <1h -> NON-PREEMPTIBLE on short-a100.)
# ======================================================================
#SBATCH --job-name=gsweep
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --requeue
set -uo pipefail
: "${GPUREC_DIR:?}"; : "${ROOT:=Eury}"; : "${WARM:?}"; : "${OUTDIR:?}"; : "${PY_MODULE:=python/3.11.11}"
: "${STEPS:=200}"; : "${NSAMP:=200}"; : "${FBS:=500}"

# --- 32 principled configs: tag|fit-EXTRA_ARGS (free per-branch DTL unless noted) ---
CONFIGS=(
 "optimize|--origination optimize --prior brownian --brownian-sigma 1.0"
 "uniform|--origination uniform --prior brownian --brownian-sigma 1.0"
 "depth30|--origination optimize --origination-depth-lambda 30 --prior brownian --brownian-sigma 1.0"
 "depth100|--origination optimize --origination-depth-lambda 100 --prior brownian --brownian-sigma 1.0"
 "depth300|--origination optimize --origination-depth-lambda 300 --prior brownian --brownian-sigma 1.0"
 "depth1000|--origination optimize --origination-depth-lambda 1000 --prior brownian --brownian-sigma 1.0"
 "depth3000|--origination optimize --origination-depth-lambda 3000 --prior brownian --brownian-sigma 1.0"
 "rootclass|--origination optimize --origination-root-class --prior brownian --brownian-sigma 1.0"
 "bs0.25|--origination optimize --prior brownian --brownian-sigma 0.25"
 "bs0.5|--origination optimize --prior brownian --brownian-sigma 0.5"
 "bs0.75|--origination optimize --prior brownian --brownian-sigma 0.75"
 "bs1.5|--origination optimize --prior brownian --brownian-sigma 1.5"
 "bs2|--origination optimize --prior brownian --brownian-sigma 2.0"
 "bs3|--origination optimize --prior brownian --brownian-sigma 3.0"
 "bs4|--origination optimize --prior brownian --brownian-sigma 4.0"
 "bs6|--origination optimize --prior brownian --brownian-sigma 6.0"
 "tv10|--origination optimize --dtl-tv-lambda 10"
 "tv30|--origination optimize --dtl-tv-lambda 30"
 "tv100|--origination optimize --dtl-tv-lambda 100"
 "tv300|--origination optimize --dtl-tv-lambda 300"
 "tv1000|--origination optimize --dtl-tv-lambda 1000"
 "tv3000|--origination optimize --dtl-tv-lambda 3000"
 "tv300uniO|--origination uniform --dtl-tv-lambda 300"
 "tv1000uniO|--origination uniform --dtl-tv-lambda 1000"
 "clade|--clade-groups-from-tree --origination optimize --prior brownian --brownian-sigma 1.0"
 "cladeUniO|--clade-groups-from-tree --origination uniform"
 "optNoPrior|--origination optimize --prior none"
 "uniNoPrior|--origination uniform --prior none"
 "depth1000bs0.5|--origination optimize --origination-depth-lambda 1000 --prior brownian --brownian-sigma 0.5"
 "depth300tv300|--origination optimize --origination-depth-lambda 300 --dtl-tv-lambda 300"
 "rootclassbs0.5|--origination optimize --origination-root-class --prior brownian --brownian-sigma 0.5"
 "uniformbs0.5|--origination uniform --prior brownian --brownian-sigma 0.5"
)

cfg="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
TAG="${cfg%%|*}"; EXTRA="${cfg#*|}"

cd "$GPUREC_DIR"
export TORCH_EXTENSIONS_DIR="$GPUREC_DIR/.torch_ext"
[ -d ".venv/bin" ] && export PATH="$GPUREC_DIR/.venv/bin:$PATH"
module load "$PY_MODULE" 2>/dev/null || true
if module load gcc/14.2.0 2>/dev/null; then
  GCC14LIB=/apps/unit/SzollosiU/gcc/14.2.0/lib64; [ -d "$GCC14LIB" ] && export LD_LIBRARY_PATH="$GCC14LIB:${LD_LIBRARY_PATH:-}"
fi
for SO in .torch_ext/preprocess_cpp/preprocess_cpp.so .torch_ext/recon_sampler/recon_sampler.so; do [ -f "$SO" ] && touch "$SO"; done
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3.11"; fi
mkdir -p "$OUTDIR"
RATES="$OUTDIR/sweep_${TAG}.rates.txt"
PNC="$OUTDIR/pnc_${TAG}"
echo "=== [$SLURM_ARRAY_TASK_ID] $TAG : fit [$EXTRA] on $(hostname) $(date) ==="

# RESUME: if this config already produced a genome sample, re-emit its SWEEPROW and skip.
if [ -s "$PNC.perspecies_events.tsv" ]; then
  echo "=== [$TAG] already done -> re-emit + skip ==="
elif [ -s "$RATES.json" ]; then
  echo "=== [$TAG] fit already done -> skip fit, sample only ==="
else
  PYTHONPATH=. "$PY" experiments/run_undine_branchwise.py --root "$ROOT" --out "$RATES" \
    --fm-mode e-only --optimizer lbfgs --steps "$STEPS" --family-batch-size "$FBS" \
    --init-from-rates "$WARM" $EXTRA 2>&1 | tail -4 || { echo "[$TAG] FIT FAILED"; exit 1; }
fi

if [ ! -s "$PNC.perspecies_events.tsv" ]; then
  echo "=== [$TAG] sample genomes ==="
  PYTHONPATH=. "$PY" experiments/per_node_copies.py --root "$ROOT" --sidecar "$RATES.json" \
    --engine cpp --n-samples "$NSAMP" --out "$PNC" 2>&1 | tail -3 || { echo "[$TAG] PNC FAILED"; exit 1; }
fi

# summary line: tag, data_logL, internal genome (copies) at root(512)/511/DPANN(510) + deep clades
"$PY" - "$RATES.json" "$PNC.perspecies_events.tsv" "$TAG" <<'PYEOF'
import sys, json
sc=json.load(open(sys.argv[1])); tsv=sys.argv[2]; tag=sys.argv[3]
dll=sc.get("data_log_likelihood_ln") or sc.get("data_logL_ln") or "NA"
rows={}
with open(tsv) as fh:
    hd=fh.readline().rstrip().split("\t"); ci={c:i for i,c in enumerate(hd)}
    for ln in fh:
        p=ln.rstrip().split("\t"); rows[int(p[ci["node"]])]=p
def cop(n):
    return float(rows[n][ci["copies"]]) if n in rows else float("nan")
lab={510:"DPANN",511:"node511",512:"root"}
print(f"SWEEPROW\t{tag}\tdata_logL_ln={dll}\troot={cop(512):.0f}\tn511={cop(511):.0f}\tDPANN={cop(510):.0f}")
PYEOF
echo "=== [$TAG] DONE ($(date)) ==="
