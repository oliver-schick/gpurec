# gpurec — quick start for Oliver

How to run the CLI and which code is involved, for two jobs:
1. **Compare gpurec vs AleRax** across AleRax model setups (global/uniform → most-complex),
   on **1 GPU and with DDP**, so you can measure agreement and run-times.
2. **Fit one species tree** with **full per-branch D,T,L + origination (O)** — a minimal
   robust recipe.

gpurec is a GPU (PyTorch+Triton) reimplementation of AleRax's undated DTL reconciliation.
Its likelihood is **faithful to AleRax**: at AleRax's own fitted rates the per-family logL
matches AleRax's `per_fam_likelihoods.txt` to **Pearson r = 0.999998**.

### 3 things to know before running
- **log₂ / NLL.** Rates are stored as log₂; the function called `compute_log_likelihood`
  returns the **negative** logL. Sidecars report `data_log_likelihood_ln` (natural-log) for
  easy comparison to AleRax.
- **`--fm-mode e-only`** is the fraction-missing mode that matches AleRax (it applies the
  missing-data factor in the extinction recursion only). Always use it.
- **Cold-start caveat (only matters for the full free per-branch + O fit, task 2):** from a
  uniform start, the joint per-branch-rate + O optimizer can settle in a poor local optimum.
  The fixes are a sensible warm-start (`--init-dlt`) and regularizing the rates
  (`--dtl-tv-lambda`). The grouped/AleRax-structured fits in task 1 do **not** have this
  issue.

### Code bits involved
| file | role |
|---|---|
| `experiments/run_undine_branchwise.py` | **the CLI** — fit one tree (1 root) and write rates + a JSON sidecar |
| `gpurec/optimization/wave_optimizer.py` | `optimize_theta_wave` — L-BFGS/Adam over rates(+O), with the priors (Brownian, fused-lasso TV, O barrier) |
| `gpurec/distributed.py` | DDP: family sharding + `all_reduce` (used via `torchrun`) |
| `experiments/eval_bigtree_at_alerax.py` | evaluate gpurec @ AleRax's exact rates → per-family logL → compare/AU |
| `experiments/agg_*.py`, `au_bigtree.py` | pure-python result readers / AU test (login-node safe) |
| `gpurec/core/{forward,backward,likelihood,terms}.py` | the engine (E fixed point, Π wave forward/backward) |

---

## Task 1 — gpurec vs AleRax across model setups (1 GPU + DDP)

AleRax outputs are extracted under
`…/williams_run/undine/alerax_ref/3_Reconciliation/This_study/5_reconcilation models/<SETUP>/Undine_C60_<ROOT>root[…]/`
with `model_parameters/model_parameters.txt` (per-branch rates) and `per_fam_likelihoods.txt`.
Available setups, simplest → most complex:

| SETUP | dir name pattern | params | O column |
|---|---|---|---|
| **global** (uniform rates) | `global/Undine_C60_<ROOT>root` | 3 (one D,L,T) | no |
| DTL_br1_O | `DTL_br1_O/Undine_C60_<ROOT>root2_OR` | ~7 DTL + 4 O | **yes** |
| **DTL_br2** (most complex, BIC-best) | `DTL_br2/Undine_C60_<ROOT>root` | 72 (per-clade D,T,L) | no |

gpurec reads the setup's clade structure straight from its `model_parameters` via
`--clade-groups <that dir>` — so the **same command works for any setup**, you just point at
a different dir. There are two comparisons:

### 1a. Likelihood fidelity (fast, no optimization)
Evaluate gpurec at AleRax's exact fitted rates and correlate per-family logL with AleRax's.
```bash
cd /work/SzollosiU/gergely-szollosi/gpurec-cpp && source ../williams_run/gpurec_env.sh
python experiments/eval_bigtree_at_alerax.py --model DTL_br2   --fm-mode e-only --out /tmp/ev_br2.json
python experiments/eval_bigtree_at_alerax.py --model DTL_br1_O --fm-mode e-only --out /tmp/ev_br1O.json
python experiments/agg_evalAX.py        # prints Pearson r per root (expect ~0.999998)
```
(`--model global` works once you add `global` to the `eval_bigtree_at_alerax.py` model list —
one line; DTL_br2/DTL_br1_O are already wired.)

### 1b. Run-time comparison: gpurec **optimizes** the same setup (1 GPU)
This is the apples-to-apples timing run — gpurec fits the rates under the AleRax model
structure. `--init-from-alerax` seeds at AleRax's fitted rates so gpurec optimizes in the
**same optimum** AleRax found (important for the complex setups; harmless for global):
```bash
ROOT=Eury
AXDIR="/work/SzollosiU/gergely-szollosi/williams_run/undine/alerax_ref/3_Reconciliation/This_study/5_reconcilation models"
# most complex (DTL_br2, no O -> uniform O):
python experiments/run_undine_branchwise.py --root $ROOT \
  --clade-groups "$AXDIR/DTL_br2/Undine_C60_${ROOT}root" --init-from-alerax \
  --origination uniform --families 0 --fm-mode e-only --prior none \
  --family-batch-size 500 --steps 200 --dtype float64 --out /tmp/g_br2_${ROOT}.rates.txt
# with origination (DTL_br1_O):
python experiments/run_undine_branchwise.py --root $ROOT \
  --clade-groups "$AXDIR/DTL_br1_O/Undine_C60_${ROOT}root2_OR" --init-from-alerax \
  --origination optimize --families 0 --fm-mode e-only --prior none \
  --family-batch-size 500 --steps 200 --dtype float64 --out /tmp/g_br1O_${ROOT}.rates.txt
# global (uniform rates):
python experiments/run_undine_branchwise.py --root $ROOT \
  --clade-groups "$AXDIR/global/Undine_C60_${ROOT}root" \
  --origination uniform --families 0 --fm-mode e-only --prior none \
  --family-batch-size 500 --steps 200 --dtype float64 --out /tmp/g_global_${ROOT}.rates.txt
```
The log prints **per-step and total wall-time**; the sidecar has `data_log_likelihood_ln`
and the fitted `rates` to diff against AleRax. To match compute, fix `--steps` (or a tolerance)
and use the same model + all families (`--families 0`).

**Same run on multiple GPUs (DDP)** — identical command, just launch with `torchrun`:
```bash
torchrun --standalone --nproc_per_node=4 \
  experiments/run_undine_branchwise.py --root $ROOT \
  --clade-groups "$AXDIR/DTL_br2/Undine_C60_${ROOT}root" --init-from-alerax \
  --origination uniform --families 0 --fm-mode e-only --prior none \
  --family-batch-size 500 --steps 200 --dtype float64 --out /tmp/g_br2_${ROOT}_ddp4.rates.txt
```
DDP shards families across GPUs and `all_reduce`s the gradient — result is byte-identical to
1-GPU (validated, dNLL 9e-13); the forward/backward (the per-step cost) scales ~linearly, so
this is your scaling/run-time measurement. (SLURM: `#SBATCH --gres=gpu:a100:N --ntasks=1`,
then the `torchrun --nproc_per_node=N …` line.)

**Measured 1-GPU compute** (A100-80GB, S=513, 7059 families, float64, batch 500): ~**57 s/step**
(forward ~9 s, Π-backward ~29 s, E-adjoint CG ~14 s); a grouped fit early-stops in ≈28 steps →
**~28 min**; expect ≈ /N with N GPUs under DDP.

---

## Task 2 — fit one species tree with full per-branch D,T,L + O (minimal robust recipe)

Goal: free per-branch rates **and** origination, on a single tree, without (a) over-fitting
the 513×3 rates or (b) the cold-start local optimum. Recipe:

- **`--dtl-tv-lambda <λ>`** — fused-lasso/total-variation prior on the per-branch rates. It
  fuses neighbouring branches → **piecewise-constant rates = a data-driven clade grouping**
  (no groups specified, no over-parametrization). λ tunes the effective number of rate
  classes (on the big tree λ≈1e3 gave ~9 classes at the grouped-model likelihood; λ→0 = fully
  free). Brownian (`--prior brownian`) smooths but does **not** group (stays 513 distinct).
- **`--init-dlt "D,L,T"`** — warm-start every branch at a sensible global rate (avoids the
  cold-start trap). Use rough values for your clade, e.g. archaeal `0.1,0.3,0.15`.
- **`--origination optimize`** for free per-branch O, **`--fm-mode e-only`**, batch 500, float64.

```bash
python experiments/run_undine_branchwise.py --root <ROOT> \
  --origination optimize \
  --init-dlt 0.1,0.3,0.15 \
  --dtl-tv-lambda 1000 \
  --families 0 --min-species 1 --fm-mode e-only --prior none \
  --family-batch-size 500 --steps 250 --dtype float64 --out OUT.rates.txt
```
- Read the result: `python experiments/agg_reg.py` reports the fit and **eff#rates** (the
  number of distinct per-branch rate tuples — your check that TV grouped them, e.g. ~10–20
  rather than 513).
- **O is the one to sanity-check.** Free O is the part that can settle in a poor optimum from
  cold. If the fitted O looks degenerate (mass spiked on one branch) or the rooting looks off,
  either (a) drop O with `--origination uniform` (robust DTL rates, no O), or (b) add a mild
  anti-concentration barrier `--origination-dirichlet 30000 --origination-barrier-kind simpson`,
  or (c) warm-start from a grouped fit via `--init-from-rates <prev sidecar.json>`.
- This runs the **same on multiple GPUs** by prefixing `torchrun --standalone
  --nproc_per_node=N` (see Task 1).
- **Generic (non-Undine) input:** `experiments/gpurec_fit.py --species tree.nwk --genes
  '*.ale' …` takes an arbitrary species tree + gene families directly. It already does DDP;
  **small port needed**: it doesn't yet expose `--init-dlt`, `--dtl-tv-lambda`,
  `--init-from-rates`, `--free-origination` (they live in `optimize_theta_wave` already — copy
  the argparse + pass-through from `run_undine_branchwise.py`).

---

## Paths & deploy
- **Code on the cluster:** `/work/SzollosiU/gergely-szollosi/gpurec-cpp` (your fork
  `oliver-schick/gpurec`, branch `cpp-rust-free`). Env: `…/williams_run/gpurec_env.sh`.
- **Deploy from a laptop:** push to `cpp-rust-free`, then on the cluster
  `git fetch origin && git reset --hard origin/cpp-rust-free`; if a C++ rebuild kicks off,
  `touch .torch_ext/preprocess_cpp/preprocess_cpp.so`.
- **Data:** `…/williams_run/data/3_Reconciliation/This_study/` (trees in `4_species_tree/`,
  gene families `.ale` in `3_UFBOOTs/ufboot_for_alerax/`, `fraction_missing`).
- **AleRax reference:** `…/williams_run/undine/alerax_ref/.../5_reconcilation models/{global,DTL_br1_O,DTL_br2}/`.
- **Cluster:** OIST Saion, `largegpu` = 8×A100-80GB cap, 12 h wall (`#SBATCH --time<=12:00:00`,
  `module load python/3.11.11`).
