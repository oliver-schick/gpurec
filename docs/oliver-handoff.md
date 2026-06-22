# gpurec ↔ AleRax handoff (for Oliver)

Status update + how-to for (a) the CLI incl. multi-GPU DDP, (b) running the
AleRax-equivalent fits **in the right basin** with comparable compute so run-times are
measurable, and (c) a minimal robust per-branch recipe for an unseen dataset.

---

## 0. What we learned (the one thing that matters for comparisons)

gpurec's likelihood **is faithful to AleRax** on the big tree (This_study/Undine C60,
257 taxa → S=513, 7059 families): evaluated at AleRax's *exact* per-branch rates, the
per-family logL matches AleRax's `per_fam_likelihoods.txt` at **Pearson r = 0.999998**
(`experiments/eval_bigtree_at_alerax.py`), and gpurec reproduces AleRax's Euryarchaeota
root exactly (DTL_br1_O → Eury BP=1.0).

**But the optimization landscape is bistable** (in the origination O, coupled to DTL):
- a **deep / Eury basin** (data logL ≈ −1.642M ln, deep cluster {Eury,TackA,DPANN} on top), and
- a **shallow / SGA basin** (≈ −1.712M ln, small-genome roots on top).

The **initialization** picks the basin. AleRax lands in the deep basin via its
model + L-BFGS-B early-stopping; gpurec from a **cold uniform init falls into the SGA
basin**. So **for any gpurec↔AleRax comparison you must put gpurec in the deep basin**,
otherwise you are comparing different optima. The clean, controlled way is
`--init-from-alerax` (warm-start at AleRax's fitted rates) + `--clade-groups` (AleRax's
exact grouping). Cold-start basin-finding (global-rate warm-up + a TV/fused-lasso
complexity homotopy) is under active development; see §4.

Other facts that bite:
- **`--fm-mode e-only`** is the correct fraction-missing mode (AleRax applies `_fm` only
  in the extinction recursion, not the Π leaf boundary; the supplement double-counts).
- **Everything is log₂ (bits)**; `compute_log_likelihood` returns **NLL** despite the name.
- Per-branch fits want **`--family-batch-size 500`** at S=513 to bound memory.

---

## 1. The drivers / CLI

Two entry points; both write a `*.rates.txt` (AleRax-style `node D L T [O]`) + a
`*.rates.txt.json` sidecar (theta/omega/rates, `data_log_likelihood_ln`, etc.).

### A. `experiments/run_undine_branchwise.py` — the developed big-tree driver
This is where all the recent work lives (clade-grouping, AleRax warm-start, fused-lasso
TV, basin-finding, DDP). It is wired to the Undine layout:
- tree `…/This_study/4_species_tree/Undine_C60_<ROOT>root_short_name.nw`
- gene families `…/This_study/3_UFBOOTs/ufboot_for_alerax/*.ale`
- AleRax reference `…/williams_run/undine/alerax_ref/3_Reconciliation/This_study/5_reconcilation models/{DTL_br2,DTL_br1_O}/Undine_C60_<ROOT>root[2_OR]/`

Key flags (full list: `--help`):

| flag | meaning |
|---|---|
| `--root <ROOT>` | candidate root (Eury, Cluster2, DPANN, MHH, …, 15 total) |
| `--families 0` | use all (else cap; for quick tests) |
| `--fm-mode e-only` | fraction-missing in E only (the correct mode) |
| `--origination {uniform,optimize}` | fixed p^O=1/S vs fit per-branch/grouped O |
| `--clade-groups <dir>` | **AleRax's exact grouping** (distinct rate tuples) from a per-root model_parameters dir |
| `--free-origination` | with `--clade-groups`: keep **free per-branch O** (needed for the O depth/root priors) |
| `--init-from-alerax` | **warm-start θ(+ω) at AleRax's fitted rates** → lands in the deep basin |
| `--init-dlt "D,L,T"` | global-rate warm-up (seed every branch at one rate) |
| `--init-from-rates <sidecar.json>` | warm-start from a previous fit (chains the annealing stages) |
| `--dtl-tv-lambda <λ>` | **fused-lasso / TV** on per-branch DTL → piecewise-constant = auto clade-grouping (no overparam) |
| `--origination-dirichlet <c> --origination-barrier-kind simpson` | anti-concentration O barrier |
| `--family-batch-size 500 --steps N --dtype float64 --out PATH` | optimizer/output |

### B. `experiments/gpurec_fit.py` — the generic-input CLI
For an **arbitrary dataset**: `--species <tree.nwk> --genes <glob|dir>` (auto-detects
`.ale` vs Newick gene-tree samples), optional `--gene-map` / `--species-sep` (default
`_`-prefix), `--fraction-missing FILE --fm-mode e-only`, model flags (`--origination`,
`--barrier-kind/--barrier-c`, `--clade-groups DIR | --clade-groups-from-tree
--dtl-clades {all,paper}`, `--prior`), `--optimizer/--steps/--dtype/--family-batch-size`,
`--out`, `--per-family-out`. It already does DDP + `grad_reduction="sum"`.
**TODO (small port):** `gpurec_fit.py` does **not yet expose** `--init-dlt`,
`--init-from-rates`, `--dtl-tv-lambda`, `--free-origination` — they exist in the
optimizer (`optimize_theta_wave`) and in `run_undine_branchwise.py`; copy the argparse
+ pass-through to use the basin-finding/TV recipe on a generic dataset.

---

## 2. Multi-GPU (DDP)

Family-sharded data parallelism, **manual `all_reduce(SUM)`** of nll+grad (the gradient is
the hand-written implicit VJP, no autograd graph, so torch DDP hooks don't apply). Core:
`gpurec/distributed.py` (`maybe_init_distributed`, `shard_families`, `all_reduce_sum_`),
wired into `optimize_theta_wave(distributed=…, grad_reduction="sum")` and both drivers.

Run on N GPUs of one node with **torchrun** — no code change, just launch:
```bash
torchrun --standalone --nproc_per_node=4 \
  experiments/run_undine_branchwise.py --root Eury --clade-groups <dir> --init-from-alerax \
  --origination optimize --families 0 --fm-mode e-only --family-batch-size 500 \
  --steps 200 --dtype float64 --out OUT.rates.txt
```
- Correctness: loss = Σ_families, E is family-independent (recomputed per rank), the
  E-adjoint solve is linear with a shared operator ⇒ sum-of-shards == whole. **Validated
  byte-identical** (Part C, 2-rank NCCL: dNLL 9e-13, dGrad at the CG solver floor;
  `experiments/test_ddp_equivalence.py`).
- Speedup is **near-linear in the forward/backward** (the per-step cost; the all-reduce is
  a tiny `[S,3]`+scalars). Caveat: each rank recomputes E (cheap, ~0.02 s vs ~30 s/step
  for the Π backward).
- Triton+multiproc: set a **per-rank `TRITON_CACHE_DIR`** (see the test) so ranks don't
  corrupt a shared JIT cache.

---

## 3. (i) AleRax-equivalent runs **in the right basin** + comparable compute

Goal: gpurec and AleRax fitting the **same model in the same (deep) basin**, so wall-times
are a fair comparison. Use AleRax's exact grouping and warm-start at its rates (this is the
controlled deep-basin config — it reproduces AleRax's Eury, data logL ≈ −1.642M):

```bash
DD=/work/SzollosiU/gergely-szollosi/williams_run
AX="$DD/undine/alerax_ref/3_Reconciliation/This_study/5_reconcilation models/DTL_br1_O/Undine_C60_Euryroot2_OR"
python experiments/run_undine_branchwise.py --root Eury \
  --clade-groups "$AX" --init-from-alerax --origination optimize \
  --families 0 --min-species 1 --fm-mode e-only --prior none \
  --family-batch-size 500 --steps 200 --dtype float64 \
  --out $DD/undine/CMP_Eury.rates.txt
```
To make compute **comparable to AleRax** (which runs L-BFGS-B to a loose tolerance):
- **Same model** = AleRax's DTL_br1_O grouping (`--clade-groups` on the `2_OR` dir) →
  matches AleRax's ~7 DTL classes + 4 O classes. (For the BIC-best 72-param model use the
  `DTL_br2/Undine_C60_<ROOT>root` dir; it has **no O column** → run `--origination uniform`.)
- **Same data** = all 7059 families (`--families 0`).
- **Match the work**: fix `--steps` (or match a convergence tolerance) and report wall-time
  **per fit and per step**. AleRax = single config; gpurec needs the 15 roots for a rooting
  comparison → run them as a SLURM array (one root per A100, `--array=0-14%8`).

**Measured gpurec compute (1× A100-80GB, S=513, 7059 fams, float64, batch 500):**
- **~57 s/step** (forward ~9 s, Π-backward ~29 s, E-adjoint CG ~14 s, θ-vjp ~0 s).
- Clade-grouped DTL_br1_O fit, AleRax-init, early-stops ≈ 28 steps → **~28 min/root**.
- Full per-branch DTL (1539 free params, 250-step cap) → **~0.5–2.5 h/root** (CG-bound;
  high variance across roots).
- 15-root sweep on 8 A100s ≈ **~1 h** (grouped) to a few h (full per-branch).
- DDP: divide the per-step time ≈ linearly by #GPUs (forward/backward is family-parallel).

So for the writeup: report gpurec wall-time/step and per-fit vs AleRax's, at matched model
+ data + convergence, on 1 GPU and N GPUs (DDP scaling).

---

## 4. (ii) Minimal robust per-branch recipe for an **unseen dataset**

For a new dataset there is no AleRax reference to warm-start from, so you must avoid the
cold-start SGA trap **and** the per-branch over-parametrization. The robust recipe:

1. **Regularize per-branch with fused-lasso/TV** (`--dtl-tv-lambda`). This drives adjacent
   branch rates together → **piecewise-constant = data-driven clade grouping**, no group
   spec needed, no overparam. (On the big tree, λ≈1e3 collapsed 513 free rates → ~9
   classes at ≈ the hand-grouped likelihood; the λ knob sets the effective #groups.)
   FD-validated to 2e-9.
2. **Global-rate warm-up** (`--init-dlt`) — the single most effective basin lever in our
   screen (it got ~5× closer to the deep basin than any strong prior, which *backfire*).
   Use sensible global rates for the clade, e.g. archaeal `D,L,T ≈ 0.1,0.3,0.15`.
3. **`--fm-mode e-only`**, **`--family-batch-size 500`**, **`--dtype float64`**, L-BFGS.

Minimal robust command (per-branch DTL, regularized, warm-started):
```bash
python experiments/run_undine_branchwise.py --root <R> \
  --origination uniform \            # safest: uniform O. (free O is bistable; see note)
  --init-dlt 0.1,0.3,0.15 \          # global-rate warm-up (avoid the cold SGA trap)
  --dtl-tv-lambda 1000 \            # fused-lasso: auto-grouped per-branch, no overparam
  --families 0 --min-species 1 --fm-mode e-only --prior none \
  --family-batch-size 500 --steps 200 --dtype float64 --out OUT.rates.txt
```
- **Origination note:** `--origination uniform` is the *robust default* — it removes the
  bistable-O risk and still gives good DTL rates, but loses the O contribution to rooting.
  If you need O (for rooting), use `--origination optimize` and reach the deep basin via a
  warm-start or the TV/complexity-anneal (below); free O from cold can degenerate.
- **Generic input:** for a non-Undine dataset, use `gpurec_fit.py` once the four flags in
  §1B are ported; the recipe is identical.
- **Cold-start rooting (in development):** the end-to-end pipeline is global-rate warm-up →
  **strong→weak TV homotopy** (anneal complexity: ~1 group → free per-branch) → pure-ML O,
  driven per root by `run_undine_branchwise.py … --init-dlt … --dtl-tv-lambda <λ>` chained
  with `--init-from-rates` over a decreasing-λ schedule (see `TVANNEAL.sbatch`). The aim is
  to recover Eury *cold* with auto-grouped branch-wise rates and no final prior; verdict
  pending. `agg_tvanneal.py` reads the result.

---

## 5. Where things are

- **Cluster code checkout** (deploy target): `/work/SzollosiU/gergely-szollosi/gpurec-cpp`
  (fork `oliver-schick/gpurec`, branch `cpp-rust-free`). Env: `williams_run/gpurec_env.sh`.
  Deploy = push from the Mac, then `git fetch origin && git reset --hard origin/cpp-rust-free`;
  if it triggers a C++ rebuild, `touch .torch_ext/preprocess_cpp/preprocess_cpp.so`.
- **Data**: `…/williams_run/data/3_Reconciliation/This_study/` (trees, .ale, fraction_missing).
- **AleRax reference**: `…/williams_run/undine/alerax_ref/.../5_reconcilation models/{DTL_br2,DTL_br1_O}/`.
- **Aggregators** (`gpurec-cpp/experiments/`, pure-python, login-node-safe):
  `agg_evalAX.py` (engine fidelity r vs AleRax), `agg_axgrp.py` (deep-basin ranking),
  `agg_fullbasin.py` (full DTL in basin), `agg_reg.py` (TV/Brownian + eff#groups),
  `agg_tvanneal.py` (cold TV-anneal), `au_bigtree.py` (AU test on AleRax per_fam).
- **Validation**: `eval_bigtree_at_alerax.py` (gpurec @ AleRax rates → per-fam → AU),
  `test_ddp_equivalence.py` (DDP correctness).

Key memories (auto-loaded): `project-undine-bigtree-rooting`, `project-gpurec-ddp-gradfix`,
`reference-huang-archaea-rooting-paper`, `reference-saion-*` (cluster ops, 8×A100 cap, 12h wall).
