# Fitting DTL(O) rates — CLI reference

gpurec fits per-branch **Duplication / Transfer / Loss (+ Origination)** rates by
maximum likelihood (or MAP, with a prior) over ALE gene-family CCPs on a **fixed rooted
species tree**. It writes an AleRax-comparable `model_parameters.txt`-style rates file
plus a `.json` sidecar carrying the **prior-free** data log-likelihood (the statistic
used for rooting comparisons).

> **Requires a CUDA GPU (Triton).** The optimiser runs the Triton wave engine — real
> runs go on the Saion A100s. Only `--preflight` (species mapping / wiring check) and the
> C++ `.ale` preprocessing run CPU-only.

## Which entry point

| entry point | use it when |
|---|---|
| **`gpurec fit`** (or `gpurec-fit`) | **General / recommended.** Any rooted tree + a directory of `.ale` families — you pass `--species-tree` and `--ale-dir` explicitly. Thin wrapper over the proven branch-wise fitter, so it accepts every flag in the [full reference](#full-flag-reference) below. |
| **`experiments/run_undine_branchwise.py`** | The big-tree **This_study / Undine C60** preset (`--root <name>` path convention). This is the **canonical argparse** — the table below is its flag set. |
| **`experiments/run_williams_branchwise.py`** | The **Williams 2017** `.ale` preset (per-root, `--data-dir` layout). See [differences](#williams-branch-wise-fit). |
| **Python API** (`GeneReconModel`) | [Fit from code](#fitting-from-python-api), not the shell. |

---

## `gpurec fit` — the general CLI

The dataset-agnostic command. Installed as the console script `gpurec-fit` and as the
`gpurec fit` subcommand (both → `gpurec/cli/fit.py:main`, a thin wrapper that runs the
validated `run_undine_branchwise.py` driver). **Run it from a gpurec repo checkout.**

**Required (general mode):** pass **`--species-tree <Newick>`** *and* **`--ale-dir <dir>`**.
(Or use a dataset preset's `--root` instead, which derives the paths from `--data-dir`.)

```bash
# General per-branch DTLO fit on an explicit tree + .ale directory
gpurec fit \
  --species-tree sp.nwk --ale-dir ale/ \
  --origination optimize --fm-mode e-only \
  --family-batch-size 500 --steps 250 \
  --out fit.rates.txt

# Hold origination FIXED at AleRax's values; optimise only D/L/T
gpurec fit \
  --species-tree sp.nwk --ale-dir ale/ \
  --origination fixed \
  --origination-fixed-from model_parameters.txt \
  --origination-fixed-atree labelled_sp.nwk \
  --steps 200 --out fit_Ofixed.rates.txt

# Constrained clade-grouped DTL_br + structured O (no AleRax output needed)
gpurec fit \
  --species-tree sp.nwk --ale-dir ale/ \
  --clade-groups-from-tree \
  --family-batch-size 300 --pibar-mode uniform \
  --steps 250 --out fit_cladegroups.rates.txt
```

Because `gpurec fit` wraps `run_undine_branchwise.py`, every flag in the next section
applies. `gpurec fit --help` prints them all.

---

## Full flag reference

The canonical argparse, shared by `gpurec fit` and `run_undine_branchwise.py`. No flag
is strictly *required* on the Undine preset (defaults fit `--root root` and write to
`results/undine_<root>_branchwise.rates.txt`); for an arbitrary dataset supply
`--species-tree` + `--ale-dir`. `--origination fixed` **requires** `--origination-fixed-from`.

### Data / root

| flag | default | meaning |
|---|---|---|
| `--root` | `root` | Root LABEL (output naming + the root-mass origination prior); one of the preset's known roots. |
| `--species-tree` | `None` | General: explicit rooted species tree (Newick); overrides the `--root`/`--data-dir` convention. |
| `--ale-dir` | `None` | General: directory of ALEobserve `.ale` CCP files; overrides `--data-dir`/`--ccp-dir`. |
| `--data-dir` | preset | Base data dir for the path convention. |
| `--ccp-dir` | `3_UFBOOTs/ufboot_for_alerax` | Subdir (under `--data-dir`) of `.ale` CCPs. |
| `--min-species` | `1` | Drop families with fewer mapped species. |
| `--families` | `0` | Limit #families (0 = all; for quick tests). |
| `--shuffle-seed` | `-1` | Shuffle `.ale` order with this seed before the `--families` cut (-1 = sorted). |
| `--family-batch-size` | `0` | Mini-batch families for fwd/bwd to bound GPU memory (0 = all; use a few hundred on the big tree). |
| `--dtype` | `float64` | `float32` / `float64`. |
| `--preflight` | off | Load a sample of `.ale`, check species mapping, then exit (CPU; no CUDA/Triton). |

### Optimiser

| flag | default | meaning |
|---|---|---|
| `--steps` | `200` | Optimiser iterations. |
| `--optimizer` | `lbfgs` | `lbfgs` / `adam` / `sgd`. (`--origination optimize` needs `lbfgs`.) |
| `--pibar-mode` | `uniform` | Transfer term mode: `uniform` (large-tree default) / `dense` / `topk`. |
| `--init-rate` | `0.1` | Uniform per-branch DTL init rate. |

### Prior (Brownian / total-variation)

| flag | default | meaning |
|---|---|---|
| `--prior` | `none` | `none` / `brownian` (relaxed-clock smoothness coupling adjacent branch log-rates). |
| `--brownian-sigma` | `1.0` | Brownian step sd (log2 units). |
| `--brownian-root-sigma` | `5.0` | Brownian prior sd at the root. |
| `--dtl-tv-lambda` | `0.0` | Fused-lasso / TV prior on per-branch DTL (smoothed-L1 on parent–child log2-rate diffs → piecewise-constant rates). Use WITHOUT `--clade-groups`. 0 = off. |
| `--dtl-tv-eps` | `1e-3` | Pseudo-Huber smoothing scale (log2 units) for `--dtl-tv-lambda`. |

### Fraction-missing

| flag | default | meaning |
|---|---|---|
| `--fm-mode` | `off` (Williams: `both`) | `both` (AleRax supplement) / `e-only` (source-faithful) / `off`. |
| `--no-fraction-missing` | off | Force `--fm-mode off`. |
| `--fraction-missing` | `None` | General: explicit fraction_missing file (else the `--data-dir` default). |

### Origination — `uniform` | `optimize` | `fixed`

| flag | default | meaning |
|---|---|---|
| `--origination` | `uniform` | `uniform` (pᴼ = 1/S), `optimize` (free per-branch pᴼ = softmax(ω); L-BFGS only), or `fixed` (held at an external model). |
| `--origination-fixed-from` | `None` | **Required with `--origination fixed`.** AleRax `model_parameters.txt` to read per-branch origination from, HELD FIXED while D/L/T optimise (e.g. the paper's DTL_br1_O Eury model, where DPANN has a real O). |
| `--origination-fixed-atree` | `None` | Labelled tree matching `--origination-fixed-from` (default: the `--root` tree). |
| `--origination-l2` | `0.0` | L2 / ridge on the origination logits ω (shrink pᴼ toward uniform). |
| `--origination-depth-lambda` | `0.0` | Vertical-evolution prior `λ·E_pᴼ[depth]` — pulls origination mass to the root. |
| `--origination-root-lambda` | `0.0` | Root-mass prior `λ·(1 − pᴼ_root)` (depth-agnostic). Use INSTEAD of `--origination-depth-lambda`. |
| `--origination-dirichlet` | `0.0` | Anti-concentration barrier strength c (forbids the single-branch reroot degeneracy). |
| `--origination-barrier-kind` | `meanlog` | Barrier divergence: `meanlog` / `simpson` / `renyi2`. |
| `--origination-vertical-rho` | `0.0` (Williams: `0.95`) | Verticality of the barrier target (0 = uniform / pure anti-concentration). |
| `--origination-vertical-decay` | `2.0` | Geometric decay base of the vertical target with depth (ρ>0). |
| `--init-omega-depth-scale` | `0.0` | Structured-O seed: init free ω = −depth·scale (origination starts concentrated at the root). |
| `--free-origination` | off | With `--clade-groups`: keep FREE per-branch ω (don't group O into AleRax's O categories). Required for the root-mass/depth priors. |

### Clade-grouping (constrained DTL)

| flag | default | meaning |
|---|---|---|
| `--clade-groups-from-tree` | off | Constrained DTL grouped by smallest enclosing NAMED clade (paper's per-clade DTL_br) + structured O {root, 2 children, DPANN, rest}. No AleRax output needed. |
| `--all-named-clades` | off | Use ALL named internal clades (~30) instead of the paper's DTL_br2 whitelist (22). |
| `--clade-groups` | `None` | AleRax's EXACT grouping: a per-root dir with `species_trees/starting_species_tree.newick` + `model_parameters/model_parameters.txt`. Use INSTEAD of `--clade-groups-from-tree` to match AleRax's parametrisation. |
| `--preflight-groups` | off | With `--clade-groups-from-tree`: build + print the grouping, then exit before fitting. |

### Warm-start

| flag | default | meaning |
|---|---|---|
| `--init-from-rates` | `None` | Warm-start θ (+ free ω) from a previous fit's `*.rates.txt.json` sidecar (chains annealing stages). |
| `--init-from-alerax` | off | With `--clade-groups`: seed θ (and ω) at AleRax's fitted per-branch rates instead of uniform. |
| `--init-dlt` | `None` | Global-rate warm-up `'D,L,T'` seeding every branch (instead of uniform `--init-rate`), e.g. `'0.07,0.30,0.17'`. |

### Output

| flag | default | meaning |
|---|---|---|
| `--out` | `results/undine_<root>_branchwise.rates.txt` | Output rates path (a `.json` sidecar is written alongside). |

### Examples (Undine preset)

```bash
# (1) Free per-branch D/L/T + optimised origination, Brownian smoothness prior
python experiments/run_undine_branchwise.py \
  --root Eury --origination optimize \
  --optimizer lbfgs --steps 250 --pibar-mode uniform \
  --prior brownian --brownian-sigma 1.0 \
  --fm-mode e-only --family-batch-size 500 --dtype float64 \
  --out results/undine_Eury_freeDLT_optO_brownian.rates.txt

# (2) Held-fixed-O fit: free D/L/T, origination pinned to AleRax DTL_br1_O
python experiments/run_undine_branchwise.py \
  --root Eury --origination fixed \
  --origination-fixed-from   ".../DTL_br1_O/Undine_C60_Euryroot2_OR/model_parameters/model_parameters.txt" \
  --origination-fixed-atree  ".../DTL_br1_O/Undine_C60_Euryroot2_OR/species_trees/starting_species_tree.newick" \
  --prior brownian --brownian-sigma 1.0 \
  --init-from-rates AXgrp_DTL_br1_O_Eury.rates.txt.json \
  --fm-mode e-only --family-batch-size 500 --steps 250 \
  --out results/undine_Eury_freeDLT_fixedO.rates.txt

# (3) Clade-grouped DTL matching AleRax's exact parametrisation, seeded at AleRax's rates
python experiments/run_undine_branchwise.py \
  --root Eury --clade-groups /path/to/alerax/Eury_DTL_br2 --init-from-alerax \
  --optimizer lbfgs --steps 250 --family-batch-size 500 \
  --out results/undine_Eury_cladegrouped_alerax.rates.txt
```

---

## Williams branch-wise fit

`experiments/run_williams_branchwise.py` — the **Williams 2017** preset. Always
specieswise; `.ale`-input; per-root. Mirrors most of the reference above, with these
differences:

- **Input layout** is locked to the Williams convention: tree at
  `--data-dir/rooted_phylogeny/<root>`, `.ale`s globbed from `--data-dir`
  (+ `--extra-ale-dir`). The dataset-agnostic overrides (`--species-tree`, `--ale-dir`,
  `--fraction-missing`, `--ccp-dir`, `--shuffle-seed`) are **undine-only**.
- **Held-fixed origination is NOT available here.** Williams `--origination` is only
  `{uniform, optimize}` — the `--origination fixed` / `--origination-fixed-from` /
  `--origination-fixed-atree` path exists only in the Undine driver.
- **Clade-grouping**: Williams supports only the explicit AleRax-dir form `--clade-groups`
  (+ `--init-from-alerax`); the `--clade-groups-from-tree` / `--all-named-clades` /
  `--preflight-groups` / `--free-origination` flags are undine-only.
- **Defaults differ**: `--root Eury`, `--min-species 4` (AleRax's cutoff), `--fm-mode both`,
  `--origination-vertical-rho 0.95`. It adds `--rate-bounds 'MIN,MAX'` (L-BFGS-B box
  bounds on linear rates).

```bash
# Free per-branch DTL(O) fit, Eury root, optimise origination
python experiments/run_williams_branchwise.py \
  --root Eury --origination optimize --optimizer lbfgs \
  --steps 300 --out results/williams_Eury_branchwise.rates.txt

# Replicate AleRax's clade-grouped "branch wise" model, seeded at AleRax's rates
python experiments/run_williams_branchwise.py \
  --root Eury --clade-groups "/path/to/AleRax/branch wise/Eury" \
  --init-from-alerax --origination optimize \
  --origination-root-lambda 100 --dtype float64
```

---

## Fitting from Python (API)

Build a model straight from Newick trees, then optimise `theta` with any `torch.optim`
optimiser. `GeneReconModel.forward()` (i.e. `model()`) returns the **negative
log-likelihood** as a differentiable scalar, so the standard `loss.backward()` /
`opt.step()` loop works. `theta` is in **log₂-rate** space; read fitted natural-space
rates from `model.rates` (`= 2**theta`).

```python
import torch
from gpurec.api.model import GeneReconModel

model = GeneReconModel.from_trees(
    species_tree="sp.nwk",
    gene_trees=["g1.nwk", "g2.nwk"],
    mode="global",          # "global" | "specieswise" | "genewise"
    pibar_mode="uniform",   # fast default; "dense" for the full Pi @ Tᵀ
    device="cuda",
)
opt = torch.optim.Adam(model.parameters(), lr=0.1)
for _ in range(200):
    opt.zero_grad()
    loss = model()          # NLL (a loss); model.nll() is the same
    loss.backward()
    opt.step()
    model.clamp_theta_()    # floor rates at 1e-10

D, L, T = model.rates.tolist()   # fitted natural-space rates
ll = model.log_likelihood()      # +log-likelihood (float)
```

`from_trees(species_tree, gene_trees, ...)` is the one-liner entry point (`species_tree`
and `gene_trees` required; optional `theta_init_rates=(D, L, T)` seeds in natural space).
`pairwise` mode is not supported by `GeneReconModel`; in `genewise` mode,
`model.nll_per_family()` returns a `[G]` vector.

For the low-level path (L-BFGS, mini-batching, priors, **origination**), call
`gpurec.optimization.wave_optimizer.optimize_theta_wave(...)`, whose `origination=`
argument selects `'uniform'` (fixed pᴼ = 1/S, default), `'optimize'` (jointly fit
per-branch ω; L-BFGS only), or `'fixed'` (hold pᴼ at a supplied `omega_init` while
fitting only D/L/T). It returns a dict with `'theta'`, `'rates'`, `'log_likelihood'`,
`'negative_log_likelihood'`, `'history'` (plus `'omega'` / `'origination'` when O is
optimised or held fixed).

---

## Output files & notes

Every CLI fit writes two files (rank 0 only, under DDP):

- **`<out>` (rates txt)** — AleRax-comparable. Header `# node D L T O` when `--origination`
  is `optimize` or `fixed`, else `# node D L T`; one line per species branch.
- **`<out>.json` (sidecar)** — the full `command`, dataset/root/`S`/`names`, family counts,
  run config, fitted `theta_log2` + `rates`, `origination_prob` (when O is optimised/fixed),
  timing, and likelihoods in log2 **and** ln — including the **prior-free**
  `data_log_likelihood_ln` / `data_log_likelihood_log2` (the rooting-comparison statistic)
  alongside the penalised `negative_log_likelihood_*`.

Everything is **log₂ (bits)** internally; `theta` is log₂-rates. `compute_log_likelihood`
returns an NLL (the forward output is treated as a loss throughout).
