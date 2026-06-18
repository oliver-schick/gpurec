# Replicating AleRax branch-wise DTL rates with gpurec on Williams et al. 2017

**Status:** branch `cpp-rust-free`. Runs on OIST Saion (A100; torch 2.6.0+cu124, triton 3.2.0, gcc 11.2.1).
**TL;DR:** gpurec reproduces AleRax's per-branch DTL rates well **without** fraction-missing. The fraction-missing implementation is **faithful to the AleRax supplement (no sign / `1-x` error)**, but enabling it **inflates the inferred loss rate** — a real, expected consequence of the observability/ascertainment denominator in the UndatedDTL likelihood, *not* a bug. The open question is whether AleRax behaves identically; **this needs an AleRax run to confirm (see "For Oliver").**

---

## 1. Goal

Estimate **branch-wise** (per-species-branch) duplication/transfer/loss rates with gpurec on the Williams 2017 archaeal dataset, *using the per-leaf fraction-missing term*, and compare to the AleRax `branch wise` reference.

- **Data** (Saion `…/Williams_et_al_2017/`): 5,446 `.ale` CCP gene families; 60-taxon `Eury`-rooted species tree (S=119 nodes); per-species `fraction_missing` (60 leaves). Complete archaea60 set adds `archaea60/small_fams` (25,790 tiny 1–3-leaf families → ~31,236 total).
- **Reference:** `reconciliation models/branch wise/Eury/model_parameters/model_parameters.txt` (per-branch `node D L T O`; 17 distinct rate tuples → a *clade-grouped* model).
- **Rate / column convention** (verified against gpurec's own writer/parser): AleRax `model_parameters.txt` is `node D L T` (loss before transfer); gpurec `theta[S,3]` axis order `[D, L, T]`, `rate = 2**theta`.

## 2. What was implemented (this branch, on top of `1f3efc0`)

1. **Fraction-missing in the production GPU wave path.** `1f3efc0` added it only to the CPU/legacy path (`likelihood.py` E solver, `legacy.py` Pi). This branch threads `leaf_E` / `leaf_obs_log = log2(1-p_obs) = log2(fraction_missing)` through `wave_optimizer → forward.py (Π leaf boundary) + backward.py + implicit_grad.py (E-adjoint)`. No Triton-kernel edits (dense leaf-term route). `model.py` derives `self.leaf_E` from a `fraction_missing` vector.
2. **CG-solver fix** (`linear_solvers.py`). Fraction-missing makes the implicit-gradient **E-adjoint operator strongly non-symmetric** (`max|A−Aᵀ| ≈ 0.5` vs `0.06`). The old `_cg` returned `success=True` on `maxiter` exhaustion regardless of residual, so the GMRES fallback never fired and gradients were corrupted (sign-flip on the loss column). Now `_cg` reports success only on actual convergence → GMRES fires for the non-symmetric system. No-fraction-missing path unchanged.
3. **Soft Brownian (Thorne–Kishino–Painter) rate prior** (`core/tree_prior.py`, wired into `optimize_theta_wave`). Per-axis Gaussian on log-rate increments along the species tree (`Δ²/2σ²` + root anchor), O(N), with per-element L-BFGS-B **box bounds** to mirror AleRax's bounded optimizer. Adapted from the `recount` project's `brownian_prior.tex`; it makes the MAP objective coercive and is the principled fix for the unidentifiable free-per-branch loss (§5).
4. **Driver** `experiments/run_williams_branchwise.py`: `.ale` + tree + `fraction_missing` → specieswise optimization → AleRax-format rates. Flags: `--prior`, `--brownian-sigma`, `--rate-bounds`, `--no-fraction-missing`, `--min-species`, `--extra-ale-dir` (complete set), `--preflight`.

## 3. Validation (all passed)

- **Forward bit-exact:** wave + fraction-missing == legacy CPU reference (which implements AleRaxSupp.tex) to `|ΔNLL| = 0` at fm=0.3; fm=0 reproduces baseline exactly.
- **Gradient (finite-difference) with fm on:** dense max rel-err **1.3e-9**, uniform **1.5e-5**; fm=0 gradient bitwise-identical to the no-fm path.
- **Brownian prior:** analytic gradient vs FD max rel-err **1.8e-9** (scalar and per-axis σ); σ→∞ recovers free-per-branch; parent array verified against the tree topology.

## 4. Results — gpurec vs AleRax (Eury, 3,946 families with ≥4 species)

AleRax `branch wise` Eury reference (background tuple): **D≈0.075, L≈0.22, T≈0.14**.

**Controlled ±fraction-missing (identical settings, σ=1 Brownian prior):**

| | D median | L median | T median |
|---|---|---|---|
| gpurec **−fm** | 0.057 | **0.39** | 0.17 |
| gpurec **+fm** | 0.076 | **4.80** | 0.063 |
| **AleRax ref** | 0.075 | **0.22** | 0.14 |

**gpurec without fraction-missing matches the AleRax reference** (D, L, T all close). With fraction-missing, loss inflates ~12×.

**Full-data configs, all WITH fm** (loss inflated regardless of regularization):

| config | D median | L median | L max |
|---|---|---|---|
| loose prior (σ=1) | 0.070 | 29.5 | 129064 |
| tight prior (σ=0.25) | 0.071 | 17.8 | 1072 |
| AleRax-style (σ=0.25 + bounds [1e-10,10]) | 0.066 | **10 (pegged at bound)** | 10 |

## 5. The fraction-missing finding (the headline)

**The implementation is faithful to AleRaxSupp.tex — there is no `1-fraction_missing` / sign / wrong-column error.** Verified line-by-line and re-verified independently:

- E (extinction) leaf boundary, supp L116 `E_l = 1-p_obs`, used as the *single factor* `p^S·E_l` in the terminal-branch E (L111): gpurec sets `E_s1=log2(fraction_missing), E_s2=0` → S-term `= log2(p^S·fraction_missing)`. ✓
- Π leaf boundary, supp L159 `Π_{l,γ}=σ+(1-σ)(1-p_obs)`, used as `p^S·Π_{l,γ}` (L147): gpurec seeds species-leaf columns with `log2(fraction_missing)` then overrides the mapped (σ=1) cell with `log2(1)=0`. ✓
- The file genuinely *is* `fraction_missing = 1-p_obs` (e.g. `Kcryp 0.0029` = near-complete genome; nonsensical as `p_obs`); `leaf_E = log2(file_value)` with no `1-x`. ✓

**Why enabling it inflates loss — the observability/ascertainment denominator.** The likelihood is `[Σ_e p^O_e Π_{e,Γ}] / [Σ_e p^O_e (1-E_e)]` (supp L91/L97). Fraction-missing raises branch extinction `E_e`, shrinking `(1-E_e)`; the optimizer is therefore rewarded for *higher* extinction, which it manufactures via higher D/T/L. Finite-difference decomposition of `dNLL/dL` at L=0.05:

| | `dNLL/dL` | numerator term | denominator term |
|---|---|---|---|
| fm = 0 | **+11.7** (pushes L down) | −12.9 (dominates) | −1.2 |
| fm on (subset) | **−1.7** (pushes L **up**) | +0.4 (flat) | **−1.3 (dominates)** |

The loss-rate gradient **flips sign** when fm is on; the numerator goes flat and the ascertainment denominator drives loss up. (Data *fit* still improves with fm: min NLL 7.6 → −0.9 — it is specifically the inferred *rate* that inflates.) This is the well-known coupling between a sampling correction and inflated rate estimates, present in AleRax's UndatedDTL by construction; gpurec reproduces it.

**Two consequences:**
1. With fm, **free per-branch loss is unidentifiable / runs to the boundary** → it *must* be regularized (the Brownian prior) or constrained. This is the real justification for the prior.
2. gpurec **without** fm matches the reference → **the AleRax `branch wise` reference was evidently run with `p_obs=1` (no fraction-missing)**. There is no fm-enabled AleRax reference in the dataset, so comparing gpurec-*with*-fm to it is apples-to-oranges.

## 6. ⚠️ For Oliver — please run AleRax to settle this

We cannot tell from the committed reference files whether AleRax used fraction-missing, and there is no AleRax binary in this repo to check. **Please run AleRax (UndatedDTL, PER-SPECIES / branch-wise) on the Williams `Eury` data (the ≥4-species families) twice:**

1. **Without** fraction-missing (`p_obs=1`) → should reproduce the existing `branch wise/Eury/model_parameters.txt`. This confirms the published reference's config.
2. **With** the `Williams_et_al_2017/fraction_missing` file (per-species `perSpeciesMissing`) → **does AleRax's inferred loss also inflate?**

If AleRax-with-fm *also* inflates loss, gpurec is faithful to AleRax (and the inflation is an intrinsic property of the model + ascertainment correction — worth a methods note). If AleRax-with-fm does **not** inflate loss, then AleRax handles the observation term differently from the supplement as we read it, and we need to find where. Either way, this AleRax run is the missing piece.

## 7. Open / not-yet-resolved

- Complete archaea60 set (`+ small_fams`, ~31,236 families) runs ±fm are in flight (results to append).
- The AleRax reference is **clade-grouped** (17 rate classes); gpurec's specieswise is free-per-branch + prior. A tight prior approximates grouping but is not identical; matching the exact grouping is future work.
- Comparison currently aligns the 60 **leaf** branches by species name; internal-branch alignment via `_alerax_label_map` is available but not yet wired into the report.

## 8. Reproduce

```bash
# on Saion, A100:
source /work/SzollosiU/gergely-szollosi/williams_run/env.sh   # module load python/3.11.11 gcc/11.2.1 + venv
# validate fraction-missing wave path (== legacy + FD gradient):
PYTHONPATH=. python experiments/validate_fraction_missing_wave.py
# branch-wise rates, with vs without fraction-missing:
PYTHONPATH=. python experiments/run_williams_branchwise.py --root Eury --prior brownian --brownian-sigma 0.25
PYTHONPATH=. python experiments/run_williams_branchwise.py --root Eury --no-fraction-missing --prior brownian --brownian-sigma 0.25
# compare to AleRax reference:
python experiments/compare_williams_alerax.py --gpurec <rates.txt> \
  --alerax ".../branch wise/Eury/model_parameters/model_parameters.txt" --tree ".../rooted_phylogeny/Eury"
```
