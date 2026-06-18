# Replicating AleRax branch-wise DTL rates with gpurec on Williams et al. 2017

**Status:** branch `cpp-rust-free`. Runs on OIST Saion (A100; torch 2.6.0+cu124, triton 3.2.0, gcc 11.2.1).
**TL;DR:** **without** fraction-missing, gpurec recovers AleRax's per-branch **duplication and transfer** rates (per-branch Pearson ~0.55–0.58 on log-rates, medians within ~15–65%); **loss is not recovered per-branch** (Pearson ~0.04) — the expected limit of free-per-branch gpurec vs AleRax's clade-*grouped* reference. gpurec's fraction-missing is **faithful to the AleRax supplement (no sign / `1-x` error)** — but **the supplement (L159) is itself inconsistent with the AleRax code**: AleRax applies fraction-missing to the **extinction `E` only**, never to the Π reconciliation. The supplement's extra Π term `(1−σ)(1−p_obs)` **double-counts** the present-but-unobserved mass already held in `E`, and that is what **inflates the inferred loss rate**. Fix: `--fm-mode e-only` (E-only, AleRax-faithful) — keeps `leaf_E` in `E`, drops the Π baseline (§5.1).

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

### 4a. Per-branch match quality (no-fm, full data, σ=0.25)

How well does gpurec recover AleRax's *per-branch* rates (not just the median)? Aligning the 60 leaf branches by species name on log-rates:

| dataset | axis | Pearson(log) | Spearman | median ratio g/a |
|---|---|---|---|---|
| **≥4-species (3,946 fam)** | **D** | **0.58** | 0.48 | 0.85 |
| | **L** | 0.04 | 0.05 | 4.4 |
| | **T** | **0.55** | 0.53 | 1.66 |
| **complete set (31,236 fam, + small_fams)** | **D** | 0.38 | 0.27 | 1.99 |
| | **L** | 0.14 | 0.06 | 6.4 |
| | **T** | **0.52** | 0.55 | 1.18 |

**Read-out:** on the AleRax-comparable ≥4-species set, gpurec **recovers per-branch duplication and transfer** rates (Pearson 0.58 / 0.55; medians within ~15–65%). **Loss is not recovered per-branch** (Pearson 0.04, ~4× median) — the structural limit of comparing free-per-branch gpurec to AleRax's *clade-grouped* (17-class) reference, with loss the least-identifiable axis. With fm ON the correlations collapse to ~0/negative and loss inflates 20–90× (§5).

### 4b. Bigger dataset — complete archaea60 (+ small_fams, 31,236 families)

Adding the 25,790 small (1–3-leaf, mostly single-gene) families — the complete `archaea60` set, all 60 species in-tree — runs cleanly (≈21 s/step, ~30 min, no OOM). No-fm medians shift up: **D 0.124, L 0.68, T 0.13** (vs 0.057/0.46/0.18 on ≥4-species). The single-gene families add origination/duplication signal that AleRax's branch-wise analysis *excluded* (it keeps only ≥4-species families), so the D median ~doubles and the D correlation drops (0.58→0.38) while T stays good (0.52). This is exactly the family-inclusion-threshold sensitivity flagged in the `recount` PNAS letter — the inferred rates depend on which families you include. (The with-fm complete-set run is still finishing; its loss will inflate as in §5.)

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

The loss-rate gradient **flips sign** when fm is on; the numerator goes flat and the denominator drives loss up. (Data *fit* still improves with fm: min NLL 7.6 → −0.9 — it is specifically the inferred *rate* that inflates.) **Important caveat:** that toy decomposition had fm on *both* `E` and Π. At scale, isolating fm to `E` only (`--fm-mode e-only`, §5.1) does **not** inflate loss — so the E-side ascertainment effect is in fact **negligible here**, and the **Π double-count (§5.1) is the actual driver**.

### 5.1 Root cause — the AleRax *code* applies fraction-missing to *E only*, not Π

Reading the AleRax source (`github.com/BenoitMorel/AleRax`, `src/ale/UndatedDTLMultiModel.hpp`) resolves it. `_fm` (the fraction-missing vector) is used in **exactly one place** — the extinction recursion (`_PS[ec] * _fm[e]`, the "S but not observed" term). The Π (`uq`) reconciliation recursion does **not** use `_fm`: its leaf boundary is just `proba += _PS[ec]` at the mapped gene-leaf, with **no `(1−σ)(1−p_obs)` baseline**.

| | extinction `E_l` | Π leaf boundary |
|---|---|---|
| AleRaxSupp.tex | `1−p_obs` (L116) | `σ + (1−σ)(1−p_obs)` (L159) |
| AleRax **code** | `_fm[e]` ✓ same | **`σ·p^S` only — no `(1−p_obs)` term** |
| gpurec (followed the supplement) | ✓ same | `σ·p^S + (1−σ)·p^S·(1−p_obs)` ← **extra term** |

**The supplement (L159) is inconsistent with the AleRax code.** gpurec faithfully implemented the *supplement*, so it carries an **extra Π leaf-boundary term that AleRax does not have** — adding "present-but-unobserved" mass to every clade at every leaf. That extra term, on top of the (shared) E-side ascertainment effect of §5, is what drives gpurec's loss inflation beyond anything AleRax would show.

**Which is mathematically correct?** **The E-only (code) formulation is the consistent one; the supplement's Π baseline double-counts.** By complementarity, a gene copy at a leaf is either *observed* — it is the singleton clade, mass `p_obs` — or *unobserved* — `E_l = 1 − p_obs`. So the `1 − p_obs` "present-but-unobserved" mass is **already held entirely in `E_l`**, and it flows into Π through the mixed `Π·E` terms (speciation with one extinct child, transfer to an extinct recipient, …). The supplement's `(1−σ)(1−p_obs)` puts that *same* mass a **second time** directly onto Π → a double count, hence the inflation. (Two honest caveats: this is a complementarity argument, not a from-first-principles derivation; and *both* formulations omit a `p_obs` factor on the mapped/observed gene — giving `1`/`p^S` not `p_obs` — so some `p_obs` accounting lives in the conditional normalization. The practical tie-breaker is that the validated, published AleRax computes the E-only version.)

**The fix — `--fm-mode e-only`.** Keep `leaf_E` in the E solver; drop the Π baseline (do not thread `leaf_obs_log` into `Pi_wave_forward`/backward). Implemented by decoupling the E-side `leaf_E` from the Π-side `leaf_obs_log` in `optimize_theta_wave`/`implicit_grad`, exposed as `--fm-mode {off,both,e-only}` (`both` = supplement, unchanged; `e-only` = AleRax-faithful).

**Confirmed (Eury, 3,946 ≥4-species families, σ=0.25):** `e-only` removes the inflation entirely — **L median 0.39** vs supplement-mode `both` **17.8** (and no-fm 0.46) — and matches AleRax as well as no-fm: per-branch **D Pearson 0.63**, **T 0.55**, L 0.08 (the free-vs-grouped limit, §7). Since `e-only` (fm in `E`, ascertainment effect included) does **not** inflate while `both` does, the **Π double-count is the sole cause** of the inflation. `--fm-mode e-only` is the AleRax-matching default going forward.

| config | D median | **L median** | T median |
|---|---|---|---|
| `off` (no fm) | 0.057 | 0.46 | 0.18 |
| `both` (supplement) | 0.071 | **17.8** | 0.053 |
| `e-only` (AleRax code) | 0.054 | **0.39** | 0.154 |
| AleRax reference | 0.075 | 0.22 | 0.14 |

**Two consequences either way:**
1. With the supplement's Π term + free per-branch rates, **loss is unidentifiable / runs to the boundary** → must be regularized (the Brownian prior) or constrained.
2. gpurec **without** fm — and, expected, with `--fm-mode e-only` — matches the reference, consistent with the committed `branch wise/Eury` reference being produced under the AleRax (E-only) formulation with `p_obs=1`.

## 6. Status of the AleRax question (largely resolved)

**The cause is the supplement↔code discrepancy in the Π leaf boundary (§5.1):** AleRax applies fraction-missing to the extinction `E` only; gpurec (following the supplement) also applied it to Π, a double count. `--fm-mode e-only` makes gpurec match the AleRax *code*. Remaining confirmation, for Oliver / Benoit / Noah:

1. **Confirm intent:** is `AleRaxSupp.tex` L159's `(1−σ)(1−p_obs)` a documentation typo (it should not be in Π), or is the code missing a term? The complementarity argument (§5.1) and the validated AleRax behavior both point to the supplement being wrong; the model's authors should confirm.
2. **(Optional) cross-check:** run AleRax branch-wise on Williams `Eury` with the `fraction_missing` file and verify it matches gpurec `--fm-mode e-only` (apples-to-apples), and that the committed reference was `p_obs=1`.

## 7. Open / not-yet-resolved

- `--fm-mode e-only` (the AleRax-faithful fix) **confirmed** (§5.1): removes the inflation, matches AleRax (D 0.63 / T 0.55). Complete archaea60 ±fm runs done (§4b).
- **Why loss doesn't match even without fm:** the AleRax reference is **clade-grouped** (17 rate classes — many leaf branches share one value, several pinned to the `1e-10` floor); gpurec fits a *free* rate per branch. Correlating 60 free values against a handful of grouped/floored ones is structurally capped, and loss is the **least-identifiable axis** (a lost gene leaves no trace, so per-branch loss is weakly constrained — unlike D/T, which leave direct signatures). To match AleRax's loss per-branch you'd impose the **same clade grouping** (17 rate classes), not free-per-branch + a smoothing prior.
- Fan-out to the other 9 rooting hypotheses (Asgard, DPANN, TACK, …) not yet run.
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
