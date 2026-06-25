# gpurec vs AleRax — Williams 2017 benchmark results

Head-to-head of **gpurec** (GPU; PyTorch + Triton) against **AleRax** (CPU/MPI) on the
Williams et al. 2017 archaeal reconciliation dataset, run on OIST clusters
(gpurec → Saion A100, AleRax → Deigo). Same model, same families, same fixed
species tree — only the engine differs. Run 2026-06-24/25.

A rendered one-page version is in `summary.html` (and was published as a Claude
artifact).

## Headline

**Full dataset, global rates, 3,946 families, DPANN root:**

| tool | hardware | families | wall time | ms/family |
|------|----------|----------|-----------|-----------|
| **gpurec** | 1× A100-80GB | 3,946 | **489 s** (8m 09s) | 124 |
| **AleRax** | 64 CPU ranks / 8 nodes | 3,946 | **1,999 s** (33m 19s) | 507 |

→ **gpurec ≈ 4.1× faster**, and computes the **same likelihood** (Pearson
*r* = 0.999997 on per-family log-likelihood, gpurec evaluated at AleRax's own
fitted global rates over all 5,446 families; the small constant offset is a
CCP-normalization convention, independent of the rates).

**The full per-branch model (357 params) — the one AleRax can't run at scale:**
gpurec fits all 3,946 families in **23 min** (1,396 s, 53 L-BFGS steps); AleRax
exceeded an hour on just 177 families (1/22 of the data) and was stopped. gpurec's
branch-wise likelihood still matches AleRax's *shipped* reference at
*r* = **0.999995** (per-branch D/T/L + origination O).

### Correctness — both models validated

| model | params | likelihood vs AleRax (per-family logL, r) |
|-------|--------|-------------------------------------------|
| global      | 3   | **0.999997** |
| branch-wise | 357 | **0.999995** (per-branch D/T/L + O) |

## All runs

| run | tool | families | wall | speed-up |
|-----|------|----------|------|----------|
| global · full   | gpurec | 3,946 | 489 s   | **4.1×** |
| global · full   | AleRax | 3,946 | 1,999 s |          |
| global · pilot  | gpurec | 177   | 34 s    | **2.5×** |
| global · pilot  | AleRax | 177   | 84 s    |          |
| branch-wise · full | gpurec | 3,946 | 1,396 s (23 min) | **gpurec only** |
| branch-wise · full | AleRax | —     | infeasible (>1 h on 177) | |

The speed-up grows with scale (gpurec amortizes fixed GPU/setup cost). **Global is
AleRax's most competitive model.** On the full **per-branch** model (357 params)
gpurec fits all 3,946 families in 23 min; AleRax exceeded an hour on just 177
families and was stopped — yet gpurec's branch-wise likelihood still matches
AleRax's reference at *r* = 0.999995, so it's validated, not just faster.

A free per-branch fit (uniform origination) on the full set is well-behaved
(median D/L/T = 0.026 / 0.36 / 0.094; max ~18, no degenerate spikes) — see
`bin/40_run_gpurec.sh` with `MODE=specieswise`.

## What made it a fair comparison

- **Identical families** — both ran the same 3,946 (5,446 minus <4-species, the
  filter both tools apply).
- **Same model** — `UndatedDTL`, global rates, uniform origination, fixed DPANN
  tree (`--species-tree-search SKIP` / gpurec fixed tree).
- **Same missing-data term** — gpurec `--fm-mode e-only` ↔ AleRax
  `--fraction-missing-file`, both filtered to the 60 tree species (AleRax
  otherwise silently corrupts one entry; see `bin/30_run_alerax.sh`).
- **Same optimizer family** — L-BFGS on both, run to each tool's own convergence.
- **Identical CCP inputs** — both consume the same ALE conditional-clade `.ale`
  files; no re-derivation between tools.
- **Pinned hardware** — gpurec on one A100-80GB (driver-verified, never the mixed
  older-GPU partition).

## On AleRax parallelization

AleRax was given **64 MPI ranks**, but it parallelizes per-family and reported a
**recommended maximum of ~20 cores** for the full set (load balance ≈ 0.32 — a few
large families dominate). So it was *over*-provisioned, not core-starved: more CPUs
would not lower its wall time materially. The 4.1× is therefore a fair (if anything
AleRax-favorable) comparison.

## Setup

- **Dataset**: Williams et al. 2017 archaeal reconciliation — 5,446 ALE CCP gene
  families, 60-taxon DPANN-rooted species tree (Zenodo `17360806`,
  `3_Reconciliation/Williams_et_al_2017`).
- **gpurec**: Saion, 1× NVIDIA A100-80GB, PyTorch 2.6 + Triton, float64.
- **AleRax**: Deigo, v1.4.1, 64 MPI ranks across 8 EPYC nodes.
- **Measurement**: wall-clock via `ruse` + each tool's own timer; correctness via
  per-family log-likelihood at matched rates.

Reproduce with the pipeline in this directory — see `README.md`.
