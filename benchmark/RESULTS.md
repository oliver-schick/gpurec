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

## All runs

| run | tool | families | wall | speed-up |
|-----|------|----------|------|----------|
| global · full   | gpurec | 3,946 | 489 s   | **4.1×** |
| global · full   | AleRax | 3,946 | 1,999 s |          |
| global · pilot  | gpurec | 177   | 34 s    | **2.5×** |
| global · pilot  | AleRax | 177   | 84 s    |          |
| per-species · pilot | gpurec | 177 | 210 s | **≫** |
| per-species · pilot | AleRax | 177 | > 1 h (stopped) | |

The speed-up grows with scale (gpurec amortizes fixed GPU/setup cost). **Global is
AleRax's most competitive model**: on per-species (per-branch) rates AleRax
exceeded an hour on just 177 families and was stopped, while gpurec finished in
210 s.

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
