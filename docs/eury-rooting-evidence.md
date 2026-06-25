# Eury rooting — evidence, commands, and the origination dependence (big tree)

Reference appendix for `clade-vs-branchwise.{tex,pdf}` and `bigtree-branchwise.{tex,pdf}`.
Compiles every rooting analysis run on the big tree (This_study / Undine C60), the exact
commands, and the full 15-root outputs, so the Eury result and its origination dependence are
reproducible. Companion memory: `project-bigtree-clade-vs-branchwise`.

## Setup

- **Species tree**: `Undine_C60_<root>root_short_name.nw` — 257 taxa = S=513 branches; 15
  candidate roots (each a different rooting edge): `Alti AMD Asgard Cluster2 DPANN Eury
  HalobacThermopl Kor MHH Micra5 MicraDia TACA TackA TAC UndineClu2`.
- **Data**: 7,059 gene families (`3_UFBOOTs/ufboot_for_alerax/*.ale`), fraction-missing e-only.
- **Eury and TackA are 3 branches apart** (path: Eury[96 taxa] → root → non-Eury ancestor
  *node 511*[161] → TackA[84]); the rooting question is a deep, basal, ~3-branch region.
- **AU test** = Shimodaira (2002) multiscale bootstrap on per-family logL; B as noted.
  Engine fidelity to the AleRax reference per-family logL: Pearson r = 0.999998.

---

## A. Models that ROOT AT EURY (comparable gradient-descent / clade fits)

These are single gradient-descent (or fixed-rate) fits with the *same* parameter space at
every root → per-root maximized log-likelihoods are directly comparable → the AU is trustworthy.

### Headline table (from `clade-vs-branchwise.tex`, Table 1; AU at B=10⁶/scale)

data logL (×10³ nats) and AU support, all 15 roots, three rate models:

| root | uni-clade logL / AU | n-u clade logL / AU | **branchwise (FULLbasin)** logL / AU |
|---|---|---|---|
| **Euryarchaeota** | −1645.0 / **0.65** | −1642.4 / **1.00** | **−1631.5 / 1.00** |
| HalobacThermopl | −1644.7 / <.05 | −1647.8 / <.05 | −1635.1 / 0.00 |
| TackA | −1645.1 / <.05 | −1643.5 / <.05 | −1635.5 / 0.00 |
| MHH | −1644.7 / **0.48** | −1649.5 / <.05 | −1637.7 / 0.00 |
| Kor | −1645.7 / <.05 | −1648.0 / <.05 | −1637.9 / 0.00 |
| TACA | −1646.1 | −1648.4 | −1638.4 / 0.00 |
| DPANN | −1645.2 | −1644.3 | −1639.7 / 0.00 |
| TAC | −1647.0 | −1648.9 | −1640.0 / 0.00 |
| Asgard | −1647.0 | −1649.0 | −1641.1 / 0.00 |
| MicraDia | −1645.3 | −1646.3 | −1642.3 / 0.00 |
| AMD | −1645.7 | −1647.6 | −1642.5 / 0.00 |
| UndineClu2 | −1645.5 | −1648.0 | −1643.2 / 0.00 |
| Alti | −1645.0 | −1648.3 | −1643.3 / 0.00 |
| Cluster2 | −1645.4 | −1648.4 | −1643.7 / 0.00 |
| Micra5 | −1645.5 | −1648.9 | −1644.1 / 0.00 |

- `uni-clade` = clade DTL + **uniform** O (≈ AleRax DTL_br2, no-origination BIC-best) → 95% set
  **{Euryarchaeota (0.65), MHH (0.48)}**.
- `n-u clade` = clade DTL + **fitted** O (≈ DTL_br1_O) → **Euryarchaeota AU = 1.00** (won all 10⁷ resamples).
- `branchwise` (FULLbasin) = **per-branch** DTL + fitted O → **Euryarchaeota AU = 1.00**, and the
  best fit (−1631.5; +10,888 nats over n-u clade) — *but* the branch-wise rates do not change the
  rooting (every other root AU=0). **The rooting comes from the ORIGINATION term, not rate granularity.**

### Commands

```bash
# Clade-grouped per-root logL + AU (reads AleRax 5_reconcilation models per_fam_likelihoods.txt;
# runs on the Mac / a compute node, torch CPU). High-B (10^6) cleans low-B AU artifacts.
PYTHONPATH=. python experiments/au_highb.py            # B=1e6 multiscale AU, uni-clade + n-u clade
PYTHONPATH=. python experiments/au_bigtree.py          # B=2e4 version (DTL_br2 + DTL_br1_O)
PYTHONPATH=. python experiments/eval_bigtree_at_alerax.py   # gpurec @ AleRax exact rates, per-root logL+AU

# Branch-wise (FULLbasin) per-family logL -> AU (free per-branch DTL + fitted O, one fit/root):
#   the fit:  experiments/run_undine_branchwise.py --root <R> --origination optimize --prior brownian ...
#   per-fam:  experiments/au_fullbasin.py   ->  FULLbasin_perfam.npz  (re-bootstrappable without GPU)
PYTHONPATH=. python experiments/au_fullbasin.py        # branchwise AU, B=1e6

# LACA genome-size dissection (uniform vs fitted O vs branchwise):
PYTHONPATH=. python experiments/laca_compare.py        # 96 -> 772 -> 903 families at root
```

### A.2 Basin depth — the Eury optimum is the DEEPEST; alternatives and cold starts are shallower

Two independent senses in which the *other* rooting basins are shallower than Eury's.

**(i) Within the FULLbasin model, every alternative root's optimum is shallower** (Table 1,
branchwise data logL, ×10³ nats; gap from Eury):

| root | branchwise logL | shallower than Eury by |
|---|---|---|
| **Euryarchaeota** | **−1631.5** | — (deepest) |
| HalobacThermopl | −1635.1 | −3,600 nats |
| TackA | −1635.5 | −4,000 |
| MHH | −1637.7 | −6,200 |
| DPANN | −1639.7 | −8,200 |
| … | … | … |
| Micra5 | −1644.1 | −12,600 |

→ Eury is the ML optimum by **+3,568 nats** over the next root (FULLbasin AU = {Eury} only;
`au_fullbasin.py`). MHH — the AleRax DTL_br2 co-equal #2 (9.7 nats behind Eury) — is **6,167 nats
behind** under FULLbasin and AU-rejected.

**(ii) The deep Eury basin is reachable only by warm-starting; cold starts fall into shallower
basins (and root SGA/TackA, not Eury).** Three independent *uniform-init* runs converge to the
SAME shallow basin; warm-starting at the paper's (AleRax) rates reaches the deep Eury basin:

| start | basin total logL (×10³ nats) | rooting | reached |
|---|---|---|---|
| **warm @ AleRax rates** (AXgrp `--init-from-alerax`) | **≈ −1,642** (clade) / **−1,631.5** (FULLbasin) | **Eury #1** | the **deep** basin |
| cold uniform-init (BTroot, BARtest, VERTtest — 3 runs) | ≈ −1,712 | Cluster2/SGA #1, **Eury last** | shallow (−70k) |
| cold global-rate warm-up (free per-branch + free O) | ≈ −1,664 | Cluster2 #1, **Eury #11** | shallow near-tie (−22k) |
| cold EM-on-origination (`em_origination.py`, uniform-O) | near-tie | SGA #1, **Eury #6** | shallow near-tie |

Every cold basin-finder tried — uniform init, anti-concentration Simpson barrier, root-mass /
depth-O priors + homotopy, global-rate warm-up, fused-lasso TV-anneal, and EM-on-O — **fails to
reach the deep Eury basin** (lands at −1,664k…−1,712k, SGA-favouring). The deep Eury optimum needs
the AleRax structured-rate warm start. **This is why the rfx-centroid and 2-class-O fits in §B root
TackA:** the rfx centroid is one of these *shallow* basins, not the deep Eury optimum — so warm-
starting any branchwise (or EM) fit from the paper's DTL_br2 rates (§B uniform-O-paper, §C) is the
methodologically correct way to test the deep basin. Commands/jobs: `eval_bigtree_at_alerax.py`
(AXgrp, `--clade-groups <AleRax dir> --init-from-alerax`), `au_fullbasin.py`, cold jobs
`{BARtest,VERTtest,ROOTCOLD}_*`, `em_origination.py` (cold-O sweep). Full record:
memory `project-undine-bigtree-rooting`.

---

## B. Models that ROOT AT TackA — the ORIGINATION dependence (this session)

The same engine, but with a **different origination treatment** (the rfx EM's *shared* `p^O`, or a
constrained 2-class O), flips the root to **TackA** and rejects Eury. This is the same bistability
the table above shows (rooting is set by O): the no-511-dump shared origination prefers TackA.
All are gradient/EM fits, AU at B=2×10⁴ (run on `short-a100`).

### rfx — gridded-rate family-mixture + shared p^O (`em_origination.py --rooting-rfx`)
95% set **{TackA 0.55, DPANN 0.46, (HalobacThermopl 0.30 = BP-0 regression artifact)}**; **Eury REJECTED (AU 0.0014)**.

| root | AU | | root | AU |
|---|---|---|---|---|
| TackA | 0.5546 ✓ | | Eury | **0.0014 ✗** |
| DPANN | 0.4574 ✓ | | AMD/TACA/MHH/… | 0.0000 ✗ |
| HalobacThermopl | 0.3007 (artifact) | | Asgard | 0.0000 ✗ |

### control — free branchwise DTL, O pinned to rfx p^O (`stageB_fiteval_saion.sbatch` + `au_branchwise.py`)
95% set **{TackA}** only (AU 1.0000, BP 1.0); **Eury REJECTED**. (12/15 roots; 3 timed out — see §D.)

### rootclass — free branchwise DTL + 2-class O {root, uniform} (`rootclass_au_saion.sbatch`)
95% set **{TackA 0.87, (Alti 0.26, Asgard 0.14 = BP-0 artifacts)}**; **Eury REJECTED**. (13/15 roots.)

### Commands

```bash
# rfx (Stage A) rooting marginal logL + per-root AU:
sbatch ... slurm/rooting_rfx_saion.sbatch        # em_origination.py --rooting-rfx --roots <R>
PYTHONPATH=. python experiments/au_rfx.py        # AU over rfx_<root>.npz perfam_logL  -> au_rfx.json

# control / rootclass / uniform-O: free branchwise DTL with a given O, per root, then AU:
sbatch ... slurm/stageB_fiteval_saion.sbatch     # --origination fixed --origination-fixed-po-npz rfx_<R>.npz
sbatch ... slurm/rootclass_au_saion.sbatch       # --origination optimize --origination-root-class
sbatch ... slurm/unifO_au_saion.sbatch           # --origination uniform           (cooking — fills below)
#   each task: rfx_init_rates.py (warm start) -> run_undine_branchwise.py -> eval_perfam_undine.py
PYTHONPATH=. python experiments/au_branchwise.py <dir>   # AU over perfam_<root>.json -> au_branchwise.json
```

### uniform-O (free branchwise DTL + flat 1/S origination) — RUNNING
The decisive cross-check: flat O is the *no-bloat* origination, and the clade table above puts
uniform-O at **{Eury, MHH}**. If uniform-O branchwise lands on Eury, it gives Eury-rooting *and* a
de-bloated genome in one consistent gradient model. **[result to be appended when `unifOAU` lands.]**

---

## C. EM both-het (branch + family heterogeneous) — genome only, NOT cross-root rooting

`stage_b_partition.py` (root-specific partition into K rate components) + per-component branchwise
fit + `stage_b_concat_au.py`. **Why it is not used for rooting:** unlike a single gradient fit, the
EM uses a *root-specific partition* (different #components / family assignments per root) wrapped
around per-component gradient M-steps, so per-root marginal logLs are **not on a common footing** —
the cross-root comparison is unreliable (empirically: a premature run returned garbage, AMD-best on
3,705/7,059 common families). It IS the best-fitting model (Eury data logL −2,321,490 bits, +25k
over rfx) and gives the **biological genome at Eury (DPANN ≈ 1121** vs FULLbasin's bloated 2819).
Cross-root both-het ranking (`bhAU3`) is reported for completeness only. **[bhAU3 to be appended.]**

---

## D. Interpretation

1. **Rooting (comparable fits): Eury.** Clade-grouped (fitted O) and fully branch-wise (FULLbasin)
   both give **Eury AU = 1.00**; uniform-O gives the **{Eury, MHH}** region. The branch-wise rates
   improve the fit at every root (+10,888 nats, and CV +1.66/fam out-of-sample → real signal, not
   over-fitting) **without changing the rooting**.
2. **The rooting is set by the ORIGINATION term, and it is bistable.** Fitted/clade/uniform O →
   Eury; the rfx *shared* no-dump O and the 2-class O → TackA — only **3 branches away**, with node
   511 between. So the disagreement is a deep-region O-dependent branch flip, not a gross conflict.
3. **The same O that gives Eury (free/fitted) over-fits the genome** (node-511 dump → DPANN ≈ 2.8k,
   non-biological). The EM branch+family-heterogeneity model with shared O **de-bloats it** (DPANN ≈
   1.1k) — used at the adopted Eury root, since EM is not reliably comparable across roots.

## E. Reproducibility — scripts & sbatch

| purpose | script / sbatch |
|---|---|
| clade AU (high-B) | `experiments/au_highb.py`, `experiments/au_bigtree.py`, `experiments/eval_bigtree_at_alerax.py` |
| branchwise FULLbasin AU | `experiments/au_fullbasin.py` (→ `FULLbasin_perfam.npz`) |
| rfx rooting + AU | `experiments/em_origination.py --rooting-rfx`, `experiments/au_rfx.py`, `slurm/rooting_rfx_saion.sbatch` |
| branchwise per-O fits + AU | `experiments/run_undine_branchwise.py`, `experiments/eval_perfam_undine.py`, `experiments/au_branchwise.py`; `slurm/{stageB_fiteval,rootclass_au,unifO_au}_saion.sbatch` |
| warm start (Stage A centroid) | `experiments/rfx_init_rates.py` |
| EM both-het | `experiments/stage_b_partition.py`, `experiments/stage_b_concat_au.py`, `slurm/stageB_full_saion.sbatch` |
| LACA dissection / genome | `experiments/laca_compare.py`, `experiments/per_node_copies.py` |
