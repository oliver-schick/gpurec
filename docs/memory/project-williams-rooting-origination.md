---
name: project-williams-rooting-origination
description: "SOLVED: why gpurec disagreed with AleRax on the Williams archaeal root -- origination (O) heterogeneity is the deciding factor; the discrepancy was a missing model term, not a gpurec bug"
metadata:
  node_type: memory
  type: project
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

RESOLVED 2026-06-19: why gpurec rooted Williams (archaea, 5446 .ale families, 10
candidate roots) on Cluster2 (the small-genome-attraction artefact) while AleRax
roots on Euryarchaeota ("Eury"). The answer is the ORIGINATION probability O, not
any gpurec bug. See [[project-williams-branchwise-replication]].

KEY DIAGNOSTIC CHAIN (all on Saion, Williams data, e-only fm, fixed rates -- no
optimization, to bypass optimizer convergence):
1. AleRax's OWN models, ranked by sum of per_fam_likelihoods.txt:
   - global -> Cluster2 best (Eury 6th); family-wise -> Cluster2 best (Eury 6th).
   - "branch wise" -> Eury best, then TackA, DPANN (Cluster2 4th).
   So even AleRax mis-roots under HOMOGENEOUS models -- the artefact is real and
   only its branch-heterogeneous model escapes it. gpurec-global == AleRax-global
   (per-family logL Pearson r=0.999997), so gpurec is a FAITHFUL likelihood engine.
2. AleRax "branch wise" is NOT free-per-branch: it is the `all_others_sp_O`
   parametrization = 17 named-clade (D,L,T) categories + per-clade ORIGINATION O,
   where DPANN/Eury/TackA are "DTLO" (own O) and all other clades share one O
   ("all others"). The O column in model_parameters.txt SUMS TO 1 -> O is a
   per-branch origination PROBABILITY distribution (matches gpurec's softmax
   origination p^O_e). Recovered the per-branch (D,L,T,O) by leaf-set mapping
   (experiments/clade_groups.py: branch_params_from_alerax / group_index_for_*).
3. gpurec free-per-branch (no prior) -> Cluster2 best (Eury 6th): naive
   heterogeneity does NOT fix it (over-parametrizes; all roots within ~48 ln).
4. gpurec clade-grouped DTL (17 cats) with UNIFORM O -> Cluster2 best (Eury 6th).
   STILL wrong. gpurec optimum != AleRax optimum (offset varied -3.59..-4.38/fam).
5. gpurec at AleRax's EXACT (D,L,T), UNIFORM O (eval_branchwise_rates.py) ->
   Kor/Asgard best, Eury 8th, DPANN LAST. The (D,L,T) rates alone are insufficient.
6. gpurec at AleRax's EXACT (D,L,T) AND O (--alerax-origination, log_pO=log2(O))
   -> TOP-3 = TackA, Eury, DPANN (the DTLO clades), Cluster2 demoted to 5th.
   MATCHES AleRax's top-3 {Eury,TackA,DPANN}. gpurec-AleRax offset pins to a near
   CONSTANT ~-3.7..-3.8 nat/family (the CCP-normalization offset) instead of
   varying -> ranking matches. Eury is 2nd, ~107 ln (0.02/fam) behind TackA --
   within the residual offset noise; the correct CLUSTER is recovered.

CONCLUSION: the discrepancy was a MISSING MODEL TERM (origination heterogeneity),
not an implementation error. gpurec optimized progressively richer models
(global -> free-per-branch -> clade-grouped DTL) but NEVER origination
heterogeneity (always uniform p^O=1/S), and origination is exactly what lifts the
DTLO clades (DPANN/Eury/TackA) above the small-genome-attraction artefact
(Cluster2). The mechanism: clade-specific origination accounts for where gene
families actually originate; without it, genomically-reduced clades (DPANN
Cluster2, lots of "absences") spuriously attract the root.

USER GUIDANCE (confirmed): "the likelihoods are the same if computed at AleRax
rates, try setting them FIXED in gpurec" (done -> reproduces Eury cluster); "O is
the origination probability" (confirmed: O sums to 1, a per-branch distribution).
The paper (biorxiv 2025.11.11.687807) PDF body renders as garbled/encoded font --
unreadable via the Read tool; rely on the supplement + empirics + user.

NEXT: to make gpurec INDEPENDENTLY recover Eury (optimize, not eval at AleRax
rates), implement GROUPED ORIGINATION in optimize_theta_wave -- optimize a per-O-
category omega (DPANN/Eury/TackA own, all-others shared) jointly with grouped DTL,
analogous to the group_index reparam already added for theta. Then re-run all 10
roots. Also worth checking the residual ~0.1/fam offset variation (gpurec vs
AleRax origination math) to get Eury cleanly #1 over TackA.
