---
name: project-rooting-two-channel-design
description: Principled archaeal-rooting design — suppress BOTH over-fitting channels (loss/SGA + origination degeneracy) with constrained DTL + anti-concentration origination + EVIDENCE arbiter + empirical-Bayes; the answer is a root REGION not a point
metadata: 
  node_type: memory
  type: project
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

The Williams/archaeal rooting bias is TWO over-fitting channels, both the same trick — "explain-away gene absences by CONCENTRATING events instead of paying for losses":

1. **LOSS/DTL channel (SGA)**: rigid loss rates → absences in reduced-genome lineages read as "never had it" → root migrates to the small-genome clade (Cluster2/DPANN). Cure: loss-rate heterogeneity (clade-structured DTL_br, or Brownian-shrunk per-branch), dimension-controlled.
2. **ORIGINATION channel (degeneracy)**: free origination collapses ALL mass onto ONE deep branch (vertex, maxP^O→1 at depth 2–3 — measured), "born deep, never present above" → selectively inflates SGA roots by ~900 nats. Cure: anti-concentration BARRIER on p^O (Dirichlet c·KL(π‖p^O), π SPREAD/near-uniform — the BARRIER not the verticality does the work; depth-based verticality FAILED because the artifact lives at depth 1–2 where a vertical prior also rewards it).

Every single-lever fix this session FAILED (depth prior, root-mass prior, free-model evidence) — each closes one channel, the other re-opens the artifact. The TWO that worked: l2=100 ridge (→Eury #1) and barrier-Dirichlet c=1000 (→deep cluster), both = constrained 18-DTL-class + strong anti-concentration O.

PRINCIPLED DESIGN (Williams lock-in, session 2026-06-20, job 4636749): constrained DTL structure (channel 1) + origination barrier strength c (channel 2) + **LAPLACE EVIDENCE** as arbiter (NOT point-MLE — it over-fits at EVERY prior) + **EMPIRICAL BAYES** (c*=argmax_c logZ per root). Success = deep cluster {Eury,DPANN,TackA,Halo≈MHH} tops the evidence ranking hands-off, SGA roots demoted. The truth is a root REGION (~150–300 nats spread) NOT a sharp peak — a sharp single winner means an unsuppressed bias.

AleRax's Eury is FRAGILE: NOT the gpurec-ML (|g|≈1100 at AleRax's point; gpurec descends ~3900 nats off it to the degenerate vertex). AleRax recovers Eury via L-BFGS-B EARLY STOPPING (loose ~23-nat tol) in the good basin + the BIC-constrained model. Eury lives on a regularized ridge, not the global optimum.

**Why:** single priors kept failing because the bias is two-coordinate. **How to apply (big tree):** use DTL_br2 (72-param, BIC-best) + origination {root, 2 children, DPANN} few-classes OR the Dirichlet barrier; rank roots by evidence/AU-test; NEVER free per-branch DTL or per-branch O (re-opens both biases). gpurec ≡ AleRax likelihood (per-family r=0.999996). See [[project-williams-rooting-origination]] [[reference-huang-archaea-rooting-paper]].

EVIDENCE-CONSISTENCY CAVEAT (found 2026-06-20, the EB lock-in): the Laplace evidence MUST use the SAME prior the MAP was fit with. `experiments/laplace_evidence.py` uses a tau-RIDGE prior, so for the barrier-fit MAPs it just rewards data fit → PREFERS WEAK c (under-suppresses) → EB-over-c picks the wrong (degenerate) end. FIX (`experiments/agg_eb_barrier.py`): barrier-consistent log Z = data_logL_ln + ln2·c·mean(log2 p^O*) [the Dirichlet barrier log-prior, penalizes concentrated p^O*] + occam(from EBev); reuses the EBev Occam, no new runs. So when EB-selecting any regularization strength by evidence, the evidence's prior must match the regularizer — else it selects toward the un-regularized (artifact) end. (Point-MLE at barrier c=3000 already gives Eury #1 on Williams.) Lock-in pending: agg_eb_barrier.py runs once EBev c=1000/3000 land.
