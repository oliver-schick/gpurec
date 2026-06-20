---
name: project-gpurec-recon-sampler
description: "gpurec native reconciliation sampler (gpurec/core/sampler.py, CPU backtrack) + the families-at-root = origination-at-root identity for LACA genome-size readout"
metadata: 
  node_type: memory
  type: project
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

Built `gpurec/core/sampler.py` (session 2026-06-20): native CPU stochastic backtracking reconciliation sampler, mirrors AleRax `MultiModel::backtrace`. At each (clade,species) cell resample one event (S/SL/D/T/TL) ∝ its contribution to Π (the `core/terms.py` decomposition); **DL and TL-lost are UNRECORDED self-loops** = the undated geometric series (resample same cell); origination sampled ∝ p^O_e·Π[root_clade,e]; T/TL recipient ∝ transfer_mat[d,·]·Π. Validated on a synthetic forced case (1000/1000 valid). Uses the legacy/dense forward Π (Triton-free, CPU). Output writers (ALE `.uml_rec` + genome-wide per-branch event summary + family×branch presence table + optional recPhyloXML, the user's requested format) NOT yet built.

KEY IDENTITY: "families present at the root" = families that **originated at the root** (nothing exists above the root). `PP_f(root) = p^O_root·Π_f(root) / Σ_e p^O_e·Π_f(e)` — the (1−E) survival is the likelihood's family-INDEPENDENT conditioning and CANCELS from the origination posterior (audited mathematically correct). So families-at-root needs ONLY the origination draw, not the full backtrace. `experiments/families_at_root.py` computes it (expected ΣPP, 1k-sampled, and PP≥0.5 count; sampled≈expected to <0.1%). For l2=100 Williams rates: deep-cluster roots give LARGE LACA (Eury 1567, TackA 1468, DPANN 1299 families) vs SGA roots tiny (Cluster2 135, AMD 128) — the rooting story AS ancestral genome size. Non-root ancestral presence needs the full backtrace.

NB: existing `gpurec.api.sample_reconciliations` is an AleRax-BINARY bridge passing only D,L,T (no origination, no branch-wise) — insufficient for the DTLO model; would need extending. See [[project-rooting-two-channel-design]].
