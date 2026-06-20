---
name: project-gpurec-transfer-to
description: "gpurec recipient ('transfer-to') transfer parametrization — weight the transfer recipient sum by w_r; SEPARABLE so O(C·S) efficiency preserved; gpurec/core/transfer_to.py validated, engine integration partial"
metadata: 
  node_type: memory
  type: project
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

Transfer-to = replace the donor transfer rate with a per-RECIPIENT receptivity weight w_r; weight the recipient sum in the T/TL likelihood terms by w_r. **KEY EFFICIENCY INSIGHT (the whole point):** w_r is recipient-SEPARABLE → scale Pi_exp (and expE) by w_r ONCE before the existing uniform-mode `row_sum − ancestor_sum`, plus an O(S) per-donor normalization `−log2(Σ_{r∉anc(d)} w_r)`. Preserves O(C·S), NO [S,S] matrix. (A general donor×recipient T[d,r] would be O(C·S²); recipient-only stays O(C·S). Rank-1 `a_d·w_r` = the "both" model is also separable — `a_d` folds into log_pT.)

DONE (session 2026-06-20): `gpurec/core/transfer_to.py` — `recipient_uniform_setup(ω,ancestors_T)→(w, recip_mt_offset)` (efficient) and `build_recipient_transfer_mat(ω,ancestors_T,log_pT)` (dense ref). Validated efficient==dense==brute-force==additive(ω=0) to 1e-15. Additive `recipient_w` hooks added to `forward._compute_Pibar_inline` + `likelihood.E_step` (None ⇒ byte-identical). Math: `Pibar[c,d]=log_pT[d]+log2(Σ_{r∉anc(d)} w_r Π[c,r] / Σ_{r∉anc(d)} w_r)`.

REMAINING: thread `recipient_w`+recip-mt through `Pi_wave_forward`+`E_fixed_point`+the Triton `wave_pibar_uniform_*` kernels (the fast path, GPU-test-bound); `extract_parameters` log_pT separation; `backward.py` VJP for ω; sampler recipient draw ∝ w_r·Π; driver `--transfer-recipient`.

**How to fit ω cheaply:** the autograd bridge is theta-only (no free grad w.r.t. transfer_mat), so fitting needs the backward VJP OR autograd-unrolled legacy. SHORTCUT (no engine change): set `transfer_mat_unnormalized = ω-broadcast` (masked to valid recipients via Recipients_mat) → the existing dense/legacy path fits ω via plain autograd; that's the "both" model (per-branch log_pT = a_d). User wants PURE transfer-to first (global donor T), then "both". See [[project-rooting-two-channel-design]] (transfer-to refines the loss/SGA channel 1).
