# Session handoff — gpurec archaeal rooting (2026-06-20)

## ⚡ LATEST UPDATE (offline handoff, ~13:30 JST) — READ THIS FIRST
- **Cluster access without VPN:** `ssh -J oist saion …` and `scp -o "ProxyJump oist" …`
  go through the `login.oist.jp` bastion (host `oist`). Direct `ssh saion` needs the VPN.
- **EBev lock-in (4636789): ~18/40.** c=10 done, c=100 nearly, **c=1000/3000 just starting**
  (array tasks 20–39, ~1–2 h left). It continues on the cluster while I'm offline.
- **When EBev finishes, run BOTH aggregators** (in `…/williams_run`):
  - `python3 agg_eb.py` — the τ-ridge evidence ranking. **CAVEAT: this prefers WEAK c**
    (it rewards data fit, ignores origination concentration) → would under-suppress.
    Partial c=10 result: DPANN #1; deep cluster {DPANN,TackA,Eury}+SGA in a ~160-nat top
    group; Asgard/Kor/TAC/Halo demoted ~430–1390 nats.
  - `python3 agg_eb_barrier.py` — **the FIXED, barrier-consistent ranking** (use this one).
    `log Z = data_logL_ln + ln2·c·mean(log2 p^O*) [Dirichlet barrier log-prior] + occam(EBev)`
    on the cmlD0 c∈{1000,3000} MAPs — penalizes concentrated p^O*, no new GPU runs.
  - **Pass = the barrier-consistent EB picks strong c and floats {Eury,DPANN,TackA,Halo}**
    (point-MLE at c=3000 already = Eury #1).
- **Transfer-to fit (`experiments/fit_transfer_to.py`) OOMs on V100** even at 100 families
  (autograd-unroll of the legacy fixed points is memory-heavy). To run it: use **A100**, a
  **small `--families`** (≤~300), and/or add `torch.utils.checkpoint` on the Pi/E fixed-point
  loops (or `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). It is *functionally correct*
  (donor vs transfer-to, fits recipient ω via autograd) — just memory-bound.

You're continuing work on **gpurec** (GPU DTL reconciliation, replicating AleRax) for
**archaeal root inference**. The auto-loaded memories have the deep context; read
especially `project-rooting-two-channel-design`, `reference-huang-archaea-rooting-paper`,
`project-gpurec-recon-sampler`, `project-gpurec-transfer-to`. This file is the ACTIVE state
+ what to do next.

## The one thing to do first
The **principled-rooting lock-in** is mid-flight. On Saion:
- Job **4636789 (EBev)**: Laplace evidence `log Z` for each (root × anti-concentration `c` ∈
  {10,100,1000,3000}) on the constrained 18-DTL+4-O Williams structure. Running on **A100**
  (V100 OOMs — always use largegpu for the Williams backward). Outputs
  `/work/SzollosiU/gergely-szollosi/williams_run/cleanml/EBev_{c}_{root}.evidence.json`.
- When ≥~36/40 land: `ssh saion 'cd /work/SzollosiU/gergely-szollosi/williams_run && python3 agg_eb.py'`
  → prints the `log Z(root,c)` table + the **empirical-Bayes rooting** (per root pick
  `c*=argmax_c logZ`, rank roots by max logZ).
- **Success criterion (the lock-in):** deep cluster {Eury, DPANN, TackA, Halo≈MHH} tops the
  EVIDENCE ranking hands-off, SGA roots (Cluster2/AMD/Asgard) demoted; answer is a *region*
  (~150–300 nats), not a sharp peak. NB the point-MLE already gives **Eury #1 at c=3000**
  (cmlD0 runs, 20/20 done) — evidence should confirm + further demote Cluster2/AMD (higher
  maxP^O → more Occam).

If the lock-in passes → the recipe is locked; proceed to the big tree.

## Open work, prioritized
1. **Big tree (Undine C60, S=513, 7059 fams, 15 roots)** under the locked recipe:
   DTL_br2 (72-param, BIC-best, per paper) OR the 18→clade structure + the **Dirichlet
   origination barrier** (`--origination-dirichlet c --origination-vertical-rho 0`, EB-select
   c by evidence) + rank by evidence/AU-test. NEVER free per-branch DTL or per-branch O.
   Driver: `experiments/run_undine_branchwise.py` (has the priors). Needs the Lanczos
   evidence path for k≈2000 (laplace_evidence already supports it).
2. **Transfer-to backward** (`task #23`): the recipient-weight forward is validated
   (`gpurec/core/transfer_to.py`, 1e-15) + additive hooks in `forward._compute_Pibar_inline`
   / `likelihood.E_step`. REMAINING: thread `recipient_w` through `Pi_wave_forward` +
   `E_fixed_point` + the Triton `wave_pibar_uniform_*` kernels (GPU-test-bound), the
   `backward.py` VJP for ω, `--transfer-recipient` flag. Quick-fit shortcut (no engine change):
   set `transfer_mat_unnormalized = ω-broadcast` (masked) → dense/legacy autograd fits ω.
   User wants PURE transfer-to first (global donor T), then "both" (a_d·w_r).
3. **Reconciliation sampler outputs** (`task #22`): `gpurec/core/sampler.py` backtrack is
   validated; build the writers — per-family ALE `.uml_rec`, genome-wide per-branch event
   summary, family×branch presence table, optional recPhyloXML (off by default). Forward Π via
   legacy/dense (Triton-free). `experiments/families_at_root.py` already gives the root readout
   (= origination-at-root; for l2=100 Williams: Eury 1567 vs Cluster2 135 families = LACA size).

## Ops / conventions
- Cluster repo `/work/SzollosiU/gergely-szollosi/gpurec-cpp`; runs in `…/williams_run`.
  Deploy edited .py via `scp` to the repo (no C++ rebuild). `source gpurec_env.sh` cds to repo.
- **A100 (largegpu) for Williams backward/evidence**; V100 (gpu) only for forward-only evals.
  8-A100 cap — let jobs PEND; be considerate of other users; throttle arrays (`%4`–`%6`).
- SSH: ControlMaster multiplexing is set up (minimize connection storms; the `bind 8888`
  warning is harmless). Grep out `bind|channel_setup|Could not request` from ssh output.
- Python one-liners over ssh: pipe a local script via stdin (`ssh saion 'python3 -' < f.py`) —
  inline single-quotes collide with the outer ssh quote.
- gpurec ≡ AleRax likelihood (per-family r=0.999996). log2/bits everywhere. forward.py imports
  Triton unconditionally → not Mac-importable; legacy.py + transfer_to.py + sampler.py are CPU.
- Local repo: 2 commits this session (`77d5237` code, `48fce29` memory snapshot in
  `docs/memory/`), branch `cpp-rust-free`, **not pushed**.

## User preferences (standing)
- Report results AS IS, minimal interpretation ("I can tell better than you can").
- Principled > ad-hoc; the user is the method's author (ALE/AleRax/reconciliation).
- Minimize SSH storms; respect the A100 cap; OK to let jobs pend.

## Live background pollers (this session, may not carry over)
- `bntq6a001` watching EBev → fires when evidence lands (then run `agg_eb.py`).
