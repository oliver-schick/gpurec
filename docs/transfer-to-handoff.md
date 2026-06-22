# gpurec — "transfer-to" (per-recipient receptivity) handoff

What the transfer-to work is, how to run it from the CLI today, and how to test it with
simulation. Companion to `docs/oliver-handoff.md`.

---

## 1. The model

Standard gpurec (and AleRax) draws a transfer's **recipient uniformly** over the eligible
branches. **Transfer-to** adds a per-recipient **receptivity weight** `w_r > 0`: some
branches are better transfer *acceptors* than others (e.g. ecologically/physically
accessible clades). The recipient of a transfer from donor `d` is drawn

```
P(recipient = r | donor d) = w_r / Σ_{r' ∉ anc(d)}  w_{r'}      (r not an ancestor of d)
```

`w_r = 2^{ω_r}`; the free parameter is **ω ∈ ℝ^S** (log₂ receptivity), made identifiable by
`mean(ω) = 0` (geometric mean of weights = 1). It enters the **Transfer (T)** gain term and
the **Transfer-Loss (TL)** extinction term as a weighted recipient sum:

```
Π̄[c,d] = log_pT[d] + log₂( Σ_{r∉anc(d)} w_r·Π[c,r]  /  Σ_{r∉anc(d)} w_r )      # T gain
Ē[d]   = log_pT[d] + log₂( Σ_{r∉anc(d)} w_r·E[r]    /  Σ_{r∉anc(d)} w_r )      # TL
```

**Why it stays cheap (the whole point):** `w_r` multiplies only the *recipient* axis, so it
is **separable** — scale `Π` (and `E`) by `w` **once**, reuse the existing uniform-mode
`row_sum − ancestor_sum`, and add a per-donor normalization `−log₂(Σ_{r∉anc(d)} w_r)` (an
`O(S)` precompute). Cost stays **`O(C·S)`** — no `[S,S]` transfer matrix. A general
donor×recipient matrix `T[d,r]` would be `O(C·S²)`. The rank-1 "**both**" model
`a_d·w_r` (per-donor rate × per-recipient weight) is *also* separable (the `a_d` folds into
`log_pT`).

---

## 2. Code map & status

| piece | file | status |
|---|---|---|
| Model math + efficient setup | `gpurec/core/transfer_to.py` | ✅ done, validated to 1e-15 |
| Efficient-path hooks (`recipient_w=` kwarg) | `gpurec/core/forward.py` (`_compute_Pibar_inline`), `gpurec/core/likelihood.py` (`E_step`) | ✅ wired in the **dense/uniform python** path; `None` ⇒ byte-identical |
| **Fitting CLI** (works today) | `experiments/fit_transfer_to.py` | ✅ via legacy dense + autograd |
| Triton wave fast-path with `recipient_w` | `gpurec/core/kernels/wave_*` | ⛔ TODO |
| Implicit backward VJP for ω (fast path) | `gpurec/core/backward.py` | ⛔ TODO (uniform mode has no transfer grad) |
| Sampler recipient draw ∝ `w_r` | `gpurec/core/sampler.py` (`_sample_recipient`) | ⛔ 1-line TODO (see §4) |
| Flag in the main CLI / `gpurec_fit.py` | `cli/reconcile.py`, `experiments/gpurec_fit.py` | ⛔ TODO |

**Public API** (`gpurec/core/transfer_to.py`):
- `normalize_omega(omega[S]) -> omega - omega.mean()` — identifiability.
- `recipient_uniform_setup(omega[S], ancestors_T) -> (w[S], recip_mt_offset[S])` — the efficient
  setup: linear weights `w=2^ω` + per-donor `−log₂Σw`. Feed `w` as `recipient_w=` to the
  uniform-mode forward/E hooks.
- `build_recipient_transfer_mat(omega[S], ancestors_T, log_pT[S]) -> (transfer_mat[S,S], max[S,1])`
  — dense reference (for validation only).

**Why fitting works today without the fast-path/VJP:** the autograd shortcut. Set the
recipient-logit matrix to a recipient-only broadcast,
`transfer_mat_unnormalized[d,r] = ω_r` at valid (non-ancestor) `r`, `−∞` else; then
`extract_parameters` + the **legacy dense fixed points** compute the w-weighted recipient
sum and **plain torch autograd through the unrolled iterations** gives `∂logL/∂ω`. Memory is
`O(C·S²)` (unrolled), so subset families with `--families`.

---

## 3. Run it from the CLI

The working entry point is `experiments/fit_transfer_to.py` (legacy dense + autograd; no
Triton/backward changes needed). Run on an A100.

```bash
cd /work/SzollosiU/gergely-szollosi/gpurec-cpp && source ../williams_run/gpurec_env.sh

# PURE transfer-to: global donor T rate + per-recipient receptivity ω (fit theta + ω)
python experiments/fit_transfer_to.py --root Eury --mode transfer-to --transfer pure \
  --families 500 --steps 80 --lr 0.05 --fm-mode e-only

# "BOTH" model: per-branch donor T rate a_d AND per-recipient ω (rank-1, still O(C·S))
python experiments/fit_transfer_to.py --root Eury --mode transfer-to --transfer both \
  --families 500 --steps 80 --lr 0.05 --fm-mode e-only

# baseline for comparison: current donor-only model (no recipient weights)
python experiments/fit_transfer_to.py --root Eury --mode donor --families 500 --steps 80
```
Flags: `--root` (species tree under the dataset's rooted_phylogeny), `--mode {donor,transfer-to}`,
`--transfer {pure,both}`, `--families N` (subset for the O(C·S²) autograd memory),
`--steps`, `--lr`, `--fm-mode e-only`, `--seed`. It reports the converged data logL; compare
`mode=donor` vs `transfer-to` (Δ logL / a likelihood-ratio or AIC test = "does receptivity
heterogeneity help?") and inspect the fitted `ω` (which clades are strong acceptors).

**Production CLI (TODO for a clean integration):** expose `--transfer-recipient` on
`run_undine_branchwise.py` / `gpurec_fit.py` by (a) threading `recipient_w` (from
`recipient_uniform_setup`) through `Pi_wave_forward` + `E_fixed_point` + the Triton
`wave_pibar_uniform_*` kernels (the O(C·S) fast path), and (b) adding the implicit backward
VJP for ω in `backward.py`. Until then, `fit_transfer_to.py` (dense autograd, subset
families) is the way.

---

## 4. Testing it with simulation

**Situation:** core gpurec has **no forward DTL simulator** — it is a likelihood/inference
engine. The Rust `rustree` bindings *did* expose `simulate_dtl_per_species_iter(...)`
(see `tests/integration/test_e2e_alerax.py`) but they were **removed in the rust-free
build**. `gpurec/core/sampler.py` is a *backward* sampler (samples reconciliations of
already-observed families), not a generator. So a simulator test needs one of:

### (a) Recovery test — the gold standard (needs a forward simulator)
1. **Simulator with recipient bias.** Restore `rustree`'s DTL simulator, *or* write a small
   Python forward DTL sim (≈100 lines): walk the species tree root→tips, run a D/T/L
   birth-death per branch; at each **transfer** event draw the recipient **∝ w_r** (the true
   receptivity). Set a clear ground truth, e.g. `w_r = 5` for one target clade, `1` elsewhere.
2. Simulate a few hundred families → CCPs (single-tree CCPs are fine: `io/ale.py` /
   the C++ amalgamator).
3. **Fit** with `fit_transfer_to.py --mode transfer-to` on the simulated families.
4. **Check recovery:** `ω̂_r` should be elevated on the target clade and ≈ the true `ω`;
   `mode=donor` should fit worse (Δ logL > 0 for transfer-to). Sweep the true bias strength
   to confirm monotonic recovery and an unbiased estimate at `w=1` (null).

### (b) Sampler-consistency test — lightweight, no forward sim, 1-line change
The backward sampler already draws recipients `∝ transfer_mat[d,r]·Π[c,r]`
(`sampler.py:121-123`, `w = T * pir`). Add the weight:
```python
# _sample_recipient(...), accept recipient_w[S]; line ~123:
w = T * pir * (recipient_w if recipient_w is not None else 1.0)
```
Then: fit (or impose) a known `ω`, sample many reconciliations on real families, **tabulate
the realized transfer-recipient frequencies**, and check they match the `w_r·Π`-weighted
prediction. This validates that the sampler and the transfer-to *likelihood* use the **same**
recipient weighting (a self-consistency / "does the fitted ω mean what we think" check) — it
doesn't recover a ground truth but needs only existing data + the 1-line edit.

**Recommended:** (b) first (cheap sanity that the math is wired coherently), then (a) for a
real identifiability validation once a forward sim is available.

---

## 5. Validation already in place
`transfer_to.py` self-tests assert **efficient (`recipient_uniform_setup` + uniform Π̄) ==
dense (`build_recipient_transfer_mat`) == brute-force == additive (ω=0 ⇒ byte-identical to
the current model)** to ~1e-15 (log₂ space, `_safe_log2`, deep-ancestor cancellation guarded
by `clamp_min(tiny)`). So the *forward* recipient-weight likelihood is trusted; the open work
is the efficient backward (VJP) + Triton fast path + the production CLI flag.

(Cluster/deploy notes are in `docs/oliver-handoff.md` §Paths & deploy.)
