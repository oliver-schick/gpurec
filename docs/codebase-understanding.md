# gpurec — Codebase Understanding

A reference for a new senior engineer. Everything below is grounded in the subsystem analyses and end-to-end traces, cross-checked against source where load-bearing (citations are `file:line`).

> _Provenance: synthesized from a 17-way parallel deep-read of the repo. 15 of 17 subsystem readers completed; the **`tests` (§5) and `rustree` sections were reconstructed from cross-references** (their dedicated readers were rate-limited) — treat those two as less directly verified and confirm against source. `CLAUDE.md` is the condensed, auto-loaded map of this document._

---

## 1. What it is & the math

**gpurec is a GPU (PyTorch + Triton) reimplementation of AleRax's Undated-DTL gene-tree/species-tree reconciliation.** Given (a) a species tree, (b) gene families expressed as ALE conditional-clade-probabilities (CCPs), and (c) DTL rates, it computes the marginal log-likelihood of the families under a Duplication/Transfer/Loss model **and** its gradient w.r.t. the rates, so the rates can be optimized.

### The DTL/ALE model
Each gene family is summarized as a CCP: a set of **clades** (subsets of gene leaves) and **splits** (a parent clade resolving into a left/right child clade), each split weighted by how many sampled gene trees support it (→ a conditional split probability). This is the ALE amalgamation — it integrates over gene-tree uncertainty and over all rootings. The reconciliation places each clade onto each **species** branch and scores four events along the species tree:

- **S (speciation):** the two child clades descend into the two species daughters.
- **D (duplication):** both child clades stay in the same species.
- **T (transfer):** one child stays local, the other is received from a donor elsewhere (the "Pibar" term).
- **L (loss):** a lineage dies on a branch.

### The two nested fixed points
1. **E — extinction probability per species branch.** `E[s]` = probability a single gene lineage on branch `s` leaves no observed descendant. Solved by Banach iteration of `E_step` (`likelihood.py:20`, driver `E_fixed_point` at `:86`). This is the **outer** fixed point and is solved **first**; it produces `E`, the two species-children extinction values `E_s1/E_s2`, and the transfer-loss aggregate `Ebar`.
2. **Pi[clade, species] — the reconciliation probability** that a clade is realized at a species branch. Solved by the **wave** algorithm `Pi_wave_forward` (`forward.py:393`). This is the **inner** fixed point; it consumes `E/Ebar/E_s1/E_s2` from step 1.

The marginal likelihood is then read off the **root clade** rows of `Pi`: `compute_log_likelihood` (`likelihood.py:168`).

### Implicit differentiation
Gradients are computed by the **adjoint method**, never by unrolling the fixed-point iterations. For a fixed point `x = f(x, θ)`, the cotangent solves the transpose system `(I − Jᵀ) λ = dL/dx`. There are two such solves — one for the Pi fixed point (Neumann series / GMRES, `Pi_wave_backward` in `backward.py`) and one for the E fixed point (CG with GMRES fallback, `_e_adjoint_and_theta_vjp` in `implicit_grad.py`) — then both are pushed back through `extract_parameters` to produce `grad_theta`. Memory cost is O(1) in iteration count.

### Conventions you must internalize
- **Everything is in log2 (bits) space.** `exp2`/`log2`/`logsumexp2`/`logaddexp2` everywhere, never natural log. `theta = log2(rate)`. `_THETA_MIN = log2(1e-10) ≈ −33.2`.
- **`compute_log_likelihood` returns NLL, not LL.** Verified at `likelihood.py:180`: `return -(numerator - denominator)`. The leading minus is real; the name is a misnomer. Every downstream consumer treats the result as a loss to minimize. Negating it again "to get the likelihood" is a bug. `GeneReconModel.log_likelihood()` flips the sign back for the user.
- **The one exception to the log2 rule:** CCP split probabilities are produced by the C++/`.ale` loaders in **natural log**, and converted to log2 by the consumer at `model.py:96` (multiply by `1/ln2`). Do not "fix" the loaders to emit log2.

---

## 2. The computational pipeline

### Forward chain (theta → NLL)
The authoritative path is in the autograd bridge `_GeneReconFunction.forward` (`api/autograd.py:153`); `GeneDataset.compute_likelihood_batch` (`core/model.py:398`) and `optimize_theta_wave._forward_backward` (`wave_optimizer.py:213`) are parallel call sites with the same ordering.

1. **`extract_parameters` / `extract_parameters_uniform`** (`extract_parameters.py:5/:105`) — map `theta` (log2-rate logits) to normalized log2 probabilities `log_pS, log_pD, log_pL` plus the factored transfer matrix `(transfer_mat, max_transfer_mat)`. A fixed `0` logit is prepended for the speciation category, so `theta` parametrizes D/L/T relative to S. Verified column order: `result[...,0]=log_pS, [...,1]=log_pD, [...,2]=log_pL` (`extract_parameters.py:12-14`).
2. **`E_fixed_point`** (`likelihood.py:86`) — Banach-iterate `E_step` to convergence (sup-norm `< tol`), init `E = log2(0.5) = −1.0`, warm-startable. Returns `{E, iterations, E_s1, E_s2, E_bar}` (note: dict key is `E_bar`, the local name is `Ebar`).
3. **Wave scheduling** — collate families into one batched CCP (`collate_gene_families`, `batching.py:7`), schedule per-family topological waves (`compute_clade_waves`, `scheduling.py:21`), merge into cross-family waves (`collate_wave` / `collate_wave_cross`), split oversized waves (`split_phase_waves`), and build the wave-ordered Pi layout (`build_wave_layout`, `batching.py:478`).
4. **`Pi_wave_forward`** (`forward.py:393`) — solve the Pi fixed point in wave order; returns `Pi` (original order), `Pi_wave_ordered`, `Pibar_wave_ordered`, `uniform_pibar_row_max`, iteration count.
5. **`compute_log_likelihood`** (`likelihood.py:168`) — root marginal: `logsumexp2(Pi[root])  − log2(S)` (uniform root prior) minus survival denominator `log2(1 − mean_s 2^{E_s})`, negated → per-family NLL.

### Backward chain (loss → grad_theta)
Driven by `_GeneReconFunction.backward` (`api/autograd.py:259`, `@once_differentiable`), which dispatches by mode.

1. **`Pi_wave_backward`** (`backward.py:386`, `@torch.no_grad`) — Stage 1 Pi adjoint. Seed root adjoint `accumulated_rhs[root] = −softmax(Pi[root])` (verified `backward.py:925`, the minus encodes the NLL). Loop waves **root→leaves** (`for k in K-1..0`); per wave solve the block-diagonal self-loop adjoint `(I − J_self^T) v = rhs` (Neumann for dense/topk, GMRES for uniform), accumulate 7 partial gradients, then push cross-clade DTS and cross-Pibar adjoints into child rows. Returns `v_Pi`, `grad_E`, `grad_Ebar`, `grad_E_s1`, `grad_E_s2`, `grad_log_pD`, `grad_log_pS`, `grad_max_transfer_mat` (+ `grad_transfer_mat` in dense/topk). **It does NOT return grad_theta.**
2. **`_e_adjoint_and_theta_vjp`** (`implicit_grad.py:101`) — Stage 2. Assemble the E-adjoint RHS `q_E` = `grad_E` + the direct `dNLL/dE` (through the survival denominator) + the Ebar/E_s1/E_s2 chains. Solve `(I − G_E^T) w = q_E` with `_cg` (GMRES fallback). Route both the direct parameter sensitivities (`grad_theta_pi`) and the E-adjoint multiplier `w` (`gtheta_E`) back through `extract_parameters` via two separate `torch.autograd.grad` passes. `grad_theta = grad_theta_pi + gtheta_E`.
3. `_GeneReconFunction.backward` scales `grad_theta` by `grad_output` and returns `(grad_theta, None, None)`.

The public wrapper `implicit_grad_loglik_vjp_wave` (`implicit_grad.py:20`) bundles steps 1–2 for the shared/global path; the genewise optimizer calls them separately.

---

## 3. Module-by-module reference

### Zone: data / model
- **`core/model.py` — `GeneDataset`** (`:23`). The lower-level data entry point: JIT-loads the C++ preprocessor, turns Newick paths into CCP tensors (single / disk-cache / multi modes), stores per-family `theta` on CPU, and runs the `@torch.no_grad` batched wave forward (`compute_likelihood_batch`, `:398`). `set_params(idx, D, T, L)` stores `log2([D, L, T])` — note the **D,L,T** storage order despite the **D,T,L** argument list (`:211`). Depended on by the whole API layer, the CLI, the optimizers, and the `.ale` loader path.
- **`core/preprocess_cpp.py`** — `lru_cache(1)` JIT loader (`_load_extension`, `:17`) compiling the C++ sources with `-O3 -fopenmp`. This is where the OpenMP flags live (not in `pyproject.toml`); macOS needs libomp tweaks here.
- **`core/_helpers.py`** — `_safe_exp2_ratio` (returns 0 where `a==-inf`), `_seg_logsumexp_host` (CPU/Triton segmented LSE), NVTX/profiler context managers.
- **`core/_logmatmul_compat.py`** — one-time import shim for the external `logmatmul` library (works around its top-level `src` package colliding with this repo's). Exposes `HAS_LOGMATMUL` and three callables; guard everything on `HAS_LOGMATMUL`.

### Zone: params / terms / log2 primitives
- **`core/extract_parameters.py`** (`:5`, `:105`) — the θ→probabilities map and the single most important contract in the codebase (see §4). The `__main__` block is a self-contained shape-assertion test (`python -m gpurec.core.extract_parameters`).
- **`core/terms.py`** — `gather_E_children` (`:12`) is the only **live** function (scatters species-child E into a 2*S parent-slot layout, −inf fill); `gather_Pi_children`, `compute_DTS`, `compute_DTS_L` are **legacy-only** (serve `legacy.py`).
- **`core/log2_utils.py`** — numerical foundation: `logsumexp2` (`:51`), `logaddexp2` (`:64`), `log2_softmax` with custom autograd `_Log2Softmax` (`:71`/`:104`), and `_safe_log2_internal` (`:30`, exported as `_safe_log2`, returns −inf with zero gradient at x≤0). Three deliberate NaN-avoidance mechanisms — do not inline or "simplify" them.

### Zone: E + likelihood + legacy
- **`core/likelihood.py`** — `E_step`/`E_fixed_point` (live), `compute_log_likelihood` (the single definition of the NLL sign for the whole codebase). Polymorphic shapes: `E` is `[S]` (shared) or `[N,S]` (per-gene).
- **`core/legacy.py`** — `Pi_step`/`Pi_fixed_point`, the dense full-`[C,S]` baseline. **Production-dead**; exists only as the oracle for `tests/unit/test_wave_vs_fp.py`. Inits non-leaf Pi to `−1000.0` (a finite −inf stand-in) — do not copy that magic number.

### Zone: forward
- **`core/forward.py` — `Pi_wave_forward`** (`:393`). The dominant cost of a likelihood eval. Holds the three Pibar strategies (`_compute_Pibar_inline` `:75`, `_compute_Pibar_uniform_spmm` `:120`), cross-clade DTS reducers (`_compute_dts_cross` `:38`, `_compute_DTS_reduced` `:136`), species-helper caching (`_get_species_wave_helpers` `:160`, mutates `species_helpers['_wave_forward_species_cache']`), and a deprecated `compute_gradient_bounds`. ~80% of the file is environment-flag dispatch and fused-Triton call sites with eager fallbacks; the two `use_global_pibar` branches (overlap-streams vs serial) **duplicate** a ~180-line dispatch ladder — any change must be made in both.

### Zone: backward
- **`core/backward.py` — `Pi_wave_backward`** (`:386`) plus the analytical VJP machinery (`_self_loop_vjp_precompute` `:158`, `_self_loop_Jt_apply` `:332`, `_gmres_self_loop_solve` `:250`) and two autograd-traceable reference helpers (`_self_loop_differentiable` `:16`, `_dts_cross_differentiable` `:90`) used only for validation. Reads ~30 `GPUREC_*` env flags; the eager fallback misrepresents the production CUDA path (which goes through `wave_backward_uniform_fused` → `dts_cross_backward_accum_fused` → `uniform_cross_pibar_vjp_tree_fused`).

### Zone: batching / scheduling
- **`core/batching.py`** — `collate_gene_families` (`:7`), `collate_wave` (`:259`, simple zip-merge), `split_phase_waves` (`:286`), `collate_wave_cross` (`:315`, the production phased cross-family priority-queue scheduler), `build_wave_layout` (`:478`). The contract boundary every CCP source must satisfy (see §4).
- **`core/scheduling.py`** — `compute_clade_waves` (`:21`, C++-phased preferred, BFS fallback `_compute_clade_waves_bfs` `:71`), `wave_stats` (`:129`).

### Zone: Triton kernels
- **`core/kernels/wave_step.py`** — the forward hot loop: one logical op (advance Pi by one DTS_L iteration) realized as many kernels trading off how `Pibar = Pi @ Tᵀ` is computed (dense log-matmul vs the uniform family: parent-walk / ancestor-cols / CSR / signed-sparse-linear / two-kernel). Workhorse `_wave_step_uniform_kernel` (`:283`), multiplexed by ~10 constexpr flags. `build_uniform_linear_operator` (`:944`) builds a host-side signed-sparse operator.
- **`core/kernels/wave_backward.py`** — the backward kernel zoo (~13 `@triton.jit` kernels + launchers), exclusively for the **uniform** Pibar mode. Implements the Neumann self-loop adjoint (`_wave_backward_uniform_kernel` `:109`), the cross-DTS VJP (several variants), and the uniform-Pibar VJP (ancestor-scatter / bottom-up tree / top-down prefix / grouped / from-staged-u_d). All variants are numerically equivalent; they exist for different tree shapes and atomic-contention regimes. The 7-tuple grad ordering `(log_pD, log_pS, E, Ebar, E_s1, E_s2, mt)` is a hard contract with `Pi_wave_backward`.
- **`core/kernels/dts_fused.py`** — `dts_fused` (`:128`): the per-split MAP. Computes `DTS_term[N,S] = log_split_probs + log2sumexp(5 DTL terms)`. No autograd (gradient supplied analytically). Uses `−1e30` as its −inf sentinel.
- **`core/kernels/scatter_lse.py`** — `seg_logsumexp` (`:370`) with custom autograd `SegLSEHdimFn` (`:301`): the per-clade REDUCE over splits sharing a parent (CSR `ptr`). Online max+sum forward, two-pass fp64-accumulated backward. Uses **true `float('-inf')`** as its sentinel — a different convention from `dts_fused` (cross-file footgun).

### Zone: C++ preprocess
- **`core/cpp/preprocess.cpp`** (2667 lines) — parses Newick, amalgamates rooted CCPs (Gamma root-splits + "above" clades), builds the sorted-split/segment-pointer arrays, enumerates species (ancestors/recipients), runs wave schedulers, and returns a torch-tensor-filled pybind dict. Production entry: `preprocess_multiple_families` (`:1221`, parses species tree once, light CCP by default). `preprocess` (`:1397`) builds the FULL CCP including the O(C²) inclusion DAG. `tree_utils.*` (Newick parser + species enumeration) and `clade_utils.*` (BitVec + wyhash dedup) are the support TUs. The `{species, ccp, leaf_row_index, leaf_col_index}` dict is THE contract `io/ale.py` reimplements.

### Zone: optimization
- **`optimization/implicit_grad.py`** — the adjoint assembly (Stage 2) and the two public grad entry points (`implicit_grad_loglik_vjp_wave` `:20`, `implicit_grad_loglik_vjp_wave_genewise` `:297`).
- **`optimization/linear_solvers.py`** — matrix-free `_cg` (`:13`, returns a success flag) and `_gmres` (`:63`, fallback). Both `@torch.no_grad`, use only a matvec closure `Av`.
- **`optimization/wave_optimizer.py` — `optimize_theta_wave`** (`:35`): one shared θ (`[3]` or `[S,3]`) over many families, three backends (Adam default, SGD, scipy L-BFGS-B), family mini-batching, OOM diagnostics, float32→float64 retry.
- **`optimization/genewise_optimizer.py` — `optimize_theta_genewise`** (`:58`): G independent θ solved simultaneously by a hand-rolled vectorized masked L-BFGS (`_lbfgs_two_loop` `:18`) with per-gene Armijo line search and per-gene convergence masking.
- **`optimization/types.py`** — `FixedPointInfo`, `LinearSolveStats` (mutated at runtime with extra timing attrs), `StepRecord`.
- **`optimization/theta_optimizer.py`** — a 10-line legacy **facade** (re-export shim); no implementation.

### Zone: API / CLI
- **`api/model.py` — `GeneReconModel`** (`:169`): the notebook-friendly `nn.Module` facade. `theta` is an `nn.Parameter`; `forward()` returns differentiable NLL. `from_trees` (`:247`) is the one-liner constructor. Custom `_apply` (`:426`) walks the cached `ReconStaticState` on `.to()`/`.cuda()` and resets `warm_E`.
- **`api/autograd.py`** — `ReconStaticState` dataclass (`:37`, holds non-diff state + mutable `warm_E`) and `_GeneReconFunction` (`:148`, the autograd bridge). Writes no new math; packages the existing pipeline behind one `torch.autograd.Function`.
- **`api/modes.py`** — `_MODE_MAP` {global, specieswise, genewise} → flags; `_mode_to_flags`; `_default_theta_init`. Pairwise is intentionally excluded.
- **`api/sampling.py`** — AleRax export bridge (pure Python, **no Rust**): a small Newick parser replicates `ensureUniqueLabels`, writes rate files, and shells out to the `alerax` binary with `--fix-rates`.
- **`cli/reconcile.py`** — the `gpurec` console script. **Uses the OLD `GeneDataset.compute_likelihood` API, NOT `GeneReconModel`**, global-mode only, prints +log_likelihood. Does not exercise the autograd/gradient path.
- **`__init__.py`** — best-effort exports `GeneReconModel`, swallowing `ImportError` so `import gpurec` succeeds even when the JIT build is missing (in which case `gpurec.GeneReconModel` is silently absent).

### Zone: io / ale + debug
- **`io/ale.py`** — pure-Python `.ale` → CCP adapter. `parse_ale_file` (`:93`) → `AleData`; `build_family_from_ale` (`:207`) emits the same per-family dict shape as `preprocess_multiple_families`; `_build_ccp_arrays` (`:334`) is a field-for-field port of the C++ `build_ccp_arrays`. Synthesizes the implicit root clade Gamma, builds internal splits from `Dip_counts` and root splits from `Bip_counts` (dedup both sides of each bipartition). Emits **natural-log** split probs (matches C++). Intentionally omits `phased_waves` (consumers fall back to BFS scheduling). Untracked working-tree feature.
- **`utils/debug.py`** — standalone tensor-debugging toolkit (stats, health checks, comparison, gradient-flow, `DebugContext`). No gpurec imports, no in-package callers.

### Zone: external
The build is **fully Rust-free** (the `rustree` crate was removed; the AleRax
sampling bridge in `api/sampling.py` is pure Python). External pieces: the
`AleRax` binary (sampling + the e2e likelihood oracle), the `logmatmul` library
(log2-space matmul, optional), and `extra/AleRax_modified` (C++ reference,
documented in `docs/alerax_explanation.md`).

---

## 4. Key mechanisms in depth

### The wave algorithm + scheduling
The inner Pi fixed point is solved Gauss-Seidel style over **topological waves**: clades are permuted so each wave occupies a contiguous `Pi[ws:we]` block, and waves are processed children-before-parents. Within a wave, a few **local self-loop iterations** re-apply the wave step in place; children's Pi/Pibar from earlier waves feed the current wave's cross-DTS term (computed once per wave, before the self-loop). Across families, per-family waves are merged into **cross-family waves** so the GPU processes many families per launch.

Scheduling has two production stages: (1) `compute_clade_waves` produces per-family waves — C++ phased waves (phase 1 = leaves, 2 = internal, 3 = roots) when available, else a Python BFS leveling that labels everything phase 0; (2) `collate_wave_cross` re-derives global dependencies and emits cross-family phased waves via a min-heap keyed by `(-split_count, family, clade)`. `build_wave_layout` then converts whichever waves into the contiguous wave-ordered Pi layout, precomputing per-wave split metadata: the eq1 (single-split, direct-copy) vs ge2 (multi-split, segment-LSE-reduced) partition, the ge2 CSR pointers, `sl/sr` child indices, `log_split_probs`, and `reduce_idx` (split → wave-local parent). Convergence is checked over "significant" entries (`Pi > −100.0`) after a 3-iteration warmup; `fixed_iters` (default 6) disables all convergence checks and host syncs.

### The three Pibar modes
`Pibar = Pi @ Tᵀ` (the transfer-aggregate term) is computed three ways, selected by `pibar_mode`:

- **`dense`** — full log-space matmul against the explicit `[S,S]` (or `[G,S,S]`) linear-space transfer matrix: `log2(2^{Pi−max} @ Tᵀ) + max + mt`. Uses cuBLAS / the `logmatmul` log-matmul. The only mode that produces `grad_transfer_mat` (a per-recipient transfer gradient). Cost O(W·S²).
- **`uniform`** — the production default. Because each donor's recipients are uniform over its **non-ancestors** (self included), `Pibar[c,s] = safe_log2(row_sum − ancestor_sum[s]) + Pi_max + mt[s]`, where `ancestor_sum` subtracts the contributions of `s` and its ancestors. **Exact, not an approximation.** Cost O(W·S·depth) via a sparse `ancestors_T` matmul; never materializes `[S,S]`. `transfer_mat` is `None` from `extract_parameters_uniform` onward — the only transfer-rate gradient is through `max_transfer_mat`. `_safe_log2` is mandatory here: fp32 cancellation can make `row_sum − ancestor_sum` slightly negative (→ −inf, not NaN), and the sparse-matmul output must be forced `.contiguous()`.
- **`topk`** — compressed: gather the top-k columns of T and do a small bmm. Forward only; the gradient path falls back to dense and is not finite-difference-validated.

Mode pervades the code: `extract_parameters` vs `extract_parameters_uniform`, the Ebar recomputation in `implicit_grad.py`, the kernel dispatch in forward/backward. Passing the wrong combination (e.g. `uniform` with a non-None `transfer_mat`) silently takes the wrong branch.

### The factored transfer-matrix representation
`extract_parameters` does NOT store `log_transfer_mat` directly. It splits it into `max_transfer_mat[i]` (the per-donor-row max, kept in **log2**) and `transfer_mat[i,j] = 2^(log_transfer_mat[i,j] − max)` (kept in **linear** space, values in (0,1], row-argmax = 1.0). The full log-domain term is reconstructed only at the Pibar step: `log2(Pi_linear @ transfer_matᵀ) + Pi_max + max_transfer_mat`. Treating `transfer_mat` as log probabilities is a classic mistake. Shape subtlety: in the genewise+specieswise+**pairwise** branch `max_transfer_mat` is `[N_sp,1]` (from the shared unnormalized matrix), but `[N_genes,N_sp,1]` in the non-pairwise branch — its rank depends on `pairwise`.

### Implicit differentiation + the adjoint solves
Two distinct Jacobians, two distinct solvers:
- **Pi self-loop** `J_self` is **block-diagonal per clade** (no cross-clade coupling inside one self-loop step; cross-clade coupling lives in the separate cross-DTS step). Solved by a truncated **Neumann series** (`neumann_terms=3`) in the dense/topk fused path; by **GMRES** (`max_iters=5, tol=1e-8`) in uniform mode, because the uniform `J_selfᵀ` spectral radius approaches 1 and Neumann diverges. The root adjoint is seeded `−softmax(Pi[root])` (the minus = NLL).
- **E fixed point** `G_E` adjoint `(I − G_E^T) w = q_E` is solved by **CG** (`linear_solvers._cg`) with **GMRES** fallback, where the matvec is built once via `torch.func.vjp` of `E_step` at the converged `E_star`. CG's success flag is load-bearing: GMRES is invoked only on CG **breakdown** (`pAp≤0` / non-finite), not on mere non-convergence.

The final theta gradient is `grad_theta_pi + gtheta_E`, obtained by re-running `extract_parameters` under `enable_grad` **twice** (two fresh `theta` tensors, two `autograd.grad` passes) — required because the outer pass runs under `@torch.no_grad`. Note: there is **no `grad_log_pL`** anywhere; the loss gradient flows implicitly through the softmax's sum-to-one constraint. `grad_Ebar` is reused twice (chained to E, and added to the `max_transfer_mat` gradient) because `Ebar = f(E, max_transfer_mat)`.

### The parameter modes
`theta` shape and meaning are governed by the `(genewise × specieswise × pairwise)` cube:

| Public mode | flags (G, Sp, P) | theta shape | meaning |
|---|---|---|---|
| `global` | (F,F,F) | `[3]` | one D/L/T for all families & species |
| `specieswise` | (F,T,F) | `[S,3]` | per-species D/L/T |
| `genewise` | (T,F,F) | `[G,3]` | per-gene D/L/T |
| genewise+specieswise | (T,T,F) | `[G,S,3]` | per-gene per-species |
| pairwise | (F,F,T) | `[2]` | D/L only; T is per-recipient, baked into the transfer matrix |

The public API (`api/modes.py`) exposes only global/specieswise/genewise. **Pairwise is unsupported through the public layer** (`GeneReconModel.__init__` raises `NotImplementedError`, sampling rejects it), and pairwise backward is untested. Pairwise also mutates `tr_mat_unnormalized -= 10.0` in place (`model.py:84`, verified) — a magic offset paired with the implicit T that any rate reconstruction must undo.

### The Triton kernel set
| Kernel | Role | Sentinel | Autograd |
|---|---|---|---|
| `dts_fused` | per-split DTL MAP (5 terms + LSE + split-prob) | `−1e30` | none (analytic) |
| `seg_logsumexp` | per-clade REDUCE over splits | `float('-inf')` | custom `SegLSEHdimFn` |
| `wave_step.*` | forward Pi-step (dense + uniform family) | `−1e30` / finfo.min | none |
| `wave_backward.*` | uniform backward (self-loop + cross-DTS + cross-Pibar VJP) | `−1e30` | none (analytic) |

The two infinity conventions across `dts_fused` (−1e30) and `scatter_lse` (true −inf) is the single most likely cross-file footgun.

### The CCP-array contract
Any CCP source — the C++ Newick amalgamator OR `io/ale.py` — must emit, per family: `C`, `N_splits`, `split_leftrights_sorted` laid out as **`[all lefts | all rights]`** (length `2N`, NOT interleaved), `log_split_probs_sorted` (**natural log**), `split_parents_sorted` (else reconstructed), `seg_parent_ids` (alias of `parents_sorted`), `ptr_ge2` (CSR over the **GE2 block only**), `num_segs_ge2/eq1/eq0`, `stop_reduce_ptr_idx = num_segs_ge2`, `end_rows_ge2`, `leaf_row_index`/`leaf_col_index`, `root_clade_id`. Clades are sorted by **descending split_count then ascending id** (so leaves with 0 splits sort to the end), and splits are stably grouped by parent rank. The global split layout for a batch is `[all-GE2 splits | all-EQ1 splits]` across families (NOT per-family contiguous); `ptr_ge2` covers only the GE2 region. The gene→species rule is `extract_species_name`: substring before the first `_` (a crude heuristic that the C++ TODO admits does not exactly replicate AleRax). Getting the GE2/EQ1 ordering or the lefts/rights layout wrong is the most error-prone way to break a new loader.

---

## 5. Testing strategy

| What | Verified against | Where |
|---|---|---|
| Wave forward Pi | legacy dense `Pi_fixed_point` oracle | `tests/unit/test_wave_vs_fp.py` (the sole consumer of `legacy.py`) |
| Forward likelihood | C++ AleRax reference | `test_e2e_alerax` (requires log2→nats + adding back `ln(S)` origination prior AleRax omits) |
| `extract_parameters` shapes | self-asserting `__main__` block | `python -m gpurec.core.extract_parameters` |
| theta gradients | finite-difference, per mode | documented FD-error matrix in `docs/history/*` |
| `seg_logsumexp` | PyTorch reference + fp64 gradcheck | `scatter_lse.py` self-tests (`:418/:439/:461`) |
| `.ale` loader | exact cross-check vs C++ amalgamator | `experiments/validate_ale_parser.py` (untracked) |

**Coverage holes (the authors' own emphasis):** end-to-end **optimizer convergence with the analytical gradient has never been validated** to confirm inferred θ matches AleRax — this is flagged as the #1 gap. Pairwise backward has zero tests; topk gradients are not FD-validated (they fall back to dense); uniform-mode pytest coverage is thin; large-S **backward** is only validated at small S (forward works at S≈20K); E-adjoint stability near spectral-radius-1 is unverified. `utils/debug.py` and the deprecated `compute_gradient_bounds` are only indirectly exercised.

---

## 6. Current state, gaps, and active work

**What works:** the wave path is the sole production Pi solver (the legacy v1 global solver was removed 2026-03-14). Forward + backward + both optimizers are built. Uniform, dense, and specieswise/genewise modes are forward+backward+FD validated; topk forward is fixed and L-BFGS-B converges on it.

**Performance (RTX 4090, from the profiles):** wave is **18.6× faster** than the legacy fixed point at S=1999 (~50 ms/family uniform vs ~200 ms dense). Forward (fixed-6) ≈ **15.36 ms**, dominated by 270 `_wave_step_uniform_kernel` launches (9.96 ms); the kernel is **instruction/memory-traffic bound**, not occupancy bound. Backward ≈ **72 ms/10-family** (down ~2.2× from the prior default), with four hot kernels: the self-loop (`_wave_backward_uniform_kernel`, DRAM-bandwidth bound, largest bucket), the Pibar-VJP tree kernel, the DTS cross-backward (atomics/divergence), and `dts_fused`. The 1000-family path is chunked (~150 families) because one resident `[C,S]` fp32 matrix would be ~51 GB; preprocessing was sped 590 s → 15.6 s by skipping the unused O(C²) inclusion DAG.

**Bottleneck verdict:** the workload is **memory-traffic and host-sync bound**, not compute. Tensor cores don't apply (log/exp/gather/scatter/tree-reduce math). fp64 is dtype-parametric (no separate path) but ~15.9× slower overall on a 4090 (weak scalar fp64, no tensor cores) — strict fp64 production wants A100/H100.

**Known gaps / tech debt:** pairwise backward (untested, blocks per-pair transfer estimation); large-S batched backward (forward works at 20K, backward not); topk gradient (FD-unvalidated); small-S wave kernel slower than legacy FP for S≤256; large-S uniform memory (~18–20 GB peak); dead specieswise+pairwise branches in `extract_parameters`; `scheduling.py` duplicating the C++ scheduler; split-parent reconstruction duplicated in three places; the two `Pi_wave_forward` dispatch ladders duplicated. **Stale docs:** `docs/reconciliation_notes.md` cites obsolete `src/reconciliation/*` paths and old function names; `docs/history/*` predates the `src/`→`gpurec/` rename and a "batched"-branch-ahead-of-main git state that has since merged. Two `.tex` files referenced by the third-pass proposals are absent.

**Apparent active front:** the recent git log (`Optimize fused backward pruning bookkeeping`, `Reuse forward Pibar row max in wave backward`, `Promote optimized wave gather defaults`, `Add/Document wave kernel proposal 6`, `wave int32 topology experiment`) plus the `docs/uniform-backward-*-profile.md` series show the active work is **micro-optimization of the wave-backward Triton kernels** — the USE_PIBAR_ROW_MAX / forward-stat-reuse path, active-mask pruning bookkeeping, device-scalar params, int32 wave topology, and DTS-staged Pibar fusion. Separately, the untracked `gpurec/io/ale.py` + `experiments/*` files are a new, in-progress `.ale` input feature.

---

## 7. Mental model & gotchas

A contributor must keep these in mind to avoid introducing bugs:

1. **`compute_log_likelihood` returns NLL** (verified `likelihood.py:180`). The root adjoint is seeded negative (`backward.py:925`) to match. Do not negate again "to get the likelihood." `model.forward()`/`nll()` are NLL; `log_likelihood()` flips it back.
2. **Everything is log2/bits** — except CCP split probs, which are natural-log out of the loaders and converted at `model.py:96`. Mixing in `torch.log_softmax`/`torch.logsumexp` (base-e) silently corrupts results. E init is `log2(0.5) = −1.0`, not `log(0.5)`.
3. **theta column order is S, D, L, T** in the softmax (`extract_parameters.py:12-14`), but `set_params(idx, D, T, L)` stores `[D, L, T]` (`model.py:211`) — the public arg list reorders L and T. The 0-prepend means theta has one fewer column than there are output probabilities.
4. **`transfer_mat` is linear, `max_transfer_mat` is log2.** The log transfer term is reconstructed only at the Pibar step. There is no `grad_log_pL` and (in uniform mode) no `grad_transfer_mat`.
5. **The forward (`compute_likelihood*`) and backward (`Pi_wave_backward`) both run under `@torch.no_grad`.** Rate gradients are analytic and **injected** (`theta.grad = grad_clean` before `opt.step()` in the Adam/SGD path), not backpropagated. Adding a torch op expecting autograd will be silently ignored.
6. **`warm_E` is mutable state** on the non-Parameter `ReconStaticState`, mutated inside `_GeneReconFunction.forward` — so `forward()` is not pure — and reset to `None` on every `.to()`/`.cuda()` (and `_apply_to_static` returns a new dataclass, so captured `static` references go stale after a device move).
7. **Sentinel conventions differ by array and kernel:** species "no child" = `S` (out of range), species parent root = `−1`, ancestor-cols padding = `−1`; `dts_fused` −inf = `−1e30`, `scatter_lse` −inf = true `float('-inf')`, `Pi` init = `finfo.min` while `Pibar`/`dts_r` init = `−inf`. Mixing these corrupts results silently.
8. **Uniform-mode stale row-max / contiguity hazards:** the `safe_log2(row_sum − ancestor_sum)` formula requires `_safe_log2` and a forced `.contiguous()` on the sparse-matmul output; the `USE_PIBAR_ROW_MAX` reuse path leaves `row_sum` meaningless (mutually exclusive with the recompute branch by construction). In the convergence-check / break path you must write **both** `Pi` and `Pibar` or `Pibar` is left stale.
9. **Wave-local vs global indexing** is the easiest correctness bug in the kernels: `v_k` is `[W,S]` indexed by wave-local `reduce_idx`, while `Pi_star` is `[C,S]` indexed by global `ws+parent_w`, and `sl/sr` child clades are global.
10. **Triton-only / CUDA-only / G==1 / fp32 / S>256 paths:** the fused backward fast path requires Triton present, `pibar_mode=='uniform'`, CUDA, fp32/64, `G==1`, and `S>256`; otherwise it silently falls back to the slower generic path. `torch.cuda.synchronize()` is called unconditionally in `wave_optimizer` and `implicit_grad` (effectively CUDA-only for those timing paths). `tf32` is force-enabled across the global-pibar forward region and only restored on the normal exit — an exception leaves it on globally.
11. **The kernel variants must stay numerically identical.** Any change to the 6 self-loop DTS_L terms or the 5 cross-clade DTS terms, or any sign/weight convention, must be replicated across **every** variant in `wave_backward.py` and **both** dispatch ladders in `forward.py`, or the chosen path silently diverges. The `aw*` buffers in the self-loop kernel are triple-purposed (softmax weights → Neumann scratch → param-grad contributions) and read the wrong meaning at the wrong line if you assume stability.
12. **Mode/shape pitfalls:** `_extract_batch_params` shared path uses `families[indices[0]]['theta']` for the whole batch (per-family `set_params` on other indices is ignored). `reduce='per_family'` is genewise-only. `_prepare_param` treats a 1-D param of length `S` as per-species but length `N` as per-split — ambiguous when `N==S`. The default tolerances differ between the two public likelihood methods (`compute_likelihood` 1e-6 vs `compute_likelihood_batch` 1e-12). The CLI uses the old `GeneDataset` API and does **not** exercise the wave/gradient pipeline.

---

Key files to start in: `gpurec/api/model.py` + `gpurec/api/autograd.py` (the ergonomic entry and the autograd boundary), `gpurec/core/likelihood.py` (the NLL definition and E solver), `gpurec/core/forward.py` (`Pi_wave_forward`, the dominant cost), `gpurec/core/backward.py` + `gpurec/optimization/implicit_grad.py` (the two-stage adjoint), and `gpurec/core/extract_parameters.py` (the θ→probability contract). For the input-side contract, read `gpurec/core/batching.py` and the C++ `gpurec/core/cpp/preprocess.cpp` / its pure-Python twin `gpurec/io/ale.py`.
