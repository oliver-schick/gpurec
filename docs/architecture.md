# gpurec Architecture

## Code Zones

| Zone | Path(s) | Description |
|------|---------|-------------|
| **Product runtime** | `gpurec/` | Installable package. Everything needed for `pip install -e .` and the `gpurec` CLI. |
| **Validation** | `tests/` | Pytest suite: `unit/`, `integration/`, `gradients/`, `kernels/`, `cli/`. |
| **Research** | `experiments/` | Standalone debug/investigation scripts. Import from `gpurec.*`. Not packaged. |
| **External** | `logmatmul/`, `extra/` | Separate libraries with their own build systems. `logmatmul/` has its own `src/` (unrelated to `gpurec/`). `extra/AleRax_modified/` is a reference C++ implementation. The build is **fully Rust-free** (`rustree` removed; AleRax sampling is pure Python). |

## Computational Pipeline

```
theta
  │
  ▼
extract_parameters  ──→  (log_pS, log_pD, log_pL, transfer_mat)
  │
  ▼
E_fixed_point       ──→  E [S]  (extinction probabilities)
  │
  ▼
Pi_wave_forward     ──→  Pi [C, S]  (clade-species reconciliation matrix)
  │
  ▼
compute_log_likelihood ──→  scalar log₂-likelihood per family
```

Backward: `Pi_wave_backward` → implicit VJP via `(I - J)ᵀ λ = dL/dx` solvers.

## Core Modules (`gpurec/core/`)

| Module | Responsibility |
|--------|---------------|
| `model.py` | `GeneDataset` class: data loading, preprocessing, high-level likelihood API |
| `likelihood.py` | E_step, E_fixed_point, compute_log_likelihood |
| `forward.py` | Pi_wave_forward, DTS cross-clade computation, Pibar strategies |
| `backward.py` | Pi_wave_backward, differentiable self-loop, VJP precomputation |
| `legacy.py` | Legacy full-matrix Pi_step / Pi_fixed_point (test baselines only) |
| `terms.py` | DTS term computation (D, T, S, L event probabilities) |
| `extract_parameters.py` | theta → log₂-space event probabilities + transfer matrix |
| `batching.py` | Multi-family collation, wave-ordered layout (v2) |
| `scheduling.py` | Wave scheduling (Python wrapper over C++ phased scheduler) |
| `log2_utils.py` | logsumexp2, logaddexp2, log2_softmax |
| `_helpers.py` | NVTX profiling markers, numerical utilities |
| `_logmatmul_compat.py` | Conditional import bridge for `logmatmul` GPU kernels |
| `preprocess_cpp.py` | JIT loader for C++ tree preprocessing extensions |
| `kernels/` | Triton GPU kernels: scatter_lse, seg_log_matmul, dts_fused, wave_step, wave_backward |

## Optimization (`gpurec/optimization/`)

| Module | Responsibility |
|--------|---------------|
| `wave_optimizer.py` | SGD/Adam optimization loop using wave forward/backward |
| `genewise_optimizer.py` | Per-gene-family L-BFGS optimization |
| `implicit_grad.py` | VJP closures, E-adjoint, transpose system assembly |
| `linear_solvers.py` | Neumann series, CG, GMRES for `(I - J)ᵀ λ = b` |
| `theta_optimizer.py` | Legacy CG/GMRES implicit gradient optimizer |
| `types.py` | FixedPointInfo, LinearSolveStats, StepRecord dataclasses |

## Parameter Modes

Six valid combinations of `transfer_mode × gene_granularity`:

| transfer_mode | genewise=False | genewise=True |
|--------------|----------------|---------------|
| uniform | global (D,T,L) | per-gene (D,T,L) |
| specieswise | per-branch (D,T,L) | per-gene × per-branch |
| pairwise | per-pair T coefficients | not implemented |

## Testing

### How components are verified

Each pipeline stage has dedicated tests that check correctness against independent baselines (legacy fixed-point iteration, ALeRax C++ reference, finite differences). Tests live under `tests/` organized by kind: `unit/`, `integration/`, `gradients/`, `kernels/`, `cli/`.

#### `extract_parameters` / `extract_parameters_uniform`

Tested indirectly by every forward-pass test (all depend on correct parameter extraction). Direct shape and consistency checks in `tests/unit/test_genewise_wave.py`:
- `test_extract_params_uniform_genewise_shapes`: correct tensor shapes for genewise/specieswise modes.
- `test_extract_params_uniform_genewise_consistency`: per-family extraction matches batched genewise extraction.

#### `E_fixed_point` / `E_step` (extinction probabilities)

No isolated unit test. Verified indirectly through every forward-pass and gradient test — an incorrect E would cause all downstream likelihood values to diverge from baselines.

#### `Pi_wave_forward` (forward pass)

The most extensively tested component. Validated against the legacy `Pi_fixed_point` baseline and against ALeRax:
- `tests/unit/test_wave_vs_fp.py`: wave output matches legacy FP on small-S (fused Triton kernel) and large-S (cuBLAS Pibar). Also benchmarks wave vs FP speed.
- `tests/unit/test_wave_v2.py`: batched wave-ordered layout (v2) matches per-family FP. Also tests the high-level `GeneDataset.compute_likelihood_batch` API.
- `tests/unit/test_cross_family_wave.py`: cross-family wave batching (up to 100 families) matches individual per-family wave and per-family FP. Includes ALeRax reference comparison.
- `tests/unit/test_genewise_wave.py`: genewise wave (uniform and specieswise Pibar) matches genewise FP on small-S and large-S.

#### `Pi_wave_backward` (backward / VJP)

- `tests/gradients/test_wave_gradient.py`: backward pass produces finite gradients, nonzero root gradient. Gradient descent step decreases NLL. `dL/d(log_pD)`, `dL/d(log_pS)`, and `dL/d(max_transfer_mat)` each match central finite differences within 0.1%.
- `tests/kernels/test_wave_backward_kernel.py`: Triton `wave_backward_uniform_fused` kernel matches PyTorch analytical path per-wave (leaf, internal, root, all-waves) at both small-S and large-S. End-to-end FD comparison for all three parameter gradients.

#### `compute_log_likelihood`

Tested in every forward-pass test above (wave vs FP, wave vs ALeRax). The integration tests (`test_e2e_alerax.py`, `test_optimizer_convergence.py`) additionally verify that the scalar likelihood matches ALeRax's output and is self-consistent with optimizer state.

#### DTS terms (`terms.py`)

No isolated unit test. Tested indirectly — DTS terms feed into `Pi_wave_forward`, so any error surfaces as a likelihood mismatch in the wave-vs-FP and wave-vs-ALeRax comparisons.

#### `batching.py` (collation, wave-ordered layout v2)

- `tests/unit/test_wave_v2.py`: `build_wave_layout` + `collate_wave` produce correct batched layouts. Batched likelihood matches per-family sequential computation.
- `tests/unit/test_cross_family_wave.py`: `collate_gene_families` + `collate_wave` at scale (2–100 families). Batched results match individual and FP baselines.

#### `scheduling.py` (wave scheduling)

- `tests/unit/test_scheduling.py`: every clade appears exactly once (coverage), parent waves strictly follow children (topological order), root is in the last wave, wave statistics are consistent.
- Also exercised by all wave forward/backward tests.

#### Triton kernels (`kernels/`)

- `tests/unit/test_seg_logsumexp.py`: `scatter_lse` kernel matches PyTorch reference (float32/float64). Backward gradients match analytical expectations.
- `tests/kernels/test_wave_backward_kernel.py`: `wave_backward_uniform_fused` kernel matches PyTorch analytical per-wave and end-to-end FD. Neumann convergence verified.
- `dts_fused`, `wave_step`, `seg_log_matmul` kernels are tested indirectly via the wave forward/backward pipeline tests.

#### `GeneDataset` / `model.py`

- `tests/unit/test_wave_v2.py`: `test_model_api_wave_matches_fp` — high-level `compute_likelihood_batch` API matches per-family computation.
- `tests/unit/test_cross_family_wave.py`: `test_model_api_wave_vs_sequential` — batched API matches sequential.
- `tests/unit/test_genewise_wave.py`: genewise dataset creation and likelihood computation.
- `tests/integration/test_e2e_alerax.py`: full data loading from simulated trees.

#### Optimization (`optimization/`)

- `tests/unit/test_optimize_genewise.py`: NLL decreases monotonically under L-BFGS, genewise results match per-family scipy L-BFGS-B within 2%, convergence masking reduces `n_active` over steps.
- `tests/integration/test_optimizer_convergence.py`: L-BFGS converges (gradient → 0), two random initializations reach the same optimum (reproducibility), recomputed likelihood matches optimizer's stored NLL (self-consistency). Optional Tier 2: inferred D/L/T rates match ALeRax within 2× ratio.
- `tests/gradients/test_wave_gradient.py`: Neumann series solver matches exact inverse on synthetic systems. Gradient step decreases NLL.

#### `legacy.py` (Pi_fixed_point)

Used as the correctness baseline in `test_wave_vs_fp.py`, `test_wave_v2.py`, `test_cross_family_wave.py`, and `test_genewise_wave.py`. Not tested in isolation — its role is to provide a trusted (slow) reference.

#### CLI (`gpurec reconcile`)

- `tests/cli/test_reconcile.py`: help output, missing-arg errors, successful runs on multiple datasets (test_trees_1, test_trees_2, test_trees_200, test_mixed_200), parameter variation, invalid paths, debug flag, CUDA device, and script existence checks. Validates that output contains "Final Results" and a plausible log-likelihood.
