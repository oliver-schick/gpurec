# CLAUDE.md — gpurec development context

> Auto-loaded each session. High-signal map of the codebase + the conventions that
> prevent bugs. **For the exhaustive deep read** (module-by-module with file:line, kernel
> internals, the adjoint solves, and a 12-point gotchas list) see
> **`docs/codebase-understanding.md`**; for project state see `docs/current-state.md`.

## What gpurec is
GPU (PyTorch + Triton) reimplementation of **AleRax**'s DTL (Duplication / Transfer /
Loss) gene-tree ↔ species-tree reconciliation. Given a species tree, gene families as
**ALE conditional-clade-probabilities (CCPs)**, and DTL rates, it computes the marginal
**log-likelihood** and its **gradient w.r.t. the rates** (for rate optimization). It is
the ALE/amalgamation framework (Szöllősi et al. 2013 — the user is the method's author)
recast as a differentiable, batched GPU computation. Repo: `github.com/SisyphusMountain/gpurec`.

## The math (orientation for the whole codebase)
- **Everything is log₂ (bits) space.** `theta` = log₂-rates; `_THETA_MIN = log2(1e-10) ≈ -33.2`.
  C++ emits ln; converted to log₂ once at load (`core/model.py:96`).
- **Two nested fixed points:** `E[s]` (extinction prob per species branch) then
  `Pi[clade, species]` (clade reconciles at branch). `E* = G(E*)`, `Π* = F(Π*, E*)`.
- **Likelihood:** `L = logsumexp(Π*[root,:]) − log|S| − log(1 − mean(E))`.
- **Gradients via IMPLICIT differentiation** (not unrolling): solve the adjoint
  `(I − Jᵀ)λ = ∂L/∂x` with Neumann / CG / GMRES → O(1) memory, exact grads.

## Repo map (where things live)
| Zone | Path | Role |
|---|---|---|
| Data/model | `core/model.py` | `GeneDataset`: load, preprocess(+cache), `compute_likelihood_batch` |
| Params/terms | `core/extract_parameters.py`, `core/terms.py`, `core/log2_utils.py` | theta→log_pS/pD/pL+transfer_mat; DTS terms; log₂ math |
| E + likelihood | `core/likelihood.py` | `E_step`/`E_fixed_point`, `compute_log_likelihood` |
| Forward (heavy) | `core/forward.py` (1.2k) | `Pi_wave_forward`, pibar modes, DTS cross/reduced |
| Backward (heavy) | `core/backward.py` (1.7k) | `Pi_wave_backward`, implicit VJP, Neumann self-loop |
| Legacy baseline | `core/legacy.py` | full-matrix `Pi_fixed_point` (test reference only) |
| Batching/sched | `core/batching.py`, `core/scheduling.py` | collation, cross-family wave layout, BFS/phased scheduler |
| Triton kernels | `core/kernels/{wave_step,wave_backward,dts_fused,scatter_lse}.py` | fused forward/backward, segmented logsumexp |
| C++ preprocess (JIT) | `core/cpp/{preprocess,tree_utils,clade_utils}.{cpp,hpp}` (2.7k) | Newick→CCP arrays + species helpers |
| Optimization | `optimization/{wave_optimizer,genewise_optimizer,implicit_grad,linear_solvers}.py` | SGD/Adam, L-BFGS, adjoint assembly, solvers |
| High-level API | `api/{model,autograd,modes,sampling}.py` | `GeneReconModel` (nn.Module), autograd bridge, AleRax sampling |
| CLI | `cli/reconcile.py` | `gpurec` command |
| **.ale loader (new)** | `io/ale.py` | classic ALEobserve `.ale` → CCP arrays (see below) |
| External | `logmatmul/`, `extra/AleRax_modified/` (gitignored) | vendored log-matmul; C++ reference. **Rust-free**: `rustree` removed; the AleRax sampling bridge is pure Python (`api/sampling.py`). |

## Computational pipeline
**Forward** (`api/autograd.py:_GeneReconFunction.forward`, or `GeneDataset.compute_likelihood_batch`):
`theta → extract_parameters[_uniform] → E_fixed_point → build_wave_layout → Pi_wave_forward → compute_log_likelihood`.
**Backward**: `implicit_grad_loglik_vjp_wave` (shared theta) or `Pi_wave_backward + _e_adjoint_and_theta_vjp(genewise=True)`:
solve `(I−F_Π^T)v = ∂L/∂Π`, `q = F_E^T v`, `(I−G_E^T)w = q`, combine → `grad_theta`.

Two API layers:
- **Low-level**: `GeneDataset` (`core/model.py`) — `compute_likelihood_batch(..., pibar_mode=…)`, `@torch.no_grad`.
- **High-level**: `GeneReconModel` (`api/model.py`) — `nn.Module`, `theta` is an `nn.Parameter`,
  `forward()` returns **NLL** so `loss.backward()` + any `torch.optim` works. `from_trees(...)` one-liner.

## Key mechanisms
- **Wave algorithm**: clades scheduled in topological waves (children before parents); families
  merged into cross-family waves (`collate_wave`); few local iterations per wave. ~18× faster than
  the legacy full-matrix fixed point at S≈2000.
- **`pibar_mode`** (transfer term `Pibar = Π @ Tᵀ`): `dense` (cuBLAS TF32), `uniform`
  (O(W·S) closed form `row_sum − ancestor_sum`, default at large S, skips the `[S,S]` matrix),
  `topk` (compressed log-matmul). Set per call.
- **Parameter modes** = `(genewise × specieswise × pairwise)` → global / per-species / per-gene.
  `api/modes.py` maps "global"/"specieswise"/"genewise" to the flags; pairwise has no gradient yet.
- **CCP-array contract** (what ANY input source must produce, per family): `ccp` dict with
  `split_parents_sorted`, `split_leftrights_sorted` ([2N], lefts‖rights), `log_split_probs_sorted`
  (**ln**, converted to log₂ on load), `split_counts`, `seg_parent_ids`/`parents_sorted`,
  `ptr`/`ptr_ge2`, `num_segs_{ge2,eq1,eq0}`, `end_rows_ge2`, `C`, `N_splits`, `root_clade_id`; plus
  per-family `leaf_row_index`/`leaf_col_index`. Built by C++ `build_ccp_arrays` OR `io/ale.py:_build_ccp_arrays`.
  Wave scheduling: provide `phased_waves` (C++) OR omit it → pure-Python BFS fallback
  (`scheduling.py:_compute_clade_waves_bfs`) recomputes waves from the split arrays.

## Inputs & the .ale loader
- **Native input is Newick** (gene trees). The C++ `amalgamate_clades_and_splits` builds the rooted
  CCP (synthesizes the full-leaf root clade Γ + root splits Γ→(g,Γ\g) for every bipartition, plus
  internal splits). Leaf→species = prefix before first `_` (`extract_species_name`).
- **`.ale` is NOT native.** `io/ale.py` (validated) loads classic **ALEobserve** `.ale`:
  ALE `Dip_counts`→internal splits, ALE `Bip_counts`→root splits Γ→(g,Γ\g), synthesize Γ.
  Validated **bit-exact** vs the C++ amalgamator on the same tree sample
  (`experiments/validate_ale_parser.py`). See memory `project_ale_loader.md`.
- **archaea60 dataset** (local `/Users/ssolo/src/archaea60`, full copy via `ales.tgz`): 31,236
  families = `small_fams/` (25,790 tiny 1–3-leaf, ~70% single-gene C=1/N=0) + `treedists/` (5,446
  real multi-tree CCPs, up to 668 leaves). Loader handles all of them (0 failures, exact norm).
  **Decision: use ALL families (no filtering).** Still missing: the **60-taxon archaeal species tree**.

## Conventions & gotchas (read before editing)
1. **`compute_log_likelihood` returns NLL** despite its name (leading `−`, `likelihood.py:180`). The
   whole stack treats the forward output as a *loss*.
2. **log₂ everywhere** — not natural log. `theta` is log₂-rates.
3. **uniform-mode stale `unnorm_row_max`**: `max_transfer_mat` uses a row-max computed once at dataset
   init; as T→0 during optimization this can yield `-inf + +inf = NaN` (see `diagnosis.md`).
4. **Triton/CUDA-only**: `core/forward.py` imports `triton` unconditionally → the package can't even
   import on a machine without Triton (no arm64-macOS wheel). Real runs go on **Saion A100s**
   (see memory `reference_saion_howto.md`). The C++ preprocessing + `io/ale.py` + `batching.py`
   are Triton-free and DO run on the Mac (CPU).
5. **macOS C++ build**: needs `pip install ninja`; Apple clang rejects the hard-coded `-fopenmp`
   (only `#pragma omp`, no `omp_*` calls) → compile without `-fopenmp` + `-I/opt/homebrew/opt/libomp/include`.
6. **Shell `*.ale` globs blow past ARG_MAX** at 25k files — count with `find`, not `ls *.ale`.
7. **float32 precision floor**: rates plateau before AleRax's 1e-10 (NLL difference unrepresentable);
   not a correctness bug (`diagnosis.md`).

## Build · test · run
```bash
pip install -e ".[triton,dev]"     # triton optional in metadata but effectively required to import
pytest tests/                      # unit / integration / gradients / kernels / cli
gpurec --species sp.nwk --gene g.nwk --delta 0.01 --tau 0.01 --lambda 0.01
```
First import JIT-compiles the C++ extension (`core/preprocess_cpp.py`). Tests verify against three
baselines: legacy fixed-point, AleRax reference output (`tests/data/**/output/`), and finite differences.

## Current state, gaps, active front
- **Works**: wave forward/backward, batched cross-family likelihood, `optimize_theta_wave` (SGD/Adam)
  & `optimize_theta_genewise` (L-BFGS), all three pibar modes, global/specieswise/genewise modes.
- **Perf**: S=1999/10fam wave 2.6s (~200ms/fam, 18.6× vs FP); S=19999 0.26s/fam, ~18 GB peak.
- **Known gaps**: pairwise backward (no gradient); batched backward only validated S≤199; genewise
  L-BFGS never run to convergence on real data; wave slower than legacy FP at small S; ~18–20 GB at S=20K.
- **Active front** (recent git): wave-backward Triton kernel micro-optimization — fused backward
  pruning bookkeeping, Pibar row-max reuse, int32 topology, staged Pibar VJP, DTS scalar reductions.
- **In flight this session**: `.ale` loader done+validated; archaea60 fetched; next = wire
  `GeneDataset.from_ale(...)` (Triton-free dataset build testable on Mac; optimize needs Saion),
  obtain the archaea60 species tree, and confirm the C=1/N=0 single-gene forward on Saion.

## Where to look
- **`docs/codebase-understanding.md`** — the exhaustive multi-agent deep read (this file is its condensed map).
- `docs/architecture.md`, `docs/current-state.md` — authoritative (README's structure block is stale:
  lists `logmatmul/` and `core/model.py` as the front door; the `api/` layer is newer).
- `docs/{reconciliation_notes,alerax_explanation,uniform-mode}.md`, `docs/uniform-*-profile.md`, `docs/history/`.
- Auto-memory: `project_ale_loader.md` (.ale loader + archaea60 + macOS build), `reference_saion_howto.md` (cluster ops).
