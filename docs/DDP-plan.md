# gpurec multi-GPU DDP — implementation plan
_(captured from design workflow wf_1aeaccac-1cd, 2026-06-21; ready to implement next session)_

I have enough from the maps plus the two core files I read (the orchestrator/`_e_adjoint_and_theta_vjp` and the `_forward_backward` loop). Here is the concrete plan.

---

# Family-sharded data-parallel training for gpurec — implementation plan

## Decision summary (read first)

- **Manual `torch.distributed.all_reduce(SUM)`, NOT `DistributedDataParallel`.** The training gradient is the hand-written implicit VJP assigned straight to `theta.grad` (`wave_optimizer.py:1139`); the forward runs under `torch.no_grad()` (`api/autograd.py:165`) and backward is `@once_differentiable` (`api/autograd.py:260`). DDP hooks fire on an autograd graph through `nn.Parameter`s — there is no such graph reaching `theta`, so DDP would silently never sync. Confirmed by the DEVICE map.
- **Do NOT refactor `implicit_grad_loglik_vjp_wave` to hoist `q_E`** (BACKWARD-map option (a)). The existing `family_batch_size` path already runs **a per-batch E-adjoint solve and SUMs the resulting `grad_theta_b`** (`wave_optimizer.py:804`). So the *existing single-GPU semantics with `family_batch_size>0` are already "sum of per-shard adjoint solves"*. If we define correctness as **"N-GPU == 1-GPU run with the same per-shard batching"**, then each rank can run the unmodified `implicit_grad_loglik_vjp_wave` on its shard and we all-reduce(SUM) the final `grad_theta`. This is the SIMPLEST correct design and touches zero heavy kernel code.
  - The subtlety: the per-batch adjoint with local `n_fam` (`implicit_grad.py:189`) is the *denominator seed* `n_fam·∂denom/∂E`. Because the E-adjoint solve is **linear in its RHS** `q_E`, and `(I−G_E^T)` is identical on every rank (shared `E_star`), we have `Σ_shard solve(q_shard) = solve(Σ_shard q_shard)` *mathematically*. So summing per-shard `grad_theta` is exact-in-real-arithmetic. The only deviation from a single global solve is iterative-solver tolerance, and that deviation **already exists** in the single-GPU `family_batch_size>0` path. We match that path bit-for-bit (up to float assoc).
- **The one required code change to the math path:** kill the `/n_batch_accum` MEAN normalization (`wave_optimizer.py:813`, `:840`). It must become a global-count normalization or a pure SUM, or N-GPU ≠ 1-GPU by a factor. We make it a pure SUM (loss = Σ NLL) controlled by a flag.

The plan keeps `optimize_theta_wave`'s internals nearly untouched; sharding is wired at the **driver + a thin distributed helper + 3 all-reduce insertion points**.

---

## 1. Distributed setup

**New file: `gpurec/distributed.py`** (Triton-free; safe to import on Mac for unit tests).

```python
import os, torch, torch.distributed as dist

def ddp_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()

def maybe_init_distributed(device_pref="cuda"):
    """Detect torchrun env (RANK/WORLD_SIZE/LOCAL_RANK). If present, init NCCL,
    pin device to LOCAL_RANK, return (rank, world_size, local_rank, device).
    If absent, return (0, 1, 0, resolved-single-device) — single-GPU path unchanged."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, torch.device(device_pref)
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world, local, torch.device(f"cuda:{local}")

def all_reduce_sum_(t: torch.Tensor):
    if ddp_enabled():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def broadcast_(t: torch.Tensor, src=0):
    if ddp_enabled():
        dist.broadcast(t, src=src)
    return t

def barrier():
    if ddp_enabled():
        dist.barrier()

def cleanup():
    if ddp_enabled():
        dist.destroy_process_group()
```

- **Launcher: `torchrun`**, not `mp.spawn`. Reason: gpurec is one-device-per-process already (the `'cuda' -> cuda:current_device()` idiom at `forward.py:172-173`, `backward.py:632-633`, `wave_optimizer.py:23`); `torchrun --nproc_per_node=N` sets `LOCAL_RANK`, `set_device(local)` pins it, and the existing bare-`cuda` resolution then naturally points each process at its own GPU. No refactor.
- Device pinning happens **once** in `maybe_init_distributed` via `torch.cuda.set_device(local)`, before any tensor allocation.

---

## 2. Per-shard wave layout

Each rank builds its layout from its **own family sub-list** using the existing 3-step pipeline; nothing needs re-basing because `collate_gene_families` (`batching.py:7`) assigns clade offsets locally starting at 0 (LAYOUT map confirms this).

**Shard split — contiguous, work-balanced.** Use a deterministic balanced partition keyed on per-family work (clade/split count), NOT `families[rank::world]` (gotcha: archaea60 is 70% tiny + a few 668-leaf giants → naive stride is wildly unbalanced). Add to `gpurec/distributed.py`:

```python
def shard_families(families, rank, world):
    """Greedy balanced partition by per-family work (C + N_splits). Deterministic:
    every rank computes the same partition, then takes its own bucket."""
    def work(f):
        h = f['ccp_helpers']
        return int(h.get('C', 1)) + int(h.get('N_splits', 0))
    order = sorted(range(len(families)), key=lambda i: -work(families[i]))
    buckets = [[] for _ in range(world)]
    load = [0]*world
    for i in order:
        j = min(range(world), key=lambda k: load[k])
        buckets[j].append(i); load[j] += work(families[i])
    idx = sorted(buckets[rank])           # restore family order within shard
    return [families[i] for i in idx]
```

**Build the layout** by reusing the exact whole-dataset template (`api/model.py:_assemble_static_state`, lines 35-127), which already does `collate_gene_families → compute_clade_waves → collate_wave → split_phase_waves → build_wave_layout`, but pass `families_shard` in place of the full list. Concretely, in the driver:

```python
families_shard = shard_families(dataset.families, rank, world)
static_shard = _assemble_static_state(families_shard, dtype=..., device=device)  # rank-local layout
```

Prefer this template over `_lazy_batch_builder` (`wave_optimizer.py:583`) because it calls `split_phase_waves` (`batching.py:286`) for balanced kernel waves. Within a shard you may still set `family_batch_size>0` to chunk memory; that is orthogonal and identical to the single-GPU meaning.

Guard: `assert world <= len(dataset.families)` and never create an empty shard (`collate`'s `max()` over empty breaks — LAYOUT gotcha 6).

---

## 3. The all-reduce points (the correctness core)

**Design (option b) — reuse the existing per-batch adjoint, all-reduce the FINAL summed quantities.** No refactor of `implicit_grad_loglik_vjp_wave`.

Every rank, every optimizer step, in lock-step:

1. **Recompute E identically (no comms).** Each rank calls `E_fixed_point` (`wave_optimizer.py:683`) on the **replicated** `theta` (and `omega`) + full species helpers. E is family-independent (LIKELIHOOD map (1)), so all ranks get bit-identical `E_star`. **Never all-reduce E.**
2. Each rank runs `_forward_backward` on its shard → local `nll_shard` (scalar, `wave_optimizer.py:740`) and `grad_theta_shard` (`:804`), plus `grad_omega_shard` if origination.
3. **Sanitize NaNs locally first** (`grad.nan_to_num_()`) BEFORE reducing — one bad family must not NaN the cluster (DEVICE gotcha 8).
4. **Three all-reduce(SUM):**
   - `all_reduce_sum_(grad_theta_shard)`
   - `all_reduce_sum_(nll_tensor)` (wrap the python float as a CUDA tensor on `device`)
   - `all_reduce_sum_(grad_omega_shard)` (origination only)

**Where, relative to the E-adjoint solve:** the all-reduce is **after** `_forward_backward` returns and **before** the optimizer step / scipy return. The E-adjoint solve runs **inside** `_forward_backward`, per shard, on that shard's `q_E`. We do **not** touch it.

### Gradient-correctness argument (why N-shard == 1-shard)

Let the total loss be `L = Σ_f NLL_f` over all families. Decompose every shard-summable / shared quantity:

- **NLL:** `NLL_f = num_f − denom`, `num_f` per-family (`likelihood.py:221/228`), `denom` shared (depends only on `E`,`log_pO`). Each rank subtracts `denom` once per *local* family, so `Σ_ranks nll_shard = Σ_f num_f − N_total·denom = L`. Exact because `denom` is identical across ranks (E identical). ✓
- **`grad_theta`:** `∂L/∂θ` splits into (i) the **v-part** (`grad_theta_pi`, `implicit_grad.py:317`), linear in family-summed Pi-backward seeds → additive across shards; and (ii) the **w-part** (`gtheta_E`, `:353`), a VJP seeded by `w = (I−G_E^T)^{-1} q_E`. The operator `(I−G_E^T)` is shared (closes over shared `E_star`, `implicit_grad.py:254-267`); `q_E` is a **sum over the shard's families** (`grad_E` seed is family-summed via `_scatter_accum` `backward.py:1145`, plus `n_fam·∂denom/∂E` with local `n_fam`, `:189,204`). Linearity of the solve gives:
  `Σ_shard gtheta_E(solve(q_shard)) = gtheta_E(solve(Σ_shard q_shard)) = gtheta_E(solve(q_total))`.
  And `Σ_shard n_fam_shard = N_total`, so the denominator seed sums to the correct global `N_total·∂denom/∂E`. Therefore `Σ_shard grad_theta_shard = grad_theta_total`. ✓ (The only inexactness is Krylov tolerance across N solves vs one — identical in kind to the existing single-GPU `family_batch_size>0` path, which we are matching.)
- **`grad_omega`:** numerator term per-family/additive (`wave_optimizer.py:833`); denominator term uses shared `E` + `N_total` via the per-family sum (`:835` sums over local families). Summing `grad_omega_shard` recovers the global value, **provided** we drop the `/n_batch_accum` mean (next). ✓
- **Float caveat:** SUM over a different family partition reorders the reduction → match to ~float32 assoc, exact in real arithmetic (CLAUDE.md gotcha 7). Validate in float64.

### The one mandatory edit to the math path

`wave_optimizer.py:812-813` and `:839-840` currently do `grad_theta /= n_batch_accum` (a **mean**). Under SUM all-reduce this scales the gradient by `1/(local batch count)` → N-GPU ≠ 1-GPU. **Fix:** gate it behind a new `grad_reduction` arg (default keeps old behavior; DDP path sets `"sum"`):

```python
# replace lines 812-813
if grad_reduction == "mean" and n_batch_accum > 0:
    grad_theta = grad_theta / float(n_batch_accum)
# (sum -> no division; the 1-GPU reference for the DDP equality test also uses sum)
```
Same guard at `:839-840` for `grad_omega`. The correctness test (§6) runs **both** 1-GPU and N-GPU with `grad_reduction="sum"`.

---

## 4. Optimizer-loop integration (both paths)

Add a small distributed context to `optimize_theta_wave` via new args (§5). Insert all-reduce at the two grad-production sites.

### Torch Adam/SGD path (`wave_optimizer.py:1125-1144`)

After `_forward_backward` (`:1130`), before `theta.grad = grad_clean` (`:1139`):

```python
nll, grad_theta, statsG, E_out = _forward_backward(theta_t, warm_E)
grad_theta.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)   # local, before reduce
if ddp_enabled():
    nll_t = torch.tensor(float(nll), device=device, dtype=dtype)
    all_reduce_sum_(grad_theta)
    all_reduce_sum_(nll_t); nll = float(nll_t.item())
# ... existing prior application, then theta.grad = grad_theta; opt.step()
```

Lock-step is automatic: identical `theta` init (broadcast once at start, see below), identical summed grad → identical `opt.step()` → identical clamp (`:1144`). **Early-stop fix:** the `if diff < tol_theta: break` (`:1178`) must not desync; wrap with `all_reduce(MAX)` on `diff` (or broadcast rank-0's decision) so all ranks break together and never deadlock on the next collective (DEVICE gotcha 9).

### scipy L-BFGS path (`wave_optimizer.py:870-1112`)

scipy owns the loop and must see **identical** `(nll, grad)` on every rank so each rank's L-BFGS state evolves identically (no broadcast of `x` needed if the inputs match bit-for-bit). All ranks run scipy **redundantly**; the only comms is inside `forward_and_grad`.

In `forward_and_grad` (`:891`), after `_forward_backward` and after the existing prior/origination-prior application but **before** flattening to numpy (`:951-960`):

```python
# right before building grad_np / returning to scipy
if ddp_enabled():
    all_reduce_sum_(grad_theta)
    if grad_omega is not None:
        all_reduce_sum_(grad_omega)
    nll_t = torch.tensor(float(nll), device=device, dtype=dtype)
    all_reduce_sum_(nll_t); nll = float(nll_t.item())
```

Important ordering: **priors are per-rank-local and must be applied identically on every rank.** Apply the all-reduce to the *data-likelihood* `nll`/`grad` only, then add priors — OR (simpler) apply priors on rank-replicated quantities after the reduce. Since `theta`/`omega` are identical across ranks, the prior terms (`_apply_prior`, `_apply_origination_*`) are already identical on every rank; to avoid N×-counting the prior, **all-reduce the data terms first, then add the prior once** on each rank. Concretely: reduce `grad_theta`/`nll` produced by `_forward_backward` (data-only), then call `_apply_prior` after the reduce. (Move the existing `_apply_prior` call to after the all-reduce.)

- **Warm-start E** (`warm_E_ref[0]` `:927`, `warm_E` `:1132`): per-rank cells, fine — identical `theta` ⇒ identical warm E. Broadcast `theta` from rank 0 once at the very start (after init, before the loop) so float-identical initialization is guaranteed:
  ```python
  broadcast_(theta_t)            # torch path, after theta_t built (~:874)
  # scipy path: broadcast the initial x0 numpy via a tensor before scipy_minimize
  ```
- **float64 retry recursion** (`:999-1079`): the DDP args must be forwarded through the recursive `optimize_theta_wave` call (mirror how batching args are forwarded at `:1044-1048`). The new args are plain kwargs, so add them to that forwarded set.
- **Origination is L-BFGS-only** (`:345-348`); the scipy path is the one to wire for origination fits (the Undine rooting use-case). The Adam path needs no omega reduce.

---

## 5. Wiring: args, drivers, sbatch

### `optimize_theta_wave` (`wave_optimizer.py:52`)
Add kwargs (all default to single-GPU behavior):
- `distributed: bool = False` — enables the all-reduce insertions.
- `grad_reduction: str = "mean"` — `"sum"` for DDP exactness; gate `:813`/`:840`.
- `ddp_device: Optional[torch.device] = None` — the pinned per-rank device (else use `device`).

Forward all three through the float64 retry recursion (`:1044-1048`).

### Drivers
`experiments/run_undine_branchwise.py` (currently modified in the tree) and the Williams driver:
- At top of `main`: `rank, world, local, device = maybe_init_distributed()`.
- Build the dataset/families on every rank (cheap, Triton-free `io/ale.py` / C++ preprocess), then `families_shard = shard_families(dataset.families, rank, world)` and assemble the rank-local `static_shard` (§2).
- Call `optimize_theta_wave(..., device=device, distributed=(world>1), grad_reduction="sum", ddp_device=device)`.
- Only **rank 0** writes outputs / logs / checkpoints; guard file writes with `if rank == 0:`.
- `cleanup()` in a `finally`.

Add a CLI flag `--ddp` only as documentation; actual detection is env-based (`maybe_init_distributed` keys on `RANK`/`WORLD_SIZE`), so `torchrun` "just works" and a bare `python` run stays single-GPU.

### sbatch (new `slurm/undine_branchwise_ddp.sbatch`)
```bash
#SBATCH --partition=largegpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1                 # ONE task; torchrun spawns the workers
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00            # SLURM 12h cap (memory: feedback-slurm-wall)
module load python/3.11.11         # admin requirement (memory: feedback-slurm-module-python)
export NCCL_DEBUG=WARN
torchrun --standalone --nproc_per_node=4 \
    experiments/run_undine_branchwise.py --root <ROOT> ...
```
Trainer must be **resumable across re-submits** (12h cap) — checkpoint `theta`/`omega` on rank 0 each step and reload on start (memory: feedback-slurm-wall).

---

## 6. Correctness test

**New file: `experiments/test_ddp_equivalence.py`** — run the SAME tiny fit on 1 and N ranks; assert per-step `nll` and `grad` match in **float64**.

Strategy (cleanest, avoids needing 2 GPUs for the math check): make the test exercise the **reduction algebra** deterministically.

1. Pick a small fixture: ~20–40 families, S small (e.g. the `tests/data` sample or a 60-taxon subset), `dtype=torch.float64`, fixed `theta`.
2. **Reference (1 "rank"):** call `_forward_backward` on ALL families with `grad_reduction="sum"`, capture `(nll_ref, grad_ref)` for one step (no optimizer move).
3. **Sharded (N logical shards, single process):** for `world in (2,3,4)`, partition with `shard_families(families, r, world)`, run `_forward_backward` per shard (each recomputes the same `E_star` from the same `theta`), then `nll_sum = Σ nll_shard`, `grad_sum = Σ grad_shard`.
4. Assert:
   ```python
   assert abs(nll_sum - nll_ref) < 1e-9
   assert torch.allclose(grad_sum, grad_ref, atol=1e-9, rtol=0)
   ```
   (float64 ⇒ ~1e-9; in float32 expect ~1e-5, per CLAUDE.md gotcha 7.)
5. **Multi-process variant** (run under `torchrun --nproc_per_node=N` on Saion, NCCL): same fixture, each rank takes `shard_families(..., rank, world)`, real `all_reduce_sum_`, compare on rank 0 against a serialized single-rank reference `.pt`. Assert same 1e-9 bound. This validates the actual NCCL path + device pinning + lock-step.
6. **Timing comparison:** in the multi-process variant, time `K=20` full steps on the real Undine shard at S=513, ~7059 families, for `world ∈ {1,2,4}`; print wall-clock/step and speedup. Report E-recompute overhead per rank (should be a small constant since E is S-only).

The single-process test (steps 1–4) runs **on the Mac** for the layout/reduction algebra (`shard_families`, `collate`, summation) — Triton-free parts. The forward/backward kernels and the NCCL test run on Saion A100s (CLAUDE.md gotcha 4).

---

## 7. Risks / gotchas

1. **DDP wrapper is unusable here** — manual `all_reduce` only. (DEVICE map; reason in Decision summary.) Do not wrap `GeneReconModel` in DDP; its grad never flows through autograd.
2. **`/n_batch_accum` mean vs sum** (`wave_optimizer.py:813,840`) — the single biggest correctness trap. Must be SUM (or global-count) under sharding. Gated by `grad_reduction`. The 1-GPU reference in the equality test must use the SAME convention.
3. **E recompute vs broadcast** — recompute per rank (chosen): zero comms, deterministic given identical `theta`. Broadcast is the fallback if float drift between ranks is ever observed; keep `theta` bit-synced (broadcast once at start, identical summed grads, identical clamp). **Never all-reduce E** (would multiply by `world`). Same for `log_pS/pD/pL`, `transfer_mat`, `max_transfer_mat`, the survival `denom`, `leaf_E`, `log_pO`, species helpers — all family-independent, replicated, never reduced.
4. **Uneven shards** — archaea60/Undine have a few giant families among many tiny ones; `families[rank::world]` would stall the cluster on whichever rank holds the 668-leaf trees. Balance by **work (C+N_splits)**, not family count (`shard_families`). Correctness is unaffected by imbalance, only speed.
5. **warm_start_E per rank** — fine while `theta` is lock-step; would diverge if `theta` ever desyncs. Guard by the start-of-run `theta` broadcast.
6. **`family_batch_size` interaction** — orthogonal to sharding and *already* produces the "sum of per-batch adjoint solves" semantics we rely on. Keep it for per-rank memory control; it does not change the reduction algebra. The per-batch E-adjoint approximation is inherited identically by 1-GPU and N-GPU, so the equality test holds.
7. **Triton + multiprocess** — `torchrun` spawns N processes each importing `core/forward.py` (Triton). First import JIT-compiles the C++ extension and Triton kernels concurrently → possible build race on a shared cache dir. Mitigate: pre-warm the C++/Triton compile on rank 0 with a `barrier()` before the others import heavy kernels, or set per-rank `TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` (`$TMPDIR/triton_$LOCAL_RANK`). Do the same for the C++ JIT (`core/preprocess_cpp.py`).
8. **NaN poisoning** — `nan_to_num_` the LOCAL grad BEFORE all-reduce (`:1138` currently sanitizes after; move/duplicate it before the reduce). One bad family-shard otherwise NaNs every rank.
9. **Early-stop deadlock** — `diff < tol_theta` break (`:1178`) must be a collective decision (`all_reduce(MAX, diff)` or broadcast rank-0's flag) or ranks deadlock at the next `all_reduce`.
10. **scipy lock-step** — all ranks run scipy redundantly on identical reduced `(nll, grad)`; apply **priors after the reduce** so the prior is counted once, not `world×`. float64 retry recursion (`:999-1079`) must forward the new DDP kwargs.
11. **SLURM** — `--time<=12:00:00`, `module load python/3.11.11`, rank-0 checkpointing for resumability (memories: feedback-slurm-wall, feedback-slurm-module-python).

---

### Files touched
- **New:** `gpurec/distributed.py` (init/reduce/shard helpers), `experiments/test_ddp_equivalence.py`, `slurm/undine_branchwise_ddp.sbatch`.
- **Edited:** `gpurec/optimization/wave_optimizer.py` — add `distributed`/`grad_reduction`/`ddp_device` kwargs (`:52`); gate mean-norm at `:813` and `:840`; all-reduce in scipy `forward_and_grad` (~`:945`) and torch loop (`:1130-1139`); collective early-stop (`:1178`); broadcast `theta` at start (~`:874`); forward kwargs through float64 retry (`:1044-1048`). `experiments/run_undine_branchwise.py` (+ Williams driver) — `maybe_init_distributed`, `shard_families`, rank-0-only IO, `cleanup()`.
- **Reused unchanged:** `implicit_grad.py` (NO refactor — option (b)), `core/backward.py`, `core/forward.py`, `core/batching.py`, `api/model.py:_assemble_static_state` (as the per-shard layout template).