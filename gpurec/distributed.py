"""Family-sharded data-parallel helpers for gpurec (manual all-reduce, NOT torch DDP).

The training gradient is the hand-written implicit VJP assigned straight to
``theta.grad`` (no autograd graph reaches theta), so torch's DistributedDataParallel
hooks would never fire. Instead we shard the gene families across ranks, each rank
runs the unmodified forward+backward on its shard, and we ``all_reduce(SUM)`` the
per-shard NLL + grad in the optimizer loop. Correct because the loss/grad is a SUM
over families and the E-adjoint solve is LINEAR in its (family-summed) RHS with a
family-independent operator: ``Σ_shard solve(q_shard) = solve(Σ_shard q_shard)``.

Triton-free: safe to import on the Mac for the reduction-algebra unit test.
"""
from __future__ import annotations
import os
import torch

try:
    import torch.distributed as dist
    _HAVE_DIST = True
except Exception:                                   # pragma: no cover
    dist = None
    _HAVE_DIST = False


def ddp_enabled() -> bool:
    return _HAVE_DIST and dist.is_available() and dist.is_initialized()


def maybe_init_distributed(device_pref: str = "cuda"):
    """If launched under torchrun (RANK/WORLD_SIZE in env), init NCCL + pin the
    per-rank GPU and return (rank, world_size, local_rank, device). Otherwise return
    (0, 1, 0, single-device) so the bare-``python`` path is unchanged."""
    if not _HAVE_DIST or "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, torch.device(device_pref)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    ndev = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    local = int(os.environ.get("LOCAL_RANK", rank % ndev))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
        backend = "nccl"
    else:                                           # CPU multiproc (testing only)
        device = torch.device("cpu")
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world, local, device


def all_reduce_sum_(t: torch.Tensor) -> torch.Tensor:
    if ddp_enabled():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_max_(t: torch.Tensor) -> torch.Tensor:
    if ddp_enabled():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t


def broadcast_(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    if ddp_enabled():
        dist.broadcast(t, src=src)
    return t


def barrier():
    if ddp_enabled():
        dist.barrier()


def cleanup():
    if ddp_enabled():
        dist.destroy_process_group()


def _family_work(f) -> int:
    """Per-family work proxy = #clades + #splits (drives the balanced partition)."""
    h = f.get("ccp_helpers", f) if isinstance(f, dict) else f
    def _g(k, d=0):
        v = h.get(k, d) if isinstance(h, dict) else getattr(h, k, d)
        try:
            return int(v)
        except Exception:
            return int(d)
    return _g("C", 1) + _g("N_splits", 0)


def shard_families(families, rank: int, world: int):
    """Deterministic work-balanced partition (greedy longest-processing-time).
    Every rank computes the SAME buckets then returns its own (family order
    preserved within the shard). Balances by work, not count, so a few giant
    families don't stall one rank. Returns the rank's family sub-list."""
    if world <= 1:
        return list(families)
    if world > len(families):
        raise ValueError(f"world_size {world} > n_families {len(families)}; "
                         f"reduce --nproc_per_node")
    order = sorted(range(len(families)), key=lambda i: -_family_work(families[i]))
    buckets = [[] for _ in range(world)]
    load = [0] * world
    for i in order:
        j = min(range(world), key=lambda k: load[k])
        buckets[j].append(i)
        load[j] += _family_work(families[i])
    idx = sorted(buckets[rank])                     # restore within-shard order
    return [families[i] for i in idx]
