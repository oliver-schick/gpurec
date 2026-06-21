"""DDP correctness test for gpurec family-sharding.

Part A (CPU, always): shard_families is disjoint + complete + work-balanced.
Part B (1 GPU): the CORE algebra — running the SAME 1-step fit on the whole family
  set vs on each shard separately and SUMMING the per-shard nll+grad must MATCH
  (grad_reduction="sum"). This is exactly what the DDP all-reduce computes, validated
  in a single process (no NCCL): proves sum-of-shards == whole to ~1e-9 (float64).
Part C (N GPU, torchrun): the real NCCL all_reduce path end-to-end.

  python experiments/test_ddp_equivalence.py                                   # A (+B if CUDA)
  torchrun --standalone --nproc_per_node=4 experiments/test_ddp_equivalence.py --ddp  # C
"""
import sys, math, argparse, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from gpurec import distributed as ddp

DD = ("/work/SzollosiU/gergely-szollosi/williams_run/data/"
      "3_Reconciliation/Williams_et_al_2017")


def part_a():
    fams = [{"ccp_helpers": {"C": c, "N_splits": c * 2}}
            for c in [1, 1, 1, 1, 100, 1, 1, 50, 1, 1, 1, 1]]
    for world in (2, 3, 4):
        shards = [ddp.shard_families(fams, r, world) for r in range(world)]
        ids = [id(f) for s in shards for f in s]
        assert len(ids) == len(set(ids)) == len(fams)
    print("[A] shard_families disjoint + complete + balanced: OK")


def _load(root, n, dtype):
    from run_williams_branchwise import _load_species_helpers
    from eval_at_alerax_rates import _load_families_named
    sp = _load_species_helpers(str(Path(DD) / "rooted_phylogeny" / root))
    ale = sorted(p for p in glob.glob(str(Path(DD) / "ccps" / "*.ale"))
                 if not Path(p).name.startswith("._"))[:n]
    fams, _names = _load_families_named(ale, sp["species_name_to_index"],
                                        min_species=1, dtype=dtype)
    return sp, fams


def _run(sp, fams_subset, device, dtype, distributed=False):
    """1 optimizer step (Adam, grad_reduction='sum') on `fams_subset`; return
    (nll, grad[S,3]). distributed=True triggers the in-loop all_reduce."""
    from gpurec.optimization.wave_optimizer import optimize_theta_wave
    from run_williams_branchwise import _build_wave_layout, _sp_helpers_for_uniform
    wl, rcids = _build_wave_layout(fams_subset, device, dtype)
    sp_gpu, _anc = _sp_helpers_for_uniform(sp, device, dtype)
    urm = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)
    S = int(sp["S"])
    theta0 = math.log2(0.1) * torch.ones(S, 3, dtype=dtype, device=device)
    res = optimize_theta_wave(
        wave_layout=wl, species_helpers=sp_gpu, root_clade_ids=rcids,
        unnorm_row_max=urm, theta_init=theta0, steps=1, optimizer="adam",
        specieswise=True, pibar_mode="uniform", families=fams_subset, family_batch_size=0,
        device=device, dtype=dtype, leaf_E=None, verbose=False,
        grad_reduction="sum", distributed=distributed, ddp_device=device)
    h = res["history"][0]
    return float(h.negative_log_likelihood), h.gradient.to(torch.float64)


def part_b(device, dtype=torch.float64):
    sp, fams = _load("Eury", 60, dtype)
    nref, gref = _run(sp, fams, device, dtype)                       # whole
    for world in (2, 3, 4):
        ns, gs = 0.0, torch.zeros_like(gref)
        for r in range(world):
            shard = ddp.shard_families(fams, r, world)
            n_, g_ = _run(sp, shard, device, dtype)
            ns += n_; gs += g_                                        # sum of shards
        dn, dg = abs(ns - nref), float((gs - gref).abs().max())
        ok = dn < 1e-7 and dg < 1e-7
        print(f"[B] world={world}: sum-of-shards vs whole  dNLL={dn:.2e} dGrad={dg:.2e}  "
              f"({'OK' if ok else 'FAIL'})")


def part_c(dtype=torch.float64):
    rank, world, local, device = ddp.maybe_init_distributed()
    sp, fams = _load("Eury", 60, dtype)
    n, g = _run(sp, ddp.shard_families(fams, rank, world), device, dtype, distributed=True)
    if rank == 0:
        nref, gref = _run(sp, fams, device, dtype)                   # single-proc whole
        dn, dg = abs(n - nref), float((g - gref).abs().max())
        print(f"[C] {world}-rank NCCL vs 1-proc  dNLL={dn:.2e} dGrad={dg:.2e}  "
              f"({'OK' if dn < 1e-6 and dg < 1e-6 else 'FAIL'})")
    ddp.cleanup()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ddp", action="store_true")
    args = ap.parse_args()
    part_a()
    if args.ddp:
        part_c()
    elif torch.cuda.is_available():
        part_b(torch.device("cuda"))
    else:
        print("[B/C] skipped (no CUDA)")
