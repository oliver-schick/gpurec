"""SCRATCH (not committed to core): fraction-missing EFFECT-DIRECTION test.

Sweep the LOSS rate L over a grid; at each L compute the NLL via the CPU legacy
path (E_fixed_point + legacy Pi_fixed_point + compute_log_likelihood), once with
fraction_missing=0 and once with a moderate fraction_missing on the leaves.
Find argmin-L (the ML loss) in each case and decompose NLL into numerator vs
denominator contributions.

NLL (bits) = -(numerator_log2 - denominator_log2)
  numerator_log2   = logsumexp2(Pi[root]) - log2(S)        [= log2 sum_e p^O_e Pi_{e,Gamma}]
  denominator_log2 = log2(1 - mean_e 2**E_e)               [= log2 sum_e p^O_e (1-E_e)]
so  NLL = -numerator_log2 + denominator_log2.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_INV = 1.0 / math.log(2.0)
DT = torch.float64
DEV = "cpu"


def _load_ext():
    from torch.utils.cpp_extension import load
    cpp = Path(__file__).resolve().parents[1] / "gpurec" / "core" / "cpp"
    srcs = [str(cpp / f) for f in ("preprocess.cpp", "tree_utils.cpp", "clade_utils.cpp")]
    cflags = ["-O3"]
    inc = Path("/opt/homebrew/opt/libomp/include")
    if inc.exists():
        cflags.append(f"-I{inc}")
    bdir = Path.home() / ".cache" / "gpurec_macval" / "preprocess_cpp"
    bdir.mkdir(parents=True, exist_ok=True)
    return load(name="preprocess_cpp", sources=srcs, extra_cflags=cflags,
                extra_ldflags=[], build_directory=str(bdir), verbose=False)


def main() -> int:
    ext = _load_ext()
    from gpurec.core.extract_parameters import extract_parameters
    from gpurec.core.likelihood import E_fixed_point
    from gpurec.core.legacy import Pi_fixed_point
    from gpurec.core.log2_utils import logsumexp2

    import contextlib
    import gpurec.core.likelihood as _likmod
    _likmod._nvtx_range = lambda *a, **k: contextlib.nullcontext()

    import gpurec.core.legacy as _legmod

    def _seg_lse_cpu(x, ptr):
        n = int(ptr.numel()) - 1
        rows = [logsumexp2(x[int(ptr[i]):int(ptr[i + 1])], dim=0)
                if int(ptr[i + 1]) > int(ptr[i])
                else torch.full_like(x[0], float("-inf")) for i in range(n)]
        return torch.stack(rows, dim=0) if rows else \
            torch.empty((0, *x.shape[1:]), device=x.device, dtype=x.dtype)
    _legmod._seg_logsumexp_host = _seg_lse_cpu

    SP = "tests/data/test_trees_1/sp.nwk"
    G = "tests/data/test_trees_1/g.nwk"
    raw = ext.preprocess(SP, [G])
    sr, cr = raw["species"], raw["ccp"]
    S = int(sr["S"])
    C = int(cr["C"])
    sh = {
        "S": S, "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(DEV),
        "s_C12_indexes": sr["s_C12_indexes"].to(DEV),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=DT, device=DEV),
    }
    ch = {
        "split_leftrights_sorted": cr["split_leftrights_sorted"].to(DEV),
        "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=DT, device=DEV) * _INV,
        "seg_parent_ids": cr["seg_parent_ids"].to(DEV),
        "ptr_ge2": cr["ptr_ge2"].to(DEV),
        "num_segs_ge2": int(cr["num_segs_ge2"]),
        "num_segs_eq1": int(cr["num_segs_eq1"]),
        "end_rows_ge2": int(cr["end_rows_ge2"]),
        "C": C, "N_splits": int(cr["N_splits"]),
        "split_parents_sorted": cr["split_parents_sorted"].to(DEV),
    }
    li = raw["leaf_row_index"].long().to(DEV)
    lc = raw["leaf_col_index"].long().to(DEV)
    root_id = torch.tensor([int(cr["root_clade_id"])], device=DEV)

    sP = sh["s_P_indexes"]
    internal = sP[sP < S].unique()
    leaf_mask = torch.ones(S, dtype=torch.bool, device=DEV)
    leaf_mask[internal] = False
    n_leaves = int(leaf_mask.sum())

    def leaf_E_of(frac):
        fm = torch.zeros(S, dtype=DT, device=DEV)
        fm[leaf_mask] = frac
        return torch.where(fm > 0, torch.log2(fm.clamp_min(1e-300)),
                           torch.full_like(fm, float("-inf")))

    def decomposed_nll(D, L, T, leaf_E=None, leaf_obs_log=None):
        theta = torch.log2(torch.tensor([D, L, T], dtype=DT, device=DEV))
        tm = torch.log2(sh["Recipients_mat"])
        pS, pD, pL, tf, mt = extract_parameters(theta, tm, genewise=False,
                                                specieswise=False, pairwise=False)
        mv = mt.squeeze(-1) if mt.ndim == 2 else mt
        Eo = E_fixed_point(species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                           transfer_mat=tf, max_transfer_mat=mv, max_iters=8000,
                           tolerance=1e-14, warm_start_E=None, dtype=DT, device=DEV,
                           pibar_mode="dense", leaf_E=leaf_E)
        fp = Pi_fixed_point(ccp_helpers=ch, species_helpers=sh,
                            leaf_row_index=li, leaf_col_index=lc,
                            E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
                            log_pS=pS, log_pD=pD, log_pL=pL,
                            transfer_mat_T=tf.T.contiguous(), max_transfer_mat=mv,
                            max_iters=8000, tolerance=1e-14, warm_start_Pi=None,
                            device=DEV, dtype=DT,
                            leaf_obs_log=leaf_obs_log, leaf_species_mask=leaf_mask)
        Pi = fp["Pi"]
        E = Eo["E"]
        root_probs = Pi[root_id, :]
        numerator = float(logsumexp2(root_probs, dim=-1) - math.log2(Pi.shape[-1]))
        mean_E = float(torch.exp2(E).mean())
        denominator = float(math.log2(1.0 - mean_E))
        nll = -(numerator - denominator)
        return {
            "nll": nll, "numerator": numerator, "denominator": denominator,
            "mean_E": mean_E, "Emax": float(E.max()), "Emin": float(E.min()),
        }

    print(f"S={S} ({n_leaves} leaves), C={C}, root_clade={int(root_id)}")
    print("NLL = -(numerator - denominator);  num=log2 sum p^O Pi[root];  "
          "den=log2(1 - mean 2**E)\n")

    D0, T0 = 0.1, 0.1
    Lgrid = [1e-6, 1e-5, 1e-4, 0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02,
             0.025, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15, 0.2,
             0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

    fm_val = 0.5
    le_fm = leaf_E_of(fm_val)

    rows_base, rows_fm = [], []
    print(f"{'L':>7} | {'NLL(fm=0)':>12} {'num':>10} {'den':>10} {'meanE':>8} "
          f"| {'NLL(fm=.5)':>12} {'num':>10} {'den':>10} {'meanE':>8}")
    print("-" * 110)
    for L in Lgrid:
        b = decomposed_nll(D0, L, T0, leaf_E=None, leaf_obs_log=None)
        f = decomposed_nll(D0, L, T0, leaf_E=le_fm, leaf_obs_log=le_fm)
        rows_base.append((L, b))
        rows_fm.append((L, f))
        print(f"{L:>7.3f} | {b['nll']:>12.5f} {b['numerator']:>10.4f} "
              f"{b['denominator']:>10.4f} {b['mean_E']:>8.4f} "
              f"| {f['nll']:>12.5f} {f['numerator']:>10.4f} "
              f"{f['denominator']:>10.4f} {f['mean_E']:>8.4f}")

    Lstar_b, b_best = min(rows_base, key=lambda r: r[1]["nll"])
    Lstar_f, f_best = min(rows_fm, key=lambda r: r[1]["nll"])
    print("\n=== ARGMIN over the L grid (ML loss rate) ===")
    print(f"  fm=0   : L* = {Lstar_b:.4f}   NLL* = {b_best['nll']:.6f}")
    print(f"  fm=0.5 : L* = {Lstar_f:.4f}   NLL* = {f_best['nll']:.6f}")
    direction = "UP" if Lstar_f > Lstar_b else ("DOWN" if Lstar_f < Lstar_b else "UNCHANGED")
    print(f"  --> turning fraction-missing ON moves the ML loss rate {direction} "
          f"(from {Lstar_b:.4f} to {Lstar_f:.4f})")

    print("\n=== numerator/denominator decomposition at the fm=0 ML loss "
          f"(L={Lstar_b:.4f}) ===")
    b_at = decomposed_nll(D0, Lstar_b, T0, leaf_E=None, leaf_obs_log=None)
    f_at = decomposed_nll(D0, Lstar_b, T0, leaf_E=le_fm, leaf_obs_log=le_fm)
    print(f"  fm=0   : NLL={b_at['nll']:.5f}  num={b_at['numerator']:.5f}  "
          f"den={b_at['denominator']:.5f}  meanE={b_at['mean_E']:.5f}")
    print(f"  fm=0.5 : NLL={f_at['nll']:.5f}  num={f_at['numerator']:.5f}  "
          f"den={f_at['denominator']:.5f}  meanE={f_at['mean_E']:.5f}")
    print(f"  delta  : dNLL={f_at['nll'] - b_at['nll']:+.5f}  "
          f"dnum={f_at['numerator'] - b_at['numerator']:+.5f}  "
          f"dden={f_at['denominator'] - b_at['denominator']:+.5f}")
    print("  NLL = -num + den, so dNLL = -dnum + dden.")
    print(f"    contribution from numerator : {-(f_at['numerator'] - b_at['numerator']):+.5f}")
    print(f"    contribution from denominator: {+(f_at['denominator'] - b_at['denominator']):+.5f}")

    # --- Sanity (d): fraction_missing=0 reproduces the no-fm baseline EXACTLY ---
    print("\n=== SANITY (d): fm=0 explicit == baseline (no-fm) ? ===")
    le0 = leaf_E_of(0.0)
    allsame = True
    for L in (0.001, 0.05, 0.5, 2.0):
        b = decomposed_nll(D0, L, T0, leaf_E=None, leaf_obs_log=None)["nll"]
        z = decomposed_nll(D0, L, T0, leaf_E=le0, leaf_obs_log=le0)["nll"]
        same = (b == z)
        allsame &= same
        print(f"  L={L:>6.3f}: baseline={b:.10f}  fm=0={z:.10f}  identical={same}")
    print(f"  ALL identical: {allsame}")

    # --- Isolate the denominator (observability) channel: ---
    # Run with the fm E-boundary ON (so E rises and the denominator shrinks)
    # but the Pi leaf boundary OFF (leaf_obs_log=None), to see how much of the
    # NLL move is driven purely by the observability denominator vs the Pi numerator.
    print("\n=== CHANNEL ISOLATION at L=0.05 (D=T=0.1, fm=0.5) ===")
    base = decomposed_nll(D0, 0.05, T0, leaf_E=None, leaf_obs_log=None)
    both = decomposed_nll(D0, 0.05, T0, leaf_E=le_fm, leaf_obs_log=le_fm)
    e_only = decomposed_nll(D0, 0.05, T0, leaf_E=le_fm, leaf_obs_log=None)
    print(f"  baseline (no fm)        : NLL={base['nll']:.5f}  num={base['numerator']:.5f}  den={base['denominator']:.5f}")
    print(f"  E-boundary ONLY (den ch): NLL={e_only['nll']:.5f}  num={e_only['numerator']:.5f}  den={e_only['denominator']:.5f}")
    print(f"  full fm (E + Pi bndry)  : NLL={both['nll']:.5f}  num={both['numerator']:.5f}  den={both['denominator']:.5f}")
    print(f"  dNLL from E/den channel alone : {e_only['nll'] - base['nll']:+.5f}")
    print(f"  dNLL from adding Pi numerator : {both['nll'] - e_only['nll']:+.5f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
