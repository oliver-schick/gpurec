"""SCRATCH: fraction-missing effect on INFERRED LOSS rate.

Two refinements over scratch_fm_effect_direction.py:
  (A) fraction_missing applied to only a SUBSET of leaves (realistic: only some
      genomes are incomplete), sweeping L to find the ML loss rate.
  (B) joint coordinate-descent over (D, L, T) to locate the full ML optimum and
      report the inferred LOSS rate L* with vs without fm.
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
    leaf_cols = torch.nonzero(leaf_mask, as_tuple=False).flatten().tolist()

    def leaf_E_subset(frac, which_cols):
        fm = torch.zeros(S, dtype=DT, device=DEV)
        for c in which_cols:
            fm[c] = frac
        return torch.where(fm > 0, torch.log2(fm.clamp_min(1e-300)),
                           torch.full_like(fm, float("-inf")))

    def nll_of(D, L, T, leaf_E=None):
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
                            leaf_obs_log=leaf_E, leaf_species_mask=leaf_mask)
        Pi = fp["Pi"]; E = Eo["E"]
        numerator = float(logsumexp2(Pi[root_id, :], dim=-1) - math.log2(Pi.shape[-1]))
        denominator = float(math.log2(1.0 - float(torch.exp2(E).mean())))
        return -(numerator - denominator)

    # ---- (A) fm on a SUBSET (2 of 8 leaves), sweep L ----
    subset = leaf_cols[:2]
    le_sub = leaf_E_subset(0.5, subset)
    Lgrid = [1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1,
             0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0]
    print(f"(A) fm=0.5 on 2/8 leaves (species cols {subset}); D=T=0.1; sweep L")
    print(f"{'L':>8} | {'NLL(fm=0)':>11} | {'NLL(fm subset)':>15}")
    rows_b, rows_f = [], []
    for L in Lgrid:
        b = nll_of(0.1, L, 0.1, None)
        f = nll_of(0.1, L, 0.1, le_sub)
        rows_b.append((L, b)); rows_f.append((L, f))
        print(f"{L:>8.5f} | {b:>11.5f} | {f:>15.5f}")
    Lb, vb = min(rows_b, key=lambda r: r[1]); Lf, vf = min(rows_f, key=lambda r: r[1])
    print(f"  argmin: fm=0 -> L*={Lb:.5f} (NLL {vb:.5f}) | fm subset -> L*={Lf:.5f} (NLL {vf:.5f})")
    print(f"  loss-rate direction: {'UP' if Lf>Lb else ('DOWN' if Lf<Lb else 'SAME')}\n")

    # ---- (B) joint coordinate descent over (D,L,T) ----
    def joint_min(leaf_E, n_rounds=6):
        grid = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1,
                0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
        D, L, T = 0.1, 0.1, 0.1
        best = nll_of(D, L, T, leaf_E)
        for _ in range(n_rounds):
            # L axis
            L = min(grid, key=lambda x: nll_of(D, x, T, leaf_E))
            # D axis
            D = min(grid, key=lambda x: nll_of(x, L, T, leaf_E))
            # T axis
            T = min(grid, key=lambda x: nll_of(D, L, x, leaf_E))
            newbest = nll_of(D, L, T, leaf_E)
            if abs(newbest - best) < 1e-9:
                best = newbest
                break
            best = newbest
        return D, L, T, best

    Db, Lb2, Tb, vb2 = joint_min(None)
    le_all = leaf_E_subset(0.5, leaf_cols)            # fm on all leaves
    Df, Lf2, Tf, vf2 = joint_min(le_all)
    Ds, Ls2, Ts, vs2 = joint_min(le_sub)             # fm on subset
    print("(B) joint coordinate-descent ML optimum (D,L,T):")
    print(f"  fm=0       : D*={Db:.4f} L*={Lb2:.4f} T*={Tb:.4f}  NLL*={vb2:.5f}")
    print(f"  fm=0.5 all : D*={Df:.4f} L*={Lf2:.4f} T*={Tf:.4f}  NLL*={vf2:.5f}")
    print(f"  fm=0.5 2/8 : D*={Ds:.4f} L*={Ls2:.4f} T*={Ts:.4f}  NLL*={vs2:.5f}")
    print(f"  inferred LOSS L*: fm=0 -> {Lb2:.4f} | fm all -> {Lf2:.4f} | fm 2/8 -> {Ls2:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
