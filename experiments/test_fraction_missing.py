"""Implement + test fraction-missing (per-leaf p_obs) in gpurec's UndatedDTL likelihood.

Runs the CPU legacy path (E_fixed_point + legacy Pi_fixed_point + compute_log_likelihood)
on the bundled test family, so it works on the Mac (no Triton). Checks:
  1. regression  : fraction_missing = 0 reproduces the baseline likelihood EXACTLY;
  2. effect      : fraction_missing > 0 changes the likelihood;
  3. E self-consistency: the converged E at a missing leaf satisfies the terminal-branch
     fixed-point equation WITH the p^S * E_l (E_l = 1 - p_obs) term, i.e. the injection
     lands in the right place with the right (single-factor) form;
  4. continuity  : fraction_missing -> 0 returns to the baseline.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_INV = 1.0 / math.log(2.0)   # ln -> log2
DT = torch.float64
DEV = "cpu"
D, L, T = 0.1, 0.1, 0.1


def _load_ext():
    """Build the C++ preprocess extension on macOS (drop -fopenmp; add libomp include)."""
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
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.legacy import Pi_fixed_point
    from gpurec.core.log2_utils import logsumexp2

    # NVTX markers are CUDA-only and crash on a CPU/Mac torch build; no-op them.
    import contextlib
    import gpurec.core.likelihood as _likmod
    _likmod._nvtx_range = lambda *a, **k: contextlib.nullcontext()

    # _seg_logsumexp_host imports the Triton kernel at the top (defeating its own
    # CPU fallback on a Triton-less box). Route the legacy reduction to pure CPU.
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

    theta = torch.log2(torch.tensor([D, L, T], dtype=DT, device=DEV))
    tm = torch.log2(sh["Recipients_mat"])
    pS, pD, pL, tf, mt = extract_parameters(theta, tm, genewise=False, specieswise=False, pairwise=False)
    mv = mt.squeeze(-1) if mt.ndim == 2 else mt

    # species-tree leaf mask: nodes that are never internal parents
    sP = sh["s_P_indexes"]
    internal = sP[sP < S].unique()
    leaf_mask = torch.ones(S, dtype=torch.bool, device=DEV)
    leaf_mask[internal] = False
    n_leaves = int(leaf_mask.sum())

    def likelihood(leaf_E=None, leaf_obs_log=None):
        Eo = E_fixed_point(species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                           transfer_mat=tf, max_transfer_mat=mv, max_iters=4000,
                           tolerance=1e-13, warm_start_E=None, dtype=DT, device=DEV,
                           pibar_mode="dense", leaf_E=leaf_E)
        fp = Pi_fixed_point(ccp_helpers=ch, species_helpers=sh,
                            leaf_row_index=li, leaf_col_index=lc,
                            E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
                            log_pS=pS, log_pD=pD, log_pL=pL,
                            transfer_mat_T=tf.T.contiguous(), max_transfer_mat=mv,
                            max_iters=4000, tolerance=1e-13, warm_start_Pi=None,
                            device=DEV, dtype=DT,
                            leaf_obs_log=leaf_obs_log, leaf_species_mask=leaf_mask)
        nll = float(compute_log_likelihood(fp["Pi"], Eo["E"], root_id))
        return nll, Eo

    def leaf_E_of(frac):
        fm = torch.zeros(S, dtype=DT, device=DEV)
        fm[leaf_mask] = frac
        return torch.where(fm > 0, torch.log2(fm.clamp_min(1e-300)),
                           torch.full_like(fm, float("-inf")))

    print(f"S={S} species ({n_leaves} leaves), C={C} clades, D=L=T={D}")
    ok = True

    # 1. baseline vs explicit fraction_missing = 0  -> must be IDENTICAL
    nll_base, _ = likelihood()
    le0 = leaf_E_of(0.0)
    nll_fm0, _ = likelihood(leaf_E=le0, leaf_obs_log=le0)
    same = (nll_base == nll_fm0)
    print(f"[1] regression: baseline NLL={nll_base:.10f} | fm=0 NLL={nll_fm0:.10f} | "
          f"identical={same}")
    ok &= same

    # 2. fraction_missing = 0.3 on every leaf -> likelihood changes
    le3 = leaf_E_of(0.3)
    nll_fm3, Eo3 = likelihood(leaf_E=le3, leaf_obs_log=le3)
    print(f"[2] effect: fm=0.3 NLL={nll_fm3:.10f}  (delta from baseline {nll_fm3 - nll_base:+.6f})")
    ok &= (abs(nll_fm3 - nll_base) > 1e-6)

    # 3. E self-consistency at a missing leaf: E_l == logsumexp2(pS+log2(fm), pD+2E, E+Ebar, pL)
    l = int(torch.nonzero(leaf_mask, as_tuple=False)[0])
    E_l = Eo3["E"][l]
    Eb_l = Eo3["E_bar"][l]
    fm_l = torch.tensor(0.3, dtype=DT)
    terms = torch.stack([
        pS + torch.log2(fm_l),      # p^S * E_l  (single factor, the fraction-missing term)
        pD + 2 * E_l,               # p^D * E_l^2
        E_l + Eb_l,                 # p^T * E_l * Ebar_l   (pT folded into Ebar)
        pL.expand(()),              # p^L
    ])
    rhs = logsumexp2(terms, dim=0)
    resid = float(abs(rhs - E_l))
    print(f"[3] E self-consistency at leaf {l}: E_l={float(E_l):.10f} rhs={float(rhs):.10f} "
          f"|resid|={resid:.2e}")
    ok &= (resid < 1e-9)

    # 4. continuity: tiny fraction_missing -> baseline
    le_tiny, _ = likelihood(leaf_E=leaf_E_of(1e-9), leaf_obs_log=leaf_E_of(1e-9))
    print(f"[4] continuity: fm=1e-9 NLL={le_tiny:.10f}  (|delta|={abs(le_tiny - nll_base):.2e})")
    ok &= (abs(le_tiny - nll_base) < 1e-5)

    print("\nRESULT:", "PASS -- fraction-missing implemented + validated (E + legacy Pi path)"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
