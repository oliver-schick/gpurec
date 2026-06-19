"""Validate fraction-missing in the WAVE (Triton/GPU) forward + backward path.

Runs on Saion (CUDA + Triton required). Cross-checks the production wave path
against the CPU legacy fixed-point reference (itself validated bit-exact in
experiments/test_fraction_missing.py), plus a regression, a finite-difference
gradient check, and a no-speed-regression timing check.

Checks:
  [1] regression  : fraction_missing = 0 reproduces the no-missing wave NLL
                    EXACTLY (the leaf baseline collapses to -inf == sigma-only);
  [2] reference   : wave NLL(fm) == legacy fixed-point NLL(fm) (uniform & dense);
  [3] uniform==dense: the two pibar modes agree with fraction-missing active;
  [4] gradient    : implicit-grad dNLL/dtheta(fm) == central finite difference
                    of the wave NLL(fm)  (global + specieswise);
  [5] speed       : forward with fraction-missing is within tolerance of the
                    no-missing forward (the default path stays the fast one).

Usage (on a Saion V100/A100 allocation):
    PYTHONPATH=$PWD python experiments/test_fraction_missing_wave.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

_INV = 1.0 / math.log(2.0)   # ln -> log2
DEV = "cuda"
DT = torch.float64
D0, L0, T0 = 0.1, 0.1, 0.1
SP = "tests/data/test_trees_1/sp.nwk"
G = "tests/data/test_trees_1/g.nwk"
FM = 0.3


# ----------------------------------------------------------------------------
# CPU legacy reference (E_fixed_point + legacy Pi_fixed_point), fraction-missing
# ----------------------------------------------------------------------------
def _legacy_reference(frac: float) -> float:
    """NLL via the legacy full-matrix fixed point with fraction_missing=frac."""
    import contextlib
    import gpurec.core.likelihood as _likmod
    import gpurec.core.legacy as _legmod
    from gpurec.core.preprocess_cpp import _load_extension  # JIT C++ loader
    from gpurec.core.extract_parameters import extract_parameters
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.legacy import Pi_fixed_point
    from gpurec.core.log2_utils import logsumexp2

    # The legacy reference runs purely on CPU; NVTX + the Triton seg-lse kernel
    # are CUDA-only, so route them to no-op / CPU fallbacks here.
    _likmod._nvtx_range = lambda *a, **k: contextlib.nullcontext()

    def _seg_lse_cpu(x, ptr):
        n = int(ptr.numel()) - 1
        rows = [logsumexp2(x[int(ptr[i]):int(ptr[i + 1])], dim=0)
                if int(ptr[i + 1]) > int(ptr[i])
                else torch.full_like(x[0], float("-inf")) for i in range(n)]
        return torch.stack(rows, dim=0) if rows else \
            torch.empty((0, *x.shape[1:]), device=x.device, dtype=x.dtype)
    _legmod._seg_logsumexp_host = _seg_lse_cpu

    raw = _load_extension().preprocess(SP, [G])
    sr, cr = raw["species"], raw["ccp"]
    S, C = int(sr["S"]), int(cr["C"])
    cpu = torch.device("cpu")
    sh = {
        "S": S, "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(cpu),
        "s_C12_indexes": sr["s_C12_indexes"].to(cpu),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=DT, device=cpu),
    }
    ch = {
        "split_leftrights_sorted": cr["split_leftrights_sorted"].to(cpu),
        "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=DT, device=cpu) * _INV,
        "seg_parent_ids": cr["seg_parent_ids"].to(cpu),
        "ptr_ge2": cr["ptr_ge2"].to(cpu),
        "num_segs_ge2": int(cr["num_segs_ge2"]),
        "num_segs_eq1": int(cr["num_segs_eq1"]),
        "end_rows_ge2": int(cr["end_rows_ge2"]),
        "C": C, "N_splits": int(cr["N_splits"]),
        "split_parents_sorted": cr["split_parents_sorted"].to(cpu),
    }
    li = raw["leaf_row_index"].long().to(cpu)
    lc = raw["leaf_col_index"].long().to(cpu)
    root_id = torch.tensor([int(cr["root_clade_id"])], device=cpu)

    theta = torch.log2(torch.tensor([D0, L0, T0], dtype=DT, device=cpu))
    tm = torch.log2(sh["Recipients_mat"])
    pS, pD, pL, tf, mt = extract_parameters(theta, tm, genewise=False, specieswise=False, pairwise=False)
    mv = mt.squeeze(-1) if mt.ndim == 2 else mt

    sP = sh["s_P_indexes"]
    internal = sP[sP < S].unique()
    leaf_mask = torch.ones(S, dtype=torch.bool, device=cpu)
    leaf_mask[internal] = False

    fm = torch.zeros(S, dtype=DT, device=cpu)
    fm[leaf_mask] = frac
    leaf_log = torch.where(fm > 0, torch.log2(fm.clamp_min(1e-300)),
                           torch.full_like(fm, float("-inf")))
    leaf_E = leaf_log if frac > 0 else None
    leaf_obs = leaf_log if frac > 0 else None

    Eo = E_fixed_point(species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                       transfer_mat=tf, max_transfer_mat=mv, max_iters=4000,
                       tolerance=1e-13, warm_start_E=None, dtype=DT, device=cpu,
                       pibar_mode="dense", leaf_E=leaf_E)
    fp = Pi_fixed_point(ccp_helpers=ch, species_helpers=sh,
                        leaf_row_index=li, leaf_col_index=lc,
                        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
                        log_pS=pS, log_pD=pD, log_pL=pL,
                        transfer_mat_T=tf.T.contiguous(), max_transfer_mat=mv,
                        max_iters=4000, tolerance=1e-13, warm_start_Pi=None,
                        device=cpu, dtype=DT,
                        leaf_obs_log=leaf_obs, leaf_species_mask=leaf_mask)
    return float(compute_log_likelihood(fp["Pi"], Eo["E"], root_id))


def _wave_nll(model) -> float:
    with torch.no_grad():
        return float(model().item())


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA required for the wave path (run on a Saion GPU).")
        return 0

    from gpurec.api import GeneReconModel

    def build(mode, frac):
        return GeneReconModel.from_trees(
            species_tree=SP, gene_trees=[G], mode=mode, pibar_mode="uniform",
            device=DEV, dtype=DT, theta_init_rates=(D0, L0, T0),
            fraction_missing=frac, fixed_iters_Pi=None, max_iters_Pi=4000,
            tol_Pi=1e-12, max_iters_E=4000, tol_E=1e-12,
        )

    def build_dense(mode, frac):
        return GeneReconModel.from_trees(
            species_tree=SP, gene_trees=[G], mode=mode, pibar_mode="dense",
            device=DEV, dtype=DT, theta_init_rates=(D0, L0, T0),
            fraction_missing=frac, fixed_iters_Pi=None, max_iters_Pi=4000,
            tol_Pi=1e-12, max_iters_E=4000, tol_E=1e-12,
        )

    ref_base = _legacy_reference(0.0)
    ref_fm = _legacy_reference(FM)
    print(f"legacy reference: NLL(fm=0)={ref_base:.10f}  NLL(fm={FM})={ref_fm:.10f}")
    ok = True

    # [1] regression: fm=0 == no-missing, EXACTLY
    nll_none = _wave_nll(build("global", None))
    nll_fm0 = _wave_nll(build("global", 0.0))
    reg = (nll_none == nll_fm0)
    print(f"[1] regression: wave NLL(no-fm)={nll_none:.10f} | NLL(fm=0)={nll_fm0:.10f} | identical={reg}")
    ok &= reg

    # [2] reference: wave NLL(fm) == legacy NLL(fm) (uniform)
    nll_fm = _wave_nll(build("global", FM))
    d_ref = abs(nll_fm - ref_fm)
    print(f"[2] reference(uniform): wave NLL(fm)={nll_fm:.10f}  legacy={ref_fm:.10f}  |delta|={d_ref:.2e}")
    ok &= (d_ref < 1e-6)
    # also the baseline must match
    d_base = abs(nll_none - ref_base)
    print(f"    baseline: wave NLL(no-fm)={nll_none:.10f}  legacy={ref_base:.10f}  |delta|={d_base:.2e}")
    ok &= (d_base < 1e-6)

    # [3] uniform == dense with fraction-missing
    nll_fm_dense = _wave_nll(build_dense("global", FM))
    d_ud = abs(nll_fm - nll_fm_dense)
    print(f"[3] uniform==dense(fm): uniform={nll_fm:.10f}  dense={nll_fm_dense:.10f}  |delta|={d_ud:.2e}")
    ok &= (d_ud < 1e-6)

    # [4] gradient vs central finite difference (global + specieswise)
    for mode in ("global", "specieswise"):
        m = build(mode, FM)
        loss = m()
        loss.backward()
        g = m.theta.grad.detach().clone()
        # central finite difference on a single coordinate (D of the first row)
        eps = 1e-4
        with torch.no_grad():
            flat = m.theta.view(-1)
            i = 0  # first theta coordinate (delta/D of first species row or global)
            base = flat[i].item()
            flat[i] = base + eps
            fp_ = _wave_nll(m)
            flat[i] = base - eps
            fm_ = _wave_nll(m)
            flat[i] = base
        fd = (fp_ - fm_) / (2 * eps)
        ga = float(g.view(-1)[i])
        rel = abs(fd - ga) / max(1.0, abs(fd))
        print(f"[4] grad {mode:11s}: analytic[0]={ga:+.6f}  fd[0]={fd:+.6f}  rel={rel:.2e}")
        ok &= (rel < 1e-3)

    # [5] speed: fraction-missing forward must not regress vs no-missing forward
    m_none = build("global", None)
    m_fm = build("global", FM)
    for m in (m_none, m_fm):           # warmup / JIT compile
        for _ in range(3):
            _wave_nll(m)
    torch.cuda.synchronize()
    def _time(m, n=30):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            _wave_nll(m)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n
    t_none = _time(m_none)
    t_fm = _time(m_fm)
    ratio = t_fm / t_none
    print(f"[5] speed: no-fm={t_none*1e3:.3f} ms  fm={t_fm*1e3:.3f} ms  ratio={ratio:.3f}")
    # allow 25% slack on a tiny tree (leaf_term build is O(W*S), negligible at scale)
    ok &= (ratio < 1.25)

    print("\nRESULT:", "PASS -- fraction-missing validated in the wave forward+backward"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
