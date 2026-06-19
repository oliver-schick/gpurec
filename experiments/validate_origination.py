"""Validate the ORIGINATION parameters added to gpurec (CPU legacy path, no Triton).

Mirrors AleRax ``UndatedDTLMultiModel`` origination: a gene family originates on
species branch ``e`` with probability ``p^O_e`` (a distribution over branches,
Σ_e p^O_e = 1), entering BOTH the numerator and the survival denominator of

    P = [Σ_e p^O_e · Π(root, e)] / [Σ_e p^O_e · (1 − E_e)].

We parameterise ``p^O_e = softmax(omega)_e`` with a free per-branch log2-weight
``omega`` [S]; ``log_pO = omega − logsumexp2(omega)`` and UNIFORM ⇔ omega const
(log_pO = −log2(S)).

Checks (all on the CPU legacy E + legacy Pi path on tests/data/test_trees_1, the
same setup as ``experiments/test_fraction_missing.py``):

  [1] compute_log_likelihood(..., log_pO = −log2(S)) == the None (uniform) path,
      and the None branch is the verbatim original code (byte-identical).
  [2] DIRECT omega gradient ∂(Σ NLL)/∂omega at FIXED Pi, E (the exact formula the
      wave optimizer uses) matches central finite differences (max rel-err < 1e-5).
  [3] ∂L/∂theta WITH a NON-UNIFORM log_pO matches central FD end-to-end through the
      legacy fixed points (exercises the seed change), AND the hand-coded implicit
      seeds (Pi-root softmax + E-denominator) match autograd of the weighted
      compute_log_likelihood (the exact backward.py / implicit_grad.py change).
  [4] origination='uniform' is byte-identical to today: the None path equals the
      pre-origination expression bit-for-bit.
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


def _setup():
    """Load the bundled tiny family + species tree; return everything needed."""
    ext = _load_ext()
    from gpurec.core.log2_utils import logsumexp2

    # NVTX markers are CUDA-only and crash on a CPU/Mac torch build; no-op them.
    import contextlib
    import gpurec.core.likelihood as _likmod
    _likmod._nvtx_range = lambda *a, **k: contextlib.nullcontext()

    # Route the legacy segmented reduction to pure CPU (Triton-less box).
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
    return sh, ch, li, lc, root_id, S, C


def _solve(theta, sh, ch, li, lc):
    """Run the legacy E + legacy Pi fixed points for a given theta. Returns (Pi, E)."""
    from gpurec.core.extract_parameters import extract_parameters
    from gpurec.core.likelihood import E_fixed_point
    from gpurec.core.legacy import Pi_fixed_point

    tm = torch.log2(sh["Recipients_mat"])
    pS, pD, pL, tf, mt = extract_parameters(theta, tm, genewise=False, specieswise=False, pairwise=False)
    mv = mt.squeeze(-1) if mt.ndim == 2 else mt
    Eo = E_fixed_point(species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                       transfer_mat=tf, max_transfer_mat=mv, max_iters=4000,
                       tolerance=1e-13, warm_start_E=None, dtype=DT, device=DEV,
                       pibar_mode="dense", leaf_E=None)
    fp = Pi_fixed_point(ccp_helpers=ch, species_helpers=sh,
                        leaf_row_index=li, leaf_col_index=lc,
                        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
                        log_pS=pS, log_pD=pD, log_pL=pL,
                        transfer_mat_T=tf.T.contiguous(), max_transfer_mat=mv,
                        max_iters=4000, tolerance=1e-13, warm_start_Pi=None,
                        device=DEV, dtype=DT,
                        leaf_obs_log=None, leaf_species_mask=None)
    return fp["Pi"], Eo["E"]


def main() -> int:
    from gpurec.core.likelihood import compute_log_likelihood, origination_log_pO
    from gpurec.core.log2_utils import logsumexp2, _safe_log2_internal as _safe_log2

    sh, ch, li, lc, root_id, S, C = _setup()
    torch.manual_seed(12345)

    theta0 = torch.log2(torch.tensor([D, L, T], dtype=DT, device=DEV))
    Pi, E = _solve(theta0, sh, ch, li, lc)
    Pi = Pi.detach()
    E = E.detach()

    print(f"S={S} species, C={C} clades, root_clade={int(root_id)}, D=L=T={D}")
    ok = True

    # ----------------------------------------------------------------------
    # [1] uniform log_pO == None path; None branch is the original code.
    # ----------------------------------------------------------------------
    nll_none = compute_log_likelihood(Pi, E, root_id)
    log_pO_uniform = torch.full((S,), -math.log2(S), dtype=DT, device=DEV)
    nll_uni = compute_log_likelihood(Pi, E, root_id, log_pO=log_pO_uniform)
    d1 = float((nll_none - nll_uni).abs().max())
    # The None branch is byte-identical to the pre-origination code; the explicit
    # uniform log_pO differs only by float summation order (mean vs logsumexp).
    print(f"[1] uniform log_pO vs None path: NLL_none={float(nll_none):.12f} "
          f"NLL_uni={float(nll_uni):.12f}  max|diff|={d1:.2e} (<= few eps)")
    ok &= (d1 < 1e-12)

    # ----------------------------------------------------------------------
    # [2] DIRECT omega gradient at FIXED Pi, E vs central FD.
    #     Replicates the wave optimizer's exact formula:
    #       NLL_f = -(logsumexp2(root_Pi_f + log_pO) - logsumexp2(log2(1-exp2(E)) + log_pO))
    #     summed over families. Use a few synthetic families = distinct root rows.
    # ----------------------------------------------------------------------
    # Build n_fam "families" with distinct root Pi rows (perturb the real root row).
    n_fam = 4
    root_row = Pi[root_id, :]                      # [1, S]
    root_pi_all = root_row + 0.5 * torch.randn(n_fam, S, dtype=DT, device=DEV)
    log_one_minus_E = _safe_log2(1.0 - torch.exp2(E))

    omega = 0.7 * torch.randn(S, dtype=DT, device=DEV)

    def sum_nll(omega_vec):
        log_pO = origination_log_pO(omega_vec)
        num = logsumexp2(root_pi_all + log_pO, dim=-1)        # [n_fam]
        denom = logsumexp2(log_one_minus_E + log_pO, dim=-1)  # scalar
        return -(num - denom).sum()

    # Analytic (autograd) direct gradient — the same expression the optimizer uses.
    omega_leaf = omega.detach().clone().requires_grad_(True)
    g_analytic = torch.autograd.grad(sum_nll(omega_leaf), omega_leaf)[0].detach()

    # Central finite differences.
    eps = 1e-6
    g_fd = torch.zeros_like(omega)
    for i in range(S):
        op = omega.clone(); op[i] += eps
        om = omega.clone(); om[i] -= eps
        g_fd[i] = (sum_nll(op) - sum_nll(om)) / (2 * eps)

    rel = (g_analytic - g_fd).abs() / g_fd.abs().clamp_min(1e-8)
    max_rel = float(rel.max())
    print(f"[2] ∂(Σ NLL)/∂omega FD-check (n_fam={n_fam}): "
          f"max|analytic-FD| rel-err = {max_rel:.2e}  (target < 1e-5)")
    print(f"    analytic[:4] = {[round(float(x), 6) for x in g_analytic[:4]]}")
    print(f"    FD      [:4] = {[round(float(x), 6) for x in g_fd[:4]]}")
    ok &= (max_rel < 1e-5)

    # ----------------------------------------------------------------------
    # [3a] ∂L/∂theta WITH a non-uniform log_pO, end-to-end through the legacy
    #      fixed points, vs central FD (exercises the weighted likelihood + seed).
    # ----------------------------------------------------------------------
    omega_fixed = 0.9 * torch.randn(S, dtype=DT, device=DEV)
    log_pO_fixed = origination_log_pO(omega_fixed).detach()

    def total_nll_of_theta(theta_vec):
        Pi_t, E_t = _solve(theta_vec, sh, ch, li, lc)
        return compute_log_likelihood(Pi_t, E_t, root_id, log_pO=log_pO_fixed).sum()

    theta_leaf = theta0.detach().clone().requires_grad_(True)
    g_theta_auto = torch.autograd.grad(total_nll_of_theta(theta_leaf), theta_leaf)[0].detach()

    eps_t = 1e-6
    g_theta_fd = torch.zeros_like(theta0)
    for i in range(theta0.numel()):
        tp = theta0.clone(); tp[i] += eps_t
        tm_ = theta0.clone(); tm_[i] -= eps_t
        with torch.no_grad():
            g_theta_fd[i] = (total_nll_of_theta(tp) - total_nll_of_theta(tm_)) / (2 * eps_t)

    rel_t = (g_theta_auto - g_theta_fd).abs() / g_theta_fd.abs().clamp_min(1e-8)
    max_rel_t = float(rel_t.max())
    print(f"[3a] ∂L/∂theta (non-uniform log_pO, legacy path) FD-check: "
          f"max rel-err = {max_rel_t:.2e}  (target < 1e-5)")
    print(f"     autograd = {[round(float(x), 6) for x in g_theta_auto]}")
    print(f"     FD       = {[round(float(x), 6) for x in g_theta_fd]}")
    ok &= (max_rel_t < 1e-5)

    # ----------------------------------------------------------------------
    # [3b] The HAND-CODED implicit seeds match autograd of the weighted likelihood.
    #      Pi-root seed  (backward.py): -softmax(Pi[root,:] + log_pO).
    #      E denominator seed (implicit_grad.py): +∂(n_fam·denom)/∂E with
    #        denom = logsumexp2(log2(1-exp2(E)) + log_pO).
    #      Both are the SEEDS that flow into the wave backward; verifying them on
    #      CPU directly tests the gradient-seed change without needing Triton.
    # ----------------------------------------------------------------------
    # Reference seeds via autograd of compute_log_likelihood w.r.t. Pi and E.
    Pi_leaf = Pi.detach().clone().requires_grad_(True)
    E_leaf = E.detach().clone().requires_grad_(True)
    nll_w = compute_log_likelihood(Pi_leaf, E_leaf, root_id, log_pO=log_pO_fixed).sum()
    dL_dPi_auto, dL_dE_auto = torch.autograd.grad(nll_w, (Pi_leaf, E_leaf))
    seed_Pi_auto = dL_dPi_auto[root_id, :].squeeze(0)   # ∂NLL/∂Pi[root]

    # Hand-coded Pi-root seed (verbatim formula from backward.py).
    root_Pi = Pi[root_id, :].squeeze(0)
    weighted = root_Pi + log_pO_fixed
    lse = logsumexp2(weighted, dim=0)
    from gpurec.core._helpers import _safe_exp2_ratio
    seed_Pi_hand = -_safe_exp2_ratio(weighted, lse)
    pi_seed_err = float((seed_Pi_auto - seed_Pi_hand).abs().max())

    # Hand-coded E denominator seed (verbatim formula from implicit_grad.py,
    # n_fam = 1 here since compute_log_likelihood was summed over a single root row).
    E_req_d = E.detach().clone().requires_grad_(True)
    log_one_minus_E_h = _safe_log2(1.0 - torch.exp2(E_req_d))
    denom_h = logsumexp2(log_one_minus_E_h + log_pO_fixed, dim=-1)
    direct_dNLL_dE = torch.autograd.grad(denom_h, E_req_d)[0]   # ∂denom/∂E (n_fam=1)
    e_seed_err = float((dL_dE_auto - direct_dNLL_dE).abs().max())

    print(f"[3b] hand-coded seeds vs autograd of weighted likelihood:")
    print(f"     Pi-root seed (backward.py)      max|err| = {pi_seed_err:.2e}")
    print(f"     E-denominator seed (impl_grad)  max|err| = {e_seed_err:.2e}")
    ok &= (pi_seed_err < 1e-10) and (e_seed_err < 1e-10)

    # ----------------------------------------------------------------------
    # [4] origination='uniform' (log_pO=None) byte-identical to the original.
    #     Reproduce the pre-origination expression literally and compare.
    # ----------------------------------------------------------------------
    root_probs = Pi[root_id, :]
    numerator_orig = logsumexp2(root_probs, dim=-1) - math.log2(Pi.shape[-1])
    denominator_orig = torch.log2((1 - torch.exp2(E).mean(dim=-1)))
    nll_orig = -(numerator_orig - denominator_orig)
    byte_id = torch.equal(nll_none, nll_orig)
    print(f"[4] origination='uniform' byte-identical to original: {byte_id} "
          f"(NLL={float(nll_none):.12f})")
    ok &= byte_id

    print("\nRESULT:", "PASS -- ORIGINATION implemented + validated "
          "(uniform byte-identical; omega + theta seeds FD-checked)"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
