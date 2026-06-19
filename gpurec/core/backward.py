"""Backward pass: Pi_wave_backward and helpers (VJP / Neumann / GMRES)."""
import os

import torch

from .log2_utils import logsumexp2, logaddexp2, _safe_log2_internal as _safe_log2
from ._helpers import _safe_exp2_ratio  # noqa: F401

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Differentiable self-loop step (for torch.func.vjp tracing)
# ---------------------------------------------------------------------------

def _self_loop_differentiable(
    Pi_W, mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2, leaf_wt, dts_r, S,
    pibar_mode='uniform', transfer_mat_T=None, ancestors_T=None,
):
    """Pure-PyTorch differentiable self-loop step (Pibar + DTS_L).

    Computes one iteration of g_k(Pi_W) = logaddexp2(dts_r, DTS_L(Pi_W, Pibar(Pi_W))).
    Used by the backward pass to build VJP closures via torch.func.vjp.

    Args:
        Pi_W: [W, S] log2-space, requires_grad
        mt_squeezed: [S] or [W, S] max transfer mat
        DL_const, Ebar, E, SL1_const, SL2_const: [S] or [W, S] precomputed constants
        sp_child1, sp_child2: [S] species child indices
        leaf_wt: [W, S] leaf term
        dts_r: [W, S] or None, cross-clade DTS (frozen)
        S: int
        pibar_mode: 'dense', 'uniform', or 'topk'
        transfer_mat_T: [S, S] for dense/topk mode (may require grad for theta VJP)
        ancestors_T: [S, S] sparse COO, ancestors.T (for uniform mode)

    Returns:
        Pi_new: [W, S] log2-space
    """
    W = Pi_W.shape[0]

    def _expand(t):
        return t.unsqueeze(0).expand(W, -1) if t.ndim == 1 else t

    mt_w = _expand(mt_squeezed)
    DL_w = _expand(DL_const)
    Ebar_w = _expand(Ebar)
    E_w = _expand(E)
    SL1_w = _expand(SL1_const)
    SL2_w = _expand(SL2_const)

    # --- Pibar ---
    Pi_max = Pi_W.max(dim=1, keepdim=True).values
    Pi_exp = torch.exp2(Pi_W - Pi_max)
    if pibar_mode == 'uniform':
        row_sum = Pi_exp.sum(dim=1, keepdim=True)
        ancestor_sum = Pi_exp @ ancestors_T
        Pibar_W = _safe_log2(row_sum - ancestor_sum) + Pi_max + mt_w
    else:  # dense or topk
        Pibar_W = _safe_log2(Pi_exp @ transfer_mat_T) + Pi_max + mt_w

    # --- Species children of Pi_W ---
    Pi_W_pad = torch.cat([Pi_W, torch.full((W, 1), NEG_INF, device=Pi_W.device, dtype=Pi_W.dtype)], dim=1)
    Pi_s1 = Pi_W_pad[:, sp_child1.long()]
    Pi_s2 = Pi_W_pad[:, sp_child2.long()]

    # --- DTS_L: 6 terms ---
    DTS_L = torch.stack([
        DL_w + Pi_W,
        Pi_W + Ebar_w,
        Pibar_W + E_w,
        SL1_w + Pi_s1,
        SL2_w + Pi_s2,
        leaf_wt,
    ], dim=0)

    DTS_L_term = logsumexp2(DTS_L, dim=0)

    if dts_r is not None:
        return logaddexp2(dts_r, DTS_L_term)
    else:
        return DTS_L_term


# ---------------------------------------------------------------------------
# Differentiable cross-clade DTS
# ---------------------------------------------------------------------------

def _dts_cross_differentiable(
    Pi, Pibar, meta, sp_child1, sp_child2, log_pD, log_pS, S, device, dtype,
):
    """Differentiable DTS cross-clade computation for one wave.

    Same as _compute_dts_cross but uses pure PyTorch ops (no Triton)
    so that torch.func.vjp can trace through it.

    Args:
        Pi, Pibar: [C, S] full tensors (Pi requires_grad for children)
        meta: wave metadata dict
        sp_child1, sp_child2: [S] species child indices
        log_pD, log_pS: scalar or [S] event probabilities
        S: int

    Returns:
        dts_r: [W, S] reduced DTS cross-clade terms
    """
    sl = meta['sl']
    sr = meta['sr']
    wlsp = meta['log_split_probs']
    W = meta['W']
    n_ws = sl.shape[0]

    Pi_l = Pi[sl]
    Pi_r = Pi[sr]
    Pibar_l = Pibar[sl]
    Pibar_r = Pibar[sr]

    Pi_pad = torch.cat([Pi, torch.full((Pi.shape[0], 1), NEG_INF, device=device, dtype=dtype)], dim=1)
    Pi_l_s1 = Pi_pad[sl][:, sp_child1.long()]
    Pi_l_s2 = Pi_pad[sl][:, sp_child2.long()]
    Pi_r_s1 = Pi_pad[sr][:, sp_child1.long()]
    Pi_r_s2 = Pi_pad[sr][:, sp_child2.long()]

    DTS = torch.stack([
        log_pD + Pi_l + Pi_r,
        Pi_l + Pibar_r,
        Pi_r + Pibar_l,
        log_pS + Pi_l_s1 + Pi_r_s2,
        log_pS + Pi_r_s1 + Pi_l_s2,
    ], dim=0)

    dts_term = wlsp + logsumexp2(DTS, dim=0)

    reduce_idx = meta['reduce_idx']
    reduce_expand = reduce_idx.unsqueeze(1).expand(n_ws, S)

    seg_max = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    seg_max.scatter_reduce_(0, reduce_expand, dts_term.detach(), reduce='amax',
                            include_self=True)

    # Avoid -inf - (-inf) -> NaN when a whole segment/species slice is unreachable.
    # In that case seg_max is -inf and all corresponding dts_term are -inf, so using
    # a finite shift (0) yields exp2(-inf)=0 and the reduced result remains -inf.
    seg_max_safe = torch.where(seg_max == NEG_INF, torch.zeros_like(seg_max), seg_max)
    shifted = torch.exp2(dts_term - seg_max_safe[reduce_idx])
    seg_sum = torch.zeros((W, S), device=device, dtype=dtype)
    seg_sum.scatter_add_(0, reduce_expand, shifted)
    dts_r = _safe_log2(seg_sum) + seg_max

    return dts_r


# ---------------------------------------------------------------------------
# VJP precompute
# ---------------------------------------------------------------------------

def _self_loop_vjp_precompute(
    Pi_star, Pibar_star, dts_r,
    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
    sp_child1, sp_child2, leaf_wt, S,
    pibar_mode, transfer_mat_T, ancestors_T,
):
    """Precompute softmax weights and Pibar VJP ingredients for one wave.

    Evaluates the self-loop g(Pi) at Pi=Pi_star and caches all quantities
    needed by _self_loop_Jt_apply (Neumann VJP) and param VJP.
    Called ONCE per wave.
    """
    W = Pi_star.shape[0]
    device, dtype = Pi_star.device, Pi_star.dtype

    def _expand(t):
        return t.unsqueeze(0).expand(W, -1) if t.ndim == 1 else t

    mt = _expand(mt_w)
    DL = _expand(DL_w)
    Ebar = _expand(Ebar_w)
    E = _expand(E_w)
    SL1 = _expand(SL1_w)
    SL2 = _expand(SL2_w)

    Pi_max = Pi_star.max(dim=1, keepdim=True).values
    p_prime = torch.exp2(Pi_star - Pi_max)

    if pibar_mode == 'uniform':
        row_sum = p_prime.sum(dim=1, keepdim=True)
        anc_sum = p_prime @ ancestors_T
        pibar_denom = row_sum - anc_sum
    else:
        pibar_matmul = p_prime @ transfer_mat_T

    Pi_pad = torch.cat([Pi_star, torch.full((W, 1), NEG_INF, device=device, dtype=dtype)], dim=1)
    Pi_s1 = Pi_pad[:, sp_child1.long()]
    Pi_s2 = Pi_pad[:, sp_child2.long()]

    terms = torch.stack([
        DL + Pi_star,
        Pi_star + Ebar,
        Pibar_star + E,
        SL1 + Pi_s1,
        SL2 + Pi_s2,
        leaf_wt,
    ], dim=0)

    DTS_L = logsumexp2(terms, dim=0)

    if dts_r is not None:
        Pi_new = logaddexp2(dts_r, DTS_L)
        w_L = _safe_exp2_ratio(DTS_L, Pi_new)
    else:
        w_L = torch.ones(W, S, device=device, dtype=dtype)

    w_terms = _safe_exp2_ratio(terms, DTS_L.unsqueeze(0))

    sc1 = sp_child1.long()
    sc2 = sp_child2.long()
    valid1 = sc1 < S
    valid2 = sc2 < S

    result = {
        'w_L': w_L,
        'w_terms': w_terms,
        'p_prime': p_prime,
    }
    if pibar_mode == 'uniform':
        pos = pibar_denom > 0
        inv_denom = torch.where(pos, 1.0 / torch.where(pos, pibar_denom, torch.ones_like(pibar_denom)),
                                torch.zeros_like(pibar_denom))
        result['pibar_inv_denom'] = inv_denom
    else:
        pos = pibar_matmul > 0
        inv_matmul = torch.where(pos, 1.0 / torch.where(pos, pibar_matmul, torch.ones_like(pibar_matmul)),
                                 torch.zeros_like(pibar_matmul))
        result['pibar_inv_matmul'] = inv_matmul
        result['pibar_matmul'] = pibar_matmul
    if valid1.any():
        result['sc1_valid'] = valid1
        result['sc1_idx'] = sc1[valid1].unsqueeze(0)
    if valid2.any():
        result['sc2_valid'] = valid2
        result['sc2_idx'] = sc2[valid2].unsqueeze(0)
    return result


# ---------------------------------------------------------------------------
# GMRES solver for (I - J^T) v = rhs
# ---------------------------------------------------------------------------

def _gmres_self_loop_solve(
    rhs, ingredients, sp_child1, sp_child2, S, W,
    pibar_mode, transfer_mat_T, ancestors_T,
    max_iters=30, tol=1e-5,
):
    """Solve (I - J_self^T) v = rhs via GMRES.

    Used when spectral radius of J_self^T is close to 1 (e.g., pibar_mode='uniform'),
    making the Neumann series diverge. Returns v [W, S].
    """
    n = W * S
    b = rhs.reshape(n)
    beta = b.norm()
    if beta < 1e-30:
        return rhs.clone()

    V = [b / beta]
    H = torch.zeros(max_iters + 1, max_iters, device=rhs.device, dtype=rhs.dtype)

    cs = torch.zeros(max_iters, device=rhs.device, dtype=rhs.dtype)
    sn = torch.zeros(max_iters, device=rhs.device, dtype=rhs.dtype)
    g = torch.zeros(max_iters + 1, device=rhs.device, dtype=rhs.dtype)
    g[0] = beta

    converged_j = 0
    for j in range(max_iters):
        vj_2d = V[j].reshape(W, S)
        Jt_vj = _self_loop_Jt_apply(
            vj_2d, ingredients, sp_child1, sp_child2, S, W,
            pibar_mode, transfer_mat_T, ancestors_T,
        )
        w = (vj_2d - Jt_vj).reshape(n)

        for i in range(j + 1):
            H[i, j] = w.dot(V[i])
            w = w - H[i, j] * V[i]
        H[j + 1, j] = w.norm()

        if H[j + 1, j] > 1e-14:
            V.append(w / H[j + 1, j])
        else:
            V.append(torch.zeros_like(w))

        for i in range(j):
            temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
            H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
            H[i, j] = temp

        denom = (H[j, j] ** 2 + H[j + 1, j] ** 2).sqrt()
        if denom > 1e-14:
            cs[j] = H[j, j] / denom
            sn[j] = H[j + 1, j] / denom
        else:
            cs[j] = 1.0
            sn[j] = 0.0

        H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
        H[j + 1, j] = 0.0
        temp = cs[j] * g[j] + sn[j] * g[j + 1]
        g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1]
        g[j] = temp

        converged_j = j + 1
        if abs(float(g[j + 1])) / float(beta) < tol:
            break

    m = converged_j
    y = torch.zeros(m, device=rhs.device, dtype=rhs.dtype)
    for i in range(m - 1, -1, -1):
        y[i] = (g[i] - H[i, i + 1:m] @ y[i + 1:m]) / H[i, i] if H[i, i].abs() > 1e-14 else 0.0

    v = torch.zeros(n, device=rhs.device, dtype=rhs.dtype)
    for i in range(m):
        v = v + float(y[i]) * V[i]

    return v.reshape(W, S)


# ---------------------------------------------------------------------------
# Analytical J^T application
# ---------------------------------------------------------------------------

def _self_loop_Jt_apply(
    v, ingredients, sp_child1, sp_child2, S, W,
    pibar_mode, transfer_mat_T, ancestors_T,
):
    """Apply J_self^T @ v analytically using precomputed ingredients.

    This is the VJP of one self-loop step g(Pi) = logaddexp2(dts_r, DTS_L(Pi)).
    The Jacobian J = dg/dPi is block-diagonal per clade (no cross-clade coupling
    in the self-loop). Each block captures:
      - diagonal: d(DTS_L)/d(Pi) through DL+Pi and Pi+Ebar terms
      - Pibar path: d(DTS_L)/d(Pibar) * d(Pibar)/d(Pi) through Pibar+E term
      - speciation: d(DTS_L)/d(Pi_s1) * d(Pi_s1)/d(Pi) scatter through SL terms
    """
    w_L = ingredients['w_L']
    w_terms = ingredients['w_terms']
    p_prime = ingredients['p_prime']

    alpha = v * w_L

    result = alpha * (w_terms[0] + w_terms[1])

    v_Pibar = alpha * w_terms[2]

    if pibar_mode == 'uniform':
        u_d = v_Pibar * ingredients['pibar_inv_denom']
        A = u_d.sum(dim=1, keepdim=True)
        correction = (ancestors_T @ u_d.T).T
        result = result + p_prime * (A - correction)
    else:  # dense / topk
        u_mr = v_Pibar * ingredients['pibar_inv_matmul']
        result = result + p_prime * (u_mr @ transfer_mat_T.T)

    sc1_valid = ingredients.get('sc1_valid')
    sc2_valid = ingredients.get('sc2_valid')
    sc1_idx = ingredients.get('sc1_idx')
    sc2_idx = ingredients.get('sc2_idx')

    if sc1_valid is not None:
        src = alpha * w_terms[3]
        idx = sc1_idx.expand(W, -1) if sc1_idx.shape[0] == 1 else sc1_idx
        result.scatter_add_(1, idx, src[:, sc1_valid])
    if sc2_valid is not None:
        src = alpha * w_terms[4]
        idx = sc2_idx.expand(W, -1) if sc2_idx.shape[0] == 1 else sc2_idx
        result.scatter_add_(1, idx, src[:, sc2_valid])

    return result


# ---------------------------------------------------------------------------
# Pi wave backward
# ---------------------------------------------------------------------------

@torch.no_grad()
def Pi_wave_backward(
    wave_layout,
    Pi_star_wave,
    Pibar_star_wave,
    E, Ebar, E_s1, E_s2,
    log_pS, log_pD, log_pL,
    max_transfer_mat,
    species_helpers,
    root_clade_ids_perm,
    device, dtype,
    *,
    neumann_terms=3,
    pruning_threshold=1e-6,
    use_pruning=True,
    pibar_mode='uniform',
    transfer_mat=None,
    ancestors_T=None,
    family_idx=None,
    uniform_pibar_row_max=None,
    leaf_obs_log=None,
    leaf_species_mask=None,
):
    """Wave-decomposed backward pass for implicit gradient computation.

    Computes dL/dPi via Neumann series per wave (root→leaves), then
    accumulates parameter gradients.  Always operates in batched mode
    internally; a single gene tree (family_idx=None) is handled as G=1.

    Args:
        wave_layout: dict from build_wave_layout()
        Pi_star_wave: [C, S] converged Pi in wave-ordered space
        Pibar_star_wave: [C, S] converged Pibar in wave-ordered space
        E, Ebar, E_s1, E_s2: [S] or [G, S] species extinction
        log_pS, log_pD, log_pL: scalar/[S] or [G]/[G, S] event probabilities
        max_transfer_mat: [S] or [G, S] log2-space
        species_helpers: species tree helpers
        root_clade_ids_perm: Long[F] root clade IDs in wave-ordered space
        device, dtype: target device/dtype
        neumann_terms: number of Neumann series terms (default 3)
        pruning_threshold: linear-space adjoint magnitude threshold for pruning
        use_pruning: whether to prune waves with negligible adjoint gradient
        pibar_mode: 'dense', 'uniform', or 'topk'
        transfer_mat: [S, S] linear-space transfer matrix (for dense mode)
        ancestors_T: [S, S] sparse CSR = ancestors.T (for uniform mode)
        family_idx: Long[C] clade→family mapping. None → auto-wrapped as G=1.
        uniform_pibar_row_max: optional [C] final forward-side row max values
            for uniform Pibar. Used only by opt-in fused cross-Pibar VJP paths.

    Returns:
        dict with:
            'v_Pi': [C, S] adjoint vector for Pi (wave-ordered)
            'grad_E': [S] or [G, S] gradient contribution from Pi adjoint to E
            'grad_log_pS': [S] or [G, S] gradient wrt log_pS
            'grad_log_pD': [S] or [G, S] gradient wrt log_pD
            'grad_max_transfer_mat': [S] or [G, S] gradient wrt max_transfer_mat
            'grad_transfer_mat': [S, S] gradient wrt transfer_mat (dense mode only)
    """
    # Fused Triton backward kernels (optional)
    try:
        from .kernels.wave_backward import (
            wave_backward_uniform_fused,
            dts_cross_backward_fused,
            dts_cross_backward_accum_fused,
            dts_cross_backward_accum_grouped_fused,
            uniform_cross_pibar_vjp_fused,
            uniform_cross_pibar_vjp_tree_fused,
            uniform_cross_pibar_vjp_tree_from_ud_fused,
            uniform_cross_pibar_vjp_tree_prefix_fused,
            uniform_cross_pibar_vjp_tree_grouped_fused,
            uniform_cross_pibar_vjp_grouped_tree_fused,
            pibar_row_stats_fused,
            active_mask_from_rhs_absmax_fused,
        )
        _HAS_FUSED_BACKWARD = True
    except ImportError:
        _HAS_FUSED_BACKWARD = False

    wave_metas = wave_layout['wave_metas']
    C, S = Pi_star_wave.shape
    K = len(wave_metas)

    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers['s_C12_indexes']
    p_cpu = sp_P_idx.cpu().long()
    c_cpu = sp_c12_idx.cpu().long()
    mask_c1 = p_cpu < S
    sp_child1_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child2_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child1_cpu[p_cpu[mask_c1]] = c_cpu[mask_c1]
    sp_child2_cpu[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1]
    sp_child1 = sp_child1_cpu.to(device)
    sp_child2 = sp_child2_cpu.to(device)

    fused_cross_pibar_vjp_enabled = (
        os.environ.get("GPUREC_FUSED_CROSS_PIBAR_VJP", "1") != "0"
        and _HAS_FUSED_BACKWARD
        and pibar_mode == 'uniform'
        and dtype in (torch.float32, torch.float64)
        and device.type == 'cuda'
    )
    fused_cross_pibar_vjp_impl = os.environ.get(
        "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL", "tree"
    ).lower()
    prefix_cross_pibar_vjp_impl = fused_cross_pibar_vjp_impl in ("prefix", "tree_prefix")
    tree_cross_pibar_vjp_impl = fused_cross_pibar_vjp_impl in (
        "tree",
        "prefix",
        "tree_prefix",
    )
    grouped_cross_pibar_vjp_enabled = (
        os.environ.get("GPUREC_GROUPED_CROSS_PIBAR_VJP", "0") != "0"
    )
    grouped_cross_pibar_reduce_impl = os.environ.get(
        "GPUREC_GROUPED_CROSS_PIBAR_REDUCE_IMPL", "torch"
    ).lower()
    grouped_cross_pibar_use_active = (
        os.environ.get("GPUREC_GROUPED_CROSS_PIBAR_USE_ACTIVE", "1") != "0"
    )
    dts_pibar_ud_fusion_enabled = (
        os.environ.get("GPUREC_DTS_PIBAR_UD_FUSION", "1") != "0"
    )
    dts_pibar_ud_min_splits = int(
        os.environ.get("GPUREC_DTS_PIBAR_UD_MIN_SPLITS", "0")
    )
    cross_pibar_row_stats_enabled = (
        os.environ.get("GPUREC_CROSS_PIBAR_ROW_STATS", "0") != "0"
    )
    kernelized_active_mask_enabled = (
        os.environ.get("GPUREC_KERNELIZED_ACTIVE_MASK", "1") != "0"
        and _HAS_FUSED_BACKWARD
        and pibar_mode == 'uniform'
        and device.type == 'cuda'
        and dtype in (torch.float32, torch.float64)
    )
    kernelized_backward_dts_enabled = (
        os.environ.get("GPUREC_KERNELIZED_BACKWARD_DTS", "1") != "0"
        and device.type == 'cuda'
    )
    fused_dts_backward_accum_enabled = (
        os.environ.get("GPUREC_FUSED_DTS_BACKWARD_ACCUM", "1") != "0"
    )
    dts_backward_accum_impl = os.environ.get(
        "GPUREC_DTS_BACKWARD_ACCUM_IMPL", "direct"
    ).lower()
    grouped_dts_backward_accum_enabled = (
        os.environ.get("GPUREC_GROUPED_DTS_BACKWARD_ACCUM", "0") != "0"
        or dts_backward_accum_impl in (
            "grouped",
            "grouped_all",
            "child_grouped",
            "child_grouped_all",
        )
    )
    noatomic_dts_backward_accum_enabled = (
        os.environ.get("GPUREC_NOATOMIC_DTS_BACKWARD_ACCUM", "0") != "0"
        or dts_backward_accum_impl in (
            "noatomic",
            "noatomic_all",
            "nonatomic",
            "nonatomic_all",
        )
    )
    merged_dts_backward_accum_enabled = (
        os.environ.get("GPUREC_MERGED_DTS_BACKWARD_ACCUM", "1") != "0"
        or dts_backward_accum_impl in (
            "merged",
            "merged_s",
            "merge_s",
            "merged_s_term",
        )
    )
    grouped_dts_backward_accum_all = dts_backward_accum_impl in (
        "grouped_all",
        "child_grouped_all",
    )
    noatomic_dts_backward_accum_all = dts_backward_accum_impl in (
        "noatomic_all",
        "nonatomic_all",
    )
    grouped_dts_backward_min_splits = int(
        os.environ.get("GPUREC_GROUPED_DTS_BACKWARD_MIN_SPLITS", "8192")
    )
    grouped_dts_backward_min_fanout = float(
        os.environ.get("GPUREC_GROUPED_DTS_BACKWARD_MIN_FANOUT", "64")
    )
    fused_uniform_backward_enabled = (
        os.environ.get("GPUREC_FUSED_UNIFORM_BACKWARD", "1") != "0"
    )
    fused_uniform_backward_view_rhs = (
        os.environ.get("GPUREC_FUSED_UNIFORM_BACKWARD_VIEW_RHS", "1") != "0"
    )
    backward_pruning_row_stats_enabled = (
        os.environ.get("GPUREC_BACKWARD_PRUNING_ROW_STATS", "0") != "0"
    )
    wave_topology_int32_enabled = (
        os.environ.get("GPUREC_WAVE_TOPOLOGY_INT32", "1") != "0"
        and device.type == 'cuda'
        and dtype in (torch.float32, torch.float64)
    )
    dts_reduction_accum_impl = os.environ.get(
        "GPUREC_DTS_BACKWARD_REDUCTION_ACCUM", "scalar"
    ).strip().lower()
    dts_reduction_accum_scalar_enabled = dts_reduction_accum_impl in (
        "1",
        "true",
        "yes",
        "on",
        "scalar",
        "scalars",
        "all",
        "full",
    )
    dts_reduction_accum_mt_enabled = dts_reduction_accum_impl in (
        "mt",
        "grad_mt",
        "all",
        "full",
    )
    dts_reduction_accum_min_splits = int(
        os.environ.get("GPUREC_DTS_BACKWARD_REDUCTION_ACCUM_MIN_SPLITS", "8192")
    )
    _compute_dts_cross_kernelized = None
    if kernelized_backward_dts_enabled:
        from .forward import _compute_dts_cross as _compute_dts_cross_kernelized

    ancestor_cols = None
    level_parents = None
    sp_parent = None
    depth_nodes = None
    if fused_cross_pibar_vjp_enabled:
        target_device = torch.device(device)
        if target_device.type == 'cuda' and target_device.index is None:
            target_device = torch.device('cuda', torch.cuda.current_device())
        cache = species_helpers.get('_wave_forward_species_cache')
        if cache is not None and int(cache.get('S', -1)) == int(S):
            cached_ancestor_cols = cache.get('ancestor_cols')
            if torch.is_tensor(cached_ancestor_cols) and cached_ancestor_cols.device == target_device:
                ancestor_cols = cached_ancestor_cols
            cached_level_parents = cache.get('level_parents')
            if torch.is_tensor(cached_level_parents) and cached_level_parents.device == target_device:
                level_parents = cached_level_parents
            cached_sp_parent = cache.get('sp_parent')
            if torch.is_tensor(cached_sp_parent) and cached_sp_parent.device == target_device:
                sp_parent = cached_sp_parent
            cached_depth_nodes = cache.get('depth_nodes')
            if torch.is_tensor(cached_depth_nodes) and cached_depth_nodes.device == target_device:
                depth_nodes = cached_depth_nodes

        if (
            ancestor_cols is None
            or (tree_cross_pibar_vjp_impl and level_parents is None)
            or (prefix_cross_pibar_vjp_impl and (sp_parent is None or depth_nodes is None))
        ):
            sp_parent_cpu = torch.full((S,), -1, dtype=torch.long)
            sp_parent_cpu[c_cpu[mask_c1]] = p_cpu[mask_c1]
            sp_parent_cpu[c_cpu[~mask_c1]] = p_cpu[~mask_c1] - S
            if sp_parent is None:
                sp_parent = sp_parent_cpu.to(target_device)

            parent_values = sp_parent_cpu.tolist()
            ancestor_lists = []
            max_ancestor_depth = 0
            for s_idx in range(S):
                cur = s_idx
                depth = 0
                ancestors = []
                while cur >= 0:
                    ancestors.append(cur)
                    depth += 1
                    if depth > S:
                        raise RuntimeError("Cycle detected in species parent pointers")
                    cur = parent_values[cur]
                ancestor_lists.append(ancestors)
                max_ancestor_depth = max(max_ancestor_depth, depth)

            ancestor_cols_cpu = torch.full((S, max_ancestor_depth), -1, dtype=torch.long)
            for s_idx, ancestors in enumerate(ancestor_lists):
                ancestor_cols_cpu[s_idx, :len(ancestors)] = torch.tensor(ancestors, dtype=torch.long)
            if ancestor_cols is None:
                ancestor_cols = ancestor_cols_cpu.T.contiguous().to(target_device)

            if tree_cross_pibar_vjp_impl and level_parents is None:
                child1_values = sp_child1_cpu.tolist()
                child2_values = sp_child2_cpu.tolist()
                levels = [-1] * S

                for s_idx in range(S):
                    if levels[s_idx] >= 0:
                        continue
                    stack = [(s_idx, False)]
                    while stack:
                        node, expanded = stack.pop()
                        if levels[node] >= 0:
                            continue
                        c1 = child1_values[node]
                        c2 = child2_values[node]
                        if not expanded:
                            stack.append((node, True))
                            if c2 < S and levels[c2] < 0:
                                stack.append((c2, False))
                            if c1 < S and levels[c1] < 0:
                                stack.append((c1, False))
                            continue
                        child_levels = []
                        if c1 < S:
                            child_levels.append(levels[c1])
                        if c2 < S:
                            child_levels.append(levels[c2])
                        levels[node] = (max(child_levels) + 1) if child_levels else 0

                max_level = max(levels) if levels else 0
                level_lists = []
                max_level_width = 1
                for level in range(1, max_level + 1):
                    parents = [
                        s_idx for s_idx, node_level in enumerate(levels)
                        if node_level == level
                        and (child1_values[s_idx] < S or child2_values[s_idx] < S)
                    ]
                    if parents:
                        level_lists.append(parents)
                        max_level_width = max(max_level_width, len(parents))

                level_parents_cpu = torch.full(
                    (max(len(level_lists), 1), max_level_width),
                    -1,
                    dtype=torch.long,
                )
                for level, parents in enumerate(level_lists):
                    level_parents_cpu[level, :len(parents)] = torch.tensor(parents, dtype=torch.long)
                level_parents = level_parents_cpu.contiguous().to(target_device)

            if prefix_cross_pibar_vjp_impl and depth_nodes is None:
                depths = [-1] * S

                def _species_depth(s_idx):
                    cached_depth = depths[s_idx]
                    if cached_depth >= 0:
                        return cached_depth
                    parent = parent_values[s_idx]
                    depth = 0 if parent < 0 else _species_depth(parent) + 1
                    depths[s_idx] = depth
                    return depth

                for s_idx in range(S):
                    _species_depth(s_idx)
                max_depth = max(depths) if depths else 0
                depth_lists = [
                    [s_idx for s_idx, depth in enumerate(depths) if depth == level]
                    for level in range(max_depth + 1)
                ]
                max_depth_width = max((len(nodes) for nodes in depth_lists), default=1)
                depth_nodes_cpu = torch.full(
                    (max(len(depth_lists), 1), max_depth_width),
                    -1,
                    dtype=torch.long,
                )
                for level, nodes in enumerate(depth_lists):
                    if nodes:
                        depth_nodes_cpu[level, :len(nodes)] = torch.tensor(nodes, dtype=torch.long)
                depth_nodes = depth_nodes_cpu.contiguous().to(target_device)

            if cache is not None and int(cache.get('S', -1)) == int(S):
                if level_parents is not None:
                    cache['level_parents'] = level_parents
                if sp_parent is not None:
                    cache['sp_parent'] = sp_parent
                if depth_nodes is not None:
                    cache['depth_nodes'] = depth_nodes

    sp_parent_wave = (
        sp_parent.to(dtype=torch.int32).contiguous()
        if wave_topology_int32_enabled and torch.is_tensor(sp_parent)
        else sp_parent
    )

    # Auto-wrap single-family inputs into batched format (G=1).
    _auto_wrapped = family_idx is None
    if _auto_wrapped:
        family_idx = torch.zeros(C, dtype=torch.long, device=device)
        E = E.unsqueeze(0)
        Ebar = Ebar.unsqueeze(0)
        E_s1 = E_s1.unsqueeze(0)
        E_s2 = E_s2.unsqueeze(0)
        log_pS = log_pS.unsqueeze(0)
        log_pD = log_pD.unsqueeze(0)
        log_pL = log_pL.unsqueeze(0)
        max_transfer_mat = max_transfer_mat.unsqueeze(0)

    mt_squeezed = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 2 else max_transfer_mat

    transfer_mat_T = None
    if pibar_mode in ('dense', 'topk') and transfer_mat is not None:
        transfer_mat_T = transfer_mat.T.contiguous()

    G = log_pD.shape[0]

    def _to_clade(p):
        r = p[family_idx]
        return r.unsqueeze(-1) if r.ndim == 1 else r

    can_use_fused_uniform_backward = (
        fused_uniform_backward_enabled
        and _HAS_FUSED_BACKWARD
        and G == 1
        and pibar_mode == 'uniform'
        and dtype in (torch.float32, torch.float64)
        and device.type == 'cuda'
        and S > 256
    )

    # Shared/global mode has one parameter/E row for every clade.  Keeping the
    # constants as [S] avoids materializing several [C, S] copies before the
    # wave loop, which is both slow and the main source of backward OOMs.
    if _auto_wrapped:
        mt_shared = mt_squeezed[0]
        E_shared = E[0]
        Ebar_shared = Ebar[0]
        E_s1_shared = E_s1[0]
        E_s2_shared = E_s2[0]
        log_pD_shared = log_pD[0]
        log_pS_shared = log_pS[0]
        DL_shared = 1.0 + log_pD_shared + E_shared
        SL1_shared = log_pS_shared + E_s2_shared
        SL2_shared = log_pS_shared + E_s1_shared
        mt_clade = E_clade = Ebar_clade = log_pD_clade = log_pS_clade = None
        DL_const = SL1_const = SL2_const = None
    else:
        mt_shared = E_shared = Ebar_shared = E_s1_shared = E_s2_shared = None
        log_pD_shared = log_pS_shared = None
        DL_shared = SL1_shared = SL2_shared = None
        mt_clade = _to_clade(mt_squeezed)
        E_clade = _to_clade(E)
        Ebar_clade = _to_clade(Ebar)
        log_pD_clade = _to_clade(log_pD)
        log_pS_clade = _to_clade(log_pS)
        DL_const = 1.0 + log_pD_clade + E_clade
        SL1_const = log_pS_clade + _to_clade(E_s2)
        SL2_const = log_pS_clade + _to_clade(E_s1)

    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']
    leaf_species_index = wave_layout.get('leaf_species_index')

    # Fraction-missing leaf boundary (mirror of Pi_wave_forward): off-leaf
    # species-leaf columns carry log2(1 - p_obs_l) instead of -inf. The leaf
    # term is constant in Pi, so this only reconstructs the correct forward
    # weights; the in-kernel/dense-index sigma paths can't represent it, so we
    # fall back to the explicit [W,S] leaf_term path when it is active.
    fraction_missing_active = leaf_obs_log is not None and leaf_species_mask is not None
    leaf_baseline_row = None
    if fraction_missing_active:
        leaf_baseline_row = torch.full((S,), NEG_INF, device=device, dtype=dtype)
        _lm_fm = leaf_species_mask.to(device)
        leaf_baseline_row[_lm_fm] = leaf_obs_log.to(device=device, dtype=dtype)[_lm_fm]

    use_uniform_leaf_index = bool(
        os.environ.get("GPUREC_BACKWARD_LEAF_INDEX", "1") != "0"
        and _auto_wrapped
        and pibar_mode == 'uniform'
        and device.type == 'cuda'
        and leaf_species_index is not None
        and not fraction_missing_active
    )
    uniform_leaf_logp = None
    if use_uniform_leaf_index:
        use_scalar_leaf_logp = (
            os.environ.get("GPUREC_SCALAR_LEAF_LOGP", "0") != "0"
            and log_pS_shared.ndim == 0
        )
        uniform_leaf_logp = (
            log_pS_shared.contiguous()
            if use_scalar_leaf_logp
            else (
                log_pS_shared.expand(S).contiguous()
                if log_pS_shared.ndim == 0
                else log_pS_shared.contiguous()
            )
        )
    fused_wave_param_accum_enabled = (
        os.environ.get("GPUREC_FUSED_WAVE_PARAM_ACCUM", "1") != "0"
    )
    no_cpu_pruning = (
        os.environ.get("GPUREC_BACKWARD_NO_CPU_PRUNING", "0") != "0"
    )
    device_pruning_requested = (
        os.environ.get("GPUREC_DEVICE_PRUNING", "0") != "0"
    )

    if wave_topology_int32_enabled:
        sp_child1_wave = sp_child1_cpu.to(device=device, dtype=torch.int32)
        sp_child2_wave = sp_child2_cpu.to(device=device, dtype=torch.int32)
        leaf_species_index_wave = (
            leaf_species_index.to(device=device, dtype=torch.int32).contiguous()
            if torch.is_tensor(leaf_species_index)
            else leaf_species_index
        )
    else:
        sp_child1_wave = sp_child1
        sp_child2_wave = sp_child2
        leaf_species_index_wave = leaf_species_index
    dense_leaf_mask_from_index = (
        os.environ.get("GPUREC_DENSE_LEAF_MASK_FROM_INDEX", "0") != "0"
        and leaf_species_index is not None
        and not (can_use_fused_uniform_backward and use_uniform_leaf_index)
        and not fraction_missing_active
    )
    leaf_species_lanes = (
        torch.arange(S, device=device)
        if dense_leaf_mask_from_index else None
    )
    leaf_zero = (
        torch.tensor(0.0, device=device, dtype=dtype)
        if dense_leaf_mask_from_index else None
    )
    leaf_neg_inf = (
        torch.tensor(NEG_INF, device=device, dtype=dtype)
        if dense_leaf_mask_from_index else None
    )

    def _get_leaf_mask(ws, we):
        W = we - ws
        if dense_leaf_mask_from_index:
            return torch.where(
                leaf_species_index[ws:we].unsqueeze(1) == leaf_species_lanes.unsqueeze(0),
                leaf_zero,
                leaf_neg_inf,
            )

        if leaf_baseline_row is not None:
            lwt = leaf_baseline_row.unsqueeze(0).expand(W, S).clone()
        else:
            lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
        mask = (leaf_row_index >= ws) & (leaf_row_index < we)
        if mask.any():
            lwt[leaf_row_index[mask] - ws, leaf_col_index[mask]] = 0.0
        return lwt

    def _get_leaf_wt(ws, we):
        leaf_mask = _get_leaf_mask(ws, we)
        if _auto_wrapped:
            return log_pS_shared + leaf_mask
        return log_pS_clade[ws:we] + leaf_mask

    n_waves_total = K
    n_waves_skipped = 0
    n_clades_total = C
    n_clades_skipped = 0
    device_pruning_clades_total = 0
    device_pruning_waves_total = 0
    device_pruning_active_counts = []
    device_pruning_wave_active_flags = []

    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    for r in root_clade_ids_perm:
        r = int(r)
        root_Pi = Pi_star_wave[r]
        lse = logsumexp2(root_Pi, dim=0)
        accumulated_rhs[r] = -_safe_exp2_ratio(root_Pi, lse)

    grad_log_pD = torch.zeros_like(log_pD)
    grad_log_pS = torch.zeros_like(log_pS)
    grad_mt = torch.zeros_like(mt_squeezed)
    grad_E_acc = torch.zeros_like(E)
    grad_Ebar_acc = torch.zeros_like(Ebar)
    grad_transfer_mat_acc = torch.zeros(S, S, device=device, dtype=dtype) if pibar_mode in ('dense', 'topk') else None
    grad_E_s1_acc = torch.zeros_like(E_s1)
    grad_E_s2_acc = torch.zeros_like(E_s2)
    cross_pibar_row_stats = None
    forward_pibar_row_max = None
    reuse_forward_pibar_stats_enabled = (
        os.environ.get("GPUREC_REUSE_FORWARD_PIBAR_STATS", "0") != "0"
        and uniform_pibar_row_max is not None
        and torch.is_tensor(uniform_pibar_row_max)
        and uniform_pibar_row_max.numel() == C
        and fused_cross_pibar_vjp_enabled
        and pibar_mode == 'uniform'
        and _auto_wrapped
        and mt_shared is not None
        and mt_shared.ndim == 1
        and dtype in (torch.float32, torch.float64)
        and device.type == 'cuda'
        and S > 256
    )
    if reuse_forward_pibar_stats_enabled:
        forward_pibar_row_max = uniform_pibar_row_max.contiguous()
    elif (
        dts_pibar_ud_fusion_enabled
        and uniform_pibar_row_max is not None
        and torch.is_tensor(uniform_pibar_row_max)
        and uniform_pibar_row_max.numel() == C
        and pibar_mode == 'uniform'
        and device.type == 'cuda'
        and dtype in (torch.float32, torch.float64)
    ):
        forward_pibar_row_max = uniform_pibar_row_max.contiguous()
    if (
        cross_pibar_row_stats_enabled
        and not reuse_forward_pibar_stats_enabled
        and fused_cross_pibar_vjp_enabled
        and pibar_mode == 'uniform'
        and dtype in (torch.float32, torch.float64)
        and device.type == 'cuda'
        and S > 256
    ):
        cross_pibar_row_stats = pibar_row_stats_fused(Pi_star_wave)

    def _compute_active_mask(rhs):
        if kernelized_active_mask_enabled:
            threshold = pruning_threshold if use_pruning else 0.0
            return active_mask_from_rhs_absmax_fused(
                rhs, threshold, use_pruning=use_pruning
            )
        clade_max = rhs.abs().max(dim=1).values
        if use_pruning:
            return clade_max >= pruning_threshold
        return clade_max > 0

    for k in range(K - 1, -1, -1):
        meta = wave_metas[k]
        ws = meta['start']
        we = meta['end']
        W = meta['W']

        use_fused = can_use_fused_uniform_backward
        rhs_slice = accumulated_rhs[ws:we]
        # The fused uniform kernel treats rhs as read-only, and this wave's
        # later cross-DTS/Pibar adjoints accumulate into child rows.
        rhs_k = (
            rhs_slice
            if (use_fused and fused_uniform_backward_view_rhs)
            else rhs_slice.clone()
        )
        no_cpu_pruning_wave = no_cpu_pruning and use_fused
        device_pruning_wave = device_pruning_requested and use_fused
        kernel_pruning_wave = no_cpu_pruning_wave or device_pruning_wave
        active_mask = None
        active_mask_for_kernels = None

        if kernel_pruning_wave:
            if use_pruning or device_pruning_wave:
                active_mask = _compute_active_mask(rhs_k)
            if active_mask is not None:
                active_mask = active_mask.contiguous()
                active_mask_for_kernels = active_mask
            if device_pruning_wave:
                device_pruning_clades_total += W
                device_pruning_waves_total += 1
                device_pruning_active_counts.append(active_mask.sum())
                device_pruning_wave_active_flags.append(active_mask.any())
            n_active = W
            use_compact = False
            active_idx = None
            rhs_active = rhs_k
        else:
            active_mask = _compute_active_mask(rhs_k)

            if not active_mask.any():
                n_waves_skipped += 1
                n_clades_skipped += W
                continue

            if use_fused:
                # The fused Triton path does not consume compact row indices.
                # Keep optional row statistics out of the production path
                # because sum().item() synchronizes the wave loop.
                if backward_pruning_row_stats_enabled:
                    n_active = int(active_mask.sum().item())
                    n_clades_skipped += (W - n_active)
                else:
                    n_active = W
            else:
                n_active = int(active_mask.sum().item())
                n_clades_skipped += (W - n_active)

        Pi_W_star = Pi_star_wave[ws:we].detach()

        leaf_wt = None if (use_fused and use_uniform_leaf_index) else _get_leaf_wt(ws, we)

        if meta['has_splits']:
            reduce_idx = meta['reduce_idx']
            if _auto_wrapped:
                log_pD_dts = log_pD_shared
                log_pS_dts = log_pS_shared
            else:
                log_pD_dts = log_pD_clade[ws + reduce_idx]
                log_pS_dts = log_pS_clade[ws + reduce_idx]
            with torch.no_grad():
                if _compute_dts_cross_kernelized is not None:
                    dts_r = _compute_dts_cross_kernelized(
                        Pi_star_wave.detach(), Pibar_star_wave.detach(), meta,
                        sp_child1, sp_child2, log_pD_dts, log_pS_dts, S, device, dtype,
                        active_mask=active_mask_for_kernels,
                    )
                else:
                    dts_r = _dts_cross_differentiable(
                        Pi_star_wave.detach(), Pibar_star_wave.detach(), meta,
                        sp_child1, sp_child2, log_pD_dts, log_pS_dts, S, device, dtype,
                    )
        else:
            dts_r = None

        if _auto_wrapped:
            mt_w = mt_shared
            DL_w = DL_shared
            E_w = E_shared
            Ebar_w = Ebar_shared
            SL1_w = SL1_shared
            SL2_w = SL2_shared
        else:
            mt_w = mt_clade[ws:we]
            DL_w = DL_const[ws:we]
            E_w = E_clade[ws:we]
            Ebar_w = Ebar_clade[ws:we]
            SL1_w = SL1_const[ws:we]
            SL2_w = SL2_const[ws:we]

        if not kernel_pruning_wave and not use_fused:
            use_compact = (n_active < W)
            if use_compact:
                active_idx = active_mask.nonzero(as_tuple=True)[0]
                rhs_active = rhs_k[active_idx]
            else:
                active_idx = None
                rhs_active = rhs_k
        elif not kernel_pruning_wave:
            use_compact = False
            active_idx = None
            rhs_active = rhs_k

        # Per-wave family indices for scatter accumulation.
        fi_w = family_idx[ws:we]
        fi_expand = fi_w.unsqueeze(1).expand(W, S)

        def _scatter_accum(acc, contrib):
            if G == 1:
                if acc.ndim == 1:
                    acc[0] += contrib.sum()
                else:
                    acc[0] += contrib.sum(dim=0)
                return
            if acc.ndim == 1:
                acc.scatter_add_(0, fi_w, contrib.sum(dim=1))
            else:
                acc.scatter_add_(0, fi_expand, contrib)

        if use_fused:
            accum_param_grads = None
            if fused_wave_param_accum_enabled and _auto_wrapped:
                accum_param_grads = (
                    grad_log_pD,
                    grad_log_pS,
                    grad_E_acc[0],
                    grad_Ebar_acc[0],
                    grad_E_s1_acc[0],
                    grad_E_s2_acc[0],
                    grad_mt[0],
                )
            # G=1: extract shared [S] constants for the fused kernel.
            v_k, aw0, aw1, aw2, aw345, aw3, aw4 = wave_backward_uniform_fused(
                Pi_star_wave, Pibar_star_wave, ws, W, S,
                dts_r, rhs_k,
                mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                sp_child1_wave, sp_child2_wave, leaf_wt,
                neumann_terms=neumann_terms,
                leaf_species_idx=leaf_species_index_wave if use_uniform_leaf_index else None,
                leaf_logp=uniform_leaf_logp if use_uniform_leaf_index else None,
                accum_param_grads=accum_param_grads,
                active_mask=active_mask_for_kernels,
                sp_parent=sp_parent_wave,
                pibar_row_max=forward_pibar_row_max,
            )

            if accum_param_grads is None:
                _scatter_accum(grad_log_pD, aw0)
                _scatter_accum(grad_log_pS, aw345)
                _scatter_accum(grad_E_acc, aw0 + aw2)
                _scatter_accum(grad_Ebar_acc, aw1)
                _scatter_accum(grad_E_s1_acc, aw4)
                _scatter_accum(grad_E_s2_acc, aw3)
                _scatter_accum(grad_mt, aw2)

        else:
            Pibar_W_star = Pibar_star_wave[ws:we]
            ingredients = _self_loop_vjp_precompute(
                Pi_W_star, Pibar_W_star, dts_r,
                mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                sp_child1, sp_child2, leaf_wt, S,
                pibar_mode, transfer_mat_T, ancestors_T,
            )

            if use_compact:
                compact_ing = {
                    'w_L': ingredients['w_L'][active_idx],
                    'w_terms': ingredients['w_terms'][:, active_idx],
                    'p_prime': ingredients['p_prime'][active_idx],
                }
                if 'pibar_inv_denom' in ingredients:
                    compact_ing['pibar_inv_denom'] = ingredients['pibar_inv_denom'][active_idx]
                if 'pibar_inv_matmul' in ingredients:
                    compact_ing['pibar_inv_matmul'] = ingredients['pibar_inv_matmul'][active_idx]
                if 'pibar_matmul' in ingredients:
                    compact_ing['pibar_matmul'] = ingredients['pibar_matmul'][active_idx]
                for key in ('sc1_valid', 'sc1_idx', 'sc2_valid', 'sc2_idx'):
                    if key in ingredients:
                        compact_ing[key] = ingredients[key]
                solve_ing = compact_ing
                solve_W = n_active
            else:
                solve_ing = ingredients
                solve_W = W

            if pibar_mode == 'uniform':
                v_k = _gmres_self_loop_solve(
                    rhs_active, solve_ing, sp_child1, sp_child2, S, solve_W,
                    pibar_mode, transfer_mat_T, ancestors_T,
                    max_iters=5, tol=1e-8,
                )
            else:
                v_k = rhs_active.clone()
                term = rhs_active
                for _n in range(neumann_terms):
                    term = _self_loop_Jt_apply(
                        term, solve_ing, sp_child1, sp_child2, S, solve_W,
                        pibar_mode, transfer_mat_T, ancestors_T,
                    )
                    v_k = v_k + term

            if use_compact:
                v_k_full = torch.zeros(W, S, device=device, dtype=dtype)
                v_k_full[active_idx] = v_k
                v_k = v_k_full

            alpha_full = v_k * ingredients['w_L']
            wt = ingredients['w_terms']

            aw0 = alpha_full * wt[0]
            aw1 = alpha_full * wt[1]
            aw2 = alpha_full * wt[2]
            aw3 = alpha_full * wt[3]
            aw4 = alpha_full * wt[4]
            aw5 = alpha_full * wt[5]

            _scatter_accum(grad_log_pD, aw0)
            _scatter_accum(grad_log_pS, aw3 + aw4 + aw5)
            _scatter_accum(grad_E_acc, aw0 + aw2)
            _scatter_accum(grad_Ebar_acc, aw1)
            _scatter_accum(grad_E_s1_acc, aw4)
            _scatter_accum(grad_E_s2_acc, aw3)
            _scatter_accum(grad_mt, aw2)

            if pibar_mode in ('dense', 'topk') and grad_transfer_mat_acc is not None:
                v_Pibar_full = alpha_full * wt[2]
                matmul_r = ingredients['pibar_matmul']
                mr_safe = torch.where(matmul_r > 0, matmul_r, torch.ones_like(matmul_r))
                u_mr = torch.where(matmul_r > 0, v_Pibar_full / mr_safe, torch.zeros_like(v_Pibar_full))
                grad_transfer_mat_acc = grad_transfer_mat_acc + u_mr.T @ ingredients['p_prime']

        if meta['has_splits'] and dts_r is not None:
            sl = meta['sl']
            sr = meta['sr']
            wlsp = meta['log_split_probs']
            reduce_idx = meta['reduce_idx']
            n_ws = sl.shape[0]

            # Fused kernel currently supports only scalar log_pD/log_pS (shared-param case).
            # For specieswise/genewise tensors, use the generic path below.
            fused_scalar_params = (log_pD.numel() == 1 and log_pS.numel() == 1)
            used_fused_pibar_vjp = False
            used_fused_direct_pi_accum = False
            used_dts_mt_reduction_accum = False
            used_dts_pibar_ud_fusion = False

            if use_fused and fused_scalar_params:
                # G=1: pass shared params to fused kernel.
                if fused_dts_backward_accum_enabled:
                    dts_accum_threshold_match = (
                        n_ws >= grouped_dts_backward_min_splits
                        and (n_ws / max(W, 1)) >= grouped_dts_backward_min_fanout
                    )
                    use_noatomic_dts_accum = (
                        noatomic_dts_backward_accum_enabled
                        and (noatomic_dts_backward_accum_all or dts_accum_threshold_match)
                    )
                    use_grouped_dts_accum = (
                        grouped_dts_backward_accum_enabled
                        and (grouped_dts_backward_accum_all or dts_accum_threshold_match)
                    )
                    group_children = group_inverse = None
                    if use_noatomic_dts_accum or use_grouped_dts_accum:
                        group_children = meta.get('_dts_accum_group_children')
                        group_inverse = meta.get('_dts_accum_group_inverse')
                        if (
                            not torch.is_tensor(group_children)
                            or not torch.is_tensor(group_inverse)
                            or group_children.device != sl.device
                            or group_inverse.device != sl.device
                        ):
                            all_children = torch.cat((sl, sr), dim=0)
                            group_children, group_inverse = torch.unique(
                                all_children,
                                sorted=True,
                                return_inverse=True,
                            )
                            group_children = group_children.contiguous()
                            group_inverse = group_inverse.contiguous()
                            meta['_dts_accum_group_children'] = group_children
                            meta['_dts_accum_group_inverse'] = group_inverse
                    child_rows_unique = (
                        torch.is_tensor(group_children)
                        and int(group_children.shape[0]) == int(2 * n_ws)
                    )
                    if use_noatomic_dts_accum and child_rows_unique:
                        (grad_Pibar_l, grad_Pibar_r,
                         param_pD, param_pS) = dts_cross_backward_accum_fused(
                            Pi_star_wave, Pibar_star_wave, v_k, ws,
                            sl, sr, reduce_idx, wlsp,
                            log_pD, log_pS,
                            sp_child1, sp_child2, accumulated_rhs, S,
                            active_mask=active_mask_for_kernels,
                            use_atomics=False,
                        )
                    elif use_grouped_dts_accum:
                        (grad_Pibar_l, grad_Pibar_r,
                         param_pD, param_pS) = dts_cross_backward_accum_grouped_fused(
                            Pi_star_wave, Pibar_star_wave, v_k, ws,
                            sl, sr, reduce_idx, wlsp,
                            log_pD, log_pS,
                            sp_child1, sp_child2, accumulated_rhs, S,
                            active_mask=active_mask_for_kernels,
                            group_children=group_children,
                            group_inverse=group_inverse,
                        )
                    else:
                        reduction_threshold_match = (
                            n_ws >= dts_reduction_accum_min_splits
                        )
                        pibar_ud_fusion_match = (
                            dts_pibar_ud_fusion_enabled
                            and fused_cross_pibar_vjp_enabled
                            and fused_cross_pibar_vjp_impl == "tree"
                            and level_parents is not None
                            and forward_pibar_row_max is not None
                            and mt_shared is not None
                            and mt_shared.ndim == 1
                            and n_ws >= dts_pibar_ud_min_splits
                        )
                        use_dts_reduction_accum_scalar = (
                            dts_reduction_accum_scalar_enabled
                            and reduction_threshold_match
                        )
                        use_dts_reduction_accum_mt = (
                            (
                                dts_reduction_accum_mt_enabled
                                and reduction_threshold_match
                            )
                            or pibar_ud_fusion_match
                        )
                        used_dts_mt_reduction_accum = use_dts_reduction_accum_mt
                        used_dts_pibar_ud_fusion = pibar_ud_fusion_match
                        (grad_Pibar_l, grad_Pibar_r,
                         param_pD, param_pS) = dts_cross_backward_accum_fused(
                            Pi_star_wave, Pibar_star_wave, v_k, ws,
                            sl, sr, reduce_idx, wlsp,
                            log_pD, log_pS,
                            sp_child1, sp_child2, accumulated_rhs, S,
                            active_mask=active_mask_for_kernels,
                            merge_s_term=merged_dts_backward_accum_enabled,
                            grad_log_pD=grad_log_pD,
                            grad_log_pS=grad_log_pS,
                            grad_mt=(
                                grad_mt
                                if grad_mt.ndim == 1
                                else grad_mt[0]
                            ),
                            accum_param_reductions=use_dts_reduction_accum_scalar,
                            accum_mt_reduction=use_dts_reduction_accum_mt,
                            output_pibar_ud=pibar_ud_fusion_match,
                            mt_squeezed=mt_shared,
                            pibar_row_max=forward_pibar_row_max,
                        )
                    used_fused_direct_pi_accum = True
                    grad_Pi_l = grad_Pi_r = None
                else:
                    (grad_Pi_l, grad_Pi_r, grad_Pibar_l, grad_Pibar_r,
                     param_pD, param_pS) = dts_cross_backward_fused(
                        Pi_star_wave, Pibar_star_wave, v_k, ws,
                        sl, sr, reduce_idx, wlsp,
                        log_pD, log_pS,
                        sp_child1, sp_child2, S,
                        active_mask=active_mask_for_kernels,
                    )

                # Accumulate into G=1 row. The direct DTS accumulation path can
                # optionally do these reductions in-kernel.
                if not (
                    used_fused_direct_pi_accum
                    and param_pD is None
                    and param_pS is None
                ):
                    grad_log_pD[0] += param_pD.sum()
                    grad_log_pS[0] += param_pS.sum()
                if not (
                    used_fused_direct_pi_accum
                    and used_dts_mt_reduction_accum
                ):
                    mt_contrib = grad_Pibar_l.sum(dim=0) + grad_Pibar_r.sum(dim=0)
                    if grad_mt.ndim == 1:
                        grad_mt[0] += mt_contrib.sum()
                    else:
                        grad_mt[0] += mt_contrib

            else:
                Pi_l = Pi_star_wave[sl]
                Pi_r = Pi_star_wave[sr]
                Pibar_l = Pibar_star_wave[sl]
                Pibar_r = Pibar_star_wave[sr]
                neg_inf_col = torch.full((Pi_star_wave.shape[0], 1), NEG_INF, device=device, dtype=dtype)
                Pi_col_pad = torch.cat([Pi_star_wave, neg_inf_col], dim=1)
                Pi_l_s1 = Pi_col_pad[sl][:, sp_child1.long()]
                Pi_l_s2 = Pi_col_pad[sl][:, sp_child2.long()]
                Pi_r_s1 = Pi_col_pad[sr][:, sp_child1.long()]
                Pi_r_s2 = Pi_col_pad[sr][:, sp_child2.long()]

                fi_splits = family_idx[ws + reduce_idx]
                _pD_s = log_pD[fi_splits]
                if _pD_s.ndim == 1:
                    _pD_s = _pD_s.unsqueeze(-1)
                _pS_s = log_pS[fi_splits]
                if _pS_s.ndim == 1:
                    _pS_s = _pS_s.unsqueeze(-1)

                DTS_5 = torch.stack([
                    _pD_s + Pi_l + Pi_r,
                    Pi_l + Pibar_r,
                    Pi_r + Pibar_l,
                    _pS_s + Pi_l_s1 + Pi_r_s2,
                    _pS_s + Pi_r_s1 + Pi_l_s2,
                ], dim=0)

                Pi_parent = Pi_W_star[reduce_idx]
                combined = wlsp + DTS_5
                v_k_parent = v_k[reduce_idx]
                grad_DTS_5 = v_k_parent.unsqueeze(0) * _safe_exp2_ratio(
                    combined, Pi_parent.unsqueeze(0))

                fi_split_expand = fi_splits.unsqueeze(1).expand(n_ws, S)
                if grad_log_pD.ndim == 1:
                    grad_log_pD.scatter_add_(0, fi_splits, grad_DTS_5[0].sum(dim=1))
                    grad_log_pS.scatter_add_(0, fi_splits, (grad_DTS_5[3] + grad_DTS_5[4]).sum(dim=1))
                else:
                    grad_log_pD.scatter_add_(0, fi_split_expand, grad_DTS_5[0])
                    grad_log_pS.scatter_add_(0, fi_split_expand, grad_DTS_5[3] + grad_DTS_5[4])
                child_ids_dts = torch.cat([sl, sr])
                fi_ch = family_idx[child_ids_dts]
                fi_ch_expand = fi_ch.unsqueeze(1).expand(2 * n_ws, S)
                grad_mt.scatter_add_(0, fi_ch_expand,
                                     torch.cat([grad_DTS_5[2], grad_DTS_5[1]], dim=0))

                if pibar_mode in ('dense', 'topk') and grad_transfer_mat_acc is not None:
                    v_Pibar_ch = torch.cat([grad_DTS_5[2], grad_DTS_5[1]], dim=0)
                    child_ids = torch.cat([sl, sr])
                    Pi_ch = Pi_star_wave[child_ids]
                    Pi_max_ch = Pi_ch.max(dim=1, keepdim=True).values
                    p_prime_ch = torch.exp2(Pi_ch - Pi_max_ch)
                    matmul_ch = p_prime_ch @ transfer_mat_T
                    mc_safe = torch.where(matmul_ch > 0, matmul_ch, torch.ones_like(matmul_ch))
                    u_mc = torch.where(matmul_ch > 0, v_Pibar_ch / mc_safe, torch.zeros_like(v_Pibar_ch))
                    grad_transfer_mat_acc = grad_transfer_mat_acc + u_mc.T @ p_prime_ch

                grad_Pi_l = grad_DTS_5[0] + grad_DTS_5[1]
                grad_Pi_r = grad_DTS_5[0] + grad_DTS_5[2]
                grad_Pibar_l = grad_DTS_5[2]
                grad_Pibar_r = grad_DTS_5[1]

                sc1 = sp_child1.long()
                sc2 = sp_child2.long()
                valid1 = sc1 < S
                valid2 = sc2 < S
                if valid1.any():
                    idx1 = sc1[valid1]
                    grad_Pi_l.scatter_add_(1, idx1.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[3][:, valid1])
                    grad_Pi_r.scatter_add_(1, idx1.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[4][:, valid1])
                if valid2.any():
                    idx2 = sc2[valid2]
                    grad_Pi_r.scatter_add_(1, idx2.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[3][:, valid2])
                    grad_Pi_l.scatter_add_(1, idx2.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[4][:, valid2])

            if not used_fused_direct_pi_accum:
                accumulated_rhs.index_add_(0, sl, grad_Pi_l)
                accumulated_rhs.index_add_(0, sr, grad_Pi_r)

            if (
                use_fused
                and fused_scalar_params
                and fused_cross_pibar_vjp_enabled
                and ancestor_cols is not None
            ):
                if used_dts_pibar_ud_fusion:
                    uniform_cross_pibar_vjp_tree_from_ud_fused(
                        Pi_star_wave,
                        grad_Pibar_l,
                        grad_Pibar_r,
                        sl,
                        sr,
                        sp_child1,
                        sp_child2,
                        level_parents,
                        accumulated_rhs,
                        S,
                        active_mask=active_mask_for_kernels,
                        reduce_idx=reduce_idx,
                        pibar_row_max=forward_pibar_row_max,
                    )
                elif (
                    grouped_cross_pibar_vjp_enabled
                    and fused_cross_pibar_vjp_impl == "tree"
                    and level_parents is not None
                    and active_mask_for_kernels is None
                ):
                    if grouped_cross_pibar_reduce_impl in ("triton", "kernel"):
                        group_children = meta.get('_cross_pibar_group_children')
                        group_inverse = meta.get('_cross_pibar_group_inverse')
                        if (
                            not torch.is_tensor(group_children)
                            or not torch.is_tensor(group_inverse)
                            or group_children.device != sl.device
                            or group_inverse.device != sl.device
                        ):
                            all_children = torch.cat((sl, sr), dim=0)
                            group_children, group_inverse = torch.unique(
                                all_children,
                                sorted=True,
                                return_inverse=True,
                            )
                            group_children = group_children.contiguous()
                            group_inverse = group_inverse.contiguous()
                            meta['_cross_pibar_group_children'] = group_children
                            meta['_cross_pibar_group_inverse'] = group_inverse
                        uniform_cross_pibar_vjp_tree_grouped_fused(
                            Pi_star_wave,
                            grad_Pibar_l,
                            grad_Pibar_r,
                            group_children,
                            group_inverse,
                            ancestor_cols,
                            sp_child1,
                            sp_child2,
                            level_parents,
                            accumulated_rhs,
                            S,
                            active_mask=(
                                active_mask
                                if grouped_cross_pibar_use_active
                                else active_mask_for_kernels
                            ),
                            reduce_idx=reduce_idx,
                        )
                    else:
                        all_children = torch.cat([sl, sr])
                        all_pibar_grad = torch.cat([grad_Pibar_l, grad_Pibar_r])
                        unique_children, inverse = torch.unique(
                            all_children, sorted=True, return_inverse=True
                        )
                        grouped_pibar_grad = torch.zeros(
                            (unique_children.shape[0], S),
                            device=device,
                            dtype=dtype,
                        )
                        grouped_pibar_grad.index_add_(0, inverse, all_pibar_grad)
                        uniform_cross_pibar_vjp_grouped_tree_fused(
                            Pi_star_wave,
                            unique_children,
                            grouped_pibar_grad,
                            ancestor_cols,
                            sp_child1,
                            sp_child2,
                            level_parents,
                            accumulated_rhs,
                            S,
                            row_stats=cross_pibar_row_stats,
                        )
                elif (
                    prefix_cross_pibar_vjp_impl
                    and level_parents is not None
                    and depth_nodes is not None
                    and sp_parent is not None
                ):
                    uniform_cross_pibar_vjp_tree_prefix_fused(
                        Pi_star_wave,
                        grad_Pibar_l,
                        grad_Pibar_r,
                        sl,
                        sr,
                        sp_parent,
                        sp_child1,
                        sp_child2,
                        depth_nodes,
                        level_parents,
                        accumulated_rhs,
                        S,
                        active_mask=active_mask_for_kernels,
                        reduce_idx=reduce_idx,
                        row_stats=cross_pibar_row_stats,
                    )
                elif fused_cross_pibar_vjp_impl == "tree" and level_parents is not None:
                    uniform_cross_pibar_vjp_tree_fused(
                        Pi_star_wave,
                        grad_Pibar_l,
                        grad_Pibar_r,
                        sl,
                        sr,
                        ancestor_cols,
                        sp_child1,
                        sp_child2,
                        level_parents,
                        accumulated_rhs,
                        S,
                        active_mask=active_mask_for_kernels,
                        reduce_idx=reduce_idx,
                        row_stats=cross_pibar_row_stats,
                        Pibar_star=Pibar_star_wave,
                        mt_squeezed=mt_shared,
                        pibar_row_max=forward_pibar_row_max,
                    )
                else:
                    uniform_cross_pibar_vjp_fused(
                        Pi_star_wave,
                        grad_Pibar_l,
                        grad_Pibar_r,
                        sl,
                        sr,
                        ancestor_cols,
                        accumulated_rhs,
                        S,
                        active_mask=active_mask_for_kernels,
                        reduce_idx=reduce_idx,
                        row_stats=cross_pibar_row_stats,
                    )
                used_fused_pibar_vjp = True

            if not used_fused_pibar_vjp:
                all_children = torch.cat([sl, sr])
                all_pibar_grad = torch.cat([grad_Pibar_l, grad_Pibar_r])

                nz = all_pibar_grad.abs().sum(dim=1) > 0
                if nz.any():
                    nz_children = all_children[nz]
                    u = all_pibar_grad[nz]
                    Pi_ch = Pi_star_wave[nz_children]
                    Pi_max_p = Pi_ch.max(dim=1, keepdim=True).values
                    p_prime = torch.exp2(Pi_ch - Pi_max_p)

                    if pibar_mode == 'uniform':
                        anc_sum = p_prime @ ancestors_T
                        denom = p_prime.sum(dim=1, keepdim=True) - anc_sum
                        denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))
                        u_d = torch.where(denom > 0, u / denom_safe, torch.zeros_like(u))
                        A = u_d.sum(dim=1, keepdim=True)
                        correction = (ancestors_T @ u_d.T).T
                        pi_from_pibar = p_prime * (A - correction)
                    else:
                        matmul_r = p_prime @ transfer_mat_T
                        mr_safe = torch.where(matmul_r > 0, matmul_r, torch.ones_like(matmul_r))
                        u_mr = torch.where(matmul_r > 0, u / mr_safe, torch.zeros_like(u))
                        pi_from_pibar = p_prime * (u_mr @ transfer_mat_T.T)

                    accumulated_rhs.index_add_(0, nz_children, pi_from_pibar)

    if device_pruning_active_counts:
        n_device_active = int(torch.stack(device_pruning_active_counts).sum().item())
        n_device_waves_active = int(torch.stack(device_pruning_wave_active_flags).sum().item())
        n_clades_skipped += device_pruning_clades_total - n_device_active
        n_waves_skipped += device_pruning_waves_total - n_device_waves_active

    result = {
        'v_Pi': accumulated_rhs,
        'grad_E': grad_E_acc,
        'grad_Ebar': grad_Ebar_acc,
        'grad_E_s1': grad_E_s1_acc,
        'grad_E_s2': grad_E_s2_acc,
        'grad_log_pD': grad_log_pD,
        'grad_log_pS': grad_log_pS,
        'grad_max_transfer_mat': grad_mt,
        'n_waves_total': n_waves_total,
        'n_waves_skipped': n_waves_skipped,
        'n_waves_processed': n_waves_total - n_waves_skipped,
        'n_clades_total': n_clades_total,
        'n_clades_skipped': n_clades_skipped,
        'n_clades_active': n_clades_total - n_clades_skipped,
    }
    if grad_transfer_mat_acc is not None:
        result['grad_transfer_mat'] = grad_transfer_mat_acc

    # Unwrap G=1 results back to original shapes.
    if _auto_wrapped:
        for key in ('grad_E', 'grad_Ebar', 'grad_E_s1', 'grad_E_s2',
                     'grad_log_pD', 'grad_log_pS', 'grad_max_transfer_mat'):
            result[key] = result[key][0]

    return result
