"""Implicit gradient: build VJP closures & solve the two transpose systems."""
from __future__ import annotations

from typing import Optional

import time

import torch
from torch import Tensor
from torch import func as tfunc

from gpurec.core.likelihood import E_step
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.log2_utils import _safe_log2_internal as _safe_log2, logsumexp2 as _logsumexp2
from gpurec.core.extract_parameters import extract_parameters, extract_parameters_uniform

from .linear_solvers import _cg, _gmres


class _PiLeafUnset:
    """Sentinel: caller did NOT pass ``leaf_obs_log`` to the implicit-grad VJP.

    Lets us distinguish "default the Pi backward boundary to ``leaf_E``" (legacy
    single-tensor / supplement behavior, byte-identical to before) from an explicit
    ``leaf_obs_log=None`` that turns the Pi backward leaf boundary OFF
    (AleRax-faithful "e-only"), independent of the E adjoint's ``leaf_E``.
    """


_PI_LEAF_UNSET = _PiLeafUnset()


@torch.no_grad()
def implicit_grad_loglik_vjp_wave(
    wave_layout,
    species_helpers,
    *,
    Pi_star_wave: torch.Tensor,
    Pibar_star_wave: torch.Tensor,
    E_star: torch.Tensor,
    E_s1: torch.Tensor,
    E_s2: torch.Tensor,
    Ebar: torch.Tensor,
    log_pS: torch.Tensor,
    log_pD: torch.Tensor,
    log_pL: torch.Tensor,
    max_transfer_mat: torch.Tensor,
    root_clade_ids_perm: torch.Tensor,
    theta: torch.Tensor,
    unnorm_row_max: torch.Tensor,
    specieswise: bool,
    device: torch.device,
    dtype: torch.dtype,
    neumann_terms: int = 3,
    use_pruning: bool = True,
    pruning_threshold: float = 1e-6,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    gmres_restart: int = 40,
    pibar_mode: str = 'uniform',
    transfer_mat: Optional[torch.Tensor] = None,
    transfer_mat_unnormalized: Optional[torch.Tensor] = None,
    ancestors_T: Optional[torch.Tensor] = None,
    uniform_pibar_row_max: Optional[torch.Tensor] = None,
    leaf_E: Optional[torch.Tensor] = None,
    leaf_obs_log=_PI_LEAF_UNSET,
    log_pO: Optional[torch.Tensor] = None,
):
    """Compute ∇θ logL using wave-decomposed backward pass + E adjoint.

    Steps:
    1. Pi backward: wave-by-wave Neumann series (root→leaves)
    2. E adjoint: solve (I - G_E^T) w = q via CG/GMRES
    3. θ gradient: VJP through extract_parameters

    ``leaf_E`` is the fraction-missing boundary for the E (extinction) adjoint
    ONLY: it is threaded into the ``E_step`` calls in ``_e_adjoint_and_theta_vjp``.
    ``leaf_obs_log`` is the fraction-missing boundary for the Pi (reconciliation)
    backward leaf boundary ONLY: it is forwarded to ``Pi_wave_backward``. They are
    decoupled so an AleRax-faithful run can keep fraction-missing in E while
    dropping the supplement's extra Pi baseline.

    Backward-compat: if ``leaf_obs_log`` is left at its sentinel default (caller did
    not pass it), it defaults to ``leaf_E`` so existing callers that pass only
    ``leaf_E`` keep the previous "both" behavior, byte-identical to before. An
    explicit ``leaf_obs_log=None`` turns the Pi backward boundary off.

    ``log_pO`` (optional [S]) is the per-branch ORIGINATION log-probability
    ``log2(p^O_e)``. Origination enters compute_log_likelihood's numerator (the
    root reconciliation sum) and denominator (the survival term), so it changes
    the THETA-gradient SEEDs only: the Pi root seed (origination-weighted softmax,
    forwarded to ``Pi_wave_backward``) and the E denominator seed
    (origination-weighted survival, forwarded to ``_e_adjoint_and_theta_vjp``). It
    does NOT enter the E or Pi fixed-point maps. None (default) -> uniform
    p^O_e = 1/S, byte-identical to before.

    Returns (grad_theta, pi_backward_info).
    """
    # Backward-compat: unset leaf_obs_log -> mirror leaf_E (legacy "both" behavior).
    if isinstance(leaf_obs_log, _PiLeafUnset):
        leaf_obs_log = leaf_E
    # --- Step 1: Pi backward (can be pre-computed for batched mode) ---
    torch.cuda.synchronize()
    _t_pi_bwd_0 = time.perf_counter()
    pi_bwd = Pi_wave_backward(
        wave_layout=wave_layout,
        Pi_star_wave=Pi_star_wave,
        Pibar_star_wave=Pibar_star_wave,
        E=E_star, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        max_transfer_mat=max_transfer_mat,
        species_helpers=species_helpers,
        root_clade_ids_perm=root_clade_ids_perm,
        device=device, dtype=dtype,
        neumann_terms=neumann_terms,
        use_pruning=use_pruning,
        pruning_threshold=pruning_threshold,
        pibar_mode=pibar_mode,
        transfer_mat=transfer_mat,
        ancestors_T=ancestors_T,
        uniform_pibar_row_max=uniform_pibar_row_max,
        leaf_obs_log=leaf_obs_log,
        log_pO=log_pO,
    )
    torch.cuda.synchronize()
    _t_pi_bwd = time.perf_counter() - _t_pi_bwd_0

    grad_theta, statsG = _e_adjoint_and_theta_vjp(
        pi_bwd, E_star, Ebar, E_s1, E_s2,
        log_pS, log_pD, log_pL,
        max_transfer_mat, species_helpers, root_clade_ids_perm,
        theta, unnorm_row_max, specieswise,
        device, dtype,
        cg_tol=cg_tol, cg_maxiter=cg_maxiter, gmres_restart=gmres_restart,
        pibar_mode=pibar_mode, transfer_mat=transfer_mat,
        transfer_mat_unnormalized=transfer_mat_unnormalized,
        ancestors_T=ancestors_T,
        leaf_E=leaf_E,
        log_pO=log_pO,
    )
    statsG.pi_bwd_time = _t_pi_bwd
    return grad_theta, statsG


def _e_adjoint_and_theta_vjp(
    pi_bwd,
    E_star, Ebar, E_s1, E_s2,
    log_pS, log_pD, log_pL,
    max_transfer_mat, species_helpers, root_clade_ids_perm,
    theta, unnorm_row_max, specieswise,
    device, dtype,
    *,
    genewise=False,
    cg_tol=1e-8, cg_maxiter=500, gmres_restart=40,
    pibar_mode='uniform',
    transfer_mat=None, transfer_mat_unnormalized=None, ancestors_T=None,
    leaf_E=None,
    log_pO=None,
):
    """E adjoint solve + theta VJP from pre-computed Pi backward result.

    ``leaf_E`` (optional [S]) is the fraction-missing leaf extinction boundary
    ``log2(1 - p_obs_l)`` for the E (extinction) side ONLY. It MUST be threaded into
    every ``E_step`` call below so the (I - G_E^T) operator and the theta->E VJP use
    the same fixed-point map as the forward E solve; otherwise the E adjoint
    gradient is inconsistent. The Pi (reconciliation) leaf boundary
    (``leaf_obs_log``) is decoupled and handled by the caller
    (``implicit_grad_loglik_vjp_wave`` -> ``Pi_wave_backward``); it is intentionally
    NOT used here, since this function performs no Pi backward pass.

    Takes pi_bwd dict (from Pi_wave_backward) and completes the gradient
    computation through E adjoint solve and extract_parameters VJP.
    """
    # --- Step 2: E adjoint ---
    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers['s_C12_indexes']

    # Direct dNLL/dE from likelihood denominator.
    #
    # NLL = -(numerator - denominator), so the denominator term enters NLL with a
    # +sign. The direct dNLL/dE we seed here is +∂(denominator)/∂E summed over the
    # families that share this E. With ORIGINATION, the denominator is the
    # origination-weighted survival term (matching compute_log_likelihood):
    #   denom = logsumexp2(log2(1 - exp2(E)) + log_pO)
    # whereas the uniform path keeps the byte-identical mean form
    #   denom = log2(1 - mean(exp2(E))).
    # The two agree exactly for log_pO = -log2(S); we keep the original expression
    # in the None branch so the uniform gradient is byte-identical to before.
    n_fam = root_clade_ids_perm.numel()
    E_req_d = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        if log_pO is None:
            denom = torch.log2(1.0 - torch.exp2(E_req_d).mean(dim=-1))
        else:
            # Origination-weighted survival; _safe_log2 guards E -> 0.
            log_one_minus_E = _safe_log2(1.0 - torch.exp2(E_req_d))
            denom = _logsumexp2(log_one_minus_E + log_pO, dim=-1)
        # Shared-E mode: denominator contributes once per family => n_fam * denom.
        # Genewise mode: E is per-family [G, S], so each row contributes once.
        if E_req_d.ndim > 1 and E_req_d.shape[0] == n_fam:
            direct_obj = denom.sum()
        else:
            direct_obj = (n_fam * denom).sum() if denom.ndim > 0 else (n_fam * denom)
        direct_dNLL_dE = torch.autograd.grad(direct_obj, E_req_d)[0]
    q_E = pi_bwd['grad_E'].clone() + direct_dNLL_dE

    # Chain Ebar gradient through E_step's Ebar computation
    if pi_bwd['grad_Ebar'].abs().max() > 0:
        E_req2 = E_star.detach().requires_grad_(True)
        with torch.enable_grad():
            mt_sq = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 1 else max_transfer_mat
            if pibar_mode in ('dense', 'topk') and transfer_mat is not None:
                # Ebar[i] = log2(sum_j(transfer_mat[i,j] * exp2(E[j]))) + max_E + mt[i]
                max_E = E_req2.max(dim=-1, keepdim=True).values
                expE = torch.exp2(E_req2 - max_E)
                Ebar_recomp = _safe_log2(torch.einsum("ij,j->i", transfer_mat, expE)) + max_E.squeeze(-1) + mt_sq
            else:
                # uniform: Ebar[s] = log2(sum(exp2(E)) - ancestor_sum) + max_E + mt[s]
                max_E = E_req2.max(dim=-1, keepdim=True).values
                expE = torch.exp2(E_req2 - max_E)
                if expE.ndim == 1:
                    expE_2d = expE.unsqueeze(0)
                    row_sum = expE_2d.sum(dim=-1, keepdim=True)
                    ancestor_sum = (expE_2d @ ancestors_T).contiguous()
                    Ebar_recomp = _safe_log2((row_sum - ancestor_sum).squeeze(0)) + max_E.squeeze(-1) + mt_sq
                else:
                    row_sum = expE.sum(dim=-1, keepdim=True)
                    ancestor_sum = (expE @ ancestors_T).contiguous()
                    Ebar_recomp = _safe_log2(row_sum - ancestor_sum) + max_E + mt_sq
            ebar_to_e = torch.autograd.grad(
                Ebar_recomp, E_req2,
                grad_outputs=pi_bwd['grad_Ebar'],
                retain_graph=False,
            )[0]
        q_E = q_E + ebar_to_e

    # Chain E_s1, E_s2 gradients through gather_E_children
    if pi_bwd['grad_E_s1'].abs().max() > 0 or pi_bwd['grad_E_s2'].abs().max() > 0:
        E_req3 = E_star.detach().requires_grad_(True)
        with torch.enable_grad():
            from gpurec.core.terms import gather_E_children
            E_s12 = gather_E_children(E_req3, sp_P_idx, sp_c12_idx)
            E_s1_r, E_s2_r = torch.chunk(E_s12, 2, dim=-1)
            E_s1_r = E_s1_r.view(E_req3.shape)
            E_s2_r = E_s2_r.view(E_req3.shape)
            total = (E_s1_r * pi_bwd['grad_E_s1']).sum() + (E_s2_r * pi_bwd['grad_E_s2']).sum()
            es_to_e = torch.autograd.grad(total, E_req3, retain_graph=False)[0]
        q_E = q_E + es_to_e

    # Solve (I - G_E^T) w = q_E via CG/GMRES
    torch.cuda.synchronize()
    _t_qE = time.perf_counter()

    def G_E_fun(E_in):
        """E_step as a function of E only."""
        return E_step(
            E_in, sp_P_idx, sp_c12_idx,
            log_pS, log_pD, log_pL,
            transfer_mat, max_transfer_mat, pibar_mode=pibar_mode,
            ancestors_T=ancestors_T,
            leaf_E=leaf_E,
        )[0]

    # Build VJP for G_E
    E_req_g = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        _, vjpG = tfunc.vjp(G_E_fun, E_req_g)

    E_shape = E_star.shape
    q_flat = q_E.reshape(-1)

    def AG_flat(w_flat):
        wE = w_flat.view(E_shape).contiguous()
        gE, = vjpG(wE.clone())
        return (wE - gE).reshape(-1)

    w_flat, statsG, okG = _cg(AG_flat, q_flat, tol=cg_tol, maxiter=cg_maxiter)
    if not okG:
        w_flat, statsG = _gmres(AG_flat, q_flat, tol=cg_tol, restart=gmres_restart, maxiter=cg_maxiter)
        statsG.fallback_used = True

    torch.cuda.synchronize()
    _t_cg = time.perf_counter() - _t_qE

    wE = w_flat.view(E_shape)

    # --- Step 3: theta gradient through extract_parameters ---
    grad_mt_total = pi_bwd['grad_max_transfer_mat'] + pi_bwd['grad_Ebar']

    theta_req = theta.detach().requires_grad_(True)
    with torch.enable_grad():
        if pibar_mode in ('dense', 'topk') and transfer_mat_unnormalized is not None:
            log_pS_r, log_pD_r, log_pL_r, transfer_mat_r, mt_r_raw = extract_parameters(
                theta_req, transfer_mat_unnormalized,
                genewise=genewise, specieswise=specieswise, pairwise=False,
            )
            mt_r = mt_r_raw.squeeze(-1) if mt_r_raw.ndim == 2 else mt_r_raw
            param_loss = (
                (log_pS_r * pi_bwd['grad_log_pS']).sum() +
                (log_pD_r * pi_bwd['grad_log_pD']).sum() +
                (mt_r * grad_mt_total).sum()
            )
            # Add transfer_mat gradient contribution (dense/topk)
            grad_transfer_mat = pi_bwd.get('grad_transfer_mat')
            if grad_transfer_mat is not None:
                param_loss = param_loss + (transfer_mat_r * grad_transfer_mat).sum()
        else:
            # uniform: no theta-dependent transfer_mat
            log_pS_r, log_pD_r, log_pL_r, _, mt_r = extract_parameters_uniform(
                theta_req, unnorm_row_max, specieswise=specieswise, genewise=genewise,
            )
            param_loss = (
                (log_pS_r * pi_bwd['grad_log_pS']).sum() +
                (log_pD_r * pi_bwd['grad_log_pD']).sum() +
                (mt_r * grad_mt_total).sum()
            )
        grad_theta_pi = torch.autograd.grad(param_loss, theta_req, retain_graph=False)[0]

    # E adjoint contribution to theta
    theta_req2 = theta.detach().requires_grad_(True)
    with torch.enable_grad():
        if pibar_mode in ('dense', 'topk') and transfer_mat_unnormalized is not None:
            log_pS_r2, log_pD_r2, log_pL_r2, transfer_mat_r2, mt_r2_raw = extract_parameters(
                theta_req2, transfer_mat_unnormalized,
                genewise=genewise, specieswise=specieswise, pairwise=False,
            )
            mt_r2 = mt_r2_raw.squeeze(-1) if mt_r2_raw.ndim == 2 else mt_r2_raw

            def G_E_theta(th_pS, th_pD, th_pL, th_tm, th_mt):
                return E_step(
                    E_star.detach(), sp_P_idx, sp_c12_idx,
                    th_pS, th_pD, th_pL, th_tm, th_mt, pibar_mode='dense',
                    leaf_E=leaf_E,
                )[0]

            E_from_theta = G_E_theta(log_pS_r2, log_pD_r2, log_pL_r2, transfer_mat_r2, mt_r2)
        else:
            # uniform: no theta-dependent transfer_mat
            log_pS_r2, log_pD_r2, log_pL_r2, _, mt_r2 = extract_parameters_uniform(
                theta_req2, unnorm_row_max, specieswise=specieswise, genewise=genewise,
            )

            def G_E_theta(th_pS, th_pD, th_pL, th_mt):
                return E_step(
                    E_star.detach(), sp_P_idx, sp_c12_idx,
                    th_pS, th_pD, th_pL, None, th_mt,
                    pibar_mode='uniform', ancestors_T=ancestors_T,
                    leaf_E=leaf_E,
                )[0]

            E_from_theta = G_E_theta(log_pS_r2, log_pD_r2, log_pL_r2, mt_r2)

        gtheta_E = torch.autograd.grad(
            E_from_theta, theta_req2,
            grad_outputs=wE,
            retain_graph=False,
        )[0]

    torch.cuda.synchronize()
    _t_theta_vjp = time.perf_counter() - _t_qE - _t_cg  # remainder after CG
    # Stash sub-timings on statsG for the caller
    statsG.cg_time = _t_cg
    statsG.theta_vjp_time = _t_theta_vjp

    grad_theta = (grad_theta_pi + gtheta_E).detach()
    return grad_theta, statsG


@torch.no_grad()
def implicit_grad_loglik_vjp_wave_genewise(
    families,
    species_helpers,
    *,
    E_all: Tensor,
    E_s1_all: Tensor,
    E_s2_all: Tensor,
    Ebar_all: Tensor,
    log_pS_all: Tensor,
    log_pD_all: Tensor,
    log_pL_all: Tensor,
    mt_all: Tensor,
    theta_stack: Tensor,
    unnorm_row_max: Tensor,
    specieswise: bool,
    device: torch.device,
    dtype: torch.dtype,
    pibar_mode: str = 'uniform',
    neumann_terms: int = 3,
    use_pruning: bool = True,
    pruning_threshold: float = 1e-6,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    gmres_restart: int = 40,
    local_iters: int = 2000,
    local_tolerance: float = 1e-3,
    ancestors_T=None,
    leaf_E=None,
    leaf_obs_log=None,
):
    """Genewise implicit gradient via per-family Pi forward + backward loop.

    Each family gets its own wave layout, Pi forward pass, and backward pass
    with per-gene scalar/[S] parameters. The E adjoint is solved per-family.

    Parameters
    ----------
    families : list of dict
        Per-family data, each with 'ccp_helpers', 'leaf_row_index',
        'leaf_col_index', 'root_clade_id'.
    species_helpers : dict
        Shared species tree helpers.
    E_all, E_s1_all, E_s2_all, Ebar_all : Tensor [G, S]
        Per-gene converged E quantities.
    log_pS_all, log_pD_all, log_pL_all : Tensor [G] or [G, S]
        Per-gene event probabilities.
    mt_all : Tensor [G, S]
        Per-gene max_transfer_mat.
    theta_stack : Tensor [G, 3] or [G, S, 3]
        Per-gene rate parameters.
    unnorm_row_max : Tensor [S]
        Row maxima of unnormalized transfer matrix.
    specieswise : bool
        Whether rates are per-species.
    local_iters, local_tolerance : int, float
        Pi forward convergence parameters.

    Returns
    -------
    (grad_theta_stack, stats_list) : (Tensor [G, ...], list[LinearSolveStats])
    """
    from gpurec.core.forward import Pi_wave_forward
    from gpurec.core.batching import collate_gene_families, build_wave_layout
    from gpurec.core.scheduling import compute_clade_waves

    G = len(families)
    grad_thetas = []
    all_stats = []

    for g in range(G):
        fam = families[g]

        # Slice per-gene quantities → [S]
        E_g = E_all[g]
        E_s1_g = E_s1_all[g]
        E_s2_g = E_s2_all[g]
        Ebar_g = Ebar_all[g]
        mt_g = mt_all[g]
        theta_g = theta_stack[g]
        pS_g = log_pS_all[g]
        pD_g = log_pD_all[g]
        pL_g = log_pL_all[g]

        # Build single-family batch for wave_layout
        single_item = {
            'ccp': fam['ccp_helpers'],
            'leaf_row_index': fam['leaf_row_index'],
            'leaf_col_index': fam['leaf_col_index'],
            'root_clade_id': int(fam['root_clade_id']),
        }
        single_batched = collate_gene_families([single_item], dtype=dtype, device=device)

        waves_g, phases_g = compute_clade_waves(fam['ccp_helpers'])
        wave_layout_g = build_wave_layout(
            waves=waves_g, phases=phases_g,
            ccp_helpers=single_batched['ccp'],
            leaf_row_index=single_batched['leaf_row_index'],
            leaf_col_index=single_batched['leaf_col_index'],
            root_clade_ids=single_batched['root_clade_ids'],
            device=device, dtype=dtype,
        )

        # Pi forward for this family (needed for backward)
        Pi_out_g = Pi_wave_forward(
            wave_layout=wave_layout_g, species_helpers=species_helpers,
            E=E_g, Ebar=Ebar_g, E_s1=E_s1_g, E_s2=E_s2_g,
            log_pS=pS_g, log_pD=pD_g, log_pL=pL_g,
            transfer_mat=None, max_transfer_mat=mt_g,
            device=device, dtype=dtype,
            local_iters=local_iters, local_tolerance=local_tolerance,
            pibar_mode=pibar_mode,
        )

        # Backward: per-family gradient
        grad_theta_g, statsG = implicit_grad_loglik_vjp_wave(
            wave_layout_g, species_helpers,
            Pi_star_wave=Pi_out_g['Pi_wave_ordered'],
            Pibar_star_wave=Pi_out_g['Pibar_wave_ordered'],
            E_star=E_g, E_s1=E_s1_g, E_s2=E_s2_g, Ebar=Ebar_g,
            log_pS=pS_g, log_pD=pD_g, log_pL=pL_g,
            max_transfer_mat=mt_g,
            root_clade_ids_perm=wave_layout_g['root_clade_ids'],
            theta=theta_g,
            unnorm_row_max=unnorm_row_max,
            specieswise=specieswise,
            device=device, dtype=dtype,
            neumann_terms=neumann_terms,
            use_pruning=use_pruning,
            pruning_threshold=pruning_threshold,
            cg_tol=cg_tol, cg_maxiter=cg_maxiter,
            gmres_restart=gmres_restart,
            pibar_mode=pibar_mode,
            ancestors_T=ancestors_T,
            leaf_E=leaf_E,
            leaf_obs_log=leaf_obs_log,
        )

        grad_thetas.append(grad_theta_g)
        all_stats.append(statsG)

    return torch.stack(grad_thetas, dim=0), all_stats
