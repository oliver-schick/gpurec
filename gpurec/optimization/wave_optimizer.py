"""Wave-based optimizer (uses wave forward + wave backward)."""
from __future__ import annotations

import math
import time
from typing import List

import torch

from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.extract_parameters import extract_parameters, extract_parameters_uniform

from .types import FixedPointInfo, LinearSolveStats, StepRecord
from .implicit_grad import implicit_grad_loglik_vjp_wave


def _cuda_mem_diag(device) -> str:
    """Best-effort CUDA memory diagnostics string."""
    try:
        if not torch.cuda.is_available():
            return "cuda_unavailable"
        idx = device.index if hasattr(device, "index") and device.index is not None else torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(idx)
        alloc_b = torch.cuda.memory_allocated(idx)
        reserved_b = torch.cuda.memory_reserved(idx)
        return (
            f"free={free_b / (1024**3):.2f}GiB total={total_b / (1024**3):.2f}GiB "
            f"allocated={alloc_b / (1024**3):.2f}GiB reserved={reserved_b / (1024**3):.2f}GiB"
        )
    except Exception as exc:
        return f"mem_diag_unavailable({exc})"


class _PiLeafUnset:
    """Sentinel: caller did NOT pass ``leaf_obs_log`` at all.

    Lets ``optimize_theta_wave`` distinguish two cases that both want a Python
    ``None`` at the call boundary:

    * ``leaf_obs_log`` unset (this sentinel) -> default it to ``leaf_E`` so legacy
      callers that pass only ``leaf_E`` keep the "both" (supplement) behavior,
      byte-identical to before.
    * ``leaf_obs_log=None`` passed explicitly -> Pi leaf boundary OFF (the
      AleRax-faithful "e-only" mode), independent of ``leaf_E``.
    """


_PI_LEAF_UNSET = _PiLeafUnset()


def optimize_theta_wave(
    wave_layout,
    species_helpers,
    root_clade_ids,
    unnorm_row_max,
    theta_init,
    *,
    transfer_mat_unnormalized=None,
    steps: int = 200,
    lr: float = 0.2,
    tol_theta: float = 1e-3,
    e_max_iters: int = 2000,
    e_tol: float = 1e-8,
    neumann_terms: int = 4,
    use_pruning: bool = True,
    pruning_threshold: float = 1e-6,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    gmres_restart: int = 40,
    specieswise: bool = False,
    device=None,
    dtype=torch.float64,
    pibar_mode: str = 'uniform',
    family_batch_size: int = 0,
    wave_layout_batches=None,
    families=None,
    stochastic_batches: bool = False,
    stochastic_seed: int = 0,
    optimizer: str = 'adam',
    momentum: float = 0.9,
    verbose: bool = False,
    leaf_E: torch.Tensor | None = None,
    leaf_obs_log=_PI_LEAF_UNSET,
    brownian_sigma=None,
    brownian_root_sigma: float = 5.0,
    brownian_mu=None,
    parent_index: torch.Tensor | None = None,
    theta_bounds=None,
    origination: str = 'uniform',
    omega_init: torch.Tensor | None = None,
    origination_sigma=None,
    origination_root_sigma: float = 5.0,
    origination_mu: float = 0.0,
    origination_decouple_root: bool = True,
):
    """Optimize theta using wave forward/backward + implicit gradient.

    Parameters
    ----------
    wave_layout : dict
        From build_wave_layout().
    species_helpers : dict
        Species tree helpers.
    root_clade_ids : Tensor
        Root clade IDs (original ordering) for likelihood computation.
    unnorm_row_max : Tensor [S]
        Row maxima of unnormalized transfer matrix.
    theta_init : Tensor [3] or [S, 3]
        Initial rate parameters (log-space).
    steps : int
        Maximum optimization iterations (ignored for 'lbfgs').
    lr : float
        Learning rate (ignored for 'lbfgs').
    optimizer : str
        'adam', 'sgd', or 'lbfgs'.
    family_batch_size : int
        Optional family mini-batch size for global/specieswise gradient accumulation.
        Effective when ``wave_layout_batches`` is provided by caller.
    wave_layout_batches : list[tuple[dict, Tensor]] | None
        Optional prebuilt batches of ``(wave_layout_batch, root_clade_ids_batch)``.
        When provided, forward/backward runs batch-by-batch and accumulates NLL/gradient.
    families : list[dict] | None
        Optional original family dicts. When provided with ``family_batch_size > 0``
        and ``wave_layout_batches is None``, batch wave layouts are built lazily per step.
    stochastic_batches : bool
        If True (and optimizer='sgd'), sample a random family batch each step
        instead of accumulating all family batches.
    stochastic_seed : int
        RNG seed used for stochastic family-batch sampling.
    momentum : float
        Momentum for SGD (default 0.9, ignored for adam/lbfgs).
    leaf_E : Tensor [S] | None
        Optional fraction-missing boundary, log2(1 - p_obs_l) = log2(fraction
        missing) at missing species-leaves and -inf elsewhere, applied to the E
        (extinction) solve ONLY: passed as leaf_E to E_fixed_point and to the E
        adjoint E_steps. This is the AleRax ``_fm`` term (the "S but not observed"
        extinction contribution). None (default) -> every gene observed in the E
        recursion (byte-identical to before).
    leaf_obs_log : Tensor [S] | None
        Optional fraction-missing boundary applied to the Pi (reconciliation) leaf
        boundary ONLY: passed as leaf_obs_log to Pi_wave_forward / Pi_wave_backward.
        This is the EXTRA ``(1-sigma)(1-p_obs)`` Pi baseline from the AleRax
        *supplement* (AleRaxSupp.tex L159) which the AleRax *source* does NOT apply.
        Decoupled from ``leaf_E`` so an AleRax-faithful run can keep fraction-missing
        in E while dropping it from Pi.

        Backward-compat: if ``leaf_obs_log`` is None but ``leaf_E`` is set, it
        defaults to ``leaf_E`` so callers passing only ``leaf_E`` get the previous
        "both" (supplement) behavior, byte-identical to before. To get the
        AleRax-faithful "E-only" mode, pass ``leaf_obs_log=None`` explicitly via the
        sentinel (see ``_PiLeafUnset`` below) — i.e. set ``leaf_E`` and leave the Pi
        boundary off. The driver exposes this through ``--fm-mode e-only``.
    brownian_sigma : float | Tensor [3] | None
        Std (log2 units) of the time-uniform TKP Brownian rate prior coupling
        adjacent branches' log-rates. None (default) disables the prior and keeps
        the path byte-identical to before. Scalar or per-axis [3] (D, L, T).
        Only applies in specieswise mode (theta [S,3]); ignored otherwise.
    brownian_root_sigma : float | Tensor [3]
        Std (log2 units) of the root-anchor term (propriety). Default 5.0.
    brownian_mu : float | Tensor [3] | None
        Root-anchor centre (log2 units). None (default) -> use the theta_init
        root row (i.e. log2(init_rate)) so the prior is centred at the init.
    parent_index : LongTensor [S] | None
        Parent of each species node (root maps to -1/self), from
        ``species_parent_index(species_helpers)``. Required when brownian_sigma
        is not None.
    theta_bounds : tuple | None
        Optional box constraints (log2 units) on theta for the L-BFGS-B path,
        AleRax-style. ``(lo, hi)`` scalars applied to every element, or a tuple
        of two arrays broadcastable to the flattened theta. None (default) keeps
        the current behavior (lower bound = log2(1e-10), no upper bound).
    origination : str
        ORIGINATION strategy (matching AleRax). 'uniform' (default): every species
        branch originates with equal probability p^O_e = 1/S (omega absent;
        byte-identical to before). 'optimize': a FREE per-branch origination
        log2-weight ``omega`` [S] is jointly optimised with theta. The per-branch
        origination probability is the softmax p^O_e = 2^omega_e / Σ_e 2^omega_e
        (log_pO = omega - logsumexp2(omega)), which enters compute_log_likelihood's
        numerator AND survival denominator. Only the L-BFGS path supports
        'optimize': the scipy vector becomes [theta_flat (3S or S*3); omega (S)],
        the implicit theta gradient uses the origination-weighted seeds, and the
        omega gradient is the DIRECT ∂(Σ NLL)/∂omega at fixed Pi, E.
    omega_init : Tensor [S] | None
        Initial origination log2-weights (only used when origination='optimize').
        None (default) -> omega = 0 (uniform p^O_e = 1/S).
    origination_sigma : float | None
        Std (log2 units) of a time-uniform TKP Brownian prior coupling adjacent
        branches' origination log2-weights ``omega`` along the species tree
        (the same prior used for the DTL rates, applied to omega [S,1]). None
        (default) leaves origination FREE/unregularized. Small sigma -> omega
        smoothed toward a constant -> p^O_e -> uniform (1/S), matching AleRax's
        near-constant origination column. Only used when origination='optimize'.
        The Brownian increment penalty is shift-invariant, so it respects the
        softmax gauge; the root anchor pins the otherwise-free global shift.
    origination_root_sigma : float
        Root-anchor std (log2 units) for the omega prior. Default 5.0. IGNORED
        when ``origination_decouple_root`` is True (the root is left free).
    origination_mu : float
        Root-anchor centre (log2 units) for the omega prior. Default 0.0 (the
        anchor value is a gauge choice; softmax makes any global shift a no-op).
    origination_decouple_root : bool
        If True (default), the ROOT is decoupled from the omega prior: the
        root's incident edges are cut (its children are NOT smoothed toward the
        root) and the root anchor is dropped, so the root origination weight is
        FREE — set by the data, not pulled toward the rest of the tree.
        Origination at the root (genes present in the LCA / ancestral genome) is
        qualitatively different from lineage-specific gene birth, so it should
        not be coupled to the per-lineage origination rates. Non-root branches
        are still smoothed within their clades. If False, the root participates
        in the prior like any node (coupled to its children + anchored).

    Returns
    -------
    dict with 'theta', 'rates', 'log_likelihood', 'negative_log_likelihood', 'history'.
    When origination='optimize', also 'omega' [S] (log2-weights) and 'origination'
    [S] (normalised p^O_e).
    """
    if device is None:
        device = theta_init.device

    # Resolve the Pi leaf boundary. Backward-compat: if the caller did NOT pass
    # ``leaf_obs_log`` at all, default it to ``leaf_E`` so the legacy single-tensor
    # path (fraction-missing in BOTH E and Pi, the supplement behavior) is
    # byte-identical to before. An explicit ``leaf_obs_log=None`` turns the Pi leaf
    # boundary OFF (AleRax-faithful "e-only"), independent of ``leaf_E``.
    if isinstance(leaf_obs_log, _PiLeafUnset):
        leaf_obs_log = leaf_E

    _THETA_MIN = math.log2(1e-10)

    # --- ORIGINATION setup --------------------------------------------------
    # 'uniform' (default): omega absent, log_pO=None everywhere -> byte-identical
    #   to the pre-origination path (uniform p^O_e = 1/S).
    # 'optimize': a free per-branch origination log2-weight omega [S] is jointly
    #   optimised; log_pO = omega - logsumexp2(omega) is the softmax log-prob.
    from gpurec.core.likelihood import origination_log_pO as _origination_log_pO
    if origination not in ('uniform', 'optimize'):
        raise ValueError(f"origination must be 'uniform' or 'optimize', got {origination!r}")
    _opt_origination = origination == 'optimize'
    # Determine S (number of species branches) for the omega vector.
    _S_branches = int(species_helpers['S'])
    if _opt_origination:
        if omega_init is None:
            _omega0 = torch.zeros(_S_branches, dtype=dtype, device=device)
        else:
            _omega0 = omega_init.to(device=device, dtype=dtype).reshape(-1).clone()
        if _omega0.numel() != _S_branches:
            raise ValueError(
                f"omega_init must have S={_S_branches} entries, got {_omega0.numel()}"
            )
        if optimizer != 'lbfgs':
            raise NotImplementedError(
                "origination='optimize' is only supported with optimizer='lbfgs'"
            )
    else:
        _omega0 = None

    # --- Brownian (TKP) rate prior setup -----------------------------------
    # Only meaningful for specieswise theta [S,3]; skip otherwise so the
    # global/genewise paths stay byte-identical.
    _use_prior = brownian_sigma is not None
    if _use_prior and not (specieswise and theta_init.ndim == 2 and theta_init.shape[-1] == 3):
        if verbose:
            print("  [brownian prior requested but theta is not specieswise [S,3]; "
                  "skipping prior]", flush=True)
        _use_prior = False

    # Optional Brownian prior on the per-branch origination log2-weights omega.
    # Regularizes the FREE per-branch origination (otherwise as unidentifiable
    # as free per-branch loss) by coupling adjacent branches along the species
    # tree -- the SAME prior, applied to omega reshaped [S,1].
    _use_orig_prior = _opt_origination and (origination_sigma is not None)

    _brownian_prior = None
    _prior_parent_index = None
    _prior_sigma = None
    _prior_root_sigma = None
    _prior_mu = None
    _orig_sigma = None
    _orig_root_sigma = None
    _orig_mu = None
    _orig_parent_index = None
    if _use_prior or _use_orig_prior:
        from gpurec.core.tree_prior import (
            brownian_log_prior_and_grad as _brownian_prior,
            species_parent_index as _species_parent_index,
        )
        # Shared species-tree parent index (both priors live on the same tree).
        if parent_index is None:
            _prior_parent_index = _species_parent_index(species_helpers).to(device)
        else:
            _prior_parent_index = parent_index.to(device=device, dtype=torch.long)
    if _use_prior:
        _prior_sigma = brownian_sigma
        _prior_root_sigma = brownian_root_sigma
        # Default root-anchor centre = theta_init root row (== log2(init_rate)
        # for a uniform init), so the prior is centred at the initialization.
        if brownian_mu is None:
            _ti = theta_init.to(device=device, dtype=dtype)
            arange_S = torch.arange(_ti.shape[0], device=device)
            _is_root = (_prior_parent_index < 0) | (_prior_parent_index == arange_S)
            _root_ids = torch.nonzero(_is_root, as_tuple=False).flatten()
            _prior_mu = _ti[int(_root_ids[0].item())].clone()
        else:
            _prior_mu = brownian_mu
    if _use_orig_prior:
        _orig_sigma = float(origination_sigma)
        _orig_mu = float(origination_mu)
        if origination_decouple_root:
            # Decouple the root: cut its incident edges so the root and each of
            # its child-subtrees are not smoothed across the root, and drop the
            # root anchor. The root origination weight is then FREE (data-driven),
            # not pulled toward the per-lineage origination of the rest of the
            # tree. Non-root branches stay smoothed within their clades.
            _opi = _prior_parent_index.clone()
            _arange_S = torch.arange(_opi.shape[0], device=device)
            _is_root = (_opi < 0) | (_opi == _arange_S)
            _root_id = int(torch.nonzero(_is_root, as_tuple=False).flatten()[0].item())
            _opi[_opi == _root_id] = -1          # cut root -> child edges
            _orig_parent_index = _opi
            _orig_root_sigma = float('inf')       # no anchor (root left free)
        else:
            _orig_parent_index = _prior_parent_index
            _orig_root_sigma = float(origination_root_sigma)

    def _apply_prior(theta_d, nll, grad_theta):
        """Add the Brownian penalty to nll and its gradient to grad_theta."""
        if not _use_prior:
            return nll, grad_theta
        penalty, p_grad = _brownian_prior(
            theta_d, _prior_parent_index, _prior_sigma, _prior_root_sigma, _prior_mu,
        )
        return nll + float(penalty.item()), grad_theta + p_grad

    def _apply_origination_prior(omega_d, nll, grad_omega):
        """Add the Brownian penalty on omega [S,1] to nll and grad_omega [S]."""
        if not _use_orig_prior or omega_d is None or grad_omega is None:
            return nll, grad_omega
        penalty, p_grad = _brownian_prior(
            omega_d.reshape(-1, 1), _orig_parent_index,
            _orig_sigma, _orig_root_sigma, _orig_mu,
        )
        return nll + float(penalty.item()), grad_omega + p_grad.reshape(-1)

    # Precompute ancestors_T for uniform mode
    _ancestors_T = None
    if pibar_mode == 'uniform':
        anc_dense = species_helpers['ancestors_dense'].to(device=device, dtype=dtype)
        _ancestors_T = anc_dense.T.to_sparse_coo()

    _lazy_batch_ranges = None
    _lazy_batch_builder = None
    if wave_layout_batches is None and families is not None and family_batch_size and family_batch_size > 0:
        from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
        from gpurec.core.scheduling import compute_clade_waves

        fam_items = []
        fam_waves = []
        fam_phases = []
        for fam in families:
            fam_items.append({
                'ccp': fam['ccp_helpers'],
                'leaf_row_index': fam['leaf_row_index'],
                'leaf_col_index': fam['leaf_col_index'],
                'root_clade_id': int(fam['root_clade_id']),
            })
            w, p = compute_clade_waves(fam['ccp_helpers'])
            fam_waves.append(w)
            fam_phases.append(p)

        n_f = len(fam_items)
        _lazy_batch_ranges = [
            (i, min(i + int(family_batch_size), n_f))
            for i in range(0, n_f, int(family_batch_size))
        ]

        def _build_lazy_batch(batch_id: int):
            i, j = _lazy_batch_ranges[batch_id]
            sub_items = fam_items[i:j]
            sub_waves = fam_waves[i:j]
            sub_phases = fam_phases[i:j]

            batched = collate_gene_families(sub_items, dtype=dtype, device=device)
            offsets = [m['clade_offset'] for m in batched['family_meta']]
            cross_waves = collate_wave(sub_waves, offsets)

            max_n_waves = max(len(p) for p in sub_phases)
            cross_phases = []
            for k in range(max_n_waves):
                phase_k = 1
                for fp in sub_phases:
                    if k < len(fp):
                        phase_k = max(phase_k, fp[k])
                cross_phases.append(phase_k)

            family_clade_counts = [m['C'] for m in batched['family_meta']]
            family_clade_offsets = [m['clade_offset'] for m in batched['family_meta']]
            wl = build_wave_layout(
                waves=cross_waves,
                phases=cross_phases,
                ccp_helpers=batched['ccp'],
                leaf_row_index=batched['leaf_row_index'],
                leaf_col_index=batched['leaf_col_index'],
                root_clade_ids=batched['root_clade_ids'],
                device=device,
                dtype=dtype,
                family_clade_counts=family_clade_counts,
                family_clade_offsets=family_clade_offsets,
            )
            return wl, batched['root_clade_ids']

        _lazy_batch_builder = _build_lazy_batch

    def _num_layout_batches() -> int:
        if wave_layout_batches is not None:
            return len(wave_layout_batches)
        if _lazy_batch_ranges is not None:
            return len(_lazy_batch_ranges)
        return 1

    _rng = torch.Generator(device='cpu')
    _rng.manual_seed(int(stochastic_seed))

    def _select_batch_ids() -> list[int] | None:
        n_total = _num_layout_batches()
        use_stochastic = stochastic_batches and optimizer == 'sgd' and n_total > 1
        if not use_stochastic:
            return None
        sampled = int(torch.randint(0, n_total, (1,), generator=_rng).item())
        return [sampled]

    def _extract(theta_d):
        if pibar_mode in ('dense', 'topk') and transfer_mat_unnormalized is not None:
            log_pS, log_pD, log_pL, transfer_mat, mt_raw = extract_parameters(
                theta_d, transfer_mat_unnormalized,
                genewise=False, specieswise=specieswise, pairwise=False,
            )
            mt = mt_raw.squeeze(-1) if mt_raw.ndim == 2 else mt_raw
        else:
            log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
                theta_d, unnorm_row_max, specieswise=specieswise,
            )
        return log_pS, log_pD, log_pL, transfer_mat, mt

    def _forward_backward(theta_d, warm_E, selected_batch_ids=None, omega_d=None):
        """Full forward + backward.

        Returns (nll, grad_theta, statsG, E_out) when origination is uniform, and
        (nll, grad_theta, statsG, E_out, grad_omega) when omega_d is provided
        (origination='optimize'). ``grad_omega`` [S] is the DIRECT ∂(Σ NLL)/∂omega
        at FIXED Pi, E (no implicit solve): origination only enters
        compute_log_likelihood, so its omega gradient is a cheap autograd through
        the per-family numerators + the shared (E + omega) survival denominator.
        """
        # Origination log-prob log_pO = omega - logsumexp2(omega). None -> uniform.
        log_pO = _origination_log_pO(omega_d) if omega_d is not None else None
        if wave_layout_batches is not None:
            if selected_batch_ids is None:
                layout_batches = wave_layout_batches
            else:
                layout_batches = [wave_layout_batches[i] for i in selected_batch_ids]
        elif _lazy_batch_builder is not None and _lazy_batch_ranges is not None:
            if selected_batch_ids is None:
                ids = range(len(_lazy_batch_ranges))
            else:
                ids = selected_batch_ids
            layout_batches = (_lazy_batch_builder(i) for i in ids)
        else:
            layout_batches = [(wave_layout, root_clade_ids)]

        torch.cuda.synchronize()
        t_e0 = time.perf_counter()
        try:
            with torch.no_grad():
                log_pS, log_pD, log_pL, transfer_mat, mt = _extract(theta_d)

                E_out = E_fixed_point(
                    species_helpers=species_helpers,
                    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                    transfer_mat=transfer_mat, max_transfer_mat=mt,
                    max_iters=e_max_iters, tolerance=e_tol,
                    warm_start_E=warm_E,
                    dtype=dtype, device=device, pibar_mode=pibar_mode,
                    ancestors_T=_ancestors_T,
                    leaf_E=leaf_E,
                )
        except torch.OutOfMemoryError as exc:
            raise torch.OutOfMemoryError(
                f"OOM in optimize_theta_wave phase=E_fixed_point; "
                f"mode={'specieswise' if specieswise else 'global'}; "
                f"pibar_mode={pibar_mode}; {_cuda_mem_diag(device)}"
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise torch.OutOfMemoryError(
                    f"OOM in optimize_theta_wave phase=E_fixed_point; "
                    f"mode={'specieswise' if specieswise else 'global'}; "
                    f"pibar_mode={pibar_mode}; {_cuda_mem_diag(device)}"
                ) from exc
            raise

        torch.cuda.synchronize()
        t_pi0 = time.perf_counter()
        with torch.no_grad():
            nll = 0.0
            grad_theta = torch.zeros_like(theta_d)
            stats_list = []
            pi_bwd_t_total = 0.0
            cg_t_total = 0.0
            theta_t_total = 0.0
            pi_phase_time = 0.0
            grad_phase_time = 0.0
            n_batch_accum = 0
            # Origination: collect each family's root-clade Pi row [n_fam_b, S] so
            # the DIRECT omega gradient can be computed once (per-family numerators
            # + the shared survival denominator) after the wave loop.
            root_pi_rows = [] if log_pO is not None else None

            for _batch_idx, (wl_b, roots_b) in enumerate(layout_batches):
                t_pi_b0 = time.perf_counter()
                try:
                    Pi_out_b = Pi_wave_forward(
                        wave_layout=wl_b, species_helpers=species_helpers,
                        E=E_out['E'], Ebar=E_out['E_bar'],
                        E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
                        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                        transfer_mat=transfer_mat, max_transfer_mat=mt,
                        device=device, dtype=dtype, pibar_mode=pibar_mode,
                        leaf_obs_log=leaf_obs_log,
                    )
                    logL_b = compute_log_likelihood(
                        Pi_out_b['Pi'], E_out['E'], roots_b, log_pO=log_pO,
                    )
                    nll += float(logL_b.sum().item())
                    if root_pi_rows is not None:
                        # Detach: the omega gradient is evaluated at FIXED Pi.
                        root_pi_rows.append(Pi_out_b['Pi'][roots_b, :].detach())
                except torch.OutOfMemoryError as exc:
                    raise torch.OutOfMemoryError(
                        f"OOM in optimize_theta_wave phase=Pi_wave_forward; "
                        f"mode={'specieswise' if specieswise else 'global'}; "
                        f"pibar_mode={pibar_mode}; family_batch_size={family_batch_size}; {_cuda_mem_diag(device)}"
                    ) from exc
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise torch.OutOfMemoryError(
                            f"OOM in optimize_theta_wave phase=Pi_wave_forward; "
                            f"mode={'specieswise' if specieswise else 'global'}; "
                            f"pibar_mode={pibar_mode}; family_batch_size={family_batch_size}; {_cuda_mem_diag(device)}"
                        ) from exc
                    raise
                pi_phase_time += time.perf_counter() - t_pi_b0

                t_g_b0 = time.perf_counter()
                try:
                    grad_theta_b, statsG_b = implicit_grad_loglik_vjp_wave(
                        wl_b, species_helpers,
                        Pi_star_wave=Pi_out_b['Pi_wave_ordered'],
                        Pibar_star_wave=Pi_out_b['Pibar_wave_ordered'],
                        E_star=E_out['E'], E_s1=E_out['E_s1'],
                        E_s2=E_out['E_s2'], Ebar=E_out['E_bar'],
                        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                        max_transfer_mat=mt,
                        root_clade_ids_perm=wl_b['root_clade_ids'],
                        theta=theta_d,
                        unnorm_row_max=unnorm_row_max,
                        specieswise=specieswise,
                        device=device, dtype=dtype,
                        neumann_terms=neumann_terms,
                        use_pruning=use_pruning,
                        pruning_threshold=pruning_threshold,
                        cg_tol=cg_tol, cg_maxiter=cg_maxiter,
                        gmres_restart=gmres_restart,
                        pibar_mode=pibar_mode,
                        transfer_mat=transfer_mat,
                        transfer_mat_unnormalized=transfer_mat_unnormalized,
                        ancestors_T=_ancestors_T,
                        leaf_E=leaf_E,
                        leaf_obs_log=leaf_obs_log,
                        log_pO=log_pO,
                    )
                except torch.OutOfMemoryError as exc:
                    raise torch.OutOfMemoryError(
                        f"OOM in optimize_theta_wave phase=implicit_grad_loglik_vjp_wave; "
                        f"mode={'specieswise' if specieswise else 'global'}; "
                        f"pibar_mode={pibar_mode}; family_batch_size={family_batch_size}; {_cuda_mem_diag(device)}"
                    ) from exc
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise torch.OutOfMemoryError(
                            f"OOM in optimize_theta_wave phase=implicit_grad_loglik_vjp_wave; "
                            f"mode={'specieswise' if specieswise else 'global'}; "
                            f"pibar_mode={pibar_mode}; family_batch_size={family_batch_size}; {_cuda_mem_diag(device)}"
                        ) from exc
                    raise
                grad_phase_time += time.perf_counter() - t_g_b0

                grad_theta = grad_theta + grad_theta_b
                stats_list.append(statsG_b)
                pi_bwd_t_total += float(getattr(statsG_b, 'pi_bwd_time', 0.0))
                cg_t_total += float(getattr(statsG_b, 'cg_time', 0.0))
                theta_t_total += float(getattr(statsG_b, 'theta_vjp_time', 0.0))
                n_batch_accum += 1

            # Gradient accumulation normalization: average over family batches.
            if n_batch_accum > 0:
                grad_theta = grad_theta / float(n_batch_accum)

            # --- DIRECT omega gradient (origination='optimize') --------------
            # omega enters ONLY compute_log_likelihood, so ∂(Σ NLL)/∂omega at FIXED
            # Pi, E is a direct autograd through the (origination-weighted)
            # per-family numerators and the shared survival denominator — no
            # implicit solve. Mirrors compute_log_likelihood EXACTLY:
            #   NLL_f = -(logsumexp2(root_Pi_f + log_pO) - logsumexp2(log2(1-exp2(E)) + log_pO))
            grad_omega = None
            if log_pO is not None:
                from gpurec.core.log2_utils import logsumexp2 as _lse2
                from gpurec.core.log2_utils import _safe_log2_internal as _slog2
                root_pi_all = torch.cat(root_pi_rows, dim=0)  # [n_fam_total, S]
                n_fam_total = root_pi_all.shape[0]
                E_det = E_out['E'].detach()
                # log2(1 - exp2(E)) with E -> 0 guarded (matches compute_log_likelihood).
                log_one_minus_E = _slog2(1.0 - torch.exp2(E_det))
                with torch.enable_grad():
                    omega_leaf = omega_d.detach().clone().requires_grad_(True)
                    log_pO_g = omega_leaf - _lse2(omega_leaf, dim=-1, keepdim=True)
                    num = _lse2(root_pi_all + log_pO_g, dim=-1)          # [n_fam_total]
                    denom = _lse2(log_one_minus_E + log_pO_g, dim=-1)    # scalar (shared E)
                    nll_omega = -(num - denom).sum()                     # Σ_f NLL_f
                    grad_omega = torch.autograd.grad(nll_omega, omega_leaf)[0].detach()
                # Match the theta gradient's family-batch averaging so the two
                # blocks of the joint scipy gradient are on the same scale.
                if n_batch_accum > 0:
                    grad_omega = grad_omega / float(n_batch_accum)

            if stats_list:
                method0 = stats_list[0].method
                method = method0 if all(s.method == method0 for s in stats_list) else "mixed"
                statsG = LinearSolveStats(
                    method=method,
                    iters=int(sum(s.iters for s in stats_list)),
                    rel_residual=float(max(s.rel_residual for s in stats_list)),
                    fallback_used=bool(any(s.fallback_used for s in stats_list)),
                )
                setattr(statsG, 'pi_bwd_time', pi_bwd_t_total)
                setattr(statsG, 'cg_time', cg_t_total)
                setattr(statsG, 'theta_vjp_time', theta_t_total)
            else:
                statsG = LinearSolveStats("none", 0, 0.0, False)
        torch.cuda.synchronize()
        if verbose:
            pi_bwd_t = getattr(statsG, 'pi_bwd_time', 0)
            cg_t = getattr(statsG, 'cg_time', 0)
            theta_t = getattr(statsG, 'theta_vjp_time', 0)
            print(f"    breakdown: E={t_pi0-t_e0:.3f}s  Pi={pi_phase_time:.3f}s  grad={grad_phase_time:.3f}s"
                  f"  [Pi_bwd={pi_bwd_t:.3f}s  CG={cg_t:.3f}s  theta_vjp={theta_t:.3f}s]",
                  flush=True)

        if omega_d is not None:
            return nll, grad_theta, statsG, E_out, grad_omega
        return nll, grad_theta, statsG, E_out

    # --- L-BFGS-B path (scipy) ---
    if optimizer == 'lbfgs':
        import numpy as np
        from scipy.optimize import minimize as scipy_minimize

        theta_t = theta_init.to(device=device, dtype=dtype).clone()
        theta_shape = theta_t.shape
        n_theta = theta_t.numel()
        history: List[StepRecord] = []
        warm_E_ref = [None]
        eval_count = [0]

        def forward_and_grad(x_flat_np):
            # JOINT vector layout: [theta_flat (n_theta); omega (S)] when origination
            # is optimised, else just theta_flat. Split, run, re-concat the gradient.
            x_flat = torch.from_numpy(x_flat_np).to(device=device, dtype=dtype)
            theta_flat = x_flat[:n_theta]
            theta_d = theta_flat.reshape(theta_shape).clamp(min=_THETA_MIN)
            if _opt_origination:
                omega_d = x_flat[n_theta:].reshape(_S_branches)
            else:
                omega_d = None

            t_start = time.perf_counter()
            if _opt_origination:
                nll, grad_theta, statsG, E_out, grad_omega = _forward_backward(
                    theta_d, warm_E_ref[0], omega_d=omega_d,
                )
            else:
                nll, grad_theta, statsG, E_out = _forward_backward(theta_d, warm_E_ref[0])
                grad_omega = None
            nll, grad_theta = _apply_prior(theta_d, nll, grad_theta)
            nll, grad_omega = _apply_origination_prior(omega_d, nll, grad_omega)
            step_time = time.perf_counter() - t_start
            warm_E_ref[0] = E_out['E'].detach()

            nll_is_nan = math.isnan(nll)
            eval_count[0] += 1
            if verbose:
                grad_inf_str = f"{float(grad_theta.abs().max()):.3e}" if not nll_is_nan else "nan"
                nll_str = f"{nll:.4f}" if not nll_is_nan else "nan"
                e_it = int(E_out['iterations'])
                print(f"  step {eval_count[0]:3d}/{steps}  NLL={nll_str}  |g|={grad_inf_str}"
                      f"  E_iters={e_it}  t={step_time:.2f}s", flush=True)
            fp_info = FixedPointInfo(iterations_E=int(E_out['iterations']),
                                     iterations_Pi=0)
            history.append(StepRecord(
                iteration=eval_count[0], theta=theta_d.detach().cpu(),
                rates=torch.exp2(theta_d.detach()).cpu(),
                negative_log_likelihood=nll, log_likelihood=-nll,
                theta_step_inf=0.0,
                grad_infinity_norm=float(grad_theta.abs().max().item()) if not nll_is_nan else float('nan'),
                fp_info=fp_info, gradient=grad_theta.cpu(),
                solve_stats_F=LinearSolveStats("wave_neumann", neumann_terms, 0.0, False),
                solve_stats_G=statsG,
                step_time_s=step_time,
            ))

            grad_theta_np = grad_theta.reshape(-1).cpu().to(torch.float64).numpy()
            if grad_omega is not None:
                grad_omega_np = grad_omega.reshape(-1).cpu().to(torch.float64).numpy()
                grad_np = np.concatenate([grad_theta_np, grad_omega_np])
            else:
                grad_np = grad_theta_np
            np.nan_to_num(grad_np, copy=False, nan=0.0)
            # Return +inf (not NaN) so scipy's line search backtracks instead of corrupting state
            return_nll = float('inf') if nll_is_nan else float(nll)
            return return_nll, grad_np

        theta_x0 = theta_t.reshape(-1).cpu().to(torch.float64).numpy()
        if theta_bounds is None:
            # Current behavior: lower bound = log2(1e-10), no upper bound.
            theta_bounds_list = [(_THETA_MIN, None)] * len(theta_x0)
        else:
            # AleRax-style box constraints (log2 units). Accept scalar (lo, hi)
            # applied to every element, or per-element arrays broadcastable to theta.
            lo_in, hi_in = theta_bounds
            lo_arr = np.broadcast_to(np.asarray(lo_in, dtype=np.float64), theta_x0.shape)
            hi_arr = np.broadcast_to(np.asarray(hi_in, dtype=np.float64), theta_x0.shape)
            theta_bounds_list = [(float(lo_arr[i]), float(hi_arr[i])) for i in range(len(theta_x0))]
        if _opt_origination:
            # Append omega block. omega is a softmax LOGIT (shift-invariant), so it
            # is UNBOUNDED — the normalisation in log_pO = omega - logsumexp2(omega)
            # makes any global shift a no-op. Leave omega unconstrained.
            omega_x0 = _omega0.reshape(-1).cpu().to(torch.float64).numpy()
            x0 = np.concatenate([theta_x0, omega_x0])
            bounds = theta_bounds_list + [(None, None)] * len(omega_x0)
        else:
            x0 = theta_x0
            bounds = theta_bounds_list
        result = scipy_minimize(
            forward_and_grad, x0, method='L-BFGS-B', jac=True,
            bounds=bounds,
            options={'maxiter': steps, 'maxfun': steps * 3, 'ftol': 1e-12, 'gtol': 1e-6},
        )

        x_final = torch.from_numpy(result.x).to(device=device, dtype=dtype)
        theta_final = x_final[:n_theta].reshape(theta_shape)
        if _opt_origination:
            omega_final = x_final[n_theta:].reshape(_S_branches)
        else:
            omega_final = None

        # Detect float32 precision floor and retry in float64
        if dtype == torch.float32 and len(history) >= 5:
            recent_nlls = [h.negative_log_likelihood for h in history[-5:]]
            nll_range = max(recent_nlls) - min(recent_nlls)
            final_grad_inf = history[-1].grad_infinity_norm
            if final_grad_inf > 1e-2 and nll_range < 1e-4:
                if verbose:
                    print(f"  [float32 floor detected: grad_inf={final_grad_inf:.3e}, "
                          f"nll_range={nll_range:.2e}] retrying in float64 ...", flush=True)

                def _to64(x):
                    """Recursively convert floating-point tensors to float64.

                    Handles nested containers: dict, list, tuple.
                    Non-floating tensors (e.g. Long index tensors) and
                    non-tensor values (int, str, dict of strings) are left untouched.
                    """
                    if torch.is_tensor(x):
                        return x.to(torch.float64) if x.is_floating_point() else x
                    if isinstance(x, dict):
                        return {k: _to64(v) for k, v in x.items()}
                    if isinstance(x, list):
                        return [_to64(v) for v in x]
                    if isinstance(x, tuple):
                        return tuple(_to64(v) for v in x)
                    return x

                wl64 = _to64(wave_layout)
                sp64 = _to64(species_helpers)
                urm64 = _to64(unnorm_row_max)
                tm64 = _to64(transfer_mat_unnormalized) if transfer_mat_unnormalized is not None else None
                theta_init64 = theta_final.to(torch.float64)

                result64 = optimize_theta_wave(
                    wl64, sp64, root_clade_ids,
                    urm64, theta_init64,
                    transfer_mat_unnormalized=tm64,
                    steps=steps, lr=lr, tol_theta=tol_theta,
                    e_max_iters=e_max_iters, e_tol=e_tol,
                    neumann_terms=neumann_terms,
                    use_pruning=use_pruning, pruning_threshold=pruning_threshold,
                    cg_tol=cg_tol, cg_maxiter=cg_maxiter, gmres_restart=gmres_restart,
                    specieswise=specieswise, device=device,
                    dtype=torch.float64,
                    pibar_mode=pibar_mode,
                    family_batch_size=family_batch_size,
                    wave_layout_batches=wave_layout_batches,
                    families=families,
                    stochastic_batches=stochastic_batches,
                    stochastic_seed=stochastic_seed,
                    optimizer='lbfgs',
                    verbose=verbose,
                    leaf_E=leaf_E,
                    leaf_obs_log=leaf_obs_log,
                    brownian_sigma=brownian_sigma,
                    brownian_root_sigma=brownian_root_sigma,
                    # Pass the RESOLVED root-anchor centre (not None) so the
                    # float64 retry keeps the same prior centre as phase 1.
                    brownian_mu=_prior_mu if _use_prior else brownian_mu,
                    parent_index=_prior_parent_index if _use_prior else parent_index,
                    theta_bounds=theta_bounds,
                    origination=origination,
                    omega_init=omega_final if _opt_origination else None,
                    origination_sigma=origination_sigma,
                    origination_root_sigma=origination_root_sigma,
                    origination_mu=origination_mu,
                    origination_decouple_root=origination_decouple_root,
                )
                # Merge histories: float32 phase first, then float64 phase
                result64["history"] = history + result64["history"]
                return result64

        # Pure DATA negative-log-likelihood at the optimum, with the prior EXCLUDED.
        # `result.fun` is the MAP objective (data NLL + Brownian penalty); the penalty
        # depends on the species-tree topology, so it must NOT be in cross-root
        # likelihood (rooting) comparisons. Re-evaluate the forward once at the optimum
        # (with the optimised omega so the origination-weighted likelihood is used).
        _data_nll = float(_forward_backward(
            theta_final.to(device=device, dtype=dtype), None,
            omega_d=omega_final,
        )[0])

        result_dict = {
            "theta": theta_final.cpu(),
            "rates": torch.exp2(theta_final).cpu(),
            "negative_log_likelihood": result.fun,
            "log_likelihood": -result.fun,
            "data_negative_log_likelihood": _data_nll,
            "data_log_likelihood": -_data_nll,
            "history": history,
            "scipy_result": result,
        }
        if _opt_origination:
            # omega = origination log2-weights; origination = normalised p^O_e.
            result_dict["omega"] = omega_final.detach().cpu()
            result_dict["origination"] = torch.exp2(
                _origination_log_pO(omega_final)
            ).detach().cpu()
            result_dict["origination_strategy"] = origination
            if _use_orig_prior:
                result_dict["origination_prior"] = {
                    "sigma": _orig_sigma,
                    "root_sigma": _orig_root_sigma,
                    "mu": _orig_mu,
                    "decouple_root": bool(origination_decouple_root),
                }
        return result_dict

    # --- Iterative optimizers (adam, sgd) ---
    theta = torch.nn.Parameter(theta_init.to(device=device, dtype=dtype).clone())
    if optimizer == 'sgd':
        opt = torch.optim.SGD([theta], lr=lr, momentum=momentum, nesterov=False)
    else:
        opt = torch.optim.Adam([theta], lr=lr)

    history: List[StepRecord] = []
    prev_theta = theta.detach().clone()
    warm_E = None

    for it in range(1, steps + 1):
        theta_d = theta.detach()
        selected_batch_ids = _select_batch_ids()

        t_start = time.perf_counter()
        nll, grad_theta, statsG, E_out = _forward_backward(theta_d, warm_E, selected_batch_ids=selected_batch_ids)
        nll, grad_theta = _apply_prior(theta_d, nll, grad_theta)
        warm_E = E_out['E'].detach()
        iters_E = int(E_out['iterations'])

        # Optimizer step (minimize NLL)
        opt.zero_grad(set_to_none=True)
        grad_clean = grad_theta.clone()
        grad_clean.nan_to_num_(nan=0.0)
        theta.grad = grad_clean
        opt.step()
        step_time = time.perf_counter() - t_start

        with torch.no_grad():
            theta.clamp_(min=_THETA_MIN)

        # Bookkeeping
        theta_detached = theta.detach()
        diff = float(torch.max(torch.abs(theta_detached - prev_theta)).item())
        prev_theta = theta_detached.clone()
        rates = torch.exp2(theta_detached)
        grad_inf = float(grad_theta.abs().max().item())

        if verbose:
            nll_str = f"{nll:.4f}" if not math.isnan(nll) else "nan"
            print(f"  step {it:3d}/{steps}  NLL={nll_str}  |g|={grad_inf:.3e}"
                  f"  E_iters={iters_E}  t={step_time:.2f}s", flush=True)

        fp_info = FixedPointInfo(iterations_E=iters_E, iterations_Pi=0)
        statsF_dummy = LinearSolveStats("wave_neumann", neumann_terms, 0.0, False)

        history.append(
            StepRecord(
                iteration=it,
                theta=theta_detached.cpu(),
                rates=rates.cpu(),
                negative_log_likelihood=nll,
                log_likelihood=-nll,
                theta_step_inf=diff,
                grad_infinity_norm=grad_inf,
                fp_info=fp_info,
                gradient=grad_theta.cpu(),
                solve_stats_F=statsF_dummy,
                solve_stats_G=statsG,
                step_time_s=step_time,
            )
        )

        if diff < tol_theta and it > 1:
            break

    return {
        "theta": theta.detach().cpu(),
        "rates": torch.exp2(theta.detach()).cpu(),
        "negative_log_likelihood": history[-1].negative_log_likelihood if history else float('nan'),
        "log_likelihood": history[-1].log_likelihood if history else float('nan'),
        "history": history,
    }
