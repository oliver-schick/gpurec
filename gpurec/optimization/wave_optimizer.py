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
    brownian_sigma=None,
    brownian_root_sigma: float = 5.0,
    brownian_mu=None,
    parent_index: torch.Tensor | None = None,
    theta_bounds=None,
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
        Optional fraction-missing leaf boundary, log2(1 - p_obs_l) = log2(fraction
        missing) at missing species-leaves and -inf elsewhere. A single [S] tensor
        serves BOTH the E extinction boundary (passed as leaf_E to E_fixed_point and
        the E adjoint E_steps) and the Pi leaf boundary (passed as leaf_obs_log to
        Pi_wave_forward / backward), since they are identical. None (default) ->
        every gene observed (byte-identical to before).
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

    Returns
    -------
    dict with 'theta', 'rates', 'log_likelihood', 'negative_log_likelihood', 'history'.
    """
    if device is None:
        device = theta_init.device

    _THETA_MIN = math.log2(1e-10)

    # --- Brownian (TKP) rate prior setup -----------------------------------
    # Only meaningful for specieswise theta [S,3]; skip otherwise so the
    # global/genewise paths stay byte-identical.
    _use_prior = brownian_sigma is not None
    if _use_prior and not (specieswise and theta_init.ndim == 2 and theta_init.shape[-1] == 3):
        if verbose:
            print("  [brownian prior requested but theta is not specieswise [S,3]; "
                  "skipping prior]", flush=True)
        _use_prior = False
    _prior_parent_index = None
    _prior_sigma = None
    _prior_root_sigma = None
    _prior_mu = None
    if _use_prior:
        from gpurec.core.tree_prior import (
            brownian_log_prior_and_grad as _brownian_prior,
            species_parent_index as _species_parent_index,
        )
        if parent_index is None:
            _prior_parent_index = _species_parent_index(species_helpers).to(device)
        else:
            _prior_parent_index = parent_index.to(device=device, dtype=torch.long)
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

    def _apply_prior(theta_d, nll, grad_theta):
        """Add the Brownian penalty to nll and its gradient to grad_theta."""
        if not _use_prior:
            return nll, grad_theta
        penalty, p_grad = _brownian_prior(
            theta_d, _prior_parent_index, _prior_sigma, _prior_root_sigma, _prior_mu,
        )
        return nll + float(penalty.item()), grad_theta + p_grad

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

    def _forward_backward(theta_d, warm_E, selected_batch_ids=None):
        """Full forward + backward: returns (nll, grad_theta, history_record, warm_E)."""
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
                        leaf_obs_log=leaf_E,
                    )
                    logL_b = compute_log_likelihood(Pi_out_b['Pi'], E_out['E'], roots_b)
                    nll += float(logL_b.sum().item())
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

        return nll, grad_theta, statsG, E_out

    # --- L-BFGS-B path (scipy) ---
    if optimizer == 'lbfgs':
        import numpy as np
        from scipy.optimize import minimize as scipy_minimize

        theta_t = theta_init.to(device=device, dtype=dtype).clone()
        theta_shape = theta_t.shape
        history: List[StepRecord] = []
        warm_E_ref = [None]
        eval_count = [0]

        def forward_and_grad(theta_flat_np):
            theta_flat = torch.from_numpy(theta_flat_np).to(device=device, dtype=dtype)
            theta_d = theta_flat.reshape(theta_shape).clamp(min=_THETA_MIN)

            t_start = time.perf_counter()
            nll, grad_theta, statsG, E_out = _forward_backward(theta_d, warm_E_ref[0])
            nll, grad_theta = _apply_prior(theta_d, nll, grad_theta)
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

            grad_np = grad_theta.reshape(-1).cpu().to(torch.float64).numpy()
            np.nan_to_num(grad_np, copy=False, nan=0.0)
            # Return +inf (not NaN) so scipy's line search backtracks instead of corrupting state
            return_nll = float('inf') if nll_is_nan else float(nll)
            return return_nll, grad_np

        x0 = theta_t.reshape(-1).cpu().to(torch.float64).numpy()
        if theta_bounds is None:
            # Current behavior: lower bound = log2(1e-10), no upper bound.
            bounds = [(_THETA_MIN, None)] * len(x0)
        else:
            # AleRax-style box constraints (log2 units). Accept scalar (lo, hi)
            # applied to every element, or per-element arrays broadcastable to x0.
            lo_in, hi_in = theta_bounds
            lo_arr = np.broadcast_to(np.asarray(lo_in, dtype=np.float64), x0.shape)
            hi_arr = np.broadcast_to(np.asarray(hi_in, dtype=np.float64), x0.shape)
            bounds = [(float(lo_arr[i]), float(hi_arr[i])) for i in range(len(x0))]
        result = scipy_minimize(
            forward_and_grad, x0, method='L-BFGS-B', jac=True,
            bounds=bounds,
            options={'maxiter': steps, 'maxfun': steps * 3, 'ftol': 1e-12, 'gtol': 1e-6},
        )

        theta_final = torch.from_numpy(result.x).to(device=device, dtype=dtype).reshape(theta_shape)

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
                    brownian_sigma=brownian_sigma,
                    brownian_root_sigma=brownian_root_sigma,
                    # Pass the RESOLVED root-anchor centre (not None) so the
                    # float64 retry keeps the same prior centre as phase 1.
                    brownian_mu=_prior_mu if _use_prior else brownian_mu,
                    parent_index=_prior_parent_index if _use_prior else parent_index,
                    theta_bounds=theta_bounds,
                )
                # Merge histories: float32 phase first, then float64 phase
                result64["history"] = history + result64["history"]
                return result64

        return {
            "theta": theta_final.cpu(),
            "rates": torch.exp2(theta_final).cpu(),
            "negative_log_likelihood": result.fun,
            "log_likelihood": -result.fun,
            "history": history,
            "scipy_result": result,
        }

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
