"""Autograd bridge: wraps the existing implicit-gradient pipeline as a
``torch.autograd.Function`` so a notebook user can call standard
``loss.backward()`` and use any ``torch.optim`` optimizer.

The forward pass mirrors :class:`gpurec.core.model.GeneDataset.compute_likelihood_batch`
(without ``@torch.no_grad``) and the backward pass delegates to the existing
:func:`gpurec.optimization.implicit_grad.implicit_grad_loglik_vjp_wave` (or
the genewise pair ``Pi_wave_backward + _e_adjoint_and_theta_vjp(genewise=True)``).
No new gradient math is written here.

Sign convention: ``compute_log_likelihood`` actually returns NLL despite its
name (see ``gpurec/core/likelihood.py:180``). The bridge keeps the NLL
convention and returns NLL from ``forward()``, so users write
``loss = model(); loss.backward()`` directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

import torch

from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.backward import Pi_wave_backward
from gpurec.core._helpers import _nvtx_range
from gpurec.core.extract_parameters import (
    extract_parameters,
    extract_parameters_uniform,
)
from gpurec.optimization.implicit_grad import (
    implicit_grad_loglik_vjp_wave,
    _e_adjoint_and_theta_vjp,
)


@dataclass
class ReconStaticState:
    """Container for non-differentiable state shared across ``forward()`` calls.

    Built once by :class:`gpurec.api.model.GeneReconModel` from a
    :class:`GeneDataset`. Mutated only via ``warm_E`` (warm start cache) and
    via ``_apply_to_static`` when the parent module is moved (``.to``).
    """

    device: torch.device
    dtype: torch.dtype

    # Wave layout + likelihood inputs (precomputed once)
    wave_layout: dict[str, Any]
    species_helpers: dict[str, Any]
    root_clade_ids: torch.Tensor                              # Long[G] original order
    unnorm_row_max: torch.Tensor                              # [S]
    transfer_mat_unnormalized: Optional[torch.Tensor]         # [S, S] log2 (dense only)
    ancestors_T: Optional[torch.Tensor]                       # sparse COO (uniform only)

    # Mode flags (mapped from "global" / "specieswise" / "genewise")
    genewise: bool
    specieswise: bool

    # Solver knobs
    pibar_mode: str = "uniform"
    max_iters_E: int = 2000
    tol_E: float = 1e-8
    max_iters_Pi: int = 2000
    tol_Pi: float = 1e-6
    fixed_iters_Pi: Optional[int] = 6
    neumann_terms: int = 3
    use_pruning: bool = True
    pruning_threshold: float = 1e-6
    cg_tol: float = 1e-8
    cg_maxiter: int = 500
    gmres_restart: int = 40

    # Warm start cache, mutated across calls
    warm_E: Optional[torch.Tensor] = None

    # Fraction-missing leaf boundary (per-species, shared across families).
    # leaf_E == leaf_obs_log == log2(fraction_missing_l) [S], -inf for
    # internal/fully-observed species; leaf_species_mask [S] bool marks leaves.
    # All None => every gene observed (standard sigma-only boundary).
    leaf_E: Optional[torch.Tensor] = None
    leaf_obs_log: Optional[torch.Tensor] = None
    leaf_species_mask: Optional[torch.Tensor] = None


def _apply_tensor_tree(obj: Any, fn) -> Any:
    """Recursively apply ``fn`` to every Tensor inside dicts / lists / tuples.

    Mirrors :meth:`gpurec.core.model.GeneDataset._move_tensor` semantics:
    floating-point tensors get the full ``fn`` (which may change dtype),
    integer tensors only get the device portion via a stripped-dtype call.
    """
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return fn(obj)
        # For Long/Int index tensors, fn() may try to cast dtype, which would
        # break indexing. We replicate the existing pattern: only move device.
        moved = fn(obj)
        if moved.dtype != obj.dtype:
            moved = moved.to(dtype=obj.dtype)
        return moved
    if isinstance(obj, dict):
        return {k: _apply_tensor_tree(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_apply_tensor_tree(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_apply_tensor_tree(v, fn) for v in obj)
    return obj


def _apply_to_static(static: ReconStaticState, fn) -> ReconStaticState:
    """Walk a ``ReconStaticState`` and apply ``fn`` to every tensor field.

    Returns a new dataclass instance; the original is left untouched. Updates
    ``device``/``dtype`` to match the post-``fn`` state of ``unnorm_row_max``.
    """
    updated: dict[str, Any] = {}
    for f in fields(static):
        val = getattr(static, f.name)
        updated[f.name] = _apply_tensor_tree(val, fn)
    new = ReconStaticState(**updated)
    # Sync device/dtype to whatever fn produced
    probe = new.unnorm_row_max
    new.device = probe.device
    new.dtype = probe.dtype
    return new


def _extract_parameters(theta: torch.Tensor, static: ReconStaticState):
    """Replicates :meth:`GeneDataset._extract_batch_params` exactly."""
    use_uniform = static.pibar_mode == "uniform"
    if use_uniform:
        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = (
            extract_parameters_uniform(
                theta,
                static.unnorm_row_max,
                specieswise=static.specieswise,
                genewise=static.genewise,
            )
        )
        return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat
    # dense / topk
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
        theta,
        static.transfer_mat_unnormalized,
        genewise=static.genewise,
        specieswise=static.specieswise,
        pairwise=False,
    )
    if max_transfer_mat.ndim >= 2 and max_transfer_mat.shape[-1] == 1:
        max_transfer_mat = max_transfer_mat.squeeze(-1)
    return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat


class _GeneReconFunction(torch.autograd.Function):
    """``forward`` runs the existing E + Pi pipeline; ``backward`` calls the
    existing implicit gradient. Inputs other than ``theta`` are treated as
    constants by autograd (passed via the static dataclass)."""

    @staticmethod
    def forward(ctx, theta: torch.Tensor, static: ReconStaticState, reduce: str):
        if reduce not in ("sum", "per_family"):
            raise ValueError(f"reduce must be 'sum' or 'per_family', got {reduce!r}")
        if reduce == "per_family" and not static.genewise:
            raise ValueError(
                "reduce='per_family' is only valid in genewise mode."
            )

        device = static.device
        dtype = static.dtype

        with torch.no_grad():
            # 1. Extract parameters
            with _nvtx_range("forward extract parameters"):
                log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = (
                    _extract_parameters(theta, static)
                )

            # 2. E fixed-point with warm start
            with _nvtx_range("forward E fixed point"):
                E_out = E_fixed_point(
                    species_helpers=static.species_helpers,
                    log_pS=log_pS,
                    log_pD=log_pD,
                    log_pL=log_pL,
                    transfer_mat=transfer_mat,
                    max_transfer_mat=max_transfer_vec,
                    max_iters=static.max_iters_E,
                    tolerance=static.tol_E,
                    warm_start_E=static.warm_E,
                    dtype=dtype,
                    device=device,
                    pibar_mode=static.pibar_mode,
                    ancestors_T=static.ancestors_T,
                    leaf_E=static.leaf_E,
                )
                E = E_out["E"]
                E_s1 = E_out["E_s1"]
                E_s2 = E_out["E_s2"]
                Ebar = E_out["E_bar"]

            # 3. Pi wave forward
            with _nvtx_range("forward Pi waves"):
                Pi_out = Pi_wave_forward(
                    wave_layout=static.wave_layout,
                    species_helpers=static.species_helpers,
                    E=E,
                    Ebar=Ebar,
                    E_s1=E_s1,
                    E_s2=E_s2,
                    log_pS=log_pS,
                    log_pD=log_pD,
                    log_pL=log_pL,
                    transfer_mat=transfer_mat,
                    max_transfer_mat=max_transfer_vec,
                    device=device,
                    dtype=dtype,
                    local_iters=static.max_iters_Pi,
                    local_tolerance=static.tol_Pi,
                    fixed_iters=static.fixed_iters_Pi,
                    pibar_mode=static.pibar_mode,
                    return_original=False,
                    family_idx=(
                        static.wave_layout.get("family_idx") if static.genewise else None
                    ),
                    leaf_obs_log=static.leaf_obs_log,
                    leaf_species_mask=static.leaf_species_mask,
                )

            # 4. NLL: compute_log_likelihood returns NLL despite the name (see
            #    gpurec/core/likelihood.py:180). nll_vec is per-family.
            with _nvtx_range("forward root likelihood"):
                nll_vec = compute_log_likelihood(
                    Pi_out["Pi_wave_ordered"], E, static.wave_layout["root_clade_ids"]
                )

        # 5. Save state for backward.
        with _nvtx_range("forward save outputs"):
            ctx.save_for_backward(
                theta,
                Pi_out["Pi_wave_ordered"],
                Pi_out["Pibar_wave_ordered"],
                E,
                E_s1,
                E_s2,
                Ebar,
                log_pS,
                log_pD,
                log_pL,
                max_transfer_vec,
                (
                    Pi_out["uniform_pibar_row_max"]
                    if Pi_out.get("uniform_pibar_row_max") is not None
                    else theta.new_empty(0)
                ),
            )
            # transfer_mat may be None (uniform mode); store as ctx attribute.
            ctx.transfer_mat = transfer_mat
            ctx.static = static
            ctx.reduce = reduce

        # 6. Update warm-start cache (in-place mutation of the shared static).
        with _nvtx_range("forward reduce"):
            static.warm_E = E.detach()

            # 7. Reduce.
            return nll_vec.sum() if reduce == "sum" else nll_vec

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (
            theta,
            Pi_star_wave,
            Pibar_star_wave,
            E_star,
            E_s1,
            E_s2,
            Ebar,
            log_pS,
            log_pD,
            log_pL,
            max_transfer_vec,
            uniform_pibar_row_max,
        ) = ctx.saved_tensors
        transfer_mat = ctx.transfer_mat
        static: ReconStaticState = ctx.static
        wave_layout = static.wave_layout

        if static.genewise:
            # Cross-family batched genewise path. Mirrors
            # gpurec/optimization/genewise_optimizer.py:359-407 exactly.
            pi_bwd = Pi_wave_backward(
                wave_layout=wave_layout,
                Pi_star_wave=Pi_star_wave,
                Pibar_star_wave=Pibar_star_wave,
                E=E_star,
                Ebar=Ebar,
                E_s1=E_s1,
                E_s2=E_s2,
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                max_transfer_mat=max_transfer_vec,
                species_helpers=static.species_helpers,
                root_clade_ids_perm=wave_layout["root_clade_ids"],
                device=static.device,
                dtype=static.dtype,
                neumann_terms=static.neumann_terms,
                use_pruning=static.use_pruning,
                pruning_threshold=static.pruning_threshold,
                pibar_mode=static.pibar_mode,
                transfer_mat=transfer_mat,
                ancestors_T=static.ancestors_T,
                family_idx=wave_layout["family_idx"],
                leaf_obs_log=static.leaf_obs_log,
                leaf_species_mask=static.leaf_species_mask,
                uniform_pibar_row_max=(
                    uniform_pibar_row_max
                    if uniform_pibar_row_max.numel() > 0
                    else None
                ),
            )
            grad_theta, _stats = _e_adjoint_and_theta_vjp(
                pi_bwd,
                E_star,
                Ebar,
                E_s1,
                E_s2,
                log_pS,
                log_pD,
                log_pL,
                max_transfer_vec,
                static.species_helpers,
                wave_layout["root_clade_ids"],
                theta,
                static.unnorm_row_max,
                static.specieswise,
                static.device,
                static.dtype,
                genewise=True,
                cg_tol=static.cg_tol,
                cg_maxiter=static.cg_maxiter,
                gmres_restart=static.gmres_restart,
                pibar_mode=static.pibar_mode,
                transfer_mat=transfer_mat,
                transfer_mat_unnormalized=static.transfer_mat_unnormalized,
                ancestors_T=static.ancestors_T,
                leaf_E=static.leaf_E,
            )
        else:
            # Shared theta path: delegate to the public wrapper.
            grad_theta, _stats = implicit_grad_loglik_vjp_wave(
                wave_layout,
                static.species_helpers,
                Pi_star_wave=Pi_star_wave,
                Pibar_star_wave=Pibar_star_wave,
                E_star=E_star,
                E_s1=E_s1,
                E_s2=E_s2,
                Ebar=Ebar,
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                max_transfer_mat=max_transfer_vec,
                root_clade_ids_perm=wave_layout["root_clade_ids"],
                theta=theta,
                unnorm_row_max=static.unnorm_row_max,
                specieswise=static.specieswise,
                device=static.device,
                dtype=static.dtype,
                neumann_terms=static.neumann_terms,
                use_pruning=static.use_pruning,
                pruning_threshold=static.pruning_threshold,
                cg_tol=static.cg_tol,
                cg_maxiter=static.cg_maxiter,
                gmres_restart=static.gmres_restart,
                pibar_mode=static.pibar_mode,
                transfer_mat=transfer_mat,
                transfer_mat_unnormalized=static.transfer_mat_unnormalized,
                ancestors_T=static.ancestors_T,
                uniform_pibar_row_max=(
                    uniform_pibar_row_max
                    if uniform_pibar_row_max.numel() > 0
                    else None
                ),
                leaf_E=static.leaf_E,
                leaf_obs_log=static.leaf_obs_log,
                leaf_species_mask=static.leaf_species_mask,
            )

        # grad_theta is d(NLL_total)/d(theta). The forward returned NLL_total
        # (or NLL_per_family). No sign flip required.
        if ctx.reduce == "sum":
            # grad_output is a scalar.
            grad_theta = grad_theta * grad_output
        else:
            # per_family (genewise): grad_output is [G]; broadcast across the
            # remaining theta dims.
            gvec = grad_output.view((-1,) + (1,) * (theta.ndim - 1))
            grad_theta = grad_theta * gvec

        return grad_theta, None, None
