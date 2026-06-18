"""Legacy test-only baselines for validation.

These implementations are superseded by the wave-based forward pass
(``Pi_wave_forward``) and kept only for testing and validation:

- ``Pi_step`` / ``Pi_fixed_point``  – full-matrix fixed-point iteration
  (used as baseline in ``tests/unit/test_wave_vs_fp.py``)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from .terms import (
    gather_Pi_children,
    compute_DTS,
    compute_DTS_L,
)
from .log2_utils import logsumexp2, logaddexp2
from ._logmatmul_compat import HAS_LOGMATMUL as _HAS_LOGMATMUL, LogspaceMatmulFn
from ._helpers import _seg_logsumexp_host  # noqa: F401

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Pi step (legacy full-matrix iteration)
# ---------------------------------------------------------------------------

def Pi_step(
    Pi: torch.Tensor,
    ccp_helpers: dict,
    species_helpers: dict,
    log_pS: torch.Tensor,
    log_pD: torch.Tensor,
    log_pL: torch.Tensor,
    transfer_mat_T: torch.Tensor,
    max_transfer_mat: torch.Tensor,
    clade_species_map: torch.Tensor,
    E: torch.Tensor,
    Ebar: torch.Tensor,
    E_s1: torch.Tensor,
    E_s2: torch.Tensor,
    log_2: torch.Tensor,
):
    # region helpers
    split_leftrights_sorted = ccp_helpers['split_leftrights_sorted']
    log_split_probs = ccp_helpers['log_split_probs_sorted'].unsqueeze(1).contiguous()
    seg_parent_ids = ccp_helpers['seg_parent_ids']
    num_segs_ge2 = ccp_helpers['num_segs_ge2']
    num_segs_eq1 = ccp_helpers['num_segs_eq1']
    end_rows_ge2 = ccp_helpers['end_rows_ge2']
    ptr_ge2 = ccp_helpers['ptr_ge2']
    N_splits = ccp_helpers["N_splits"]
    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers["s_C12_indexes"]

    C, S = Pi.shape
    # endregion helpers

    Pi_s12 = gather_Pi_children(Pi, sp_P_idx, sp_c12_idx)

    if _HAS_LOGMATMUL and Pi.is_cuda:
        transfer_mat = transfer_mat_T.T
        Pi_T = Pi.T.contiguous()
        Pibar_T = LogspaceMatmulFn.apply(transfer_mat, Pi_T, "ieee")
        Pibar = Pibar_T.T + max_transfer_mat.squeeze(-1)
    else:
        Pi_max = torch.max(Pi, dim=1, keepdim=True).values
        Pi_minus = Pi - Pi_max
        Pi_linear = torch.exp2(Pi_minus)
        Pibar_linear = Pi_linear.mm(transfer_mat_T)
        Pibar_log = torch.log2(Pibar_linear)
        Pibar_log = Pibar_log + Pi_max
        Pibar = Pibar_log + max_transfer_mat.squeeze(-1)

    DTS_term = compute_DTS(log_pD, log_pS, Pi_s12, Pi, Pibar, log_split_probs, split_leftrights_sorted, N_splits, S)
    DTS_L_term = compute_DTS_L(log_pD, log_pS, Pi, Pibar, Pi_s12, E, Ebar, E_s1, E_s2, clade_species_map, log_2)

    DTS_reduced = torch.full((C, S), NEG_INF, device=Pi.device, dtype=Pi.dtype)
    if num_segs_ge2 > 0:
        y_ge2 = _seg_logsumexp_host(DTS_term[:end_rows_ge2], ptr_ge2)
        DTS_reduced.index_copy_(0, seg_parent_ids[:num_segs_ge2], y_ge2)
    if num_segs_eq1 > 0:
        DTS_reduced.index_copy_(
            0,
            seg_parent_ids[num_segs_ge2:num_segs_ge2 + num_segs_eq1],
            DTS_term[end_rows_ge2:end_rows_ge2 + num_segs_eq1],
        )

    return logaddexp2(DTS_reduced, DTS_L_term)


# ---------------------------------------------------------------------------
# Pi fixed point (legacy)
# ---------------------------------------------------------------------------

def Pi_fixed_point(
    ccp_helpers,
    species_helpers,
    leaf_row_index,
    leaf_col_index,
    E,
    Ebar,
    E_s1,
    E_s2,
    log_pS,
    log_pD,
    log_pL,
    transfer_mat_T,
    max_transfer_mat,
    max_iters,
    tolerance,
    warm_start_Pi,
    device,
    dtype,
    leaf_obs_log=None,
    leaf_species_mask=None,
):
    """Fixed-point solver for Pi using leaf mapping indices and current event params.

    ``leaf_obs_log`` (optional, [S]) = ``log2(1 - p_obs_l)`` = ``log2(fraction_missing_l)``
    and ``leaf_species_mask`` (optional, [S] bool) mark the species-tree leaves.
    Together they implement the fraction-missing leaf boundary
    ``Pi_{l,gamma} = sigma + (1-sigma)(1 - p_obs_l)`` (AleRaxSupp.tex). By default
    (both ``None``) ``Pi_{l,gamma} = sigma`` (every gene observed).
    """
    C = int(ccp_helpers['C'])
    S = int(species_helpers['S'])

    clade_species_map = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
    if leaf_obs_log is not None and leaf_species_mask is not None:
        # sigma=0 baseline at species-leaf columns: a clade that does not map to
        # leaf l is still present-but-unobserved with prob 1-p_obs_l. The sigma=1
        # (mapped leaf-clade) entries are overwritten to log2(1)=0 just below.
        lm = leaf_species_mask.to(device)
        clade_species_map[:, lm] = leaf_obs_log.to(device=device, dtype=dtype)[lm]
    clade_species_map[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0

    if warm_start_Pi is not None:
        Pi = warm_start_Pi
    else:
        Pi = torch.full((C, S), -1000.0, dtype=dtype, device=device)
        Pi[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0

    converged_iter = max_iters
    log_2 = torch.tensor([1.0], dtype=dtype, device=device)

    for iteration in range(max_iters):
        Pi_new = Pi_step(
            Pi,
            ccp_helpers,
            species_helpers,
            log_pS,
            log_pD,
            log_pL,
            transfer_mat_T,
            max_transfer_mat,
            clade_species_map,
            E,
            Ebar,
            E_s1,
            E_s2,
            log_2,
        )
        if torch.abs(Pi_new - Pi).max() < tolerance:
            converged_iter = iteration + 1
            Pi = Pi_new
            break
        Pi = Pi_new

    return {'Pi': Pi, 'clade_species_map': clade_species_map, 'iterations': converged_iter}
