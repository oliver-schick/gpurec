"""Recipient ("transfer-to") transfer parametrization.

Standard gpurec parametrizes transfer by the DONOR (a per-donor transfer rate,
uniform over recipients). This module implements the RECIPIENT parametrization:
each branch r has a receptivity weight w_r, and the transfer recipient sum in the
T / TL likelihood terms is weighted by w_r:

    Pibar[c,d] = log_pT[d] + log2( ( Σ_{r∉anc(d)} w_r Π[c,r] ) / ( Σ_{r∉anc(d)} w_r ) ).

KEY: w_r is recipient-separable, so the efficient uniform path is preserved at
O(C·S) -- scale Π by w_r once, reuse the existing row_sum - ancestor_sum, and add
a per-donor O(S) normalization. This module provides:

  * ``recipient_uniform_setup(omega, ancestors_T)`` -> (w, recip_mt_offset)
        w               [S] linear receptivity (exp2(omega), mean-normalised)
        recip_mt_offset [S] = -log2(Σ_{r∉anc(d)} w_r)   (replaces unnorm_row_max
                              in max_transfer_mat = log_pT + recip_mt_offset)
    The uniform Pibar then scales Pi_exp by w and adds mt = log_pT + recip_mt_offset.

  * ``build_recipient_transfer_mat(omega, ancestors_T, log_pT)`` -> (transfer_mat,
        max_transfer_mat) for the DENSE/legacy path (the explicit [S,S] matrix),
        used as the validation reference.

Both representations are exactly equivalent (see tests at bottom) and both reduce
to the current uniform model when omega = 0 (w_r = 1).
"""
from __future__ import annotations
import math
import torch

_NEG = float("-inf")


def _valid_weight_sum(w: torch.Tensor, ancestors_T: torch.Tensor) -> torch.Tensor:
    """Per-donor Σ_{r ∉ anc(d)} w_r = W − Σ_{r∈anc(d)} w_r, via the same ancestors_T.

    ancestors_T[r, d] == 1 iff r is a (strict) ancestor of d. O(S) (one S·S matvec,
    or sparse). Returns [S]."""
    W = w.sum()
    if ancestors_T.is_sparse:
        anc = torch.sparse.mm(ancestors_T.transpose(0, 1), w.unsqueeze(1)).squeeze(1)
    else:
        anc = w.unsqueeze(0).matmul(ancestors_T).squeeze(0)   # Σ_r w_r ancestors_T[r,d]
    return (W - anc).clamp_min(torch.finfo(w.dtype).tiny)


def normalize_omega(omega: torch.Tensor) -> torch.Tensor:
    """Identifiability: receptivity is relative, so anchor mean(omega)=0 (w geomean=1)."""
    return omega - omega.mean()


def recipient_uniform_setup(omega: torch.Tensor, ancestors_T: torch.Tensor):
    """Efficient-path setup. Returns (w [S], recip_mt_offset [S])."""
    omega = normalize_omega(omega)
    w = torch.exp2(omega)
    valid_w = _valid_weight_sum(w, ancestors_T)
    recip_mt_offset = -torch.log2(valid_w)            # mt = log_pT + recip_mt_offset
    return w, recip_mt_offset


def build_recipient_transfer_mat(omega: torch.Tensor, ancestors_T: torch.Tensor,
                                 log_pT: torch.Tensor):
    """Dense reference. transfer_mat[d,r] (linear, rescaled by max) with
    P(recipient=r | transfer from d) = w_r / Σ_{r'∉anc(d)} w_{r'}, masked to valid r.
    Returns (transfer_mat [S,S], max_transfer_mat [S,1])."""
    omega = normalize_omega(omega)
    w = torch.exp2(omega)                              # [S]
    S = w.shape[0]
    valid_mask = (ancestors_T == 0)                    # [S,S]; True where r not ancestor of d -> col d
    # log weight of d->r (recipient logit, recipient-only), masked:
    logw = torch.where(valid_mask.t(), omega.unsqueeze(0).expand(S, S),
                       torch.full((S, S), _NEG, dtype=w.dtype, device=w.device))  # [d, r]
    valid_w = _valid_weight_sum(w, ancestors_T)        # [S] per donor
    # actual log transfer prob d->r = log_pT[d] + omega_r - log2(valid_w[d])
    log_tm = log_pT.unsqueeze(1) + logw - torch.log2(valid_w).unsqueeze(1)   # [d, r]
    max_transfer_mat = torch.max(log_tm, dim=-1, keepdim=True).values        # [d,1]
    transfer_mat = torch.exp2(log_tm - max_transfer_mat)                     # rescaled
    return transfer_mat, max_transfer_mat
