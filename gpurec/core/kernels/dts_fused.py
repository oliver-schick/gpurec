"""Fused DTS computation: gather + 5 terms + logsumexp in one Triton kernel."""

import torch
import triton
import triton.language as tl


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


@triton.jit
def _dts_fused_kernel(
    # Full Pi and Pibar tensors: [C, S]
    Pi_ptr, Pibar_ptr,
    # Split left/right child indices: [N] each (clade indices into Pi)
    lefts_ptr, rights_ptr,
    # Species child indices: [S] each (S = sentinel for "no child")
    sp_child1_ptr, sp_child2_ptr,
    # Parameters: [S], [N], [N, 1], or [N, S] log-probabilities
    log_pD_ptr,
    log_pS_ptr,
    # Split probs: [N]
    log_split_probs_ptr,
    # Optional parent-row active mask for wave-local reduce_idx[n]
    reduce_idx_ptr,
    active_mask_ptr,
    # Output: [N, S]
    out_ptr,
    # Dimensions
    N: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    # Param modes: 0 = shared [S], 1 = per-split [N, S], 2 = per-split scalar [N] / [N, 1]
    mode_pD: tl.constexpr = 0,
    mode_pS: tl.constexpr = 0,
    USE_ACTIVE_MASK: tl.constexpr = False,
    DTYPE: tl.constexpr = tl.float32,
):
    """Compute DTS_term[i, s] = log_split_probs[i] + logsumexp2(5 DTS terms)[s].

    Fuses the gather of Pi[left], Pi[right], Pibar[left], Pibar[right]
    directly from the full [C, S] tensors, avoiding materialization of
    [N, S] intermediates and [N, S+1] padded tensors.
    """
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    # Load left/right clade indices for this split
    left_idx = tl.load(lefts_ptr + n)
    right_idx = tl.load(rights_ptr + n)
    if USE_ACTIVE_MASK:
        parent_w = tl.load(reduce_idx_ptr + n)
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            out_base = n.to(tl.int64) * S
            tl.store(out_ptr + out_base + s_offs,
                     tl.full([BLOCK_S], value=-1e30, dtype=DTYPE),
                     mask=mask)
            return

    # int64 offsets: clade/split indices are stored int32 (bandwidth), but the
    # flattened address idx*S into [C,S]/[N,S] overflows int32 once C*S or N*S
    # exceeds 2^31 (e.g. Davin S=2013 with ~1M-clade batches). Promote only the
    # multiply; s_offs (int32) then promotes to int64 in the add.
    base_l = left_idx.to(tl.int64) * S
    base_r = right_idx.to(tl.int64) * S

    # Load Pi/Pibar for left and right children directly from [C, S] tensors
    pi_l = tl.load(Pi_ptr + base_l + s_offs, mask=mask, other=-1e30)
    pi_r = tl.load(Pi_ptr + base_r + s_offs, mask=mask, other=-1e30)
    pibar_l = tl.load(Pibar_ptr + base_l + s_offs, mask=mask, other=-1e30)
    pibar_r = tl.load(Pibar_ptr + base_r + s_offs, mask=mask, other=-1e30)

    # Load parameters.  Backward often has per-split scalars as [N, 1]; avoid
    # expanding them to [N, S] just to satisfy the kernel.
    if mode_pD == 2:
        log_pD_s = tl.load(log_pD_ptr + n)
    elif mode_pD == 1:
        log_pD_s = tl.load(log_pD_ptr + n.to(tl.int64) * S + s_offs, mask=mask, other=-1e30)
    else:
        log_pD_s = tl.load(log_pD_ptr + s_offs, mask=mask, other=-1e30)

    if mode_pS == 2:
        log_pS_s = tl.load(log_pS_ptr + n)
    elif mode_pS == 1:
        log_pS_s = tl.load(log_pS_ptr + n.to(tl.int64) * S + s_offs, mask=mask, other=-1e30)
    else:
        log_pS_s = tl.load(log_pS_ptr + s_offs, mask=mask, other=-1e30)

    # Compute first 3 DTS terms
    t0 = log_pD_s + pi_l + pi_r                          # D
    t1 = pi_l + pibar_r                                  # T (l->r)
    t2 = pi_r + pibar_l                                  # T (r->l)

    # Species children for S terms: load from Pi directly with sentinel check
    c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=S)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=S)
    c1_valid = c1 < S
    c2_valid = c2 < S

    # Load Pi[left, c1], Pi[right, c2], etc. with sentinel masking
    # When c1==S (leaf species), result is -inf (no speciation possible)
    pi_l_c1 = tl.load(Pi_ptr + base_l + c1, mask=mask & c1_valid, other=-1e30)
    pi_r_c2 = tl.load(Pi_ptr + base_r + c2, mask=mask & c2_valid, other=-1e30)
    pi_r_c1 = tl.load(Pi_ptr + base_r + c1, mask=mask & c1_valid, other=-1e30)
    pi_l_c2 = tl.load(Pi_ptr + base_l + c2, mask=mask & c2_valid, other=-1e30)

    t3 = log_pS_s + pi_l_c1 + pi_r_c2                    # S
    t4 = log_pS_s + pi_r_c1 + pi_l_c2                    # S (swapped)

    # Fused logsumexp2 over 5 terms
    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m_safe = tl.where(m > -1e29, m, tl.zeros_like(m))

    s = (tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
         + tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe))

    # Add log_split_probs
    lsp = tl.load(log_split_probs_ptr + n)
    result = tl.log2(s) + m + lsp

    out_base = n.to(tl.int64) * S
    tl.store(out_ptr + out_base + s_offs, result, mask=mask)


def dts_fused(Pi, Pibar, lefts, rights,
              sp_child1, sp_child2,
              log_pD, log_pS, log_split_probs,
              out=None, active_mask=None, reduce_idx=None):
    """Fused DTS: gather + 5 terms + logsumexp + split_probs in one kernel.

    Args:
        Pi: [C, S] contiguous — full Pi tensor
        Pibar: [C, S] contiguous — full Pibar tensor
        lefts: [N] long — left child clade indices per split
        rights: [N] long — right child clade indices per split
        sp_child1, sp_child2: [S] long — species tree child indices (S=sentinel)
        log_pD, log_pS: scalar, [S], [N], [N, 1], or [N, S] event probabilities
        log_split_probs: [N, 1] or [N]
        out: optional [N, S] output buffer

    Returns:
        DTS_term: [N, S] = log_split_probs + logsumexp2(5 DTS terms)
    """
    N = lefts.shape[0]
    S = Pi.shape[1]
    if out is None:
        out = torch.empty((N, S), device=Pi.device, dtype=Pi.dtype)
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    # Flatten log_split_probs to [N]
    lsp = log_split_probs.reshape(N).contiguous()

    def _prepare_param(p):
        if p.dim() == 0:
            return p.expand(S).contiguous(), 0
        if p.dim() == 1:
            if p.numel() == S:
                return p.contiguous(), 0
            if p.numel() == N:
                return p.contiguous(), 2
        if p.dim() == 2:
            if p.shape[0] != N:
                raise ValueError(f"per-split parameter first dim must be N={N}, got {tuple(p.shape)}")
            if p.shape[1] == 1:
                return p.reshape(N).contiguous(), 2
            if p.shape[1] == S:
                return p.contiguous(), 1
        raise ValueError(
            "DTS parameters must be scalar, [S], [N], [N, 1], or [N, S]; "
            f"got shape {tuple(p.shape)} with N={N}, S={S}"
        )

    log_pD_vec, mode_pD = _prepare_param(log_pD)
    log_pS_vec, mode_pS = _prepare_param(log_pS)

    BLOCK_S = 128
    grid = (N, (S + BLOCK_S - 1) // BLOCK_S)

    _dts_fused_kernel[grid](
        Pi.contiguous(), Pibar.contiguous(),
        lefts, rights,
        sp_child1, sp_child2,
        log_pD_vec, log_pS_vec,
        lsp,
        reduce_idx if reduce_idx is not None else lefts,
        active_mask if active_mask is not None else lefts,
        out,
        N, S,
        BLOCK_S=BLOCK_S,
        mode_pD=mode_pD,
        mode_pS=mode_pS,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DTYPE=_tl_float_dtype(Pi.dtype),
    )
    return out
