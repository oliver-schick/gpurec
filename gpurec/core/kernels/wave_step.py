"""Fused Triton kernels for wave-step computation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _wave_pibar_step_kernel(
    # Pi[wt]: [W, S] log2-space (input, the current wave's Pi)
    Pi_W_ptr,
    # Transfer matrix: [S, S] (M[e, f]) — NOT transposed
    transfer_mat_ptr,
    # max_transfer_mat: [S] (mt[e])
    mt_ptr,
    # Precomputed constants: [S] each
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    # Species child indices: [S] each (index into columns, S = padding/leaf)
    sp_child1_ptr, sp_child2_ptr,
    # Leaf term: [W, S]
    leaf_term_ptr,
    # DTS_reduced: [W, S] or None (if no splits)
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    # Outputs: Pi_new [W, S], Pibar [W, S]
    Pi_new_ptr, Pibar_ptr,
    # Dimensions
    S: tl.constexpr,
    # Strides
    stride_w: tl.constexpr,       # stride for [W, S] tensors (== S)
    stride_m_row: tl.constexpr,   # stride for transfer_mat rows (== S)
    # Block sizes
    BLOCK_S: tl.constexpr,
    BLOCK_F: tl.constexpr,  # tile size for reduction over f in matmul
    FP64: tl.constexpr = False,
    stride_c: tl.constexpr = 0,
):
    """Fully fused kernel: Pibar computation + DTS_L terms + logsumexp.

    Each program instance handles one (wave_clade, species_block) pair.
    For each output species e in the block:
      1. Compute Pibar[w,e] = log2(sum_f exp2(Pi[w,f] - max_f) * M[e,f]) + max_f + mt[e]
      2. Compute 6 DTS_L terms using Pi[w,e], Pibar[w,e], children
      3. logsumexp all terms (+ DTS_reduced if present)
    """
    DTYPE = tl.float64 if FP64 else tl.float32
    NEG_LARGE = -1e300 if FP64 else -1e30

    w = tl.program_id(0).to(tl.int64)
    s_block = tl.program_id(1)

    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    base_w = w * stride_w

    # --- Step 1: Compute Pibar[w, s_offs] via log-space matmul ---
    # First pass: find max over f for stabilization
    pi_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    for f_start in range(0, S, BLOCK_F):
        f_offs = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offs < S
        pi_f = tl.load(Pi_W_ptr + base_w + f_offs, mask=f_mask, other=NEG_LARGE)
        pi_max = tl.maximum(pi_max, tl.max(pi_f, axis=0))

    # Second pass: accumulate exp2(pi_f - max) * M[e, f] for each e in s_offs
    acc = tl.zeros([BLOCK_S], dtype=DTYPE)
    for f_start in range(0, S, BLOCK_F):
        f_offs = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offs < S
        pi_f = tl.load(Pi_W_ptr + base_w + f_offs, mask=f_mask, other=NEG_LARGE)
        exp_f = tl.exp2(pi_f - pi_max)  # [BLOCK_F]

        # Load M[s_offs, f_offs] — transfer_mat[e, f]
        # M has shape [S, S], stride_m_row = S
        # For each e in s_offs, load M[e, f_offs]
        m_ptrs = transfer_mat_ptr + s_offs[:, None] * stride_m_row + f_offs[None, :]
        m_mask = mask[:, None] & f_mask[None, :]
        m_vals = tl.load(m_ptrs, mask=m_mask, other=0.0)  # [BLOCK_S, BLOCK_F]

        acc += tl.sum(m_vals * exp_f[None, :], axis=1)  # [BLOCK_S]

    mt = tl.load(mt_ptr + w * stride_c + s_offs, mask=mask, other=0.0)
    pibar_w = tl.log2(acc) + pi_max + mt  # [BLOCK_S]

    # Store Pibar
    tl.store(Pibar_ptr + base_w + s_offs, pibar_w, mask=mask)

    # --- Step 2: Load Pi[w, s_offs] and children ---
    pi_w = tl.load(Pi_W_ptr + base_w + s_offs, mask=mask, other=NEG_LARGE)

    # Load constants (stride_c=0 → shared [S], stride_c=S → per-clade [W,S])
    dl_const = tl.load(DL_const_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
    ebar = tl.load(Ebar_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
    e_val = tl.load(E_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
    sl1_const = tl.load(SL1_const_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
    sl2_const = tl.load(SL2_const_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)

    # Gather species children
    c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(Pi_W_ptr + base_w + c1, mask=mask & c1_valid, other=NEG_LARGE)
    pi_s2 = tl.load(Pi_W_ptr + base_w + c2, mask=mask & c2_valid, other=NEG_LARGE)

    # --- Step 3: 6 DTS_L terms + logsumexp ---
    t0 = dl_const + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_const + pi_s1
    t4 = sl2_const + pi_s2
    t5 = tl.load(leaf_term_ptr + base_w + s_offs, mask=mask, other=NEG_LARGE)

    M_SAFE_THRESH = -1e299 if FP64 else -1e29
    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)

    if has_splits:
        dts_r = tl.load(DTS_reduced_ptr + base_w + s_offs, mask=mask, other=NEG_LARGE)
        m = tl.maximum(m, dts_r)

    m_safe = tl.where(m > M_SAFE_THRESH, m, tl.zeros_like(m))
    s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
    s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
    if has_splits:
        s += tl.exp2(dts_r - m_safe)

    result = tl.log2(s) + m
    tl.store(Pi_new_ptr + base_w + s_offs, result, mask=mask)


def wave_pibar_step_fused(Pi_W, transfer_mat, mt_squeezed,
                          DL_const, Ebar, E, SL1_const, SL2_const,
                          sp_child1, sp_child2, leaf_term_wt,
                          Pibar_out, DTS_reduced=None):
    """Fully fused wave step: Pibar + DTS_L + logsumexp in one kernel.

    Args:
        Pi_W: [W, S] contiguous, log2-space
        transfer_mat: [S, S] contiguous
        mt_squeezed: [S] or [W, S]
        DL_const, Ebar, E, SL1_const, SL2_const: [S] or [W, S] each
        sp_child1, sp_child2: [S] long
        leaf_term_wt: [W, S] contiguous
        Pibar_out: [W, S] output buffer for Pibar (written to)
        DTS_reduced: [W, S] or None

    Returns:
        Pi_new: [W, S] log2-space
    """
    W, S = Pi_W.shape
    Pi_new = torch.empty_like(Pi_W)
    has_splits = DTS_reduced is not None
    fp64 = Pi_W.dtype == torch.float64
    stride_c = S if DL_const.ndim == 2 else 0

    BLOCK_S = 32
    BLOCK_F = min(64, triton.next_power_of_2(S))
    grid = (W, (S + BLOCK_S - 1) // BLOCK_S)

    _wave_pibar_step_kernel[grid](
        Pi_W,
        transfer_mat,
        mt_squeezed,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        leaf_term_wt,
        DTS_reduced if has_splits else Pi_W,  # dummy
        has_splits,
        Pi_new, Pibar_out,
        S,
        stride_w=S,
        stride_m_row=S,
        BLOCK_S=BLOCK_S,
        BLOCK_F=BLOCK_F,
        FP64=fp64,
        stride_c=stride_c,
    )
    return Pi_new


# Keep the old kernel for compatibility
@triton.jit
def _wave_step_kernel(
    Pi_W_ptr, Pibar_W_ptr,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    sp_child1_ptr, sp_child2_ptr,
    leaf_term_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    Pi_new_ptr,
    W: tl.constexpr, S: tl.constexpr,
    stride_ws: tl.constexpr,
    stride_c: tl.constexpr = 0,
):
    """Fused kernel: given Pi_W, Pibar_W, compute Pi_new = logsumexp2(all_terms, dim=0)."""
    w = tl.program_id(0).to(tl.int64)
    s_block = tl.program_id(1)
    BLOCK_S: tl.constexpr = 32

    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    base = w * stride_ws
    pi_w = tl.load(Pi_W_ptr + base + s_offs, mask=mask, other=-1e30)
    pibar_w = tl.load(Pibar_W_ptr + base + s_offs, mask=mask, other=-1e30)

    dl_const = tl.load(DL_const_ptr + w * stride_c + s_offs, mask=mask, other=-1e30)
    ebar = tl.load(Ebar_ptr + w * stride_c + s_offs, mask=mask, other=-1e30)
    e_val = tl.load(E_ptr + w * stride_c + s_offs, mask=mask, other=-1e30)
    sl1_const = tl.load(SL1_const_ptr + w * stride_c + s_offs, mask=mask, other=-1e30)
    sl2_const = tl.load(SL2_const_ptr + w * stride_c + s_offs, mask=mask, other=-1e30)

    c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(Pi_W_ptr + base + c1, mask=mask & c1_valid, other=-1e30)
    pi_s2 = tl.load(Pi_W_ptr + base + c2, mask=mask & c2_valid, other=-1e30)

    t0 = dl_const + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_const + pi_s1
    t4 = sl2_const + pi_s2
    t5 = tl.load(leaf_term_ptr + base + s_offs, mask=mask, other=-1e30)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)

    if has_splits:
        dts_r = tl.load(DTS_reduced_ptr + w * stride_ws + s_offs, mask=mask, other=-1e30)
        m = tl.maximum(m, dts_r)

    m_safe = tl.where(m > -1e29, m, tl.zeros_like(m))
    s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
    s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
    if has_splits:
        s += tl.exp2(dts_r - m_safe)

    result = tl.log2(s) + m
    tl.store(Pi_new_ptr + base + s_offs, result, mask=mask)


def wave_step_fused(Pi_W, Pibar_W, DL_const, Ebar, E, SL1_const, SL2_const,
                    sp_child1, sp_child2, leaf_term_wt, DTS_reduced=None):
    """Fused wave step without Pibar computation."""
    W, S = Pi_W.shape
    Pi_new = torch.empty_like(Pi_W)
    has_splits = DTS_reduced is not None
    stride_c = S if DL_const.ndim == 2 else 0

    BLOCK_S = 32
    grid = (W, (S + BLOCK_S - 1) // BLOCK_S)

    _wave_step_kernel[grid](
        Pi_W, Pibar_W,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        leaf_term_wt,
        DTS_reduced if has_splits else Pi_W,
        has_splits,
        Pi_new,
        W, S,
        stride_ws=S,
        stride_c=stride_c,
    )
    return Pi_new


# --- Fused uniform-Pibar kernel ---
# One program per clade row. Two passes:
#   Pass 1: online max+sum over Pi row (for uniform Pibar stats)
#   Pass 2: compute Pibar inline, DTS_L terms, logsumexp, convergence diff

@triton.jit
def _wave_step_uniform_kernel(
    # Global Pi tensor [C, S] — read from rows [ws : ws+W]
    Pi_ptr,
    ws,                  # wave start (clade offset)
    # Constants: [S] each
    mt_ptr,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    # Species child indices: [S] long each
    sp_child1_ptr, sp_child2_ptr,
    # Species parent index: [S] long, -1 for root
    sp_parent_ptr,
    # Padded ancestor-list table: [MAX_ANCESTOR_DEPTH, S] long
    ancestor_cols_ptr,
    # CSR ancestor matrix: row = descendant species, cols = ancestors
    ancestor_csr_indptr_ptr,
    ancestor_csr_indices_ptr,
    # Per-wave arrays: [W, S]
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    # Outputs
    Pi_new_ptr,          # [W, S]
    Pibar_out_ptr,       # [C, S] — write Pibar to rows [ws : ws+W]
    max_diff_ptr,        # [W] — per-row max |Pi_new - Pi_old| for convergence
    pibar_row_max_ptr,   # optional [C] — final row max for backward Pibar VJP
    # Dimensions
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_ANCESTOR_CSR: tl.constexpr,
    USE_ANCESTOR_COLS: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_PIBAR: tl.constexpr,
    STORE_PIBAR_ROW_MAX: tl.constexpr,
    OUTPUT_GLOBAL: tl.constexpr,
    FP64: tl.constexpr,
    stride_c: tl.constexpr = 0,
):
    """Fused kernel: uniform Pibar + DTS_L + logsumexp + convergence diff.

    Each program handles one full clade row, processing S elements in tiles.
    Pass 1 uses the online max+sum trick (single scan) for row statistics.
    Pass 2 computes Pibar inline and all DTS_L terms in one scan.

    stride_c: 0 = shared [S] constants, S = per-clade [W, S] constants.
    """
    DTYPE = tl.float64 if FP64 else tl.float32
    NEG_LARGE = -1e300 if FP64 else -1e30

    w = tl.program_id(0).to(tl.int64)
    # int64: (ws+w) is a GLOBAL clade index (up to C), so (ws+w)*S overflows int32
    # once C*S > 2^31 (Davin S=2013, ~1M-clade batches). Promote at the multiply;
    # index tensors stay int32. The [W,S] out_base (w<W, one wave) can't overflow.
    pi_base = (ws + w).to(tl.int64) * stride    # offset into global Pi/Pibar
    if OUTPUT_GLOBAL:
        out_base = pi_base            # offset into global output rows
    else:
        out_base = w * stride         # offset into [W, S] outputs

    # === Pass 1: Online max + sum over the Pi row ===
    row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        tile_max = tl.max(pi_val, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        # Rescale running sum to new max, add this tile's contribution
        row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
        row_max = new_max

    if STORE_PIBAR_ROW_MAX:
        tl.store(pibar_row_max_ptr + ws + w, tl.max(row_max, axis=0))

    # === Pass 2: Pibar + DTS_L terms + logsumexp ===
    if COMPUTE_DIFF:
        local_max_diff = tl.full([1], value=0.0, dtype=DTYPE)
    M_SAFE_THRESH = -1e299 if FP64 else -1e29

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        # Load Pi[w, s]
        pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)

        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        if USE_ANCESTOR_CSR:
            # Uniform Pibar with CSR ancestor rows. This is the fused version
            # of the sparse ancestor matmul: it reuses row_max/row_sum already
            # computed by this wave-step kernel and only performs the sparse
            # ancestor correction inline.
            row_start = tl.load(ancestor_csr_indptr_ptr + s_offs, mask=mask, other=0)
            row_end = tl.load(ancestor_csr_indptr_ptr + s_offs + 1, mask=mask, other=0)
            for k in range(0, MAX_ANCESTOR_DEPTH):
                pos = row_start + k
                anc_valid = mask & (pos < row_end)
                anc = tl.load(ancestor_csr_indices_ptr + pos, mask=anc_valid, other=-1)
                pi_anc = tl.load(Pi_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
                ancestor_sum += tl.where(anc_valid, tl.exp2(pi_anc - row_max), tl.zeros([BLOCK_S], dtype=DTYPE))
        elif USE_ANCESTOR_COLS:
            # Uniform Pibar with a precomputed [depth, species] ancestor table.
            # This avoids the loop-carried dependency of following parent
            # pointers, at the cost of one extra static index tensor.
            for k in range(0, MAX_ANCESTOR_DEPTH):
                anc = tl.load(ancestor_cols_ptr + k.to(tl.int64) * S + s_offs, mask=mask, other=-1)
                anc_valid = mask & (anc >= 0) & (anc < S)
                pi_anc = tl.load(Pi_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
                ancestor_sum += tl.where(anc_valid, tl.exp2(pi_anc - row_max), tl.zeros([BLOCK_S], dtype=DTYPE))
        else:
            # Uniform Pibar: log2(row_sum - ancestor_sum) + max + mt.
            # ancestors_dense[descendant, ancestor] includes self, so walk the
            # species parent chain starting at s and sum exp2(Pi[ancestor] - max).
            cur = s_offs.to(tl.int64)
            for _ in range(0, MAX_ANCESTOR_DEPTH):
                cur_valid = mask & (cur >= 0) & (cur < S)
                pi_anc = tl.load(Pi_ptr + pi_base + cur, mask=cur_valid, other=NEG_LARGE)
                ancestor_sum += tl.where(cur_valid, tl.exp2(pi_anc - row_max), tl.zeros([BLOCK_S], dtype=DTYPE))
                cur = tl.load(sp_parent_ptr + cur, mask=cur_valid, other=-1)

        mt = tl.load(mt_ptr + w * stride_c + s_offs, mask=mask, other=0.0)
        denom = row_sum - ancestor_sum
        pibar_w = tl.where(denom > 0.0, tl.log2(denom) + row_max + mt, NEG_LARGE)

        # Store Pibar to global tensor when this invocation is producing the
        # final Pibar rows. Fixed-iteration ping-pong uses Pibar as Pi scratch
        # and recomputes/stores final Pibar after the last iteration.
        if STORE_PIBAR:
            tl.store(Pibar_out_ptr + pi_base + s_offs, pibar_w, mask=mask)

        # Load constants (stride_c=0 → shared [S], stride_c=S → per-clade [W,S])
        dl_const = tl.load(DL_const_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
        ebar = tl.load(Ebar_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
        e_val = tl.load(E_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
        sl1_const = tl.load(SL1_const_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
        sl2_const = tl.load(SL2_const_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)

        # Gather species children
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = c1 < S
        c2_valid = c2 < S
        pi_s1 = tl.load(Pi_ptr + pi_base + c1, mask=mask & c1_valid, other=NEG_LARGE)
        pi_s2 = tl.load(Pi_ptr + pi_base + c2, mask=mask & c2_valid, other=NEG_LARGE)

        # 6 DTS_L terms
        t0 = dl_const + pi_w
        t1 = pi_w + ebar
        t2 = pibar_w + e_val
        t3 = sl1_const + pi_s1
        t4 = sl2_const + pi_s2
        if USE_LEAF_INDEX:
            leaf_species = tl.load(leaf_species_ptr + ws + w)
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=mask, other=NEG_LARGE)
            t5 = tl.where(mask & (leaf_species == s_offs), leaf_logp, NEG_LARGE)
        else:
            t5 = tl.load(leaf_term_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)

        m = tl.maximum(t0, t1)
        m = tl.maximum(m, t2)
        m = tl.maximum(m, t3)
        m = tl.maximum(m, t4)
        m = tl.maximum(m, t5)

        if has_splits:
            dts_r = tl.load(DTS_reduced_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)
            m = tl.maximum(m, dts_r)

        m_safe = tl.where(m > M_SAFE_THRESH, m, tl.zeros_like(m))
        s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
        s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
        if has_splits:
            s += tl.exp2(dts_r - m_safe)

        result = tl.log2(s) + m
        tl.store(Pi_new_ptr + out_base + s_offs, result, mask=mask)

        if COMPUTE_DIFF:
            significant = result > -100.0
            diff = tl.where(significant & mask, tl.abs(result - pi_w), tl.zeros_like(result))
            local_max_diff = tl.maximum(local_max_diff, tl.max(diff, axis=0))

    if COMPUTE_DIFF:
        tl.store(max_diff_ptr + w, tl.max(local_max_diff, axis=0))


def wave_step_uniform_fused(Pi, Pibar, ws, W, S,
                            mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                            sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                            leaf_term_wt, DTS_reduced=None, compute_diff=True,
                            leaf_species_idx=None, leaf_logp=None,
                            pibar_row_max=None):
    """Fused uniform-Pibar + wave step + convergence in one kernel.

    Computes Pibar inline using the uniform transfer matrix approximation,
    then DTS_L terms and logsumexp, plus per-row convergence diff.
    Single kernel launch per iteration, eliminating the [W,S] Pibar intermediate.

    Args:
        Pi: [C, S] global Pi tensor (reads rows [ws:ws+W])
        Pibar: [C, S] global Pibar tensor (writes rows [ws:ws+W])
        ws: wave start index
        W: wave size (number of clades)
        S: number of species
        mt_squeezed: [S] or [W, S] max_transfer_mat
        DL_const, Ebar, E, SL1_const, SL2_const: [S] or [W, S] precomputed constants
        sp_child1, sp_child2: [S] long species child indices
        sp_parent: [S] long species parent indices (-1 at root)
        max_ancestor_depth: maximum root-path length including self
        leaf_term_wt: [W, S]
        DTS_reduced: [W, S] or None

    Returns:
        Pi_new: [W, S] new Pi values
        max_diff: scalar tensor, max |Pi_new - Pi_old| across significant entries
                  when compute_diff=True; otherwise None
    """
    fp64 = Pi.dtype == torch.float64
    Pi_new = torch.empty((W, S), dtype=Pi.dtype, device=Pi.device)
    max_diff_buf = torch.empty(W, dtype=Pi.dtype, device=Pi.device) if compute_diff else Pi_new
    has_splits = DTS_reduced is not None
    stride_c = S if DL_const.ndim == 2 else 0
    use_leaf_index = leaf_species_idx is not None and leaf_logp is not None
    leaf_species_arg = leaf_species_idx if use_leaf_index else sp_parent
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt

    BLOCK_S = min(256, triton.next_power_of_2(S))
    grid = (W,)

    _wave_step_uniform_kernel[grid](
        Pi, ws,
        mt_squeezed,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        sp_parent,
        sp_parent,  # dummy ancestor table; disabled by USE_ANCESTOR_COLS=False
        sp_parent,  # dummy CSR indptr; disabled by USE_ANCESTOR_CSR=False
        sp_parent,  # dummy CSR indices; disabled by USE_ANCESTOR_CSR=False
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        DTS_reduced if has_splits else leaf_term_wt,  # dummy
        has_splits,
        Pi_new, Pibar, max_diff_buf,
        pibar_row_max if pibar_row_max is not None else Pibar,
        S,
        stride=S,
        BLOCK_S=BLOCK_S,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        COMPUTE_DIFF=bool(compute_diff),
        USE_ANCESTOR_CSR=False,
        USE_ANCESTOR_COLS=False,
        USE_LEAF_INDEX=use_leaf_index,
        STORE_PIBAR=True,
        STORE_PIBAR_ROW_MAX=bool(pibar_row_max is not None),
        OUTPUT_GLOBAL=False,
        FP64=fp64,
        stride_c=stride_c,
        num_warps=4,
    )
    max_diff = max_diff_buf.max() if compute_diff else None
    return Pi_new, max_diff


def wave_step_uniform_fused_into(Pi_in, Pi_out, Pibar, ws, W, S,
                                 mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                                 sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                                 leaf_term_wt, DTS_reduced=None,
                                 leaf_species_idx=None, leaf_logp=None):
    """Fused uniform wave step writing Pi output directly into global rows."""
    fp64 = Pi_in.dtype == torch.float64
    has_splits = DTS_reduced is not None
    stride_c = S if DL_const.ndim == 2 else 0
    use_leaf_index = leaf_species_idx is not None and leaf_logp is not None
    leaf_species_arg = leaf_species_idx if use_leaf_index else sp_parent
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt
    max_diff_buf = Pi_out

    BLOCK_S = min(256, triton.next_power_of_2(S))
    grid = (W,)
    Pi_out_rows = Pi_out.narrow(0, int(ws), int(W))

    _wave_step_uniform_kernel[grid](
        Pi_in, ws,
        mt_squeezed,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        sp_parent,
        sp_parent,
        sp_parent,
        sp_parent,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        DTS_reduced if has_splits else leaf_term_wt,
        has_splits,
        Pi_out_rows, Pibar, max_diff_buf, Pibar,
        S,
        stride=S,
        BLOCK_S=BLOCK_S,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        COMPUTE_DIFF=False,
        USE_ANCESTOR_CSR=False,
        USE_ANCESTOR_COLS=False,
        USE_LEAF_INDEX=use_leaf_index,
        STORE_PIBAR=False,
        STORE_PIBAR_ROW_MAX=False,
        OUTPUT_GLOBAL=False,
        FP64=fp64,
        stride_c=stride_c,
        num_warps=4,
    )


def wave_step_uniform_ancestor_fused(Pi, Pibar, ws, W, S,
                                    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                                     sp_child1, sp_child2, ancestor_cols,
                                     leaf_term_wt, DTS_reduced=None, compute_diff=True,
                                     leaf_species_idx=None, leaf_logp=None,
                                     pibar_row_max=None):
    """Fused uniform-Pibar + wave step using precomputed ancestor lists.

    This is equivalent to :func:`wave_step_uniform_fused`, but uses a padded
    [max_depth, S] ancestor table instead of following parent pointers inside
    the kernel.
    """
    fp64 = Pi.dtype == torch.float64
    Pi_new = torch.empty((W, S), dtype=Pi.dtype, device=Pi.device)
    max_diff_buf = torch.empty(W, dtype=Pi.dtype, device=Pi.device) if compute_diff else Pi_new
    has_splits = DTS_reduced is not None
    stride_c = S if DL_const.ndim == 2 else 0
    use_leaf_index = leaf_species_idx is not None and leaf_logp is not None
    leaf_species_arg = leaf_species_idx if use_leaf_index else ancestor_cols
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt

    BLOCK_S = min(256, triton.next_power_of_2(S))
    grid = (W,)

    _wave_step_uniform_kernel[grid](
        Pi, ws,
        mt_squeezed,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        ancestor_cols,  # dummy parent table; disabled by USE_ANCESTOR_COLS=True
        ancestor_cols,
        ancestor_cols,  # dummy CSR indptr; disabled by USE_ANCESTOR_CSR=False
        ancestor_cols,  # dummy CSR indices; disabled by USE_ANCESTOR_CSR=False
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        DTS_reduced if has_splits else leaf_term_wt,  # dummy
        has_splits,
        Pi_new, Pibar, max_diff_buf,
        pibar_row_max if pibar_row_max is not None else Pibar,
        S,
        stride=S,
        BLOCK_S=BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        COMPUTE_DIFF=bool(compute_diff),
        USE_ANCESTOR_CSR=False,
        USE_ANCESTOR_COLS=True,
        USE_LEAF_INDEX=use_leaf_index,
        STORE_PIBAR=True,
        STORE_PIBAR_ROW_MAX=bool(pibar_row_max is not None),
        OUTPUT_GLOBAL=False,
        FP64=fp64,
        stride_c=stride_c,
        num_warps=4,
    )
    max_diff = max_diff_buf.max() if compute_diff else None
    return Pi_new, max_diff


def wave_step_uniform_csr_fused(Pi, Pibar, ws, W, S,
                                mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                                sp_child1, sp_child2, ancestor_csr_indptr,
                                ancestor_csr_indices, max_ancestor_depth,
                                leaf_term_wt, DTS_reduced=None, compute_diff=True,
                                leaf_species_idx=None, leaf_logp=None,
                                pibar_row_max=None):
    """Fused uniform-Pibar + wave step using CSR ancestor rows.

    This is the hand-written Triton sparse-ancestor path. It keeps the existing
    fused row max, row sum, log/subtract, and DTS computations, and only changes
    the ancestor correction to read CSR rows directly.
    """
    fp64 = Pi.dtype == torch.float64
    Pi_new = torch.empty((W, S), dtype=Pi.dtype, device=Pi.device)
    max_diff_buf = torch.empty(W, dtype=Pi.dtype, device=Pi.device) if compute_diff else Pi_new
    has_splits = DTS_reduced is not None
    stride_c = S if DL_const.ndim == 2 else 0
    use_leaf_index = leaf_species_idx is not None and leaf_logp is not None
    leaf_species_arg = leaf_species_idx if use_leaf_index else ancestor_csr_indices
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt

    BLOCK_S = min(256, triton.next_power_of_2(S))
    grid = (W,)

    _wave_step_uniform_kernel[grid](
        Pi, ws,
        mt_squeezed,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        ancestor_csr_indices,  # dummy parent table; disabled by USE_ANCESTOR_CSR=True
        ancestor_csr_indices,  # dummy padded table; disabled by USE_ANCESTOR_COLS=False
        ancestor_csr_indptr,
        ancestor_csr_indices,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        DTS_reduced if has_splits else leaf_term_wt,  # dummy
        has_splits,
        Pi_new, Pibar, max_diff_buf,
        pibar_row_max if pibar_row_max is not None else Pibar,
        S,
        stride=S,
        BLOCK_S=BLOCK_S,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        COMPUTE_DIFF=bool(compute_diff),
        USE_ANCESTOR_CSR=True,
        USE_ANCESTOR_COLS=False,
        USE_LEAF_INDEX=use_leaf_index,
        STORE_PIBAR=True,
        STORE_PIBAR_ROW_MAX=bool(pibar_row_max is not None),
        OUTPUT_GLOBAL=False,
        FP64=fp64,
        stride_c=stride_c,
        num_warps=4,
    )
    max_diff = max_diff_buf.max() if compute_diff else None
    return Pi_new, max_diff


@triton.jit
def _wave_pibar_uniform_parent_kernel(
    Pi_ptr,
    ws,
    mt_ptr,
    sp_parent_ptr,
    Pibar_out_ptr,
    row_max_out_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    STORE_ROW_MAX: tl.constexpr,
    FP64: tl.constexpr,
    stride_c: tl.constexpr = 0,
):
    DTYPE = tl.float64 if FP64 else tl.float32
    NEG_LARGE = -1e300 if FP64 else -1e30

    w = tl.program_id(0).to(tl.int64)
    pi_base = (ws + w) * stride

    row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        tile_max = tl.max(pi_val, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
        row_max = new_max

    if STORE_ROW_MAX:
        tl.store(row_max_out_ptr + ws + w, tl.max(row_max, axis=0))

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        cur = s_offs.to(tl.int64)
        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        for _ in range(0, MAX_ANCESTOR_DEPTH):
            cur_valid = mask & (cur >= 0) & (cur < S)
            pi_anc = tl.load(Pi_ptr + pi_base + cur, mask=cur_valid, other=NEG_LARGE)
            ancestor_sum += tl.where(cur_valid, tl.exp2(pi_anc - row_max), tl.zeros([BLOCK_S], dtype=DTYPE))
            cur = tl.load(sp_parent_ptr + cur, mask=cur_valid, other=-1)

        mt = tl.load(mt_ptr + w * stride_c + s_offs, mask=mask, other=0.0)
        denom = row_sum - ancestor_sum
        pibar_w = tl.where(denom > 0.0, tl.log2(denom) + row_max + mt, NEG_LARGE)
        tl.store(Pibar_out_ptr + pi_base + s_offs, pibar_w, mask=mask)


def wave_pibar_uniform_parent_fused(Pi, Pibar, ws, W, S,
                                    mt_squeezed, sp_parent, max_ancestor_depth,
                                    row_max_out=None):
    """Compute uniform Pibar rows by walking species parent pointers."""
    fp64 = Pi.dtype == torch.float64
    stride_c = S if mt_squeezed.ndim == 2 else 0
    BLOCK_S = min(256, triton.next_power_of_2(S))
    grid = (W,)

    _wave_pibar_uniform_parent_kernel[grid](
        Pi,
        ws,
        mt_squeezed,
        sp_parent,
        Pibar,
        row_max_out if row_max_out is not None else Pibar,
        S,
        stride=S,
        BLOCK_S=BLOCK_S,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        STORE_ROW_MAX=bool(row_max_out is not None),
        FP64=fp64,
        stride_c=stride_c,
        num_warps=4,
    )


@triton.jit
def _wave_step_uniform_from_pibar_kernel(
    Pi_ptr,
    Pibar_ptr,
    ws,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    sp_child1_ptr, sp_child2_ptr,
    leaf_term_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    Pi_new_ptr,
    max_diff_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    FP64: tl.constexpr,
    stride_c: tl.constexpr = 0,
):
    DTYPE = tl.float64 if FP64 else tl.float32
    NEG_LARGE = -1e300 if FP64 else -1e30
    M_SAFE_THRESH = -1e299 if FP64 else -1e29

    w = tl.program_id(0).to(tl.int64)
    pi_base = (ws + w) * stride
    out_base = w * stride

    if COMPUTE_DIFF:
        local_max_diff = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        pibar_w = tl.load(Pibar_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)

        dl_const = tl.load(DL_const_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
        ebar = tl.load(Ebar_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
        e_val = tl.load(E_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
        sl1_const = tl.load(SL1_const_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)
        sl2_const = tl.load(SL2_const_ptr + w * stride_c + s_offs, mask=mask, other=NEG_LARGE)

        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = c1 < S
        c2_valid = c2 < S
        pi_s1 = tl.load(Pi_ptr + pi_base + c1, mask=mask & c1_valid, other=NEG_LARGE)
        pi_s2 = tl.load(Pi_ptr + pi_base + c2, mask=mask & c2_valid, other=NEG_LARGE)

        t0 = dl_const + pi_w
        t1 = pi_w + ebar
        t2 = pibar_w + e_val
        t3 = sl1_const + pi_s1
        t4 = sl2_const + pi_s2
        t5 = tl.load(leaf_term_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)

        m = tl.maximum(t0, t1)
        m = tl.maximum(m, t2)
        m = tl.maximum(m, t3)
        m = tl.maximum(m, t4)
        m = tl.maximum(m, t5)

        if has_splits:
            dts_r = tl.load(DTS_reduced_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)
            m = tl.maximum(m, dts_r)

        m_safe = tl.where(m > M_SAFE_THRESH, m, tl.zeros_like(m))
        acc = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
        acc += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
        if has_splits:
            acc += tl.exp2(dts_r - m_safe)

        result = tl.log2(acc) + m
        tl.store(Pi_new_ptr + out_base + s_offs, result, mask=mask)

        if COMPUTE_DIFF:
            significant = result > -100.0
            diff = tl.where(significant & mask, tl.abs(result - pi_w), tl.zeros_like(result))
            local_max_diff = tl.maximum(local_max_diff, tl.max(diff, axis=0))

    if COMPUTE_DIFF:
        tl.store(max_diff_ptr + w, tl.max(local_max_diff, axis=0))


def wave_step_uniform_from_pibar_fused(Pi, Pibar, ws, W, S,
                                       DL_const, Ebar, E, SL1_const, SL2_const,
                                       sp_child1, sp_child2,
                                       leaf_term_wt, DTS_reduced=None,
                                       compute_diff=True):
    """Compute a uniform-mode DTS_L step from already-stored Pibar rows."""
    fp64 = Pi.dtype == torch.float64
    Pi_new = torch.empty((W, S), dtype=Pi.dtype, device=Pi.device)
    max_diff_buf = torch.empty(W, dtype=Pi.dtype, device=Pi.device) if compute_diff else Pi_new
    has_splits = DTS_reduced is not None
    stride_c = S if DL_const.ndim == 2 else 0
    BLOCK_S = min(256, triton.next_power_of_2(S))
    grid = (W,)

    _wave_step_uniform_from_pibar_kernel[grid](
        Pi,
        Pibar,
        ws,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        leaf_term_wt,
        DTS_reduced if has_splits else leaf_term_wt,
        has_splits,
        Pi_new,
        max_diff_buf,
        S,
        stride=S,
        BLOCK_S=BLOCK_S,
        COMPUTE_DIFF=bool(compute_diff),
        FP64=fp64,
        stride_c=stride_c,
        num_warps=4,
    )
    max_diff = max_diff_buf.max() if compute_diff else None
    return Pi_new, max_diff


def wave_step_uniform_two_kernel_fused(Pi, Pibar, ws, W, S,
                                       mt_squeezed, DL_const, Ebar, E,
                                       SL1_const, SL2_const,
                                       sp_child1, sp_child2,
                                       sp_parent, max_ancestor_depth,
                                       leaf_term_wt, DTS_reduced=None,
                                       compute_diff=True):
    """Uniform Pibar followed by DTS_L update using two Triton launches."""
    wave_pibar_uniform_parent_fused(
        Pi, Pibar, ws, W, S,
        mt_squeezed, sp_parent, max_ancestor_depth,
    )
    return wave_step_uniform_from_pibar_fused(
        Pi, Pibar, ws, W, S,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        leaf_term_wt, DTS_reduced,
        compute_diff=compute_diff,
    )


def build_uniform_linear_operator(DL_const, Ebar, E, SL1_const, SL2_const, mt_squeezed,
                                  sp_parent_cpu, sp_child1_cpu, sp_child2_cpu,
                                  device, dtype):
    """Build a row-scaled signed sparse operator for uniform DTS_L.

    The operator evaluates the Pi-dependent part of DTS_L in linear space:
        row_scale[s] * (v_scaled[s] * sum(p) + M_scaled[s] @ p)
    where p = exp2(Pi - row_max).  The sparse structure is stored in a padded
    ELL layout because species-tree rows are short and fixed.
    """
    S = int(sp_parent_cpu.numel())

    ancestor_lists = []
    max_depth = 0
    for s in range(S):
        cur = s
        ancestors = []
        while cur >= 0:
            ancestors.append(cur)
            cur = int(sp_parent_cpu[cur].item())
        ancestor_lists.append(ancestors)
        max_depth = max(max_depth, len(ancestors))

    max_op_nnz = max_depth + 2
    ancestor_cols_cpu = torch.full((S, max_depth), -1, dtype=torch.long)
    op_cols_cpu = torch.zeros((S, max_op_nnz), dtype=torch.long)
    op_kind_cpu = torch.full((S, max_op_nnz), -1, dtype=torch.int8)

    for s, ancestors in enumerate(ancestor_lists):
        ancestor_cols_cpu[s, :len(ancestors)] = torch.tensor(ancestors, dtype=torch.long)

        slot = 0
        op_cols_cpu[s, slot] = s
        op_kind_cpu[s, slot] = 0  # diag + self ancestor transfer subtraction
        slot += 1

        for anc in ancestors[1:]:
            op_cols_cpu[s, slot] = anc
            op_kind_cpu[s, slot] = 1  # strict ancestor transfer subtraction
            slot += 1

        c1 = int(sp_child1_cpu[s].item())
        if c1 < S:
            op_cols_cpu[s, slot] = c1
            op_kind_cpu[s, slot] = 2
            slot += 1

        c2 = int(sp_child2_cpu[s].item())
        if c2 < S:
            op_cols_cpu[s, slot] = c2
            op_kind_cpu[s, slot] = 3

    op_cols = op_cols_cpu.to(device=device)
    op_kind = op_kind_cpu.to(device=device)
    ancestor_cols = ancestor_cols_cpu.to(device=device)

    transfer_coeff = torch.exp2(E + mt_squeezed)
    diag_coeff = torch.exp2(DL_const) + torch.exp2(Ebar)
    sl1_coeff = torch.exp2(SL1_const)
    sl2_coeff = torch.exp2(SL2_const)

    op_vals = torch.zeros((S, max_op_nnz), device=device, dtype=dtype)
    op_vals = torch.where(op_kind == 0, (diag_coeff - transfer_coeff).unsqueeze(1), op_vals)
    op_vals = torch.where(op_kind == 1, (-transfer_coeff).unsqueeze(1), op_vals)
    op_vals = torch.where(op_kind == 2, sl1_coeff.unsqueeze(1), op_vals)
    op_vals = torch.where(op_kind == 3, sl2_coeff.unsqueeze(1), op_vals)

    max_abs = torch.maximum(op_vals.abs().amax(dim=1), transfer_coeff.abs())
    max_abs = torch.where(max_abs > 0.0, max_abs, torch.ones_like(max_abs))
    row_scale = torch.log2(max_abs)
    op_vals_scaled = (op_vals / max_abs.unsqueeze(1)).contiguous()
    v_scaled = (transfer_coeff / max_abs).contiguous()

    return {
        'op_cols': op_cols.T.contiguous(),
        'op_vals': op_vals_scaled.T.contiguous(),
        'v_scaled': v_scaled,
        'row_scale': row_scale.contiguous(),
        'ancestor_cols': ancestor_cols.T.contiguous(),
    }


@triton.jit
def _wave_step_uniform_linear_kernel(
    Pi_ptr,
    ws,
    op_cols_ptr,
    op_vals_ptr,
    v_scaled_ptr,
    row_scale_ptr,
    leaf_term_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    Pi_new_ptr,
    max_diff_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    MAX_OP_NNZ: tl.constexpr,
    BLOCK_S: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    DEBUG_GUARD: tl.constexpr,
    FP64: tl.constexpr,
):
    DTYPE = tl.float64 if FP64 else tl.float32
    NEG_LARGE = -1e300 if FP64 else -1e30
    M_SAFE_THRESH = -1e299 if FP64 else -1e29

    w = tl.program_id(0).to(tl.int64)
    pi_base = (ws + w) * stride
    out_base = w * stride

    row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        tile_max = tl.max(pi_val, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
        row_max = new_max

    if COMPUTE_DIFF:
        local_max_diff = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)

        raw = tl.load(v_scaled_ptr + s_offs, mask=mask, other=0.0) * row_sum
        for k in range(0, MAX_OP_NNZ):
            cols = tl.load(op_cols_ptr + k.to(tl.int64) * S + s_offs, mask=mask, other=0)
            vals = tl.load(op_vals_ptr + k.to(tl.int64) * S + s_offs, mask=mask, other=0.0)
            pi_col = tl.load(Pi_ptr + pi_base + cols, mask=mask, other=NEG_LARGE)
            raw += vals * tl.exp2(pi_col - row_max)

        row_scale = tl.load(row_scale_ptr + s_offs, mask=mask, other=0.0)
        local_log = tl.log2(raw) + row_max + row_scale
        if DEBUG_GUARD:
            local_log = tl.where(raw > 0.0, local_log, NEG_LARGE)

        leaf = tl.load(leaf_term_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)
        m = tl.maximum(local_log, leaf)
        if has_splits:
            dts_r = tl.load(DTS_reduced_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)
            m = tl.maximum(m, dts_r)

        m_safe = tl.where(m > M_SAFE_THRESH, m, tl.zeros_like(m))
        acc = tl.exp2(local_log - m_safe) + tl.exp2(leaf - m_safe)
        if has_splits:
            acc += tl.exp2(dts_r - m_safe)

        result = tl.log2(acc) + m
        tl.store(Pi_new_ptr + out_base + s_offs, result, mask=mask)

        if COMPUTE_DIFF:
            significant = result > -100.0
            diff = tl.where(significant & mask, tl.abs(result - pi_w), tl.zeros_like(result))
            local_max_diff = tl.maximum(local_max_diff, tl.max(diff, axis=0))

    if COMPUTE_DIFF:
        tl.store(max_diff_ptr + w, tl.max(local_max_diff, axis=0))


def wave_step_uniform_linear_fused(Pi, ws, W, S, op_cols, op_vals, v_scaled, row_scale,
                                   leaf_term_wt, DTS_reduced=None, compute_diff=True,
                                   debug_guard=False):
    """Uniform DTS_L update via pre-scaled signed sparse operator."""
    fp64 = Pi.dtype == torch.float64
    Pi_new = torch.empty((W, S), dtype=Pi.dtype, device=Pi.device)
    max_diff_buf = torch.empty(W, dtype=Pi.dtype, device=Pi.device) if compute_diff else Pi_new
    has_splits = DTS_reduced is not None

    BLOCK_S = min(256, triton.next_power_of_2(S))
    grid = (W,)

    _wave_step_uniform_linear_kernel[grid](
        Pi, ws,
        op_cols,
        op_vals,
        v_scaled,
        row_scale,
        leaf_term_wt,
        DTS_reduced if has_splits else leaf_term_wt,
        has_splits,
        Pi_new,
        max_diff_buf,
        S,
        stride=S,
        MAX_OP_NNZ=op_cols.shape[0],
        BLOCK_S=BLOCK_S,
        COMPUTE_DIFF=bool(compute_diff),
        DEBUG_GUARD=bool(debug_guard),
        FP64=fp64,
        num_warps=4,
    )
    max_diff = max_diff_buf.max() if compute_diff else None
    return Pi_new, max_diff


@triton.jit
def _wave_pibar_uniform_ancestor_kernel(
    Pi_ptr,
    ws,
    mt_ptr,
    ancestor_cols_ptr,
    Pibar_out_ptr,
    row_max_out_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    BLOCK_S: tl.constexpr,
    STORE_ROW_MAX: tl.constexpr,
    FP64: tl.constexpr,
):
    DTYPE = tl.float64 if FP64 else tl.float32
    NEG_LARGE = -1e300 if FP64 else -1e30

    w = tl.program_id(0).to(tl.int64)
    pi_base = (ws + w) * stride

    row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        tile_max = tl.max(pi_val, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
        row_max = new_max

    if STORE_ROW_MAX:
        tl.store(row_max_out_ptr + ws + w, tl.max(row_max, axis=0))

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        for k in range(0, MAX_ANCESTOR_DEPTH):
            anc = tl.load(ancestor_cols_ptr + k.to(tl.int64) * S + s_offs, mask=mask, other=-1)
            anc_valid = mask & (anc >= 0) & (anc < S)
            pi_anc = tl.load(Pi_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
            ancestor_sum += tl.where(anc_valid, tl.exp2(pi_anc - row_max), tl.zeros([BLOCK_S], dtype=DTYPE))

        mt = tl.load(mt_ptr + s_offs, mask=mask, other=0.0)
        denom = row_sum - ancestor_sum
        pibar_w = tl.where(denom > 0.0, tl.log2(denom) + row_max + mt, NEG_LARGE)
        tl.store(Pibar_out_ptr + pi_base + s_offs, pibar_w, mask=mask)


def wave_pibar_uniform_fused(Pi, Pibar, ws, W, S, mt_squeezed, ancestor_cols,
                             row_max_out=None):
    """Compute final uniform Pibar rows using precomputed ancestor columns."""
    fp64 = Pi.dtype == torch.float64
    BLOCK_S = min(256, triton.next_power_of_2(S))
    grid = (W,)

    _wave_pibar_uniform_ancestor_kernel[grid](
        Pi,
        ws,
        mt_squeezed,
        ancestor_cols,
        Pibar,
        row_max_out if row_max_out is not None else Pibar,
        S,
        stride=S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        BLOCK_S=BLOCK_S,
        STORE_ROW_MAX=bool(row_max_out is not None),
        FP64=fp64,
        num_warps=4,
    )
