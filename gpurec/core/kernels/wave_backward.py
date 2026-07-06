"""Fused Triton kernels for wave-backward.

Two kernels:
1. _wave_backward_uniform_kernel: self-loop backward (Neumann VJP + param VJP)
2. _dts_cross_backward_kernel: cross-clade DTS backward (adjoint propagation + param VJP)

Both use one CTA per work-item, multi-pass over species dimension.
"""

import os

import torch
import triton
import triton.language as tl


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _device_scalar_param(param, *, device, dtype):
    """Return a one-element device tensor without extracting CUDA scalars."""
    if torch.is_tensor(param):
        if param.numel() != 1:
            raise ValueError("fused DTS backward scalar parameters must have one element")
        if param.device != device or param.dtype != dtype:
            param = param.to(device=device, dtype=dtype)
        return param.reshape(1).contiguous()
    return torch.tensor([param], device=device, dtype=dtype)


def _dts_scalar_param_args(log_pD, log_pS, *, device, dtype):
    use_device_scalars = (
        os.environ.get("GPUREC_DTS_BACKWARD_DEVICE_SCALARS", "1") != "0"
    )
    if use_device_scalars:
        return (
            _device_scalar_param(log_pD, device=device, dtype=dtype),
            _device_scalar_param(log_pS, device=device, dtype=dtype),
            True,
        )

    def _extract(param):
        if torch.is_tensor(param):
            return float(param) if param.ndim == 0 else float(param.item())
        return float(param)

    return _extract(log_pD), _extract(log_pS), False


@triton.jit
def _active_mask_from_rhs_absmax_kernel(
    rhs_ptr,          # [W, S]
    active_mask_ptr,  # [W] bool
    threshold,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    STRICT_GT: tl.constexpr,
    DTYPE: tl.constexpr,
):
    w = tl.program_id(0)
    row_base = w.to(tl.int64) * stride
    row_max = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        rhs_val = tl.load(rhs_ptr + row_base + s_offs, mask=mask, other=0.0)
        tile_max = tl.max(tl.abs(rhs_val), axis=0)
        row_max = tl.maximum(row_max, tile_max)

    if STRICT_GT:
        active = row_max > threshold
    else:
        active = row_max >= threshold
    lane = tl.arange(0, 1)
    tl.store(active_mask_ptr + w + lane, active)


def active_mask_from_rhs_absmax_fused(rhs, threshold, *, use_pruning=True):
    """Build the row activity mask for backward pruning in one Triton launch."""
    if rhs.ndim != 2:
        raise ValueError("rhs must be a 2D tensor")
    if rhs.device.type != "cuda":
        raise ValueError("active_mask_from_rhs_absmax_fused requires a CUDA tensor")
    if rhs.dtype not in (torch.float32, torch.float64):
        raise ValueError("active_mask_from_rhs_absmax_fused supports fp32/fp64 tensors")

    W, S = rhs.shape
    active_mask = torch.empty((W,), device=rhs.device, dtype=torch.bool)
    if W == 0:
        return active_mask

    BLOCK_S = min(256, triton.next_power_of_2(S))
    _active_mask_from_rhs_absmax_kernel[(W,)](
        rhs,
        active_mask,
        float(threshold),
        S,
        rhs.stride(0),
        BLOCK_S,
        STRICT_GT=bool(not use_pruning),
        DTYPE=_tl_float_dtype(rhs.dtype),
    )
    return active_mask


@triton.jit
def _wave_backward_uniform_kernel(
    # Converged values from forward pass
    Pi_star_ptr,      # [C, S] — read rows [ws:ws+W]
    Pibar_star_ptr,   # [C, S] — read rows [ws:ws+W]
    Pibar_row_max_ptr, # optional [C] forward Pi row maxima for uniform Pibar
    dts_r_ptr,        # [W, S] or None — cross-clade DTS
    has_splits: tl.constexpr,
    # Incoming adjoint
    rhs_ptr,          # [W, S]
    active_mask_ptr,  # optional [W] bool row activity mask
    # Constants [S]
    mt_ptr, DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    # Species children [S] long
    sp_child1_ptr, sp_child2_ptr,
    sp_parent_ptr,
    # Leaf term [W, S]
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    # Outputs
    v_k_ptr,          # [W, S] — Neumann-solved adjoint
    # Per-element param grad contributions [W, S] each — reduced by caller
    aw0_ptr,          # grad contribution to log_pD, E (from term 0)
    aw1_ptr,          # grad contribution to Ebar (from term 1)
    aw2_ptr,          # grad contribution to E, mt (from term 2)
    aw345_ptr,        # grad contribution to log_pS (from terms 3+4+5)
    aw3_ptr,          # grad contribution to E_s2 (from term 3)
    aw4_ptr,          # grad contribution to E_s1 (from term 4)
    # Scratch buffer for speciation scatter [W, S]
    spec_buf_ptr,
    term_buf_ptr,
    # Optional in-kernel accumulation targets for global-mode param grads.
    grad_log_pD_ptr,
    grad_log_pS_ptr,
    grad_E_ptr,
    grad_Ebar_ptr,
    grad_E_s1_ptr,
    grad_E_s2_ptr,
    grad_mt_ptr,
    # Dimensions
    ws,               # wave start offset into [C, S]
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NEUMANN_TERMS: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    ACCUM_PARAM_GRADS: tl.constexpr,
    FAST_NOSPLIT_PARAM_GRADS: tl.constexpr,
    COMPACT_PIBAR_SCRATCH: tl.constexpr,
    RECOMPUTE_PIBAR_DENOM: tl.constexpr,
    LEAF_HIT_ONLY_LOGP: tl.constexpr,
    LEAF_LOGP_SCALAR: tl.constexpr,
    USE_PIBAR_ROW_MAX: tl.constexpr,
    SPEC_GATHER: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Fused backward kernel for uniform Pibar mode.

    Per clade w, computes:
    1. Softmax weights (w_L, w_terms) from converged Pi/Pibar
    2. Neumann series: v_k = (I + J^T + (J^T)^2 + ...) @ rhs
    3. Param VJP element-wise contributions

    The Neumann J^T application needs A = sum_s(u_d[s]) — a full-row reduction.
    Each iteration uses 2 sub-passes:
      Pass A: compute u_d[s], accumulate A, write spec scatter to buffer
      Pass B: compute result[s] using A, read spec scatter from buffer
    """
    NEG_LARGE = tl.full([1], value=-1e30, dtype=DTYPE)

    w = tl.program_id(0)
    pi_base = (ws + w).to(tl.int64) * stride      # offset into [C, S]
    out_base = w.to(tl.int64) * stride             # offset into [W, S]
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + w)
        if row_active == 0:
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                mask = s_offs < S
                zero = tl.zeros([BLOCK_S], dtype=DTYPE)
                tl.store(v_k_ptr + out_base + s_offs, zero, mask=mask)
                if not ACCUM_PARAM_GRADS:
                    off = out_base + s_offs
                    tl.store(aw0_ptr + off, zero, mask=mask)
                    tl.store(aw1_ptr + off, zero, mask=mask)
                    tl.store(aw2_ptr + off, zero, mask=mask)
                    tl.store(aw345_ptr + off, zero, mask=mask)
                    tl.store(aw3_ptr + off, zero, mask=mask)
                    tl.store(aw4_ptr + off, zero, mask=mask)
            return
    else:
        row_active = True

    # ================================================================
    # Pass 1: Row statistics for uniform Pibar (same as forward)
    # ================================================================
    if USE_PIBAR_ROW_MAX:
        row_max = tl.load(Pibar_row_max_ptr + ws + w)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    else:
        row_max = tl.full([1], value=-1e30, dtype=DTYPE)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)

        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
            tile_max = tl.max(pi_val, axis=0)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
            row_max = new_max

    # ================================================================
    # Pass 2: Compute softmax weights and store to [W, S] buffers
    # We need w_L[s], w_terms[0..5][s], inv_denom[s], p_prime[s]
    # These are consumed by the Neumann loop. Since S doesn't fit in
    # registers across passes, we store per-element to global memory.
    #
    # Actually, we interleave: compute weights and immediately use them.
    # But Neumann needs full-row A, so we can't do it in one pass.
    #
    # Strategy: store weights to reusable buffers, then iterate Neumann.
    # We reuse the output buffers (aw0..aw4) as scratch during Neumann,
    # then overwrite with final param contributions.
    #
    # Stored per-element (to aw* buffers temporarily):
    #   aw0 = w_L * (w_terms[0] + w_terms[1])  — diagonal weight
    #   aw1 = w_L * w_terms[2]                  — Pibar weight
    #   aw2 = inv_denom                         — for Pibar VJP
    #   aw3 = p_prime                            — for Pibar VJP
    #   aw4 = w_L * w_terms[3]                  — SL1 weight
    #   aw345 = w_L * w_terms[4]                — SL2 weight
    # ================================================================
    M_SAFE = tl.full([1], value=-1e29, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        off = out_base + s_offs

        # Load Pi*, Pibar*
        pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
        pibar_w = tl.load(Pibar_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)

        # Load constants
        dl_c = tl.load(DL_const_ptr + s_offs, mask=mask, other=-1e30)
        ebar = tl.load(Ebar_ptr + s_offs, mask=mask, other=-1e30)
        e_val = tl.load(E_ptr + s_offs, mask=mask, other=-1e30)
        sl1_c = tl.load(SL1_const_ptr + s_offs, mask=mask, other=-1e30)
        sl2_c = tl.load(SL2_const_ptr + s_offs, mask=mask, other=-1e30)

        # Gather species children
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = c1 < S
        c2_valid = c2 < S
        pi_s1 = tl.load(Pi_star_ptr + pi_base + c1, mask=mask & c1_valid, other=-1e30)
        pi_s2 = tl.load(Pi_star_ptr + pi_base + c2, mask=mask & c2_valid, other=-1e30)

        # 6 DTS_L terms
        t0 = dl_c + pi_w
        t1 = pi_w + ebar
        t2 = pibar_w + e_val
        t3 = sl1_c + pi_s1
        t4 = sl2_c + pi_s2
        if USE_LEAF_INDEX:
            leaf_species = tl.load(leaf_species_ptr + ws + w)
            leaf_hit = mask & (leaf_species == s_offs)
            if LEAF_LOGP_SCALAR:
                leaf_logp = tl.load(leaf_logp_ptr)
                t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
            elif LEAF_HIT_ONLY_LOGP:
                t5 = tl.load(leaf_logp_ptr + s_offs, mask=leaf_hit, other=-1e30)
            else:
                leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=mask, other=-1e30)
                t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            t5 = tl.load(leaf_term_ptr + off, mask=mask, other=-1e30)

        # Logsumexp over 6 terms → DTS_L
        m = tl.maximum(t0, t1)
        m = tl.maximum(m, t2)
        m = tl.maximum(m, t3)
        m = tl.maximum(m, t4)
        m = tl.maximum(m, t5)
        m_safe = tl.where(m > M_SAFE, m, tl.zeros_like(m))
        e0 = tl.exp2(t0 - m_safe)
        e1 = tl.exp2(t1 - m_safe)
        e2 = tl.exp2(t2 - m_safe)
        e3 = tl.exp2(t3 - m_safe)
        e4 = tl.exp2(t4 - m_safe)
        e5 = tl.exp2(t5 - m_safe)
        dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
        dts_l = tl.log2(dts_l_sum) + m

        # w_L = exp2(DTS_L - Pi_new), w_terms[i] = exp2(terms[i] - DTS_L) = e_i / dts_l_sum
        if has_splits:
            dts_r = tl.load(dts_r_ptr + off, mask=mask, other=-1e30)
            pi_new_m = tl.maximum(dts_l, dts_r)
            pi_new_ms = tl.where(pi_new_m > M_SAFE, pi_new_m, tl.zeros_like(pi_new_m))
            pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
            # w_L: safe when DTS_L = -inf → w_L = 0
            w_L = tl.where(dts_l > M_SAFE, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
        else:
            w_L = tl.full(s_offs.shape, value=1.0, dtype=DTYPE)

        # Per-term softmax weights (divide by dts_l_sum, not dts_l_sum + dts_r)
        inv_sum = tl.where(dts_l_sum > 0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))
        wt0 = e0 * inv_sum
        wt1 = e1 * inv_sum
        wt2 = e2 * inv_sum
        wt3 = e3 * inv_sum
        wt4 = e4 * inv_sum
        # wt5 = e5 * inv_sum  (only needed for param VJP log_pS, computed later)

        # Pibar VJP ingredients: inv_denom = 1 / (row_sum - p_prime)
        p_prime = tl.exp2(pi_w - row_max)
        if USE_PIBAR_ROW_MAX:
            mt_val = tl.load(mt_ptr + s_offs, mask=mask, other=-1e30)
            inv_denom = tl.exp2(row_max + mt_val - pibar_w)
        else:
            denom = row_sum - p_prime
            inv_denom = tl.where(denom > 0, 1.0 / denom, tl.zeros_like(denom))

        # Store precomputed weights to scratch buffers
        diag_wt = w_L * (wt0 + wt1)        # diagonal J^T weight
        pibar_wt = w_L * wt2               # Pibar path weight
        sl1_wt = w_L * wt3                 # SL1 speciation weight
        sl2_wt = w_L * wt4                 # SL2 speciation weight

        tl.store(aw0_ptr + off, diag_wt, mask=mask)
        if COMPACT_PIBAR_SCRATCH:
            tl.store(aw1_ptr + off, pibar_wt * inv_denom, mask=mask)
        else:
            tl.store(aw1_ptr + off, pibar_wt, mask=mask)
        if (not RECOMPUTE_PIBAR_DENOM) and (not COMPACT_PIBAR_SCRATCH):
            tl.store(aw2_ptr + off, inv_denom, mask=mask)
            tl.store(aw3_ptr + off, p_prime, mask=mask)
        tl.store(aw4_ptr + off, sl1_wt, mask=mask)
        tl.store(aw345_ptr + off, sl2_wt, mask=mask)

    # ================================================================
    # Neumann series: v = rhs + J^T(rhs) + (J^T)^2(rhs) + ...
    #
    # Each J^T application on vector `term` requires:
    #   Pass A: compute u_d = term * pibar_wt * inv_denom, accumulate A = sum(u_d)
    #           and, in the default path, scatter speciation contributions.
    #   Pass B: result[s] = term[s] * diag_wt[s] + p_prime[s] * (A - u_d[s])
    #                        + speciation contribution.
    #
    # SPEC_GATHER replaces the scatter/zero/read speciation path with:
    #   spec[s] = term[parent[s]] * sl_weight[parent[s] -> s]
    # This removes one full zero pass and the speciation scatter stores, but
    # adds parent-index gathers from the current term and scratch weights.
    # ================================================================
    # Copy rhs → v_k (v_k accumulates the Neumann sum)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        rhs_val = tl.load(rhs_ptr + out_base + s_offs, mask=mask, other=0.0)
        tl.store(v_k_ptr + out_base + s_offs, rhs_val, mask=valid_mask)

    # Buffer ping-pong: iteration 0 reads rhs_ptr, even iterations write
    # spec_buf, and odd iterations write term_buf. Output buffer is zeroed
    # at the start of each iteration to avoid stale data at non-child positions.

    for _n in range(NEUMANN_TERMS):
        if not SPEC_GATHER:
            # Zero the output buffer before scatter writes.
            # Sub-pass A only writes to child positions (scatter); sub-pass B reads ALL positions.
            # Without zeroing, non-child positions would have stale data from prior iterations.
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                valid_mask = s_offs < S
                mask = valid_mask & row_active
                if _n % 2 == 0:
                    tl.store(spec_buf_ptr + out_base + s_offs,
                             tl.zeros(s_offs.shape, dtype=DTYPE), mask=mask)
                else:
                    tl.store(term_buf_ptr + out_base + s_offs,
                             tl.zeros(s_offs.shape, dtype=DTYPE), mask=mask)

        # --- Sub-pass A: accumulate A = sum_s(term * pibar_wt * inv_denom) ---
        # Also write speciation scatter contributions.
        A_acc = tl.full([1], value=0.0, dtype=DTYPE)

        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            off = out_base + s_offs

            # Load term from appropriate buffer
            if _n == 0:
                term_val = tl.load(rhs_ptr + off, mask=mask, other=0.0)
            elif _n % 2 == 1:
                term_val = tl.load(spec_buf_ptr + off, mask=mask, other=0.0)
            else:
                term_val = tl.load(term_buf_ptr + off, mask=mask, other=0.0)

            if COMPACT_PIBAR_SCRATCH:
                pibar_u_coeff = tl.load(aw1_ptr + off, mask=mask, other=0.0)
                u_d = term_val * pibar_u_coeff
            else:
                pibar_wt = tl.load(aw1_ptr + off, mask=mask, other=0.0)
                if RECOMPUTE_PIBAR_DENOM:
                    pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
                    p_prime = tl.exp2(pi_w - row_max)
                    denom = row_sum - p_prime
                    inv_denom = tl.where(denom > 0, 1.0 / denom, tl.zeros_like(denom))
                else:
                    inv_denom = tl.load(aw2_ptr + off, mask=mask, other=0.0)

                u_d = term_val * pibar_wt * inv_denom

            A_acc += tl.sum(u_d, axis=0)

            if not SPEC_GATHER:
                # Speciation scatter: write term * sl_wt to child index
                sl1_wt = tl.load(aw4_ptr + off, mask=mask, other=0.0)
                sl2_wt = tl.load(aw345_ptr + off, mask=mask, other=0.0)
                c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
                c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
                c1_valid = (c1 < S) & mask
                c2_valid = (c2 < S) & mask

                # No conflict: each child appears as target of exactly one parent.
                src1 = term_val * sl1_wt
                src2 = term_val * sl2_wt
                # Write to output buffer at child index (using the OTHER buffer)
                if _n % 2 == 0:
                    # Writing to spec_buf
                    tl.store(spec_buf_ptr + out_base + c1, src1, mask=c1_valid)
                    tl.store(spec_buf_ptr + out_base + c2, src2, mask=c2_valid)
                else:
                    tl.store(term_buf_ptr + out_base + c1, src1, mask=c1_valid)
                    tl.store(term_buf_ptr + out_base + c2, src2, mask=c2_valid)

        # --- Sub-pass B: compute J^T result using A ---
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            off = out_base + s_offs

            # Reload term and weights
            if _n == 0:
                term_val = tl.load(rhs_ptr + off, mask=mask, other=0.0)
            elif _n % 2 == 1:
                term_val = tl.load(spec_buf_ptr + off, mask=mask, other=0.0)
            else:
                term_val = tl.load(term_buf_ptr + off, mask=mask, other=0.0)

            diag_wt = tl.load(aw0_ptr + off, mask=mask, other=0.0)
            if COMPACT_PIBAR_SCRATCH:
                pibar_u_coeff = tl.load(aw1_ptr + off, mask=mask, other=0.0)
                pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
                p_prime = tl.exp2(pi_w - row_max)
                u_d = term_val * pibar_u_coeff
            else:
                pibar_wt = tl.load(aw1_ptr + off, mask=mask, other=0.0)
                if RECOMPUTE_PIBAR_DENOM:
                    pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
                    p_prime = tl.exp2(pi_w - row_max)
                    denom = row_sum - p_prime
                    inv_denom = tl.where(denom > 0, 1.0 / denom, tl.zeros_like(denom))
                else:
                    inv_denom = tl.load(aw2_ptr + off, mask=mask, other=0.0)
                    p_prime = tl.load(aw3_ptr + off, mask=mask, other=0.0)

                u_d = term_val * pibar_wt * inv_denom
            result = term_val * diag_wt + p_prime * (A_acc - u_d)

            # Add speciation contribution.
            if SPEC_GATHER:
                parent = tl.load(sp_parent_ptr + s_offs, mask=mask, other=-1)
                parent_valid = parent >= 0
                parent_off = out_base + parent
                if _n == 0:
                    parent_term = tl.load(rhs_ptr + parent_off, mask=mask & parent_valid, other=0.0)
                elif _n % 2 == 1:
                    parent_term = tl.load(spec_buf_ptr + parent_off, mask=mask & parent_valid, other=0.0)
                else:
                    parent_term = tl.load(term_buf_ptr + parent_off, mask=mask & parent_valid, other=0.0)
                parent_sl1 = tl.load(aw4_ptr + parent_off, mask=mask & parent_valid, other=0.0)
                parent_sl2 = tl.load(aw345_ptr + parent_off, mask=mask & parent_valid, other=0.0)
                parent_c1 = tl.load(sp_child1_ptr + parent, mask=mask & parent_valid, other=S)
                parent_wt = tl.where(parent_c1 == s_offs, parent_sl1, parent_sl2)
                spec_val = parent_term * parent_wt
            elif _n % 2 == 0:
                spec_val = tl.load(spec_buf_ptr + off, mask=mask, other=0.0)
            else:
                spec_val = tl.load(term_buf_ptr + off, mask=mask, other=0.0)
            result = result + spec_val

            # Store result to output buffer
            if _n % 2 == 0:
                tl.store(spec_buf_ptr + off, result, mask=mask)
            else:
                tl.store(term_buf_ptr + off, result, mask=mask)

            # Accumulate into v_k
            v_k_val = tl.load(v_k_ptr + off, mask=mask, other=0.0)
            tl.store(v_k_ptr + off, v_k_val + result, mask=mask)

    # ================================================================
    # Pass final: Param VJP contributions
    # Recompute alpha = v_k * w_L and weighted terms.
    # Store per-element contributions for the caller to reduce.
    # ================================================================
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        off = out_base + s_offs

        v_k_val = tl.load(v_k_ptr + off, mask=mask, other=0.0)

        if (ACCUM_PARAM_GRADS and FAST_NOSPLIT_PARAM_GRADS) and ((not has_splits) and (not COMPACT_PIBAR_SCRATCH)):
            diag_wt = tl.load(aw0_ptr + off, mask=mask, other=0.0)
            pibar_wt = tl.load(aw1_ptr + off, mask=mask, other=0.0)
            sl1_wt = tl.load(aw4_ptr + off, mask=mask, other=0.0)
            sl2_wt = tl.load(aw345_ptr + off, mask=mask, other=0.0)
            leaf_wt = 1.0 - diag_wt - pibar_wt - sl1_wt - sl2_wt

            dl_c = tl.load(DL_const_ptr + s_offs, mask=mask, other=-1e30)
            ebar = tl.load(Ebar_ptr + s_offs, mask=mask, other=-1e30)
            m01 = tl.maximum(dl_c, ebar)
            e0_01 = tl.exp2(dl_c - m01)
            e1_01 = tl.exp2(ebar - m01)
            frac_d = e0_01 / (e0_01 + e1_01)

            _aw0 = v_k_val * diag_wt * frac_d
            _aw1 = v_k_val * diag_wt * (1.0 - frac_d)
            _aw2 = v_k_val * pibar_wt
            _aw3 = v_k_val * sl1_wt
            _aw4 = v_k_val * sl2_wt
            _aw345 = v_k_val * (sl1_wt + sl2_wt + leaf_wt)

            tl.atomic_add(grad_log_pD_ptr, tl.sum(tl.where(mask, _aw0, 0.0), axis=0), sem="relaxed")
            tl.atomic_add(grad_log_pS_ptr, tl.sum(tl.where(mask, _aw345, 0.0), axis=0), sem="relaxed")
            tl.atomic_add(grad_E_ptr + s_offs, _aw0 + _aw2, sem="relaxed", mask=mask)
            tl.atomic_add(grad_Ebar_ptr + s_offs, _aw1, sem="relaxed", mask=mask)
            tl.atomic_add(grad_E_s1_ptr + s_offs, _aw4, sem="relaxed", mask=mask)
            tl.atomic_add(grad_E_s2_ptr + s_offs, _aw3, sem="relaxed", mask=mask)
            tl.atomic_add(grad_mt_ptr + s_offs, _aw2, sem="relaxed", mask=mask)
        else:
            # Reload Pi and Pibar to recompute weights
            # (we overwrote aw* buffers with Jt scratch data)
            pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
            pibar_w = tl.load(Pibar_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
            dl_c = tl.load(DL_const_ptr + s_offs, mask=mask, other=-1e30)
            ebar = tl.load(Ebar_ptr + s_offs, mask=mask, other=-1e30)
            e_val = tl.load(E_ptr + s_offs, mask=mask, other=-1e30)
            sl1_c = tl.load(SL1_const_ptr + s_offs, mask=mask, other=-1e30)
            sl2_c = tl.load(SL2_const_ptr + s_offs, mask=mask, other=-1e30)
            c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
            c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
            c1_valid = c1 < S
            c2_valid = c2 < S
            pi_s1 = tl.load(Pi_star_ptr + pi_base + c1, mask=mask & c1_valid, other=-1e30)
            pi_s2 = tl.load(Pi_star_ptr + pi_base + c2, mask=mask & c2_valid, other=-1e30)
            if USE_LEAF_INDEX:
                leaf_species = tl.load(leaf_species_ptr + ws + w)
                leaf_hit = mask & (leaf_species == s_offs)
                if LEAF_LOGP_SCALAR:
                    leaf_logp = tl.load(leaf_logp_ptr)
                    t5 = tl.where(leaf_hit, leaf_logp, -1e30)
                elif LEAF_HIT_ONLY_LOGP:
                    t5 = tl.load(leaf_logp_ptr + s_offs, mask=leaf_hit, other=-1e30)
                else:
                    leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=mask, other=-1e30)
                    t5 = tl.where(leaf_hit, leaf_logp, -1e30)
            else:
                t5 = tl.load(leaf_term_ptr + off, mask=mask, other=-1e30)

            # Recompute DTS_L terms and softmax weights
            t0 = dl_c + pi_w
            t1 = pi_w + ebar
            t2 = pibar_w + e_val
            t3 = sl1_c + pi_s1
            t4 = sl2_c + pi_s2
            m = tl.maximum(tl.maximum(tl.maximum(t0, t1), tl.maximum(t2, t3)), tl.maximum(t4, t5))
            m_safe = tl.where(m > -1e29, m, tl.zeros_like(m))
            e0 = tl.exp2(t0 - m_safe)
            e1 = tl.exp2(t1 - m_safe)
            e2 = tl.exp2(t2 - m_safe)
            e3 = tl.exp2(t3 - m_safe)
            e4 = tl.exp2(t4 - m_safe)
            e5 = tl.exp2(t5 - m_safe)
            dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
            inv_sum = tl.where(dts_l_sum > 0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))

            if has_splits:
                dts_r = tl.load(dts_r_ptr + off, mask=mask, other=-1e30)
                dts_l = tl.log2(dts_l_sum) + m
                pi_new_m = tl.maximum(dts_l, dts_r)
                pi_new_ms = tl.where(pi_new_m > -1e29, pi_new_m, tl.zeros_like(pi_new_m))
                pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
                w_L = tl.where(dts_l > -1e29, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
            else:
                w_L = tl.full(s_offs.shape, value=1.0, dtype=DTYPE)

            alpha = v_k_val * w_L

            # Per-element param contributions
            _aw0 = alpha * e0 * inv_sum   # → log_pD, E
            _aw1 = alpha * e1 * inv_sum   # → Ebar
            _aw2 = alpha * e2 * inv_sum   # → E, mt
            _aw3 = alpha * e3 * inv_sum   # → log_pS, E_s2
            _aw4 = alpha * e4 * inv_sum   # → log_pS, E_s1
            _aw5 = alpha * e5 * inv_sum   # → log_pS

            _aw345 = _aw3 + _aw4 + _aw5
            if ACCUM_PARAM_GRADS:
                tl.atomic_add(grad_log_pD_ptr, tl.sum(tl.where(mask, _aw0, 0.0), axis=0), sem="relaxed")
                tl.atomic_add(grad_log_pS_ptr, tl.sum(tl.where(mask, _aw345, 0.0), axis=0), sem="relaxed")
                tl.atomic_add(grad_E_ptr + s_offs, _aw0 + _aw2, sem="relaxed", mask=mask)
                tl.atomic_add(grad_Ebar_ptr + s_offs, _aw1, sem="relaxed", mask=mask)
                tl.atomic_add(grad_E_s1_ptr + s_offs, _aw4, sem="relaxed", mask=mask)
                tl.atomic_add(grad_E_s2_ptr + s_offs, _aw3, sem="relaxed", mask=mask)
                tl.atomic_add(grad_mt_ptr + s_offs, _aw2, sem="relaxed", mask=mask)
            else:
                tl.store(aw0_ptr + off, _aw0, mask=valid_mask)
                tl.store(aw1_ptr + off, _aw1, mask=valid_mask)
                tl.store(aw2_ptr + off, _aw2, mask=valid_mask)
                tl.store(aw345_ptr + off, _aw345, mask=valid_mask)
                tl.store(aw3_ptr + off, _aw3, mask=valid_mask)
                tl.store(aw4_ptr + off, _aw4, mask=valid_mask)


def wave_backward_uniform_fused(
    Pi_star, Pibar_star, ws, W, S,
    dts_r,
    rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2, leaf_term_wt,
    neumann_terms=3,
    leaf_species_idx=None,
    leaf_logp=None,
    accum_param_grads=None,
    active_mask=None,
    sp_parent=None,
    pibar_row_max=None,
):
    """Fused backward: precompute + Neumann + param VJP in one kernel per wave.

    Args:
        Pi_star: [C, S] converged Pi
        Pibar_star: [C, S] converged Pibar
        ws: wave start offset
        W: wave size
        S: number of species
        dts_r: [W, S] or None
        rhs: [W, S] incoming adjoint
        mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const: [S]
        sp_child1, sp_child2: [S] long
        leaf_term_wt: [W, S]
        neumann_terms: int
        leaf_species_idx: optional [C] row -> species leaf index, -1 for non-leaves
        leaf_logp: optional [S] log_pS values used with leaf_species_idx
        accum_param_grads: optional tuple of seven tensors
            (grad_log_pD, grad_log_pS, grad_E, grad_Ebar,
             grad_E_s1, grad_E_s2, grad_mt). When provided, the kernel
            atomically accumulates param VJP results and returns None for
            the per-element contribution tensors.

    Returns:
        v_k: [W, S] Neumann-solved adjoint
        aw0, aw1, aw2, aw345, aw3, aw4: [W, S] per-element param grad contributions
    """
    device = Pi_star.device
    dtype = Pi_star.dtype

    accum_enabled = accum_param_grads is not None
    has_splits = dts_r is not None
    use_leaf_index = leaf_species_idx is not None and leaf_logp is not None
    fast_nosplit_param_grads = (
        os.environ.get("GPUREC_FAST_NOSPLIT_PARAM_ACCUM", "0") != "0"
    )
    recompute_pibar_denom = (
        os.environ.get("GPUREC_RECOMPUTE_PIBAR_DENOM", "0") != "0"
    )
    compact_pibar_scratch_mode = (
        os.environ.get("GPUREC_COMPACT_PIBAR_SCRATCH", "1")
        .strip()
        .lower()
    )
    compact_pibar_scratch = compact_pibar_scratch_mode not in (
        "0", "false", "no", "off", ""
    )
    if compact_pibar_scratch_mode in ("leaf", "nosplit", "no_split"):
        compact_pibar_scratch = not has_splits
    if compact_pibar_scratch:
        recompute_pibar_denom = False
    leaf_hit_only_logp = (
        os.environ.get("GPUREC_LEAF_HIT_ONLY_LOGP", "0") != "0"
    )
    leaf_logp_scalar = bool(
        use_leaf_index
        and leaf_logp is not None
        and leaf_logp.numel() == 1
    )
    spec_gather = (
        os.environ.get("GPUREC_WAVE_SPEC_GATHER", "1") != "0"
        and sp_parent is not None
    )
    use_pibar_row_max = (
        os.environ.get("GPUREC_WAVE_REUSE_PIBAR_ROW_MAX", "1") != "0"
        and pibar_row_max is not None
    )

    v_k = torch.empty((W, S), device=device, dtype=dtype)
    aw0 = torch.empty((W, S), device=device, dtype=dtype)
    aw1 = torch.empty((W, S), device=device, dtype=dtype)
    need_pibar_denom_scratch = not (
        accum_enabled and (compact_pibar_scratch or recompute_pibar_denom)
    )
    aw2 = (
        torch.empty((W, S), device=device, dtype=dtype)
        if need_pibar_denom_scratch else aw0
    )
    aw345 = torch.empty((W, S), device=device, dtype=dtype)
    aw3 = (
        torch.empty((W, S), device=device, dtype=dtype)
        if need_pibar_denom_scratch else aw0
    )
    aw4 = torch.empty((W, S), device=device, dtype=dtype)
    spec_buf = torch.empty((W, S), device=device, dtype=dtype)
    term_buf = torch.empty((W, S), device=device, dtype=dtype)

    if leaf_term_wt is None:
        if not use_leaf_index:
            raise ValueError("leaf_term_wt is required when leaf_species_idx/leaf_logp are not provided")
        leaf_term_wt = leaf_logp
    leaf_species_arg = leaf_species_idx if use_leaf_index else sp_child1
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt
    if accum_enabled:
        (
            grad_log_pD_arg,
            grad_log_pS_arg,
            grad_E_arg,
            grad_Ebar_arg,
            grad_E_s1_arg,
            grad_E_s2_arg,
            grad_mt_arg,
        ) = accum_param_grads
    else:
        grad_log_pD_arg = grad_log_pS_arg = aw0
        grad_E_arg = grad_Ebar_arg = grad_E_s1_arg = grad_E_s2_arg = grad_mt_arg = aw0

    block_s_env = os.environ.get("GPUREC_WAVE_BLOCK_S", "").strip()
    if block_s_env:
        BLOCK_S = int(block_s_env)
    else:
        BLOCK_S = min(256, triton.next_power_of_2(S))
    num_warps_env = os.environ.get(
        "GPUREC_WAVE_NUM_WARPS",
        "8" if spec_gather else "",
    ).strip()
    launch_options = {}
    if num_warps_env:
        num_warps = int(num_warps_env)
        if num_warps > 0:
            launch_options["num_warps"] = num_warps

    grid = (W,)
    _wave_backward_uniform_kernel[grid](
        Pi_star, Pibar_star,
        pibar_row_max if use_pibar_row_max else Pi_star,
        dts_r if has_splits else Pi_star,  # dummy ptr when no splits
        has_splits,
        rhs,
        active_mask if active_mask is not None else rhs,
        mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        sp_parent if spec_gather else sp_child1,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        v_k,
        aw0, aw1, aw2, aw345, aw3, aw4,
        spec_buf,
        term_buf,
        grad_log_pD_arg,
        grad_log_pS_arg,
        grad_E_arg,
        grad_Ebar_arg,
        grad_E_s1_arg,
        grad_E_s2_arg,
        grad_mt_arg,
        ws, S, S, BLOCK_S,
        neumann_terms,
        USE_LEAF_INDEX=bool(use_leaf_index),
        ACCUM_PARAM_GRADS=bool(accum_enabled),
        FAST_NOSPLIT_PARAM_GRADS=bool(fast_nosplit_param_grads),
        COMPACT_PIBAR_SCRATCH=bool(compact_pibar_scratch),
        RECOMPUTE_PIBAR_DENOM=bool(recompute_pibar_denom),
        LEAF_HIT_ONLY_LOGP=bool(leaf_hit_only_logp),
        LEAF_LOGP_SCALAR=bool(leaf_logp_scalar),
        USE_PIBAR_ROW_MAX=bool(use_pibar_row_max),
        SPEC_GATHER=bool(spec_gather),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    if accum_enabled:
        return v_k, None, None, None, None, None, None
    return v_k, aw0, aw1, aw2, aw345, aw3, aw4


# =========================================================================
# Cross-clade DTS backward kernel
# =========================================================================

@triton.jit
def _dts_cross_backward_kernel(
    # Converged values [C, S]
    Pi_star_ptr,
    Pibar_star_ptr,
    # Neumann-solved adjoint [W, S]
    v_k_ptr,
    active_mask_ptr,   # optional [W] bool parent row activity mask
    # Split metadata
    sl_ptr,            # [n_ws] int64 — left child global clade index
    sr_ptr,            # [n_ws] int64 — right child global clade index
    reduce_idx_ptr,    # [n_ws] int64 — wave-local parent index
    wlsp_ptr,          # [n_ws] float — log split probability (squeezed)
    # Scalar params
    log_pD_arg,        # [1] scalar tensor or Python float
    log_pS_arg,        # [1] scalar tensor or Python float
    # Species children [S] int64
    sp_child1_ptr,
    sp_child2_ptr,
    # Outputs [n_ws, S]
    grad_Pi_l_ptr,
    grad_Pi_r_ptr,
    grad_Pibar_l_ptr,
    grad_Pibar_r_ptr,
    # Per-split param sums [n_ws]
    param_pD_ptr,
    param_pS_ptr,
    # Dimensions
    ws,                # wave start offset (parent row = ws + reduce_idx)
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DEVICE_SCALAR_PARAMS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Fused cross-clade DTS backward for uniform Pibar mode.

    For each split i, computes the VJP of the 5 cross-clade DTS terms.

    Key simplification: the chain rule through segment-logsumexp, 5-term
    logsumexp, and the DTS_L/dts_r mixing collapses to a single weight:

        v_DTS_5[t, i, s] = v_k[parent, s] * exp2(wlsp[i] + DTS_5[t,i,s] - Pi_parent[s])

    This avoids materializing intermediate [5, n_ws, S] tensors.

    Pass 1: compute direct Pi/Pibar gradients, accumulate param sums.
    Pass 2: scatter speciation gradients to child species positions.
    """
    NEG_LARGE: tl.constexpr = -1e30

    i = tl.program_id(0)  # split index

    # Load split metadata (scalar per CTA)
    sl = tl.load(sl_ptr + i)
    sr = tl.load(sr_ptr + i)
    parent_w = tl.load(reduce_idx_ptr + i)
    wlsp = tl.load(wlsp_ptr + i)
    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            out_base = i.to(tl.int64) * S
            zero_scalar = tl.zeros((1,), dtype=DTYPE)
            _scalar_off = tl.arange(0, 1)
            tl.store(param_pD_ptr + i + _scalar_off, zero_scalar)
            tl.store(param_pS_ptr + i + _scalar_off, zero_scalar)
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                mask = s_offs < S
                zero = tl.zeros([BLOCK_S], dtype=DTYPE)
                tl.store(grad_Pi_l_ptr + out_base + s_offs, zero, mask=mask)
                tl.store(grad_Pi_r_ptr + out_base + s_offs, zero, mask=mask)
                tl.store(grad_Pibar_l_ptr + out_base + s_offs, zero, mask=mask)
                tl.store(grad_Pibar_r_ptr + out_base + s_offs, zero, mask=mask)
            return
    else:
        parent_active = True

    if DEVICE_SCALAR_PARAMS:
        log_pD = tl.load(log_pD_arg).to(DTYPE)
        log_pS = tl.load(log_pS_arg).to(DTYPE)
    else:
        log_pD = log_pD_arg
        log_pS = log_pS_arg

    # Base offsets into [C, S] for child clades
    pi_l_base = sl.to(tl.int64) * stride_C
    pi_r_base = sr.to(tl.int64) * stride_C
    pibar_l_base = sl.to(tl.int64) * stride_C
    pibar_r_base = sr.to(tl.int64) * stride_C
    # Parent clade in Pi_star: row (ws + parent_w)
    parent_pi_base = (ws + parent_w).to(tl.int64) * stride_C
    # v_k is [W, S] contiguous, indexed by parent_w
    parent_vk_base = parent_w.to(tl.int64) * S
    # Output row
    out_base = i.to(tl.int64) * S

    # Accumulators for per-split param sums (1-element blocks for Triton compatibility)
    sum_pD = tl.zeros((1,), dtype=DTYPE)
    sum_pS = tl.zeros((1,), dtype=DTYPE)
    _scalar_off = tl.arange(0, 1)  # for storing 1-element blocks

    # ================================================================
    # Pass 1: Direct contributions + param sums
    # ================================================================
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & parent_active

        # Load child Pi/Pibar
        Pi_l = tl.load(Pi_star_ptr + pi_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pi_r = tl.load(Pi_star_ptr + pi_r_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_l = tl.load(Pibar_star_ptr + pibar_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_r = tl.load(Pibar_star_ptr + pibar_r_base + s_offs, mask=mask, other=NEG_LARGE)

        # Species child gathers (for speciation terms)
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask
        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE)

        # Load parent's Pi and v_k
        Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE)
        v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0)

        # DTS_5 terms
        d0 = log_pD + Pi_l + Pi_r           # D: duplication
        d1 = Pi_l + Pibar_r                 # T: transfer l→r
        d2 = Pi_r + Pibar_l                 # T: transfer r→l
        d3 = log_pS + Pi_l_s1 + Pi_r_s2    # S: speciation (c1 in l, c2 in r)
        d4 = log_pS + Pi_r_s1 + Pi_l_s2    # S: speciation (c1 in r, c2 in l)

        # Simplified weight: v_DTS_5[t] = v_k * exp2(wlsp + DTS_5[t] - Pi_parent)
        # Pi_parent >= wlsp + DTS_5[t], so exponent <= 0 and result in [0, 1].
        # Guard: when Pi_parent = -inf, all weights are 0.
        parent_valid = Pi_parent > NEG_LARGE

        w0 = tl.where(parent_valid, tl.exp2(wlsp + d0 - Pi_parent), tl.zeros_like(d0))
        w1 = tl.where(parent_valid, tl.exp2(wlsp + d1 - Pi_parent), tl.zeros_like(d1))
        w2 = tl.where(parent_valid, tl.exp2(wlsp + d2 - Pi_parent), tl.zeros_like(d2))
        w3 = tl.where(parent_valid, tl.exp2(wlsp + d3 - Pi_parent), tl.zeros_like(d3))
        w4 = tl.where(parent_valid, tl.exp2(wlsp + d4 - Pi_parent), tl.zeros_like(d4))

        vd0 = v_k_val * w0
        vd1 = v_k_val * w1
        vd2 = v_k_val * w2
        vd3 = v_k_val * w3
        vd4 = v_k_val * w4

        # Direct contributions to Pi gradients (D + T terms only; S terms via scatter in pass 2)
        tl.store(grad_Pi_l_ptr + out_base + s_offs, vd0 + vd1, mask=valid_mask)
        tl.store(grad_Pi_r_ptr + out_base + s_offs, vd0 + vd2, mask=valid_mask)
        tl.store(grad_Pibar_l_ptr + out_base + s_offs, vd2, mask=valid_mask)
        tl.store(grad_Pibar_r_ptr + out_base + s_offs, vd1, mask=valid_mask)

        # Accumulate param sums
        sum_pD += tl.sum(vd0, axis=0)
        sum_pS += tl.sum(vd3 + vd4, axis=0)

    # Store per-split param sums (use block pointer for compatibility)
    tl.store(param_pD_ptr + i + _scalar_off, sum_pD)
    tl.store(param_pS_ptr + i + _scalar_off, sum_pS)

    # ================================================================
    # Pass 2: Scatter speciation contributions to child species positions
    #
    # DTS[3] reads Pi_l[child1[s]] and Pi_r[child2[s]], so:
    #   grad_Pi_l[child1[s]] += vd3[s],  grad_Pi_r[child2[s]] += vd3[s]
    # DTS[4] reads Pi_r[child1[s]] and Pi_l[child2[s]], so:
    #   grad_Pi_r[child1[s]] += vd4[s],  grad_Pi_l[child2[s]] += vd4[s]
    #
    # Each CTA owns its output row. sp_child1/sp_child2 are injective
    # (each child has one parent), so read-modify-write is race-free.
    # ================================================================
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & parent_active

        # Recompute vd3, vd4 (speciation terms only)
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask

        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE)

        Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE)
        v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0)

        d3 = log_pS + Pi_l_s1 + Pi_r_s2
        d4 = log_pS + Pi_r_s1 + Pi_l_s2

        parent_valid = Pi_parent > NEG_LARGE
        w3 = tl.where(parent_valid, tl.exp2(wlsp + d3 - Pi_parent), tl.zeros_like(d3))
        w4 = tl.where(parent_valid, tl.exp2(wlsp + d4 - Pi_parent), tl.zeros_like(d4))
        vd3 = v_k_val * w3
        vd4 = v_k_val * w4

        # Scatter to child1 positions
        # grad_Pi_l[child1[s]] += vd3[s]
        cur = tl.load(grad_Pi_l_ptr + out_base + c1, mask=c1_valid, other=0.0)
        tl.store(grad_Pi_l_ptr + out_base + c1, cur + vd3, mask=c1_valid)
        # grad_Pi_r[child1[s]] += vd4[s]
        cur = tl.load(grad_Pi_r_ptr + out_base + c1, mask=c1_valid, other=0.0)
        tl.store(grad_Pi_r_ptr + out_base + c1, cur + vd4, mask=c1_valid)

        # Scatter to child2 positions
        # grad_Pi_r[child2[s]] += vd3[s]
        cur = tl.load(grad_Pi_r_ptr + out_base + c2, mask=c2_valid, other=0.0)
        tl.store(grad_Pi_r_ptr + out_base + c2, cur + vd3, mask=c2_valid)
        # grad_Pi_l[child2[s]] += vd4[s]
        cur = tl.load(grad_Pi_l_ptr + out_base + c2, mask=c2_valid, other=0.0)
        tl.store(grad_Pi_l_ptr + out_base + c2, cur + vd4, mask=c2_valid)


def dts_cross_backward_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    log_pD, log_pS,
    sp_child1, sp_child2,
    S,
    active_mask=None,
):
    """Fused DTS cross-clade backward: replaces both param-grad and adjoint blocks.

    Args:
        Pi_star: [C, S] converged Pi (full, not just wave slice)
        Pibar_star: [C, S] converged Pibar
        v_k: [W, S] Neumann-solved adjoint for this wave
        ws: wave start offset (int)
        sl: [n_ws] int64 — left child clade indices
        sr: [n_ws] int64 — right child clade indices
        reduce_idx: [n_ws] int64 — wave-local parent indices
        wlsp: [n_ws, 1] or [n_ws] — log split probabilities
        log_pD: scalar float — log2 duplication probability
        log_pS: scalar float — log2 speciation probability
        sp_child1, sp_child2: [S] int64 — species child indices
        S: int — number of species

    Returns:
        grad_Pi_l: [n_ws, S] gradient to Pi at left child clades (includes speciation scatter)
        grad_Pi_r: [n_ws, S] gradient to Pi at right child clades (includes speciation scatter)
        grad_Pibar_l: [n_ws, S] gradient to Pibar at left child clades
        grad_Pibar_r: [n_ws, S] gradient to Pibar at right child clades
        param_pD: [n_ws] per-split sum of v_DTS_5[0] (duplication param grad)
        param_pS: [n_ws] per-split sum of v_DTS_5[3]+v_DTS_5[4] (speciation param grad)
    """
    n_ws = sl.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype

    # Squeeze wlsp to [n_ws]
    wlsp_flat = wlsp.squeeze(-1) if wlsp.ndim > 1 else wlsp

    log_pD_arg, log_pS_arg, device_scalar_params = _dts_scalar_param_args(
        log_pD, log_pS, device=device, dtype=dtype
    )

    # Allocate outputs
    grad_Pi_l = torch.empty((n_ws, S), device=device, dtype=dtype)
    grad_Pi_r = torch.empty((n_ws, S), device=device, dtype=dtype)
    grad_Pibar_l = torch.empty((n_ws, S), device=device, dtype=dtype)
    grad_Pibar_r = torch.empty((n_ws, S), device=device, dtype=dtype)
    param_pD = torch.empty(n_ws, device=device, dtype=dtype)
    param_pS = torch.empty(n_ws, device=device, dtype=dtype)

    stride_C = Pi_star.stride(0)
    BLOCK_S = min(256, triton.next_power_of_2(S))

    grid = (n_ws,)
    _dts_cross_backward_kernel[grid](
        Pi_star, Pibar_star,
        v_k,
        active_mask if active_mask is not None else v_k,
        sl, sr, reduce_idx, wlsp_flat,
        log_pD_arg, log_pS_arg,
        sp_child1, sp_child2,
        grad_Pi_l, grad_Pi_r, grad_Pibar_l, grad_Pibar_r,
        param_pD, param_pS,
        ws, S, stride_C, BLOCK_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DEVICE_SCALAR_PARAMS=bool(device_scalar_params),
        DTYPE=_tl_float_dtype(dtype),
    )

    return grad_Pi_l, grad_Pi_r, grad_Pibar_l, grad_Pibar_r, param_pD, param_pS


@triton.jit
def _dts_cross_backward_accum_kernel(
    # Converged values [C, S]
    Pi_star_ptr,
    Pibar_star_ptr,
    # Neumann-solved adjoint [W, S]
    v_k_ptr,
    active_mask_ptr,   # optional [W] bool parent row activity mask
    # Split metadata
    sl_ptr,            # [n_ws] int64 — left child global clade index
    sr_ptr,            # [n_ws] int64 — right child global clade index
    reduce_idx_ptr,    # [n_ws] int64 — wave-local parent index
    wlsp_ptr,          # [n_ws] float — log split probability (squeezed)
    # Scalar params
    log_pD_arg,        # [1] scalar tensor or Python float
    log_pS_arg,        # [1] scalar tensor or Python float
    # Species children [S] int64
    sp_child1_ptr,
    sp_child2_ptr,
    # Outputs
    accumulated_rhs_ptr,  # [C, S], direct Pi adjoints updated atomically
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    param_pD_ptr,         # [n_ws]
    param_pS_ptr,         # [n_ws]
    grad_log_pD_ptr,      # optional scalar accumulation target
    grad_log_pS_ptr,      # optional scalar accumulation target
    grad_mt_ptr,          # optional scalar/[S] accumulation target
    pibar_ud_ptr,         # optional [2 * n_ws, S] initial Pibar VJP subtree values
    pibar_A_ptr,          # optional [2 * n_ws] row sums of pibar_ud
    mt_ptr,               # optional [S] max transfer mat for Pibar denom reuse
    pibar_row_max_ptr,    # optional [C] Pi-row max from forward uniform Pibar
    # Dimensions
    ws,                # wave start offset (parent row = ws + reduce_idx)
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_ATOMICS: tl.constexpr,
    MERGE_S_TERM: tl.constexpr,
    DEVICE_SCALAR_PARAMS: tl.constexpr,
    ACCUM_PARAM_REDUCTIONS: tl.constexpr,
    ACCUM_MT_REDUCTION: tl.constexpr,
    GRAD_MT_SCALAR: tl.constexpr,
    OUTPUT_PIBAR_UD: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """DTS cross-clade backward with direct accumulation of Pi adjoints.

    This is the same VJP as _dts_cross_backward_kernel, but it writes direct
    Pi contributions into accumulated_rhs instead of materializing
    grad_Pi_l/grad_Pi_r and relying on two PyTorch index_add_ calls.
    Pibar adjoints are still materialized because they feed the uniform Pibar
    VJP kernel.
    """
    NEG_LARGE: tl.constexpr = -1e30

    i = tl.program_id(0)

    sl = tl.load(sl_ptr + i)
    sr = tl.load(sr_ptr + i)
    parent_w = tl.load(reduce_idx_ptr + i)
    wlsp = tl.load(wlsp_ptr + i)
    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            out_base = i.to(tl.int64) * S
            ud_l_base = i.to(tl.int64) * S
            ud_r_base = (tl.program_id(0) + 0 + tl.num_programs(0)).to(tl.int64) * S
            zero_scalar = tl.zeros((1,), dtype=DTYPE)
            _scalar_off = tl.arange(0, 1)
            if not ACCUM_PARAM_REDUCTIONS:
                tl.store(param_pD_ptr + i + _scalar_off, zero_scalar)
                tl.store(param_pS_ptr + i + _scalar_off, zero_scalar)
            if OUTPUT_PIBAR_UD:
                tl.store(pibar_A_ptr + i + _scalar_off, zero_scalar)
                tl.store(pibar_A_ptr + tl.num_programs(0) + i + _scalar_off, zero_scalar)
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                mask = s_offs < S
                zero = tl.zeros([BLOCK_S], dtype=DTYPE)
                if OUTPUT_PIBAR_UD:
                    tl.store(pibar_ud_ptr + ud_l_base + s_offs, zero, mask=mask)
                    tl.store(pibar_ud_ptr + ud_r_base + s_offs, zero, mask=mask)
                else:
                    tl.store(grad_Pibar_l_ptr + out_base + s_offs, zero, mask=mask)
                    tl.store(grad_Pibar_r_ptr + out_base + s_offs, zero, mask=mask)
            return
    else:
        parent_active = True

    if DEVICE_SCALAR_PARAMS:
        log_pD = tl.load(log_pD_arg).to(DTYPE)
        log_pS = tl.load(log_pS_arg).to(DTYPE)
    else:
        log_pD = log_pD_arg
        log_pS = log_pS_arg

    pi_l_base = sl.to(tl.int64) * stride_C
    pi_r_base = sr.to(tl.int64) * stride_C
    pibar_l_base = sl.to(tl.int64) * stride_C
    pibar_r_base = sr.to(tl.int64) * stride_C
    parent_pi_base = (ws + parent_w).to(tl.int64) * stride_C
    parent_vk_base = parent_w.to(tl.int64) * S
    out_base = i.to(tl.int64) * S

    sum_pD = tl.zeros((1,), dtype=DTYPE)
    sum_pS = tl.zeros((1,), dtype=DTYPE)
    sum_ud_l = tl.zeros((1,), dtype=DTYPE)
    sum_ud_r = tl.zeros((1,), dtype=DTYPE)
    _scalar_off = tl.arange(0, 1)
    if OUTPUT_PIBAR_UD:
        row_max_l = tl.load(pibar_row_max_ptr + sl).to(DTYPE)
        row_max_r = tl.load(pibar_row_max_ptr + sr).to(DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & parent_active

        Pi_l = tl.load(Pi_star_ptr + pi_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pi_r = tl.load(Pi_star_ptr + pi_r_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_l = tl.load(Pibar_star_ptr + pibar_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_r = tl.load(Pibar_star_ptr + pibar_r_base + s_offs, mask=mask, other=NEG_LARGE)

        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask
        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE)

        Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE)
        v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0)

        d0 = log_pD + Pi_l + Pi_r
        d1 = Pi_l + Pibar_r
        d2 = Pi_r + Pibar_l
        d3 = log_pS + Pi_l_s1 + Pi_r_s2
        d4 = log_pS + Pi_r_s1 + Pi_l_s2

        parent_valid = Pi_parent > NEG_LARGE
        w0 = tl.where(parent_valid, tl.exp2(wlsp + d0 - Pi_parent), tl.zeros_like(d0))
        w1 = tl.where(parent_valid, tl.exp2(wlsp + d1 - Pi_parent), tl.zeros_like(d1))
        w2 = tl.where(parent_valid, tl.exp2(wlsp + d2 - Pi_parent), tl.zeros_like(d2))
        w3 = tl.where(parent_valid, tl.exp2(wlsp + d3 - Pi_parent), tl.zeros_like(d3))
        w4 = tl.where(parent_valid, tl.exp2(wlsp + d4 - Pi_parent), tl.zeros_like(d4))

        vd0 = v_k_val * w0
        vd1 = v_k_val * w1
        vd2 = v_k_val * w2
        vd3 = v_k_val * w3
        vd4 = v_k_val * w4

        pi_l_out = accumulated_rhs_ptr + pi_l_base + s_offs
        pi_r_out = accumulated_rhs_ptr + pi_r_base + s_offs
        if USE_ATOMICS:
            tl.atomic_add(pi_l_out, vd0 + vd1, sem="relaxed", mask=mask)
            tl.atomic_add(pi_r_out, vd0 + vd2, sem="relaxed", mask=mask)
        else:
            pi_l_cur = tl.load(pi_l_out, mask=mask, other=0.0)
            pi_r_cur = tl.load(pi_r_out, mask=mask, other=0.0)
            tl.store(pi_l_out, pi_l_cur + vd0 + vd1, mask=mask)
            tl.store(pi_r_out, pi_r_cur + vd0 + vd2, mask=mask)
        if OUTPUT_PIBAR_UD:
            mt = tl.load(mt_ptr + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
            finite_l = (Pibar_l > -1e29) & mask
            finite_r = (Pibar_r > -1e29) & mask
            inv_denom_l = tl.where(
                finite_l,
                tl.exp2(row_max_l + mt - Pibar_l),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            inv_denom_r = tl.where(
                finite_r,
                tl.exp2(row_max_r + mt - Pibar_r),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            ud_l = vd2 * inv_denom_l
            ud_r = vd1 * inv_denom_r
            tl.store(pibar_ud_ptr + i * S + s_offs, ud_l, mask=valid_mask)
            tl.store(pibar_ud_ptr + (tl.num_programs(0) + i) * S + s_offs, ud_r, mask=valid_mask)
            sum_ud_l += tl.sum(tl.where(mask, ud_l, 0.0), axis=0)
            sum_ud_r += tl.sum(tl.where(mask, ud_r, 0.0), axis=0)
        else:
            tl.store(grad_Pibar_l_ptr + out_base + s_offs, vd2, mask=valid_mask)
            tl.store(grad_Pibar_r_ptr + out_base + s_offs, vd1, mask=valid_mask)

        sum_pD += tl.sum(vd0, axis=0)
        sum_pS += tl.sum(vd3 + vd4, axis=0)
        if ACCUM_MT_REDUCTION:
            mt_contrib = vd1 + vd2
            if GRAD_MT_SCALAR:
                tl.atomic_add(
                    grad_mt_ptr + _scalar_off,
                    tl.sum(tl.where(mask, mt_contrib, 0.0), axis=0),
                    sem="relaxed",
                )
            else:
                tl.atomic_add(
                    grad_mt_ptr + s_offs,
                    mt_contrib,
                    sem="relaxed",
                    mask=mask,
                )

        if MERGE_S_TERM:
            pi_l_c1_out = accumulated_rhs_ptr + pi_l_base + c1
            pi_r_c1_out = accumulated_rhs_ptr + pi_r_base + c1
            pi_r_c2_out = accumulated_rhs_ptr + pi_r_base + c2
            pi_l_c2_out = accumulated_rhs_ptr + pi_l_base + c2
            if USE_ATOMICS:
                tl.atomic_add(pi_l_c1_out, vd3, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c1_out, vd4, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c2_out, vd3, sem="relaxed", mask=c2_valid)
                tl.atomic_add(pi_l_c2_out, vd4, sem="relaxed", mask=c2_valid)
            else:
                pi_l_c1_cur = tl.load(pi_l_c1_out, mask=c1_valid, other=0.0)
                pi_r_c1_cur = tl.load(pi_r_c1_out, mask=c1_valid, other=0.0)
                pi_r_c2_cur = tl.load(pi_r_c2_out, mask=c2_valid, other=0.0)
                pi_l_c2_cur = tl.load(pi_l_c2_out, mask=c2_valid, other=0.0)
                tl.store(pi_l_c1_out, pi_l_c1_cur + vd3, mask=c1_valid)
                tl.store(pi_r_c1_out, pi_r_c1_cur + vd4, mask=c1_valid)
                tl.store(pi_r_c2_out, pi_r_c2_cur + vd3, mask=c2_valid)
                tl.store(pi_l_c2_out, pi_l_c2_cur + vd4, mask=c2_valid)

    if ACCUM_PARAM_REDUCTIONS:
        tl.atomic_add(grad_log_pD_ptr + _scalar_off, sum_pD, sem="relaxed")
        tl.atomic_add(grad_log_pS_ptr + _scalar_off, sum_pS, sem="relaxed")
    else:
        tl.store(param_pD_ptr + i + _scalar_off, sum_pD)
        tl.store(param_pS_ptr + i + _scalar_off, sum_pS)
    if OUTPUT_PIBAR_UD:
        tl.store(pibar_A_ptr + i + _scalar_off, sum_ud_l)
        tl.store(pibar_A_ptr + tl.num_programs(0) + i + _scalar_off, sum_ud_r)

    if not MERGE_S_TERM:
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & parent_active

            c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
            c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
            c1_valid = (c1 < S) & mask
            c2_valid = (c2 < S) & mask

            Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE)
            Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE)
            Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE)
            Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE)

            Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE)
            v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0)

            d3 = log_pS + Pi_l_s1 + Pi_r_s2
            d4 = log_pS + Pi_r_s1 + Pi_l_s2

            parent_valid = Pi_parent > NEG_LARGE
            w3 = tl.where(parent_valid, tl.exp2(wlsp + d3 - Pi_parent), tl.zeros_like(d3))
            w4 = tl.where(parent_valid, tl.exp2(wlsp + d4 - Pi_parent), tl.zeros_like(d4))
            vd3 = v_k_val * w3
            vd4 = v_k_val * w4

            pi_l_c1_out = accumulated_rhs_ptr + pi_l_base + c1
            pi_r_c1_out = accumulated_rhs_ptr + pi_r_base + c1
            pi_r_c2_out = accumulated_rhs_ptr + pi_r_base + c2
            pi_l_c2_out = accumulated_rhs_ptr + pi_l_base + c2
            if USE_ATOMICS:
                tl.atomic_add(pi_l_c1_out, vd3, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c1_out, vd4, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c2_out, vd3, sem="relaxed", mask=c2_valid)
                tl.atomic_add(pi_l_c2_out, vd4, sem="relaxed", mask=c2_valid)
            else:
                pi_l_c1_cur = tl.load(pi_l_c1_out, mask=c1_valid, other=0.0)
                pi_r_c1_cur = tl.load(pi_r_c1_out, mask=c1_valid, other=0.0)
                pi_r_c2_cur = tl.load(pi_r_c2_out, mask=c2_valid, other=0.0)
                pi_l_c2_cur = tl.load(pi_l_c2_out, mask=c2_valid, other=0.0)
                tl.store(pi_l_c1_out, pi_l_c1_cur + vd3, mask=c1_valid)
                tl.store(pi_r_c1_out, pi_r_c1_cur + vd4, mask=c1_valid)
                tl.store(pi_r_c2_out, pi_r_c2_cur + vd3, mask=c2_valid)
                tl.store(pi_l_c2_out, pi_l_c2_cur + vd4, mask=c2_valid)


def dts_cross_backward_accum_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    log_pD, log_pS,
    sp_child1, sp_child2,
    accumulated_rhs,
    S,
    active_mask=None,
    use_atomics=True,
    merge_s_term=False,
    grad_log_pD=None,
    grad_log_pS=None,
    grad_mt=None,
    accum_param_reductions=False,
    accum_mt_reduction=False,
    output_pibar_ud=False,
    mt_squeezed=None,
    pibar_row_max=None,
):
    """Fused DTS backward with direct Pi-adjoint accumulation."""
    n_ws = sl.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype

    wlsp_flat = wlsp.squeeze(-1) if wlsp.ndim > 1 else wlsp
    log_pD_arg, log_pS_arg, device_scalar_params = _dts_scalar_param_args(
        log_pD, log_pS, device=device, dtype=dtype
    )

    if accum_param_reductions and (grad_log_pD is None or grad_log_pS is None):
        raise ValueError("grad_log_pD/grad_log_pS are required when accumulating DTS scalar reductions")
    if accum_param_reductions:
        if grad_log_pD.numel() != 1 or grad_log_pS.numel() != 1:
            raise ValueError("DTS scalar reduction targets must have one element")
    if accum_mt_reduction and grad_mt is None:
        raise ValueError("grad_mt is required when accumulating DTS mt reductions")
    if accum_mt_reduction and grad_mt.numel() not in (1, S):
        raise ValueError("DTS mt reduction target must have one element or S elements")
    if output_pibar_ud and (mt_squeezed is None or pibar_row_max is None):
        raise ValueError("mt_squeezed and pibar_row_max are required when outputting Pibar u_d")
    if output_pibar_ud and mt_squeezed.numel() != S:
        raise ValueError("mt_squeezed must have S elements when outputting Pibar u_d")
    if output_pibar_ud and pibar_row_max.numel() < Pi_star.shape[0]:
        raise ValueError("pibar_row_max must contain one row-max value per Pi row")

    grad_Pibar_l = None if output_pibar_ud else torch.empty((n_ws, S), device=device, dtype=dtype)
    grad_Pibar_r = None if output_pibar_ud else torch.empty((n_ws, S), device=device, dtype=dtype)
    pibar_ud = torch.empty((2 * n_ws, S), device=device, dtype=dtype) if output_pibar_ud else None
    pibar_A = torch.empty((2 * n_ws,), device=device, dtype=dtype) if output_pibar_ud else None
    param_pD = None if accum_param_reductions else torch.empty(n_ws, device=device, dtype=dtype)
    param_pS = None if accum_param_reductions else torch.empty(n_ws, device=device, dtype=dtype)
    param_pD_arg = grad_log_pD if accum_param_reductions else param_pD
    param_pS_arg = grad_log_pS if accum_param_reductions else param_pS
    dummy = pibar_ud if output_pibar_ud else grad_Pibar_l
    grad_log_pD_arg = grad_log_pD if accum_param_reductions else dummy
    grad_log_pS_arg = grad_log_pS if accum_param_reductions else dummy
    grad_mt_arg = grad_mt if accum_mt_reduction else dummy
    pibar_ud_arg = pibar_ud if output_pibar_ud else dummy
    pibar_A_arg = pibar_A if output_pibar_ud else dummy
    mt_arg = mt_squeezed.contiguous() if output_pibar_ud and not mt_squeezed.is_contiguous() else mt_squeezed
    pibar_row_max_arg = (
        pibar_row_max.contiguous()
        if output_pibar_ud and not pibar_row_max.is_contiguous()
        else pibar_row_max
    )
    mt_arg = mt_arg if output_pibar_ud else dummy
    pibar_row_max_arg = pibar_row_max_arg if output_pibar_ud else dummy
    grad_mt_scalar = bool(accum_mt_reduction and grad_mt.numel() == 1)

    stride_C = Pi_star.stride(0)
    BLOCK_S = min(256, triton.next_power_of_2(S))

    _dts_cross_backward_accum_kernel[(n_ws,)](
        Pi_star, Pibar_star,
        v_k,
        active_mask if active_mask is not None else v_k,
        sl, sr, reduce_idx, wlsp_flat,
        log_pD_arg, log_pS_arg,
        sp_child1, sp_child2,
        accumulated_rhs,
        grad_Pibar_l if grad_Pibar_l is not None else pibar_ud,
        grad_Pibar_r if grad_Pibar_r is not None else pibar_ud,
        param_pD_arg, param_pS_arg,
        grad_log_pD_arg, grad_log_pS_arg, grad_mt_arg,
        pibar_ud_arg, pibar_A_arg, mt_arg, pibar_row_max_arg,
        ws, S, stride_C, BLOCK_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ATOMICS=bool(use_atomics),
        MERGE_S_TERM=bool(merge_s_term),
        DEVICE_SCALAR_PARAMS=bool(device_scalar_params),
        ACCUM_PARAM_REDUCTIONS=bool(accum_param_reductions),
        ACCUM_MT_REDUCTION=bool(accum_mt_reduction),
        GRAD_MT_SCALAR=bool(grad_mt_scalar),
        OUTPUT_PIBAR_UD=bool(output_pibar_ud),
        DTYPE=_tl_float_dtype(dtype),
    )

    if output_pibar_ud:
        return pibar_ud, pibar_A, param_pD, param_pS
    return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS


@triton.jit
def _add_grouped_dts_pi_accum_kernel(
    group_children_ptr,   # [n_groups]
    grouped_grad_ptr,     # [n_groups, S]
    accumulated_rhs_ptr,  # [C, S]
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    group = tl.program_id(0)
    block = tl.program_id(1)

    child = tl.load(group_children_ptr + group)
    s_offs = block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    grad = tl.load(grouped_grad_ptr + group * S + s_offs, mask=mask, other=0.0)
    out = accumulated_rhs_ptr + child * stride_C + s_offs
    cur = tl.load(out, mask=mask, other=0.0)
    tl.store(out, cur + grad, mask=mask)


def dts_cross_backward_accum_grouped_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    log_pD, log_pS,
    sp_child1, sp_child2,
    accumulated_rhs,
    S,
    active_mask=None,
    group_children=None,
    group_inverse=None,
):
    """Two-stage DTS backward accumulation grouped by child clade.

    Stage 1 reuses the per-split fused DTS VJP to build local Pi/Pibar
    adjoints. Stage 2 reduces left/right Pi adjoints by child row in a compact
    scratch buffer, then adds each reduced child row once to accumulated_rhs.
    This is an opt-in high-fanout alternative to direct atomics.
    """
    n_ws = sl.shape[0]
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    (grad_Pi_l, grad_Pi_r, grad_Pibar_l, grad_Pibar_r,
     param_pD, param_pS) = dts_cross_backward_fused(
        Pi_star, Pibar_star, v_k, ws,
        sl, sr, reduce_idx, wlsp,
        log_pD, log_pS,
        sp_child1, sp_child2, S,
        active_mask=active_mask,
    )

    if n_ws == 0:
        return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS

    if group_children is None or group_inverse is None:
        all_children = torch.cat((sl, sr), dim=0)
        group_children, group_inverse = torch.unique(
            all_children,
            sorted=True,
            return_inverse=True,
        )

    group_children = group_children.contiguous()
    group_inverse = group_inverse.contiguous()
    n_groups = group_children.shape[0]
    if n_groups == 0:
        return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS

    BLOCK_S = min(256, triton.next_power_of_2(S))
    grouped_grad = torch.zeros((n_groups, S), device=Pi_star.device, dtype=Pi_star.dtype)
    use_active_mask = active_mask is not None

    _group_cross_pibar_grad_kernel[(2 * n_ws, triton.cdiv(S, BLOCK_S))](
        grad_Pi_l,
        grad_Pi_r,
        group_inverse,
        reduce_idx if reduce_idx is not None else group_inverse,
        active_mask if active_mask is not None else grouped_grad,
        grouped_grad,
        grouped_grad,
        n_ws,
        S,
        BLOCK_S,
        USE_ACTIVE_MASK=use_active_mask,
        TRACK_GROUP_ACTIVE=False,
        num_warps=4,
    )

    _add_grouped_dts_pi_accum_kernel[(n_groups, triton.cdiv(S, BLOCK_S))](
        group_children,
        grouped_grad,
        accumulated_rhs,
        S,
        accumulated_rhs.stride(0),
        BLOCK_S,
        num_warps=4,
    )

    return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS


# =========================================================================
# Uniform Pibar VJP for cross-clade gradients
# =========================================================================

@triton.jit
def _pibar_row_stats_kernel(
    Pi_star_ptr,
    row_max_ptr,
    row_sum_ptr,
    C: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Precompute row_max and shifted row_sum for uniform Pibar VJP."""
    NEG_LARGE: tl.constexpr = -1e30
    row = tl.program_id(0)

    row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    pi_base = row.to(tl.int64) * stride_C
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        tile_max = tl.max(pi_val, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
        row_max = new_max

    scalar = tl.arange(0, 1)
    tl.store(row_max_ptr + row + scalar, row_max)
    tl.store(row_sum_ptr + row + scalar, row_sum)


def pibar_row_stats_fused(Pi_star):
    """Compute compact row stats used by uniform Pibar VJP kernels."""
    C, S = Pi_star.shape
    row_max = torch.empty((C,), device=Pi_star.device, dtype=Pi_star.dtype)
    row_sum = torch.empty((C,), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    _pibar_row_stats_kernel[(C,)](
        Pi_star,
        row_max,
        row_sum,
        C,
        S,
        Pi_star.stride(0),
        BLOCK_S,
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )
    return row_max, row_sum


@triton.jit
def _uniform_cross_pibar_vjp_kernel(
    Pi_star_ptr,          # [C, S]
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    ancestor_cols_ptr,    # [MAX_ANCESTOR_DEPTH, S]
    row_max_ptr,          # optional [C]
    row_sum_ptr,          # optional [C]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    correction_buf_ptr,   # [2 * n_ws, S]
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_ROW_STATS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Apply the VJP of uniform Pibar for one cross-DTS child side.

    For a child row with incoming Pibar adjoint u:

        p = exp2(Pi - row_max)
        denom[s] = sum_f p[f] - sum_{a in ancestors(s)} p[a]
        u_d[s] = u[s] / denom[s]
        grad_Pi[f] = p[f] * (sum_s u_d[s] -
                             sum_{s: f in ancestors(s)} u_d[s])

    Each program handles either the left or right child of one split and
    atomically accumulates grad_Pi into accumulated_rhs[child].
    """
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws

    child_l = tl.load(sl_ptr + split_i)
    child_r = tl.load(sr_ptr + split_i)
    child = tl.where(is_right, child_r, child_l)
    if USE_ACTIVE_MASK:
        parent_w = tl.load(reduce_idx_ptr + split_i)
        row_active = tl.load(active_mask_ptr + parent_w)
        if row_active == 0:
            return
    else:
        row_active = True

    pi_base = child.to(tl.int64) * stride_C
    grad_base = split_i.to(tl.int64) * S
    corr_base = row.to(tl.int64) * S

    # Clear the correction row owned by this program.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        tl.store(correction_buf_ptr + corr_base + s_offs,
                 tl.zeros([BLOCK_S], dtype=DTYPE), mask=mask)

    if USE_ROW_STATS:
        row_max = tl.load(row_max_ptr + child)
        row_sum = tl.load(row_sum_ptr + child)
    else:
        # Row max and shifted row sum for p = exp2(Pi - row_max).
        row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
            tile_max = tl.max(pi_val, axis=0)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
            row_max = new_max

    # Build correction[f] = sum_{s: f ancestor of s} u_d[s].
    A = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active

        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        for k in range(0, MAX_ANCESTOR_DEPTH):
            anc = tl.load(ancestor_cols_ptr + k * S + s_offs, mask=mask, other=-1)
            anc_valid = mask & (anc >= 0) & (anc < S)
            pi_anc = tl.load(Pi_star_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
            ancestor_sum += tl.where(
                anc_valid,
                tl.exp2(pi_anc - row_max),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )

        denom = row_sum - ancestor_sum

        grad_l = tl.load(grad_Pibar_l_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_r = tl.load(grad_Pibar_r_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_u = tl.where(is_right, grad_r, grad_l)
        u_d = tl.where(denom > 0.0, grad_u / denom, tl.zeros([BLOCK_S], dtype=DTYPE))
        A += tl.sum(u_d, axis=0)

        for k in range(0, MAX_ANCESTOR_DEPTH):
            anc = tl.load(ancestor_cols_ptr + k * S + s_offs, mask=mask, other=-1)
            anc_valid = mask & (anc >= 0) & (anc < S)
            tl.atomic_add(correction_buf_ptr + corr_base + anc, u_d, sem="relaxed", mask=anc_valid)

    # Add p[f] * (A - correction[f]) into the child row's Pi adjoint.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        correction = tl.load(correction_buf_ptr + corr_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - correction)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


def uniform_cross_pibar_vjp_fused(
    Pi_star,
    grad_Pibar_l,
    grad_Pibar_r,
    sl,
    sr,
    ancestor_cols,
    accumulated_rhs,
    S,
    active_mask=None,
    reduce_idx=None,
    row_stats=None,
):
    """Fused uniform-Pibar VJP for cross-DTS child gradients.

    This replaces:
      cat(left/right Pibar grads) -> p_prime @ ancestors_T ->
      ancestors_T @ u_d.T -> index_add into accumulated_rhs.

    The operation is in-place on accumulated_rhs.
    """
    n_ws = sl.shape[0]
    if n_ws == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    correction_buf = torch.empty((2 * n_ws, S), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    _uniform_cross_pibar_vjp_kernel[(2 * n_ws,)](
        Pi_star,
        grad_Pibar_l,
        grad_Pibar_r,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else grad_Pibar_l,
        ancestor_cols,
        row_stats[0] if row_stats is not None else Pi_star,
        row_stats[1] if row_stats is not None else Pi_star,
        accumulated_rhs,
        correction_buf,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ROW_STATS=bool(row_stats is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )


@triton.jit
def _group_cross_pibar_grad_kernel(
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    group_inverse_ptr,    # [2 * n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    grouped_grad_ptr,     # [n_groups, S]
    group_active_ptr,     # optional [n_groups] bool
    n_ws: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    TRACK_GROUP_ACTIVE: tl.constexpr,
):
    side = tl.program_id(0)
    block = tl.program_id(1)
    split_i = tl.where(side < n_ws, side, side - n_ws)
    is_right = side >= n_ws

    if USE_ACTIVE_MASK:
        parent_w = tl.load(reduce_idx_ptr + split_i)
        row_active = tl.load(active_mask_ptr + parent_w)
        if row_active == 0:
            return

    group = tl.load(group_inverse_ptr + side)
    if TRACK_GROUP_ACTIVE:
        tl.store(group_active_ptr + group, 1)

    s_offs = block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    grad_base = split_i.to(tl.int64) * S
    grad_l = tl.load(grad_Pibar_l_ptr + grad_base + s_offs, mask=mask, other=0.0)
    grad_r = tl.load(grad_Pibar_r_ptr + grad_base + s_offs, mask=mask, other=0.0)
    grad = tl.where(is_right, grad_r, grad_l)
    tl.atomic_add(grouped_grad_ptr + group * S + s_offs, grad, sem="relaxed", mask=mask)


@triton.jit
def _uniform_cross_pibar_vjp_tree_grouped_kernel(
    Pi_star_ptr,          # [C, S]
    grouped_grad_ptr,     # [n_groups, S]
    group_children_ptr,   # [n_groups]
    ancestor_cols_ptr,    # [MAX_ANCESTOR_DEPTH, S]
    sp_child1_ptr,        # [S]
    sp_child2_ptr,        # [S]
    level_parents_ptr,    # [N_LEVELS, MAX_LEVEL_WIDTH]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    subtree_buf_ptr,      # [n_groups, S]
    group_active_ptr,     # optional [n_groups] bool
    n_groups: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    MAX_LEVEL_WIDTH: tl.constexpr,
    USE_GROUP_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar VJP after reducing split-side adjoints by child clade."""
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    if USE_GROUP_ACTIVE:
        group_active = tl.load(group_active_ptr + row)
        if group_active == 0:
            return

    child = tl.load(group_children_ptr + row)
    pi_base = child.to(tl.int64) * stride_C
    grad_base = row.to(tl.int64) * S
    subtree_base = row.to(tl.int64) * S

    row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        tile_max = tl.max(pi_val, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
        row_max = new_max

    A = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        for k in range(0, MAX_ANCESTOR_DEPTH):
            anc = tl.load(ancestor_cols_ptr + k * S + s_offs, mask=mask, other=-1)
            anc_valid = mask & (anc >= 0) & (anc < S)
            pi_anc = tl.load(Pi_star_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
            ancestor_sum += tl.where(
                anc_valid,
                tl.exp2(pi_anc - row_max),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )

        denom = row_sum - ancestor_sum
        grad_u = tl.load(grouped_grad_ptr + grad_base + s_offs, mask=mask, other=0.0)
        u_d = tl.where(denom > 0.0, grad_u / denom, tl.zeros([BLOCK_S], dtype=DTYPE))
        A += tl.sum(u_d, axis=0)
        tl.store(subtree_buf_ptr + subtree_base + s_offs, u_d, mask=mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
            p_offs = p_start + tl.arange(0, BLOCK_S)
            parent = tl.load(
                level_parents_ptr + level * MAX_LEVEL_WIDTH + p_offs,
                mask=p_offs < MAX_LEVEL_WIDTH,
                other=-1,
            )
            parent_valid = (parent >= 0) & (parent < S)
            c1 = tl.load(sp_child1_ptr + parent, mask=parent_valid, other=S)
            c2 = tl.load(sp_child2_ptr + parent, mask=parent_valid, other=S)
            c1_valid = parent_valid & (c1 < S)
            c2_valid = parent_valid & (c2 < S)

            parent_val = tl.load(subtree_buf_ptr + subtree_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(subtree_buf_ptr + subtree_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(subtree_buf_ptr + subtree_base + c2, mask=c2_valid, other=0.0)
            tl.store(subtree_buf_ptr + subtree_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(subtree_buf_ptr + subtree_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


@triton.jit
def _uniform_cross_pibar_vjp_tree_kernel(
    Pi_star_ptr,          # [C, S]
    Pibar_star_ptr,       # [C, S], used when reusing forward Pibar denominators
    mt_ptr,               # [S], used when reusing forward Pibar denominators
    pibar_row_max_ptr,    # [C], used when reusing forward Pibar denominators
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    ancestor_cols_ptr,    # [MAX_ANCESTOR_DEPTH, S]
    row_max_ptr,          # optional [C]
    row_sum_ptr,          # optional [C]
    sp_child1_ptr,        # [S]
    sp_child2_ptr,        # [S]
    level_parents_ptr,    # [N_LEVELS, MAX_LEVEL_WIDTH]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    subtree_buf_ptr,      # [2 * n_ws, S]
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    MAX_LEVEL_WIDTH: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_ROW_STATS: tl.constexpr,
    USE_PIBAR_DENOM_STATS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar VJP using a descendant/subtree gather.

    This computes the same correction term as _uniform_cross_pibar_vjp_kernel,
    but avoids scattering every descendant into all of its ancestors.  Instead
    it writes u_d into subtree_buf and performs a bottom-up tree reduction:

        subtree_sum[parent] = u_d[parent] + subtree_sum[child1] + subtree_sum[child2]

    After that, correction[f] is subtree_sum[f].
    """
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws

    child_l = tl.load(sl_ptr + split_i)
    child_r = tl.load(sr_ptr + split_i)
    child = tl.where(is_right, child_r, child_l)
    if USE_ACTIVE_MASK:
        parent_w = tl.load(reduce_idx_ptr + split_i)
        row_active = tl.load(active_mask_ptr + parent_w)
        if row_active == 0:
            return
    else:
        row_active = True

    pi_base = child.to(tl.int64) * stride_C
    grad_base = split_i.to(tl.int64) * S
    subtree_base = row.to(tl.int64) * S

    if USE_PIBAR_DENOM_STATS:
        row_max = tl.load(pibar_row_max_ptr + child)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    elif USE_ROW_STATS:
        row_max = tl.load(row_max_ptr + child)
        row_sum = tl.load(row_sum_ptr + child)
    else:
        row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
            tile_max = tl.max(pi_val, axis=0)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
            row_max = new_max

    # Initial subtree_buf contains u_d for every species.  Leaves are already
    # final subtree sums; internal nodes are completed by the bottom-up pass.
    A = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active

        grad_l = tl.load(grad_Pibar_l_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_r = tl.load(grad_Pibar_r_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_u = tl.where(is_right, grad_r, grad_l)
        if USE_PIBAR_DENOM_STATS:
            pibar_val = tl.load(Pibar_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
            mt = tl.load(mt_ptr + s_offs, mask=mask, other=0.0)
            finite_pibar = pibar_val > -1e29
            inv_denom = tl.where(
                finite_pibar,
                tl.exp2(row_max + mt - pibar_val),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            u_d = grad_u * inv_denom
        else:
            ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
            for k in range(0, MAX_ANCESTOR_DEPTH):
                anc = tl.load(ancestor_cols_ptr + k * S + s_offs, mask=mask, other=-1)
                anc_valid = mask & (anc >= 0) & (anc < S)
                pi_anc = tl.load(Pi_star_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
                ancestor_sum += tl.where(
                    anc_valid,
                    tl.exp2(pi_anc - row_max),
                    tl.zeros([BLOCK_S], dtype=DTYPE),
                )

            denom = row_sum - ancestor_sum
            u_d = tl.where(denom > 0.0, grad_u / denom, tl.zeros([BLOCK_S], dtype=DTYPE))
        A += tl.sum(u_d, axis=0)
        tl.store(subtree_buf_ptr + subtree_base + s_offs, u_d, mask=mask)

    tl.debug_barrier()

    # Bottom-up subtree reduction.  level_parents is ordered from parents of
    # leaves up to the root, so every child subtree has already been computed.
    for level in range(0, N_LEVELS):
        for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
            p_offs = p_start + tl.arange(0, BLOCK_S)
            parent = tl.load(
                level_parents_ptr + level * MAX_LEVEL_WIDTH + p_offs,
                mask=p_offs < MAX_LEVEL_WIDTH,
                other=-1,
            )
            parent_valid = (parent >= 0) & (parent < S) & row_active
            c1 = tl.load(sp_child1_ptr + parent, mask=parent_valid, other=S)
            c2 = tl.load(sp_child2_ptr + parent, mask=parent_valid, other=S)
            c1_valid = parent_valid & (c1 < S)
            c2_valid = parent_valid & (c2 < S)

            parent_val = tl.load(subtree_buf_ptr + subtree_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(subtree_buf_ptr + subtree_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(subtree_buf_ptr + subtree_base + c2, mask=c2_valid, other=0.0)
            tl.store(subtree_buf_ptr + subtree_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(subtree_buf_ptr + subtree_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        # Duplicate child clades across splits still require an atomic add into
        # accumulated_rhs.  The subtree correction itself is atomic-free.
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


@triton.jit
def _uniform_cross_pibar_vjp_tree_prefix_kernel(
    Pi_star_ptr,          # [C, S]
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    row_max_ptr,          # optional [C]
    row_sum_ptr,          # optional [C]
    sp_parent_ptr,        # [S]
    sp_child1_ptr,        # [S]
    sp_child2_ptr,        # [S]
    depth_nodes_ptr,      # [N_DEPTHS, MAX_DEPTH_WIDTH], top-down
    level_parents_ptr,    # [N_LEVELS, MAX_LEVEL_WIDTH], bottom-up
    accumulated_rhs_ptr,  # [C, S], updated atomically
    subtree_buf_ptr,      # [2 * n_ws, S], reused as prefix then subtree
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_DEPTHS: tl.constexpr,
    MAX_DEPTH_WIDTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    MAX_LEVEL_WIDTH: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_ROW_STATS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar VJP with top-down species-tree denominator prefixes."""
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws

    child_l = tl.load(sl_ptr + split_i)
    child_r = tl.load(sr_ptr + split_i)
    child = tl.where(is_right, child_r, child_l)
    if USE_ACTIVE_MASK:
        parent_w = tl.load(reduce_idx_ptr + split_i)
        row_active = tl.load(active_mask_ptr + parent_w)
        if row_active == 0:
            return
    else:
        row_active = True

    pi_base = child.to(tl.int64) * stride_C
    grad_base = split_i.to(tl.int64) * S
    row_base = row.to(tl.int64) * S

    if USE_ROW_STATS:
        row_max = tl.load(row_max_ptr + child)
        row_sum = tl.load(row_sum_ptr + child)
    else:
        row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
            tile_max = tl.max(pi_val, axis=0)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
            row_max = new_max

    # subtree_buf first holds prefix[s] = sum_{a in ancestors(s)} p[a].
    for depth in range(0, N_DEPTHS):
        for n_start in range(0, MAX_DEPTH_WIDTH, BLOCK_S):
            n_offs = n_start + tl.arange(0, BLOCK_S)
            node = tl.load(
                depth_nodes_ptr + depth * MAX_DEPTH_WIDTH + n_offs,
                mask=n_offs < MAX_DEPTH_WIDTH,
                other=-1,
            )
            valid = (node >= 0) & (node < S) & row_active
            pi_node = tl.load(Pi_star_ptr + pi_base + node, mask=valid, other=NEG_LARGE)
            p_node = tl.exp2(pi_node - row_max)
            parent = tl.load(sp_parent_ptr + node, mask=valid, other=-1)
            has_parent = valid & (parent >= 0) & (parent < S)
            parent_prefix = tl.load(subtree_buf_ptr + row_base + parent, mask=has_parent, other=0.0)
            tl.store(subtree_buf_ptr + row_base + node, parent_prefix + p_node, mask=valid)
        tl.debug_barrier()

    A = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active

        ancestor_sum = tl.load(subtree_buf_ptr + row_base + s_offs, mask=mask, other=0.0)
        denom = row_sum - ancestor_sum
        grad_l = tl.load(grad_Pibar_l_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_r = tl.load(grad_Pibar_r_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_u = tl.where(is_right, grad_r, grad_l)
        u_d = tl.where(denom > 0.0, grad_u / denom, tl.zeros([BLOCK_S], dtype=DTYPE))
        A += tl.sum(u_d, axis=0)
        tl.store(subtree_buf_ptr + row_base + s_offs, u_d, mask=mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
            p_offs = p_start + tl.arange(0, BLOCK_S)
            parent = tl.load(
                level_parents_ptr + level * MAX_LEVEL_WIDTH + p_offs,
                mask=p_offs < MAX_LEVEL_WIDTH,
                other=-1,
            )
            parent_valid = (parent >= 0) & (parent < S) & row_active
            c1 = tl.load(sp_child1_ptr + parent, mask=parent_valid, other=S)
            c2 = tl.load(sp_child2_ptr + parent, mask=parent_valid, other=S)
            c1_valid = parent_valid & (c1 < S)
            c2_valid = parent_valid & (c2 < S)

            parent_val = tl.load(subtree_buf_ptr + row_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(subtree_buf_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(subtree_buf_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(subtree_buf_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(subtree_buf_ptr + row_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


def uniform_cross_pibar_vjp_tree_fused(
    Pi_star,
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
    active_mask=None,
    reduce_idx=None,
    row_stats=None,
    Pibar_star=None,
    mt_squeezed=None,
    pibar_row_max=None,
):
    """Uniform-Pibar VJP using bottom-up descendant/subtree gathering."""
    n_ws = sl.shape[0]
    if n_ws == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    subtree_buf = torch.empty((2 * n_ws, S), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    n_levels = level_parents.shape[0]
    max_level_width = level_parents.shape[1]
    use_pibar_denom_stats = (
        Pibar_star is not None
        and pibar_row_max is not None
        and mt_squeezed is not None
    )
    _uniform_cross_pibar_vjp_tree_kernel[(2 * n_ws,)](
        Pi_star,
        Pibar_star if use_pibar_denom_stats else Pi_star,
        mt_squeezed if use_pibar_denom_stats else grad_Pibar_l,
        pibar_row_max if use_pibar_denom_stats else sl,
        grad_Pibar_l,
        grad_Pibar_r,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else grad_Pibar_l,
        ancestor_cols,
        row_stats[0] if row_stats is not None else Pi_star,
        row_stats[1] if row_stats is not None else Pi_star,
        sp_child1,
        sp_child2,
        level_parents,
        accumulated_rhs,
        subtree_buf,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        N_LEVELS=n_levels,
        MAX_LEVEL_WIDTH=max_level_width,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ROW_STATS=bool(row_stats is not None),
        USE_PIBAR_DENOM_STATS=bool(use_pibar_denom_stats),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )


@triton.jit
def _uniform_cross_pibar_vjp_tree_from_ud_kernel(
    Pi_star_ptr,          # [C, S]
    pibar_ud_ptr,         # [2 * n_ws, S], initial subtree values u_d
    pibar_A_ptr,          # [2 * n_ws], sum_s u_d[s] per split side
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    pibar_row_max_ptr,    # [C], Pi-row max from forward uniform Pibar
    sp_child1_ptr,        # [S]
    sp_child2_ptr,        # [S]
    level_parents_ptr,    # [N_LEVELS, MAX_LEVEL_WIDTH]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    MAX_LEVEL_WIDTH: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar VJP tree correction when DTS has already staged u_d."""
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws

    child_l = tl.load(sl_ptr + split_i)
    child_r = tl.load(sr_ptr + split_i)
    child = tl.where(is_right, child_r, child_l)
    if USE_ACTIVE_MASK:
        parent_w = tl.load(reduce_idx_ptr + split_i)
        row_active = tl.load(active_mask_ptr + parent_w)
        if row_active == 0:
            return
    else:
        row_active = True

    pi_base = child.to(tl.int64) * stride_C
    row_base = row.to(tl.int64) * S
    row_max = tl.load(pibar_row_max_ptr + child).to(DTYPE)
    A = tl.load(pibar_A_ptr + row).to(DTYPE)

    # pibar_ud is intentionally reused in-place as subtree_buf.  It already
    # contains u_d for each species from the DTS kernel.
    tl.debug_barrier()
    for level in range(0, N_LEVELS):
        for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
            p_offs = p_start + tl.arange(0, BLOCK_S)
            parent = tl.load(
                level_parents_ptr + level * MAX_LEVEL_WIDTH + p_offs,
                mask=p_offs < MAX_LEVEL_WIDTH,
                other=-1,
            )
            parent_valid = (parent >= 0) & (parent < S) & row_active
            c1 = tl.load(sp_child1_ptr + parent, mask=parent_valid, other=S)
            c2 = tl.load(sp_child2_ptr + parent, mask=parent_valid, other=S)
            c1_valid = parent_valid & (c1 < S)
            c2_valid = parent_valid & (c2 < S)

            parent_val = tl.load(pibar_ud_ptr + row_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(pibar_ud_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(pibar_ud_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(pibar_ud_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(pibar_ud_ptr + row_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


def uniform_cross_pibar_vjp_tree_from_ud_fused(
    Pi_star,
    pibar_ud,
    pibar_A,
    sl,
    sr,
    sp_child1,
    sp_child2,
    level_parents,
    accumulated_rhs,
    S,
    active_mask=None,
    reduce_idx=None,
    pibar_row_max=None,
):
    """Uniform-Pibar VJP tree correction from DTS-staged u_d."""
    n_ws = sl.shape[0]
    if n_ws == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")
    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for DTS-staged Pibar VJP")

    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)
    _uniform_cross_pibar_vjp_tree_from_ud_kernel[(2 * n_ws,)](
        Pi_star,
        pibar_ud,
        pibar_A,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else pibar_ud,
        pibar_row_max,
        sp_child1,
        sp_child2,
        level_parents,
        accumulated_rhs,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        N_LEVELS=level_parents.shape[0],
        MAX_LEVEL_WIDTH=level_parents.shape[1],
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )


def uniform_cross_pibar_vjp_tree_prefix_fused(
    Pi_star,
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
    active_mask=None,
    reduce_idx=None,
    row_stats=None,
):
    """Uniform-Pibar VJP using top-down denominator prefixes."""
    n_ws = sl.shape[0]
    if n_ws == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    subtree_buf = torch.empty((2 * n_ws, S), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    _uniform_cross_pibar_vjp_tree_prefix_kernel[(2 * n_ws,)](
        Pi_star,
        grad_Pibar_l,
        grad_Pibar_r,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else grad_Pibar_l,
        row_stats[0] if row_stats is not None else Pi_star,
        row_stats[1] if row_stats is not None else Pi_star,
        sp_parent,
        sp_child1,
        sp_child2,
        depth_nodes,
        level_parents,
        accumulated_rhs,
        subtree_buf,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        N_DEPTHS=depth_nodes.shape[0],
        MAX_DEPTH_WIDTH=depth_nodes.shape[1],
        N_LEVELS=level_parents.shape[0],
        MAX_LEVEL_WIDTH=level_parents.shape[1],
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ROW_STATS=bool(row_stats is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )


def uniform_cross_pibar_vjp_tree_grouped_fused(
    Pi_star,
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
    active_mask=None,
    reduce_idx=None,
):
    """Reduce cross-DTS Pibar adjoints by child, then run one VJP per child."""
    n_ws = grad_Pibar_l.shape[0]
    n_groups = group_children.shape[0]
    if n_ws == 0 or n_groups == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    BLOCK_S = min(256, triton.next_power_of_2(S))
    grouped_grad = torch.zeros((n_groups, S), device=Pi_star.device, dtype=Pi_star.dtype)
    track_group_active = active_mask is not None
    group_active = (
        torch.zeros((n_groups,), device=Pi_star.device, dtype=torch.bool)
        if track_group_active
        else grouped_grad
    )

    _group_cross_pibar_grad_kernel[(2 * n_ws, triton.cdiv(S, BLOCK_S))](
        grad_Pibar_l,
        grad_Pibar_r,
        group_inverse,
        reduce_idx if reduce_idx is not None else group_inverse,
        active_mask if active_mask is not None else grouped_grad,
        grouped_grad,
        group_active,
        n_ws,
        S,
        BLOCK_S,
        USE_ACTIVE_MASK=track_group_active,
        TRACK_GROUP_ACTIVE=track_group_active,
        num_warps=4,
    )

    subtree_buf = torch.empty((n_groups, S), device=Pi_star.device, dtype=Pi_star.dtype)
    stride_C = Pi_star.stride(0)
    _uniform_cross_pibar_vjp_tree_grouped_kernel[(n_groups,)](
        Pi_star,
        grouped_grad,
        group_children,
        ancestor_cols,
        sp_child1,
        sp_child2,
        level_parents,
        accumulated_rhs,
        subtree_buf,
        group_active,
        n_groups,
        S,
        stride_C,
        BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        N_LEVELS=level_parents.shape[0],
        MAX_LEVEL_WIDTH=level_parents.shape[1],
        USE_GROUP_ACTIVE=track_group_active,
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )


@triton.jit
def _uniform_cross_pibar_vjp_grouped_tree_kernel(
    Pi_star_ptr,          # [C, S]
    child_ids_ptr,        # [n_child_rows]
    grad_u_ptr,           # [n_child_rows, S], already reduced by child clade
    ancestor_cols_ptr,    # [MAX_ANCESTOR_DEPTH, S]
    row_max_ptr,          # optional [C]
    row_sum_ptr,          # optional [C]
    sp_child1_ptr,        # [S]
    sp_child2_ptr,        # [S]
    level_parents_ptr,    # [N_LEVELS, MAX_LEVEL_WIDTH]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    subtree_buf_ptr,      # [n_child_rows, S]
    n_child_rows: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    MAX_LEVEL_WIDTH: tl.constexpr,
    USE_ROW_STATS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar VJP after reducing incoming Pibar adjoints by child row."""
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    child = tl.load(child_ids_ptr + row)
    pi_base = child.to(tl.int64) * stride_C
    row_base = row.to(tl.int64) * S

    if USE_ROW_STATS:
        row_max = tl.load(row_max_ptr + child)
        row_sum = tl.load(row_sum_ptr + child)
    else:
        row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
            tile_max = tl.max(pi_val, axis=0)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
            row_max = new_max

    A = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        for k in range(0, MAX_ANCESTOR_DEPTH):
            anc = tl.load(ancestor_cols_ptr + k * S + s_offs, mask=mask, other=-1)
            anc_valid = mask & (anc >= 0) & (anc < S)
            pi_anc = tl.load(Pi_star_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
            ancestor_sum += tl.where(
                anc_valid,
                tl.exp2(pi_anc - row_max),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )

        denom = row_sum - ancestor_sum
        grad_u = tl.load(grad_u_ptr + row_base + s_offs, mask=mask, other=0.0)
        u_d = tl.where(denom > 0.0, grad_u / denom, tl.zeros([BLOCK_S], dtype=DTYPE))
        A += tl.sum(u_d, axis=0)
        tl.store(subtree_buf_ptr + row_base + s_offs, u_d, mask=mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
            p_offs = p_start + tl.arange(0, BLOCK_S)
            parent = tl.load(
                level_parents_ptr + level * MAX_LEVEL_WIDTH + p_offs,
                mask=p_offs < MAX_LEVEL_WIDTH,
                other=-1,
            )
            parent_valid = (parent >= 0) & (parent < S)
            c1 = tl.load(sp_child1_ptr + parent, mask=parent_valid, other=S)
            c2 = tl.load(sp_child2_ptr + parent, mask=parent_valid, other=S)
            c1_valid = parent_valid & (c1 < S)
            c2_valid = parent_valid & (c2 < S)

            parent_val = tl.load(subtree_buf_ptr + row_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(subtree_buf_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(subtree_buf_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(subtree_buf_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(subtree_buf_ptr + row_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


def uniform_cross_pibar_vjp_grouped_tree_fused(
    Pi_star,
    child_ids,
    grad_u,
    ancestor_cols,
    sp_child1,
    sp_child2,
    level_parents,
    accumulated_rhs,
    S,
    row_stats=None,
):
    """Uniform-Pibar VJP for pre-reduced unique child rows."""
    n_child_rows = child_ids.shape[0]
    if n_child_rows == 0:
        return

    subtree_buf = torch.empty((n_child_rows, S), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    _uniform_cross_pibar_vjp_grouped_tree_kernel[(n_child_rows,)](
        Pi_star,
        child_ids,
        grad_u,
        ancestor_cols,
        row_stats[0] if row_stats is not None else Pi_star,
        row_stats[1] if row_stats is not None else Pi_star,
        sp_child1,
        sp_child2,
        level_parents,
        accumulated_rhs,
        subtree_buf,
        n_child_rows,
        S,
        stride_C,
        BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        N_LEVELS=level_parents.shape[0],
        MAX_LEVEL_WIDTH=level_parents.shape[1],
        USE_ROW_STATS=bool(row_stats is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )
