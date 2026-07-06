# seg_lse_hdim_triton.py
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


HDIM_CONFIGS = [
    # (BLOCK_H, BLOCK_S, num_warps, num_stages)
    (32, 32, 4, 2),
    (32, 64, 4, 2),
    (32, 128, 4, 2),
    (32, 256, 8, 2),
    (64, 64, 4, 2),
    (64, 128, 4, 2),
    (64, 256, 8, 2),
    (128, 64, 4, 2),
    (128, 128, 4, 2),
    (128, 256, 8, 2),
    (256, 64, 4, 2),
    (256, 128, 8, 2),
    (256, 256, 8, 2),
]


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_H": bh, "BLOCK_S": bs}, num_warps=nw, num_stages=ns)
#         for (bh, bs, nw, ns) in HDIM_CONFIGS
#     ],
#     key=["S"],
# )
@triton.jit
def _seg_lse_hdim_kernel(
    x_ptr, ptr_ptr, y_ptr,
    G, S,
    stride_x_h, stride_x_s,
    stride_y_g, stride_y_s,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """
    Compute y[g, s] = logsumexp(x[ptr[g]:ptr[g+1], s]) for g in [0..G), s in [0..S)
    x: [H, S]
    y: [G, S]
    """
    gid = tl.program_id(0).to(tl.int64)  # segment id along H
    sid = tl.program_id(1)  # tile id along S
    if gid >= G:
        return

    # Column tile
    offs_s = sid * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    NEG_INF = tl.full((BLOCK_S,), float("-inf"), DTYPE)
    ZERO    = tl.zeros((BLOCK_S,), DTYPE)

    # Bounds for this segment along H
    h0 = tl.load(ptr_ptr + gid)
    h1 = tl.load(ptr_ptr + gid + 1)

    running_max = NEG_INF
    running_sum = tl.zeros((BLOCK_S,), DTYPE)

    # Sweep H within the segment
    start_h = h0
    while start_h < h1:
        offs_h = (start_h + tl.arange(0, BLOCK_H)).to(tl.int64)
        mask_h = offs_h < h1

        mask = mask_h[:, None] & mask_s[None, :]
        # Load tile [BLOCK_H, BLOCK_S]
        x_tile = tl.load(
            x_ptr + offs_h[:, None] * stride_x_h + offs_s[None, :] * stride_x_s,
            mask=mask,
            other=float("-inf"),
        )

        # Per-column block max/sum over H axis
        block_max = tl.max(x_tile, axis=0)
        neg_inf_col = block_max == NEG_INF
        safe_block_max = tl.where(neg_inf_col, ZERO, block_max)

        stable = tl.exp2(x_tile - safe_block_max[None, :])
        block_sum = tl.sum(stable, axis=0)

        cand_block_max = tl.where(neg_inf_col, NEG_INF, block_max)
        new_max = tl.maximum(running_max, cand_block_max)

        scaled_running = tl.where(new_max == NEG_INF, ZERO, running_sum * tl.exp2(running_max - new_max))
        scaled_block   = tl.where(new_max == NEG_INF, ZERO, block_sum   * tl.exp2(safe_block_max - new_max))
        running_sum = scaled_running + scaled_block
        running_max = new_max

        start_h += BLOCK_H

    out = tl.where(running_max == NEG_INF, NEG_INF, tl.log2(running_sum) + running_max)
    tl.store(y_ptr + gid * stride_y_g + offs_s * stride_y_s, out, mask=mask_s)


# ---------------------------
# Backward kernel: grad_x
# ---------------------------
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_H": bh, "BLOCK_S": bs}, num_warps=nw, num_stages=ns)
#         for (bh, bs, nw, ns) in HDIM_CONFIGS
#     ],
#     key=["S"],
# )
@triton.jit
def _seg_lse_hdim_bwd_kernel(
    x_ptr, ptr_ptr, gy_ptr, gx_ptr,
    G, S,
    stride_x_h, stride_x_s,
    stride_gy_g, stride_gy_s,
    stride_gx_h, stride_gx_s,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,  # tl.float32 or tl.float64 (same as x dtype)
):
    """
    Stable backward for segmented logsumexp:

      For each segment g and column s:
        m = max_h x[h,s]                 (computed in DTYPE)
        den = sum_h exp(x[h,s] - m)      (accumulated in ACC_DTYPE)
        grad_x[h,s] = grad_y[g,s] * exp(x[h,s] - m) / den

    Key points:
      - Use the SAME shift 'm' (in DTYPE) for both numerator and denominator.
      - Accumulate the denominator in higher precision (ACC_DTYPE) to reduce loss.
      - Merge blocks with running-max trick while keeping shift consistency.
    """
    gid = tl.program_id(0).to(tl.int64)  # segment id
    sid = tl.program_id(1)  # S tile id
    if gid >= G:
        return

    # Column tile
    offs_s = sid * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    # Segment bounds
    h0 = tl.load(ptr_ptr + gid)
    h1 = tl.load(ptr_ptr + gid + 1)

    # Load grad_y for this segment/columns
    gy_cols = tl.load(
        gy_ptr + gid * stride_gy_g + offs_s * stride_gy_s,
        mask=mask_s, other=0.0,
    )

    # Choose accumulator dtype: fp64 when input is fp32, else fp64 (safe default)
    ACC_DTYPE = tl.float64

    # Paired shifts: keep both fp32/64 views so comparisons are robust,
    # but the actual shift used in exp() is the DTYPE version (m_cols32).
    if DTYPE is tl.float32:
        MINF_D = float("-inf")
        m_cols32 = tl.full((BLOCK_S,), MINF_D, dtype=tl.float32)
    else:
        # DTYPE == tl.float64
        MINF_D = float("-inf")
        m_cols32 = tl.full((BLOCK_S,), MINF_D, dtype=tl.float64)
    m_cols64 = tl.full((BLOCK_S,), float("-inf"), dtype=tl.float64)

    den_cols = tl.zeros((BLOCK_S,), dtype=ACC_DTYPE)  # Σ_h exp(x - m) in high precision

    # ---------------------------
    # PASS 1: compute (m_cols, den_cols) stably
    # ---------------------------
    start_h = h0
    while start_h < h1:
        offs_h = (start_h + tl.arange(0, BLOCK_H)).to(tl.int64)
        mask_h = offs_h < h1
        mask = mask_h[:, None] & mask_s[None, :]

        x_tile = tl.load(
            x_ptr + offs_h[:, None] * stride_x_h + offs_s[None, :] * stride_x_s,
            mask=mask,
            other=float("-inf"),
        )
        # Block max in DTYPE
        block_max_d = tl.max(x_tile, axis=0)                  # DTYPE
        neg_inf_col = block_max_d == float("-inf")
        safe_block_max_d = tl.where(neg_inf_col, 0.0, block_max_d)

        # Sumexp within block relative to block_max_d; reduce in ACC_DTYPE
        stable = tl.exp2(x_tile - safe_block_max_d[None, :])   # DTYPE
        block_sum = tl.sum(stable.to(ACC_DTYPE), axis=0)      # ACC_DTYPE

        # Prepare candidates
        cand_m_d  = tl.where(neg_inf_col, float("-inf"), block_max_d)         # DTYPE
        cand_m_64 = tl.where(neg_inf_col, float("-inf"), block_max_d.to(tl.float64))

        # New running max (fp64 for comparisons), and the DTYPE shift we will use everywhere
        new_m64 = tl.maximum(m_cols64, cand_m_64)     # fp64
        new_m_d = tl.maximum(m_cols32, cand_m_d)      # DTYPE

        # Rescale old and block sums to new_m_d (convert shifts to ACC_DTYPE inside exp)
        # guarding empty columns by checking new_m64 == -inf
        scaled_old = tl.where(
            new_m64 == float("-inf"),
            0.0,
            den_cols * tl.exp2((m_cols32.to(ACC_DTYPE) - new_m_d.to(ACC_DTYPE)))
        )
        scaled_blk = tl.where(
            new_m64 == float("-inf"),
            0.0,
            block_sum * tl.exp2((safe_block_max_d.to(ACC_DTYPE) - new_m_d.to(ACC_DTYPE)))
        )

        den_cols = scaled_old + scaled_blk
        m_cols64 = new_m64
        m_cols32 = new_m_d

        start_h += BLOCK_H

    # Handle empty segments / zero denominators
    valid = den_cols > 0
    scale = tl.where(valid, gy_cols / den_cols.to(DTYPE), 0.0)

    # ---------------------------
    # PASS 2: write gradients with the SAME shift used in PASS 1
    # ---------------------------
    start_h = h0
    while start_h < h1:
        offs_h = (start_h + tl.arange(0, BLOCK_H)).to(tl.int64)
        mask_h = offs_h < h1
        mask = mask_h[:, None] & mask_s[None, :]

        x_tile = tl.load(
            x_ptr + offs_h[:, None] * stride_x_h + offs_s[None, :] * stride_x_s,
            mask=mask,
            other=float("-inf"),
        )
        # Use m_cols32 (same DTYPE shift as in denominator path)
        p = tl.exp2(x_tile - m_cols32[None, :])       # DTYPE
        gx_tile = p * scale[None, :]                 # DTYPE

        tl.store(
            gx_ptr + offs_h[:, None] * stride_gx_h + offs_s[None, :] * stride_gx_s,
            gx_tile, mask=mask,
        )

        start_h += BLOCK_H


# ---------------------------
# Python wrappers with autograd
# ---------------------------

def _seg_lse_forward_impl(
    x: torch.Tensor,
    ptr: torch.Tensor,
    block_h: int = 128,
    block_s: int = 128,
) -> torch.Tensor:
    """
    Launch forward Triton kernel (no autograd).
    """
    assert x.is_cuda and ptr.is_cuda, "x and ptr must be CUDA tensors"
    assert x.ndim == 2, "x must be [H, S]"
    assert x.is_contiguous(), "x must be contiguous"
    assert ptr.ndim == 1 and ptr.dtype == torch.long, "ptr must be int64 [G+1]"

    H, S = x.shape
    G = ptr.numel() - 1
    y = torch.empty((G, S), dtype=x.dtype, device=x.device)

    if x.dtype == torch.float32:
        DTYPE = tl.float32
    elif x.dtype == torch.float64:
        DTYPE = tl.float64
    else:
        raise TypeError("x must be float32 or float64")

    # Launch with autotuned configuration by default; manual override if both block sizes provided
    if block_h is None or block_s is None:
        grid = lambda META: (G, triton.cdiv(S, META["BLOCK_S"]))
        _seg_lse_hdim_kernel[grid](
            x, ptr, y, G, S,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            DTYPE=DTYPE,
        )
    else:
        grid = (G, (S + block_s - 1) // block_s)
        _seg_lse_hdim_kernel[grid](
            x, ptr, y, G, S,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_H=int(block_h), BLOCK_S=int(block_s), DTYPE=DTYPE,
        )
    return y


class SegLSEHdimFn(torch.autograd.Function):
    """
    Custom autograd.Function wrapper around the Triton segmented logsumexp kernels.

    Forward saves x, ptr, and y (output) for a single-pass backward:
    grad_x = grad_y * exp(x - y).
    """

    @staticmethod
    def forward(x: torch.Tensor, ptr: torch.Tensor, block_h: int = 128, block_s: int = 128):
        # Compute forward and save for backward
        y = _seg_lse_forward_impl(
            x, ptr,
            None if block_h == 0 else block_h,
            None if block_s == 0 else block_s,
        )
        return y
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, ptr, block_h, block_s = inputs
        ctx.block_h = int(block_h)
        ctx.block_s = int(block_s)
        ctx.save_for_backward(x, ptr)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, ptr = ctx.saved_tensors
        x = x.contiguous()
        ptr = ptr.contiguous()

        grad_x = torch.zeros_like(x)
        H, S = x.shape
        G = ptr.numel() - 1
        # Triton dtype
        if x.dtype == torch.float32:
            DTYPE = tl.float32
        elif x.dtype == torch.float64:
            DTYPE = tl.float64
        else:
            raise TypeError("x must be float32 or float64")

        # Grid launch (autotuned or manual if both provided)
        if ctx.block_h == 0 or ctx.block_s == 0:
            grid = lambda META: (G, triton.cdiv(S, META["BLOCK_S"]))
            _seg_lse_hdim_bwd_kernel[grid](
                x, ptr, grad_y, grad_x,
                G, S,
                x.stride(0), x.stride(1),
                grad_y.stride(0), grad_y.stride(1),
                grad_x.stride(0), grad_x.stride(1),
                DTYPE=DTYPE,
            )
        else:
            grid = (G, (S + ctx.block_s - 1) // ctx.block_s)
            _seg_lse_hdim_bwd_kernel[grid](
                x, ptr, grad_y, grad_x,
                G, S,
                x.stride(0), x.stride(1),
                grad_y.stride(0), grad_y.stride(1),
                grad_x.stride(0), grad_x.stride(1),
                BLOCK_H=int(ctx.block_h), BLOCK_S=int(ctx.block_s),
                DTYPE=DTYPE,
            )

        # Return grads for inputs: x, ptr, block_h, block_s
        return grad_x, None, None, None


def seg_logsumexp(
    x: torch.Tensor,
    ptr: torch.Tensor,
    block_h: int = 128,
    block_s: int = 128,
) -> torch.Tensor:
    """
    y[g, s] = logsumexp(x[ptr[g]:ptr[g+1], s]) for g in [0..G), s in [0..S)

    Autograd is supported:
      grad_x[h, s] = grad_y[g(h), s] * exp(x[h, s] - y[g(h), s])

    Args
    ----
    x   : [H, S] CUDA tensor (float32 or float64). Must be contiguous.
    ptr : [G+1] CUDA Long tensor, 0 <= ptr[i] <= ptr[i+1] <= H.
    block_h / block_s : Optional tiling overrides. If None, autotune decides.

    Returns
    -------
    y : [G, S] CUDA tensor, same dtype as x.
    """
    # Use 0 sentinels for "autotune" inside the autograd.Function
    bh = 0 if block_h is None else int(block_h)
    bs = 0 if block_s is None else int(block_s)
    return SegLSEHdimFn.apply(x, ptr, bh, bs)

# ---------------------------
# Internal tests (optional)
# ---------------------------

def _reference_seg_logsumexp(x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """Compute segmented logsumexp using PyTorch ops for validation."""
    G = ptr.numel() - 1
    S = x.shape[1]
    out = x.new_full((G, S), float("-inf"))
    for g in range(G):
        start = int(ptr[g].item())
        end = int(ptr[g + 1].item())
        if start == end:
            continue
        # log2-space: log2(sum(2^x))
        m = x[start:end].max(dim=0).values
        m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        out[g] = torch.log2(torch.exp2(x[start:end] - m_safe.unsqueeze(0)).sum(dim=0)) + m
    return out


def _test_forward_varying_segments() -> None:
    if not torch.cuda.is_available():
        print("[scatter_lse] Skipping forward varying segments test (CUDA required)")
        return
    from torch.testing import assert_close

    device = torch.device("cuda")
    torch.manual_seed(7)
    lengths = torch.tensor([3, 1, 4, 2], dtype=torch.long)
    ptr = torch.cat([torch.tensor([0], dtype=torch.long), lengths.cumsum(0)]).to(device)
    H = int(ptr[-1].item())
    S = 5
    x = torch.randn((H, S), dtype=torch.float32, device=device)
    inf_mask = torch.rand_like(x) < 0.1
    x = x.masked_fill(inf_mask, float("-inf"))

    ref = _reference_seg_logsumexp(x, ptr)
    out = seg_logsumexp(x.contiguous(), ptr)
    assert_close(out, ref, rtol=1e-6, atol=1e-6)


def _test_forward_empty_segments() -> None:
    if not torch.cuda.is_available():
        print("[scatter_lse] Skipping empty segment test (CUDA required)")
        return
    from torch.testing import assert_close

    device = torch.device("cuda")
    torch.manual_seed(13)
    lengths = torch.tensor([0, 2, 0, 3, 1], dtype=torch.long)
    ptr = torch.cat([torch.tensor([0], dtype=torch.long), lengths.cumsum(0)]).to(device)
    H = int(ptr[-1].item())
    S = 3
    x = torch.randn((H, S), dtype=torch.float32, device=device)
    x = x.masked_fill(torch.rand_like(x) < 0.2, float("-inf"))

    ref = _reference_seg_logsumexp(x, ptr)
    out = seg_logsumexp(x.contiguous(), ptr)
    assert torch.isneginf(out[0]).all()
    assert torch.isneginf(out[2]).all()
    assert_close(out, ref, rtol=1e-6, atol=1e-6)


def _test_gradcheck() -> None:
    if not torch.cuda.is_available():
        print("[scatter_lse] Skipping gradcheck (CUDA required)")
        return
    from torch.autograd import gradcheck

    device = torch.device("cuda")
    torch.manual_seed(23)
    lengths = torch.tensor([2, 3, 1], dtype=torch.long)
    ptr = torch.cat([torch.tensor([0], dtype=torch.long), lengths.cumsum(0)]).to(device)
    H = int(ptr[-1].item())
    S = 4
    x = torch.randn((H, S), dtype=torch.float64, device=device, requires_grad=True)

    def func(inp: torch.Tensor) -> torch.Tensor:
        return seg_logsumexp(inp, ptr)

    assert gradcheck(func, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)




def _benchmark_seg_logsumexp(
    warmup: int = 5,
    iters: int = 20,
    dtypes: Optional[Tuple[torch.dtype, ...]] = None,
) -> None:
    if not torch.cuda.is_available():
        print("[scatter_lse] Skipping benchmark (CUDA required)")
        return
    device = torch.device("cuda")
    if dtypes is None:
        dtypes = (torch.float32,)
    configs = [
        {"G": 2048, "max_len": 32, "S": 128},
        {"G": 4096, "max_len": 64, "S": 256},
        {"G": 512, "max_len": 1000, "S": 512},

    ]
    for dtype in dtypes:
        for cfg in configs:
            lengths = torch.randint(cfg["max_len"]//2, cfg["max_len"] + 1, (cfg["G"],), dtype=torch.long)
            ptr_cpu = torch.cat([torch.tensor([0], dtype=torch.long), lengths.cumsum(0)])
            ptr = ptr_cpu.to(device)
            H = int(ptr_cpu[-1].item())
            x = torch.randn((H, cfg["S"]), device=device, dtype=dtype).requires_grad_(True)
            print(f"shape of x: {x.shape}")
            torch.cuda.synchronize()
            for _ in range(max(warmup, 0)):
                if x.grad is not None:
                    x.grad.zero_()
                seg_logsumexp(x, ptr).sum().backward()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(max(iters, 1)):
                if x.grad is not None:
                    x.grad.zero_()
                seg_logsumexp(x, ptr).sum().backward()
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
            avg_ms = total_ms / max(iters, 1)
            elems = H * cfg["S"]
            rate = (elems / (avg_ms / 1000.0)) / 1e9 if avg_ms > 0 else float('nan')
            avg_len = lengths.float().mean().item()
            print(
                f"[scatter_lse] bench dtype={dtype} G={cfg['G']} avg_len={avg_len:.1f} "
                f"S={cfg['S']} -> {avg_ms:.3f} ms fwd+bwd, {rate:.2f} Gelem/s"
            )


def _run_internal_tests() -> None:
    tests = [
        ("forward varying segments", _test_forward_varying_segments),
        ("forward empty segments", _test_forward_empty_segments),
        ("gradcheck", _test_gradcheck),
    ]
    for name, fn in tests:
        print(f"[scatter_lse] Running {name} test...")
        fn()
    print("[scatter_lse] Internal tests complete.")




def _parse_bench_dtypes(kind: str) -> Tuple[torch.dtype, ...]:
    if kind == "fp32":
        return (torch.float32,)
    if kind == "fp64":
        return (torch.float64,)
    if kind == "both":
        return (torch.float32, torch.float64)
    raise ValueError(f"Unknown dtype option '{kind}'")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="seg_logsumexp utilities")
    parser.add_argument("--bench", action="store_true", help="run benchmarks after tests")
    parser.add_argument("--bench-only", action="store_true", help="skip tests and only benchmark")
    parser.add_argument("--bench-warmup", type=int, default=5, help="warmup iterations for benchmarks")
    parser.add_argument("--bench-iters", type=int, default=20, help="timed iterations for benchmarks")
    parser.add_argument(
        "--bench-dtype",
        choices=["fp32", "fp64", "both"],
        default="fp32",
        help="dtypes to benchmark",
    )
    args = parser.parse_args()

    run_tests = not args.bench_only
    run_bench = args.bench or args.bench_only

    if run_tests:
        _run_internal_tests()
    if run_bench:
        dtypes = _parse_bench_dtypes(args.bench_dtype)
        _benchmark_seg_logsumexp(
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            dtypes=dtypes,
        )


if __name__ == "__main__":
    _main()

