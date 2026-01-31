
import os

import argparse
import torch
import triton
import triton.language as tl
import cuda.tile as ct

from fla.ops.cutile import HAS_CUTILE
from fla.ops.cutile.linear_attn.chunk import (
    _cutile_fused_chunk_fwd_impl,
    _cutile_fused_chunk_fwd_split_impl,
    _cutile_fused_chunk_fwd_impl_packed_k,
    cutile_fused_chunk_fwd_kernel,
    cutile_fused_chunk_fwd_kernel_no_tma_load,
    cutile_fused_chunk_fwd_kernel_packed_k,
    cutile_fused_chunk_intra_kernel,
    cutile_fused_chunk_inter_kernel,
)
from fla.ops.cutile.linear_attn.recurrent import _cutile_recurrent_fwd_impl
from fla.ops.utils.op import exp

# -----------------------------------------------------------------------------
# TF32 explicit rounding (value-level)
# -----------------------------------------------------------------------------

@triton.jit
def _triton_round_to_tf32_kernel_rn(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Use PTX's f32->tf32 conversion instead of bit tricks.
    # cvt.rn.tf32.f32: round-to-nearest-even (sm90+).
    bits = tl.inline_asm_elementwise(
        asm="cvt.rn.tf32.f32 $0, $1;",
        constraints="=r,f",
        args=[x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    y = tl.cast(bits, tl.float32, bitcast=True)
    tl.store(y_ptr + offs, y, mask=mask)

@triton.jit
def _triton_round_to_tf32_kernel_rna(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # cvt.rna.tf32.f32: round-to-nearest-away (sm80+).
    bits = tl.inline_asm_elementwise(
        asm="cvt.rna.tf32.f32 $0, $1;",
        constraints="=r,f",
        args=[x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    y = tl.cast(bits, tl.float32, bitcast=True)
    tl.store(y_ptr + offs, y, mask=mask)


def round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    """
    Round FP32 values to TF32 (10-bit mantissa) using Triton + PTX cvt.

    Output dtype remains torch.float32, but values are TF32-rounded.
    """
    if x.dtype != torch.float32:
        return x
    x1 = x.contiguous().view(-1)
    y1 = torch.empty_like(x1)
    # Prefer RTNE when available. On sm80/sm89, use RNA for compatibility.
    major, minor = torch.cuda.get_device_capability(x.device)
    use_rn = major >= 9
    BLOCK = 1024
    grid = (triton.cdiv(x1.numel(), BLOCK),)
    if use_rn:
        _triton_round_to_tf32_kernel_rn[grid](
            x1,
            y1,
            x1.numel(),
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=1,
        )
    else:
        _triton_round_to_tf32_kernel_rna[grid](
            x1,
            y1,
            x1.numel(),
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=1,
        )
    return y1.view_as(x)

# -----------------------------------------------------------------------------
# cuTile TF32 rounding (value-level) via official ct.tfloat32 cast
# -----------------------------------------------------------------------------

def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@ct.kernel
def _cutile_tf32_cast_kernel(x, y, TILE_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(x, index=(pid,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    tile_tf32 = tile.astype(ct.tfloat32)
    tile_out = tile_tf32.astype(ct.float32)
    ct.store(y, index=(pid,), tile=tile_out)


def cutile_round_to_tf32(x: torch.Tensor, tile_size: int = 256) -> torch.Tensor:
    """
    Round FP32 values to TF32 (10-bit mantissa) using cuTile's official cast.

    Output dtype remains torch.float32, but values are TF32-rounded.
    """
    if x.dtype != torch.float32:
        return x
    orig_shape = x.shape
    x = x.contiguous().view(-1)
    y = torch.empty_like(x)
    grid = (_ceildiv(x.numel(), tile_size), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _cutile_tf32_cast_kernel,
        (x, y, tile_size),
    )
    return y.view(orig_shape)

# -----------------------------------------------------------------------------
# cuTile Fixed-Tile Wrappers (no runtime heuristics)
# -----------------------------------------------------------------------------

def cutile_chunk_fwd_fixed_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    BT: int = 64,
    BK: int = 64,
    BV: int = 64,
    use_fp32_matmul: bool = False,
    use_tma_load: bool = True,
) -> torch.Tensor:
    """
    Fixed-tile cuTile chunk kernel without size-based kernel switching.

    This keeps BT/BK/BV fully specified and avoids small-problem heuristics so
    ablation results reflect tile choices only.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K, V = *q.shape, v.shape[-1]
    NK, NV = ct.cdiv(K, BK), ct.cdiv(V, BV)
    o = v.new_empty(NK, *v.shape, dtype=torch.float32)
    dummy = torch.empty(0, device=q.device)
    grid = (NV, NK, B * H)
    kernel = cutile_fused_chunk_fwd_kernel if use_tma_load else cutile_fused_chunk_fwd_kernel_no_tma_load
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            q,
            k,
            v,
            o,
            dummy,  # h0
            dummy,  # ht
            dummy,  # cu_seqlens
            scale,
            T,
            B,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            False,  # USE_INITIAL_STATE
            False,  # STORE_FINAL_STATE
            False,  # IS_VARLEN
            use_fp32_matmul,
        ),
    )
    if NK > 1:
        o = o.sum(0).to(v)
    else:
        o = o.squeeze(0).to(v)
    return o


def cutile_chunk_fwd_packed_k_fixed_launch(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    BT: int = 64,
    BK: int = 64,
    BV: int = 64,
    use_fp32_matmul: bool = False,
) -> torch.Tensor:
    """Fixed-tile packed-K variant without runtime heuristics."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K, V = *q.shape, v.shape[-1]
    NK, NV = ct.cdiv(K, BK), ct.cdiv(V, BV)
    o = v.new_empty(NK, *v.shape, dtype=torch.float32)
    dummy = torch.empty(0, device=q.device)
    grid = (NV, NK, B * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        cutile_fused_chunk_fwd_kernel_packed_k,
        (
            q,
            k_packed,
            v,
            o,
            dummy,  # h0
            dummy,  # ht
            dummy,  # cu_seqlens
            scale,
            T,
            B,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            False,  # USE_INITIAL_STATE
            False,  # STORE_FINAL_STATE
            False,  # IS_VARLEN
            use_fp32_matmul,
        ),
    )
    if NK > 1:
        o = o.sum(0).to(v)
    else:
        o = o.squeeze(0).to(v)
    return o


def cutile_chunk_fwd_split_fixed_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    BT: int = 64,
    BK: int = 64,
    BV: int = 64,
    use_fp32_matmul: bool = False,
) -> torch.Tensor:
    """Fixed-tile split kernels without runtime heuristics."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K, V = *q.shape, v.shape[-1]
    NK, NV = ct.cdiv(K, BK), ct.cdiv(V, BV)
    o_intra = v.new_empty(NK, *v.shape, dtype=torch.float32)
    o_inter = v.new_empty(NK, *v.shape, dtype=torch.float32)
    dummy = torch.empty(0, device=q.device)
    grid = (NV, NK, B * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        cutile_fused_chunk_intra_kernel,
        (
            q,
            k,
            v,
            o_intra,
            dummy,  # cu_seqlens
            scale,
            T,
            B,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            False,  # IS_VARLEN
            use_fp32_matmul,
        ),
    )
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        cutile_fused_chunk_inter_kernel,
        (
            q,
            k,
            v,
            o_inter,
            dummy,  # h0
            dummy,  # ht
            dummy,  # cu_seqlens
            scale,
            T,
            B,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            False,  # USE_INITIAL_STATE
            False,  # STORE_FINAL_STATE
            False,  # IS_VARLEN
            use_fp32_matmul,
        ),
    )
    o = o_intra + o_inter
    if NK > 1:
        o = o.sum(0).to(v)
    else:
        o = o.squeeze(0).to(v)
    return o

# -----------------------------------------------------------------------------
# Triton Kernels (Fixed Config)
# -----------------------------------------------------------------------------

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_chunk_fwd_kernel_fixed(
    q,
    k,
    v,
    g,
    g_gamma,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Baseline adapted from `fla/ops/common/fused_chunk.py:fused_chunk_fwd_kernel`
    # (keep semantics identical; no extra fp32 casting/masking tweaks here).
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (o_i + 1)
        b_g_last = b_gamma * BT
        b_gq = exp(b_g)
        b_gk = exp(b_g_last - b_g)
        b_gn = exp(b_g_last)

    m_s = o_i[:, None] >= o_i[None, :]

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    v = v + (bos * H + i_h) * V
    o = o + (i_k * all + bos).to(tl.int64) * H * V + i_h * V

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(0, NT):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1

        # NOTE: Using TF32 dot precision. To avoid truncation bias, round inputs
        # to TF32 explicitly before invoking this kernel.
        b_s = tl.dot(b_q, b_k)

        if USE_G:
            p_g = g + (bos + o_t) * H + i_h
            b_g = tl.load(p_g, mask=(o_t < T), other=0.)
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
            b_gq = exp(b_g)
            b_gk = exp(b_g_last - b_g)
            b_gn = exp(b_g_last)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_gk = exp(b_g_last - b_g)
            b_gn = exp(b_g_last)

        if USE_G or USE_G_GAMMA:
            b_gs = tl.where(m_s & m_t, exp(b_g[:, None] - b_g[None, :]), 0)
            b_s *= b_gs
            b_o = (
                tl.dot(b_s.to(b_q.dtype), b_v) +
                tl.dot(b_q, b_h.to(b_q.dtype)) * b_gq[:, None]
            )
            b_v = (b_v * b_gk[:, None]).to(b_v.dtype)
            b_h *= b_gn
        else:
            b_s *= m_s & m_t
            b_o = (
                tl.dot(b_s.to(b_q.dtype), b_v) +
                tl.dot(b_q, b_h.to(b_q.dtype))
            )

        b_h += tl.dot(b_k, b_v)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

def fused_chunk_fwd_fixed_launch(
    q,
    k,
    v,
    scale=None,
    BT=64,
    BK=64,
    BV=64,
    explicit_tf32: bool = False,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    if explicit_tf32:
        # Explicit TF32 rounding (vs tl.dot implicit truncation).
        q = round_to_tf32(q)
        k = round_to_tf32(k)
        v = round_to_tf32(v)
    NK = triton.cdiv(K, BK)
    o = v.new_empty(NK, *v.shape, dtype=torch.float) if NK > 1 else torch.empty_like(v)
    
    grid = (triton.cdiv(V, BV), NK, B * H)
    fused_chunk_fwd_kernel_fixed[grid](
        q=q, k=k, v=v, g=None, g_gamma=None,
        o=o, h0=None, ht=None,
        cu_seqlens=None, scale=scale,
        B=B, T=T, H=H, K=K, V=V,
        BT=BT, BK=BK, BV=BV,
        num_warps=4, num_stages=3
    )
    if NK > 1:
        o = o.sum(0).to(v)
    return o

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['B', 'T'])
def fused_recurrent_fwd_kernel_fixed(
    q, k, v, g, g_gamma, gk, gv, o, h0, ht, cu_seqlens, scale,
    B, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, REVERSE: tl.constexpr,
    USE_G: tl.constexpr, USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr, USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    bos, eos = i_n * T, i_n * T + T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + bos * H*K + i_h * K + o_k
    p_k = k + bos * H*K + i_h * K + o_k
    p_v = v + bos * H*V + i_h * V + o_v
    p_o = o + (i_k * B * T + bos) * H*V + i_h * V + o_v
    
    m_k = o_k < K
    m_v = o_v < V
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)
        p_q += H*K
        p_k += H*K
        p_v += H*V
        p_o += H*V

def fused_recurrent_fwd_fixed_launch(q, k, v, scale=None, BK=64, BV=64):
    B, T, H, K, V = *k.shape, v.shape[-1]
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)
    grid = (NV, NK, B * H)
    fused_recurrent_fwd_kernel_fixed[grid](
        q=q, k=k, v=v, g=None, g_gamma=None, gk=None, gv=None, o=o, h0=None, ht=None,
        cu_seqlens=None, scale=scale,
        B=B, T=T, H=H, K=K, V=V, BK=BK, BV=BV, REVERSE=False,
        USE_G=False, USE_G_GAMMA=False, USE_GK=False, USE_GV=False,
        num_warps=4
    )
    o = o.sum(0)
    return o

# -----------------------------------------------------------------------------
# Test & Benchmark
# -----------------------------------------------------------------------------

def _make_inputs(B, T, H, K, V, dtype=torch.float32):
    torch.manual_seed(42)
    device = "cuda"
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    return q, k, v

def _check_diff(ref, res, name=""):
    diff = (ref - res).abs().max()
    print(f"[{name}] Max Diff: {diff:.6f}")
    if diff > 1e-3:
        print(f"[{name}] WARNING: Large difference!")
    return diff

def run_correctness_check():
    print("\nRunning Correctness Check...")
    B, T, H, K, V = 2, 128, 4, 64, 64
    q, k, v = _make_inputs(B, T, H, K, V)
    scale = K ** -0.5
    
    # Chunk Check
    print(f"\nChecking Chunk Mode (B={B}, T={T}, H={H}, K={K}, V={V})")
    # Use a trusted Triton baseline adapted from `fla/ops/common/fused_chunk.py`.
    out_triton_tf32 = fused_chunk_fwd_fixed_launch(
        q, k, v,
        scale=scale, BT=64, BK=32, BV=64,
        explicit_tf32=False,
    )
    out_cutile, _ = _cutile_fused_chunk_fwd_impl(
        q, k, v,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        BT=64, BK=32, BV=64,
    )
    k_packed = k.permute(0, 2, 3, 1).contiguous()
    out_cutile_packed, _ = _cutile_fused_chunk_fwd_impl_packed_k(
        q, k_packed, v,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        BT=64, BK=32, BV=64,
    )
    out_cutile_split, _ = _cutile_fused_chunk_fwd_split_impl(
        q, k, v,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        BT=64, BK=32, BV=64,
    )
    _check_diff(out_triton_tf32, out_cutile, "Chunk (single, triton TF32)")
    _check_diff(out_triton_tf32, out_cutile_packed, "Chunk (packed K, triton TF32)")
    _check_diff(out_triton_tf32, out_cutile_split, "Chunk (split, triton TF32)")
    
    # Recurrent Check
    print(f"\nChecking Recurrent Mode (B={B}, T={T}, H={H}, K={K}, V={V})")
    out_triton_r = fused_recurrent_fwd_fixed_launch(q, k, v, scale=scale, BK=32, BV=32)
    out_cutile_r, _ = _cutile_recurrent_fwd_impl(q, k, v, scale=scale, initial_state=None, output_final_state=False, cu_seqlens=None, BK=32, BV=32)
    _check_diff(out_triton_r, out_cutile_r, "Recurrent")

def _bench(fn, *args, **kwargs):
    # NOTE: keep this wrapper small; caller controls warmup/rep.
    warmup = kwargs.pop("warmup", 20)
    rep = kwargs.pop("rep", 100)
    return triton.testing.do_bench(lambda: fn(*args, **kwargs), warmup=warmup, rep=rep)

def _bench_params_from_T(T: int):
    """
    Pick reasonable benchmarking repetitions by sequence length.

    Long-context kernels can take milliseconds each; using a fixed rep=100 makes the
    benchmark unnecessarily slow (and can hit watchdog timeouts on some systems).
    """
    if T <= 1024:
        return 30, 200
    if T <= 2048:
        return 30, 150
    if T <= 4096:
        return 25, 80
    if T <= 8192:
        return 20, 40
    return 15, 20


def _ablation_inputs(preset: str = "mixed", max_T: int | None = None):
    """
    More common / practical linear-attention shapes.

    Conventions used here:
    - head_dim (K) = 64 is the most common choice in modern attention/linear-attn.
    - value_dim (V) is commonly 64, and sometimes 128 (wider values).
    - H in {4, 8, 16, 32} gives d_model in {256, 512, 1024, 2048} when K=64.
    - Two scenarios:
      - training: moderate context, larger batch; keep total tokens per step reasonable
      - inference: B=1, longer context for long-context decoding/eval
    """
    assert preset in {"training", "inference", "mixed"}
    K = 64
    value_dims_env = os.environ.get("VALUE_DIMS", "64,128")
    value_dims = sorted({int(x.strip()) for x in value_dims_env.split(",") if x.strip()})
    if not value_dims:
        value_dims = [64, 128]

    # Keep these lists short and representative; users can extend as needed.
    training = [
        # ~8k tokens per step
        (16, 512),
        (8, 1024),
        (4, 2048),
        (2, 4096),
    ]
    inference = [
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (1, 16384),
    ]

    hs = [4, 8, 16, 32]
    inputs: list[tuple[int, int, int, int, int]] = []
    if preset in {"training", "mixed"}:
        for B, T in training:
            for H in (8, 16):  # training commonly uses smaller d_model than long-context inference
                for V in value_dims:
                    inputs.append((B, T, H, K, V))
    if preset in {"inference", "mixed"}:
        for B, T in inference:
            for H in hs:
                for V in value_dims:
                    inputs.append((B, T, H, K, V))

    if max_T is not None:
        inputs = [cfg for cfg in inputs if cfg[1] <= max_T]
    return inputs


def run_benchmarks():
    if not HAS_CUTILE:
        raise RuntimeError(
            "cuTile is not available in this environment (HAS_CUTILE=False). "
            "Please build/install with cuTile enabled before running this benchmark."
        )

    print("\n" + "=" * 80)
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    print("=" * 80)
    
    preset = os.environ.get("ABLATION_PRESET", "inference")
    max_T_env = os.environ.get("MAX_T", "")
    max_T = int(max_T_env) if max_T_env.strip() else None
    inputs = _ablation_inputs(preset=preset, max_T=max_T)
    # Inference-only: exclude batch > 1.
    chunk_inputs = [cfg for cfg in inputs if cfg[0] == 1]
    # Fix head count for chunk benchmarks to avoid sweeping H.
    chunk_inputs = [cfg for cfg in chunk_inputs if cfg[2] == 4]
    # Reduce recurrent coverage to keep runtime reasonable.
    recurrent_inputs = [
        cfg for cfg in inputs
        if cfg[0] == 1 and cfg[1] in (1024, 4096, 8192) and cfg[2] in (8, 16)
    ]
    
    # Only run TF32 (FP32 inputs with TF32 tensor cores where applicable).
    dtypes = [torch.float32]

    print("\nBenchmark: Chunk Mode (Fixed Tile, Ablation)")
    print(
        f"{'Input':<52} {'Dtype':<10} {'Tile(BT,BK,BV)':<15} {'Triton(ms)':<10} "
        f"{'cuTile-fixed(ms)':<16} {'split-fixed(ms)':<16} {'packedK-fixed(ms)':<18} "
        f"{'Spd':<6} {'Split':<6} {'Pack':<6}"
    )
    for dtype in dtypes:
        for B, T, H, K, V in chunk_inputs:
            torch.cuda.empty_cache()
            try:
                q, k, v = _make_inputs(B, T, H, K, V, dtype=dtype)
            except Exception as e:
                config_str = f"B={B},T={T},H={H},K={K},V={V},D={H * K}"
                print(f"{config_str:<52} {str(dtype).replace('torch.', ''):<10} OOM during input: {e}")
                continue
            scale = K ** -0.5
            # Pre-pack K for packed-K variants (packing cost excluded from timing).
            k_packed = k.permute(0, 2, 3, 1).contiguous()

            # For ablation runs, rely on tl.dot default TF32 behavior on NVIDIA
            # (no value-level explicit rounding for Triton).
            q_triton_tf32, k_triton_tf32, v_triton_tf32 = q, k, v
            # Keep cuTile inputs as-is as well (kernel does its own casts).
            q_cutile_tf32, k_cutile_tf32, v_cutile_tf32 = q, k, v
            k_cutile_tf32_packed = k_packed
            # Keep only configurations that fit within 100KB shared memory.
            # Use smaller BT for larger BK/BV to avoid SMEM overflow while
            # still covering BK=32/64 and BV=32/64/128.
            configs = [
                (64, 32, 32),
                (64, 32, 64),
                (32, 32, 128),
                (32, 64, 32),
                (32, 64, 64),
                (32, 64, 128),
            ]
            warmup, rep = _bench_params_from_T(T)
            for BT, BK, BV in configs:
                try:
                    t_triton = _bench(
                        fused_chunk_fwd_fixed_launch,
                        q_triton_tf32, k_triton_tf32, v_triton_tf32,
                        scale=scale, BT=BT, BK=BK, BV=BV,
                        warmup=warmup, rep=rep,
                        explicit_tf32=False,
                    )
                    t_cutile = _bench(
                        cutile_chunk_fwd_fixed_launch,
                        q_cutile_tf32, k_cutile_tf32, v_cutile_tf32,
                        scale=scale,
                        BT=BT, BK=BK, BV=BV,
                        warmup=warmup, rep=rep,
                    )
                    t_cutile_split = _bench(
                        cutile_chunk_fwd_split_fixed_launch,
                        q_cutile_tf32, k_cutile_tf32, v_cutile_tf32,
                        scale=scale,
                        BT=BT, BK=BK, BV=BV,
                        warmup=warmup, rep=rep,
                    )
                    t_cutile_pack = _bench(
                        cutile_chunk_fwd_packed_k_fixed_launch,
                        q_cutile_tf32, k_cutile_tf32_packed, v_cutile_tf32,
                        scale=scale,
                        BT=BT, BK=BK, BV=BV,
                        warmup=warmup, rep=rep,
                    )
                    config_str = f"B={B},T={T},H={H},K={K},V={V},D={H * K}"
                    dtype_str = "tf32" if dtype == torch.float32 else str(dtype).replace("torch.", "")
                    print(
                        f"{config_str:<52} {dtype_str:<10} {f'{BT},{BK},{BV}':<15} "
                        f"{t_triton:<10.3f} {t_cutile:<10.3f} {t_cutile_split:<10.3f} {t_cutile_pack:<12.3f} "
                        f"{t_triton / t_cutile:<6.2f} {t_triton / t_cutile_split:<6.2f} {t_triton / t_cutile_pack:<6.2f}"
                    )
                except Exception as e:
                    print(f"Failed: {e}")
            del q, k, v, k_packed
            torch.cuda.empty_cache()

    print("\nBenchmark: Recurrent Mode (Ablation)")
    print(f"{'Input':<52} {'Dtype':<10} {'BK,BV':<10} {'Triton(ms)':<10} {'cuTile(ms)':<10} {'Speedup':<8}")
    for dtype in dtypes:
        for B, T, H, K, V in recurrent_inputs:
            torch.cuda.empty_cache()
            try:
                q, k, v = _make_inputs(B, T, H, K, V, dtype=dtype)
            except Exception as e:
                config_str = f"B={B},T={T},H={H},K={K},V={V},D={H * K}"
                print(f"{config_str:<52} {str(dtype).replace('torch.', ''):<10} OOM during input: {e}")
                continue
            scale = K ** -0.5
            # Match dk/dv sweep used in chunk benchmarks.
            configs = [(32, 64), (32, 128), (64, 64), (64, 128)]
            warmup, rep = _bench_params_from_T(T)
            for BK, BV in configs:
                try:
                    t_triton = _bench(
                        fused_recurrent_fwd_fixed_launch,
                        q, k, v,
                        scale=scale, BK=BK, BV=BV,
                        warmup=warmup, rep=rep,
                    )
                    t_cutile = _bench(
                        _cutile_recurrent_fwd_impl,
                        q, k, v,
                        scale=scale,
                        initial_state=None,
                        output_final_state=False,
                        cu_seqlens=None,
                        BK=BK, BV=BV,
                        warmup=warmup, rep=rep,
                    )
                    config_str = f"B={B},T={T},H={H},K={K},V={V},D={H * K}"
                    print(
                        f"{config_str:<52} {str(dtype).replace('torch.', ''):<10} {f'{BK},{BV}':<10} "
                        f"{t_triton:<10.3f} {t_cutile:<10.3f} {t_triton / t_cutile:<8.2f}x"
                    )
                except Exception as e:
                    print(f"Failed: {e}")
            del q, k, v
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce/ablate cuTile linear-attention kernels.")
    parser.add_argument(
        "--preset",
        choices=["training", "inference", "mixed"],
        default=os.environ.get("ABLATION_PRESET", "mixed"),
        help="Ablation shape preset (can also set ABLATION_PRESET env var).",
    )
    parser.add_argument(
        "--max-t",
        type=int,
        default=int(os.environ["MAX_T"]) if os.environ.get("MAX_T") else None,
        help="Optional max sequence length filter (can also set MAX_T env var).",
    )
    args = parser.parse_args()

    run_correctness_check()
    # Note: run_benchmarks reads env vars; keep CLI args as a convenience layer.
    if args.preset:
        os.environ["ABLATION_PRESET"] = args.preset
    if args.max_t is not None:
        os.environ["MAX_T"] = str(args.max_t)
    run_benchmarks()
