import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fla.ops.cutile import HAS_CUTILE
from fla.ops.cutile.linear_attn.chunk import cutile_fused_chunk_fwd
from fla.ops.cutile.linear_attn.packing import pack_varlen_inputs, unpack_packed_outputs
from fla.ops.common.fused_chunk import fused_chunk_fwd as triton_fused_chunk_fwd
from fla.ops.linear_attn.naive import naive_chunk_linear_attn
from fla.utils import assert_close


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")


def _require_cutile():
    if not HAS_CUTILE:
        pytest.skip("cuTile is not available in this environment.")


def _make_inputs(B, T, H, K, V, dtype):
    torch.manual_seed(42)
    device = "cuda"
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("B, T, H, K, V", [
    (1, 64, 4, 32, 32),
    (2, 128, 4, 32, 32),
    (1, 64, 2, 48, 40),
    (2, 256, 8, 64, 64),
    (1, 128, 4, 128, 128),
    (1, 64, 4, 256, 64),
])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cutile_chunk_correctness(B, T, H, K, V, dtype):
    """Test cuTile chunk against both naive and Triton implementations."""
    _require_cuda()
    _require_cutile()
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    q, k, v = _make_inputs(B, T, H, K, V, dtype)
    scale = K ** -0.5
    # Reference: naive chunk (no initial state)
    ref_naive = naive_chunk_linear_attn(q, k, v, scale=scale, normalize=False)

    # Reference: Triton fused_chunk_fwd (no initial state)
    ref_triton, _ = triton_fused_chunk_fwd(
        q=q,
        k=k,
        v=v,
        g=None,
        g_gamma=None,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        chunk_size=64,
    )

    # cuTile fused chunk
    out_cutile, _ = cutile_fused_chunk_fwd(
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        chunk_size=64,
    )

    # Compare cuTile vs naive
    assert_close("cutile_vs_naive:o", ref_naive, out_cutile, 0.001)

    # Compare cuTile vs Triton
    assert_close("cutile_vs_triton:o", ref_triton, out_cutile, 0.001)


@pytest.mark.parametrize("B, T, H, K, V", [
    (1, 64, 4, 32, 32),
    (2, 128, 4, 32, 32),
])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cutile_chunk_state_matches_triton(B, T, H, K, V, dtype):
    """Stateful comparison with Triton fused_chunk_fwd."""
    _require_cuda()
    _require_cutile()
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    q, k, v = _make_inputs(B, T, H, K, V, dtype)
    scale = K ** -0.5
    h0 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32)

    ref_triton, ref_triton_ht = triton_fused_chunk_fwd(
        q=q,
        k=k,
        v=v,
        g=None,
        g_gamma=None,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=None,
        chunk_size=64,
    )

    out_cutile, out_cutile_ht = cutile_fused_chunk_fwd(
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=None,
        chunk_size=64,
    )

    assert_close("state:triton:o", ref_triton, out_cutile, 0.001)
    assert_close("state:triton:ht", ref_triton_ht, out_cutile_ht, 0.001)


@pytest.mark.parametrize("B, S, H, K, V, lengths, padding_side", [
    (3, 128, 4, 32, 32, [64, 128, 64], "right"),
    (2, 128, 2, 48, 40, [64, 128], "left"),
    (4, 256, 4, 32, 32, [128, 64, 192, 64], "right"),
])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cutile_chunk_varlen_packed(B, S, H, K, V, lengths, padding_side, dtype):
    """Test cuTile chunk with packed varlen inputs."""
    _require_cuda()
    _require_cutile()
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    torch.manual_seed(42)
    scale = K ** -0.5

    q = torch.randn((B, S, H, K), device="cuda", dtype=dtype)
    k = torch.randn((B, S, H, K), device="cuda", dtype=dtype)
    v = torch.randn((B, S, H, V), device="cuda", dtype=dtype)
    h0 = None

    packed_q, packed_k, packed_v, cu_seqlens, _ = pack_varlen_inputs(
        q, k, v, lengths, chunk_size=64, padding_side=padding_side,
    )

    # Per-sequence reference using naive
    refs, ref_hts = [], []
    for i, L in enumerate(lengths):
        src = slice(S - L, S) if padding_side == "left" else slice(0, L)
        ref = naive_chunk_linear_attn(
            q=q[i : i + 1, src],
            k=k[i : i + 1, src],
            v=v[i : i + 1, src],
            scale=scale,
            normalize=False,
        )
        refs.append(ref)

    ref_padded = torch.zeros((B, S, H, V), device="cuda", dtype=torch.float32)
    for i, L in enumerate(lengths):
        dst = slice(S - L, S) if padding_side == "left" else slice(0, L)
        ref_padded[i, dst] = refs[i][0]

    out, _ = cutile_fused_chunk_fwd(
        q=packed_q, k=packed_k, v=packed_v,
        scale=scale, initial_state=h0, output_final_state=False,
        cu_seqlens=cu_seqlens, chunk_size=64,
    )
    out_padded = unpack_packed_outputs(out, lengths, cu_seqlens, max_len=S, padding_side=padding_side)

    # Triton packed reference
    triton_out, _ = triton_fused_chunk_fwd(
        q=packed_q,
        k=packed_k,
        v=packed_v,
        g=None,
        g_gamma=None,
        scale=scale,
        initial_state=h0,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=64,
    )
    triton_padded = unpack_packed_outputs(triton_out, lengths, cu_seqlens, max_len=S, padding_side=padding_side)

    assert_close("varlen:naive", ref_padded, out_padded, 0.001)
    assert_close("varlen:triton:o", triton_padded, out_padded, 0.001)
