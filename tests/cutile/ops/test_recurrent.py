import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fla.ops.cutile import HAS_CUTILE
from fla.ops.cutile.linear_attn.recurrent import cutile_recurrent_linear_attn_fwd
from fla.ops.cutile.linear_attn.packing import pack_varlen_inputs, unpack_packed_outputs
from fla.ops.common.fused_recurrent import fused_recurrent_fwd as triton_fused_recurrent_fwd
from fla.ops.linear_attn.naive import naive_recurrent_linear_attn
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
    (2, 64, 4, 32, 32),
    (1, 128, 8, 64, 64),
    (4, 32, 16, 16, 32),
    (1, 96, 2, 48, 40),
    (2, 256, 4, 64, 64),
    (1, 128, 4, 128, 64),
    (1, 64, 4, 256, 64),
])
@pytest.mark.parametrize("use_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cutile_recurrent_correctness(B, T, H, K, V, use_state, dtype):
    """Test cuTile recurrent against both naive and Triton implementations."""
    _require_cuda()
    _require_cutile()
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    q, k, v = _make_inputs(B, T, H, K, V, dtype)
    h0 = torch.randn(B, H, K, V, device="cuda", dtype=dtype) if use_state else None
    scale = K ** -0.5

    # Reference: naive recurrent
    ref_naive_o, ref_naive_ht = naive_recurrent_linear_attn(
        q, k, v, initial_state=h0, output_final_state=True, scale=scale
    )

    # Reference: Triton fused_recurrent_fwd
    ref_triton_o, ref_triton_ht = triton_fused_recurrent_fwd(
        q=q, k=k, v=v, g=None, g_gamma=None, gk=None, gv=None,
        scale=scale, initial_state=h0, output_final_state=True,
        reverse=False, cu_seqlens=None,
    )

    # cuTile recurrent
    tri_o, tri_ht = cutile_recurrent_linear_attn_fwd(
        q, k, v, initial_state=h0, output_final_state=True, scale=scale
    )

    # Compare cuTile vs naive
    assert_close("cutile_vs_naive:o", ref_naive_o, tri_o, 0.001)
    assert_close("cutile_vs_naive:ht", ref_naive_ht, tri_ht, 0.001)

    # Compare cuTile vs Triton
    assert_close("cutile_vs_triton:o", ref_triton_o, tri_o, 0.001)
    assert_close("cutile_vs_triton:ht", ref_triton_ht, tri_ht, 0.001)


@pytest.mark.parametrize("B, S, H, K, V, lengths, padding_side", [
    (3, 96, 4, 32, 32, [15, 64, 90], "right"),
    (2, 80, 2, 48, 40, [10, 79], "left"),
    (4, 128, 4, 64, 64, [32, 64, 96, 128], "right"),
])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cutile_recurrent_varlen_packed(B, S, H, K, V, lengths, padding_side, dtype):
    """Test cuTile recurrent with packed varlen inputs."""
    _require_cuda()
    _require_cutile()
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    torch.manual_seed(42)
    scale = K ** -0.5

    q = torch.randn((B, S, H, K), device="cuda", dtype=dtype)
    k = torch.randn((B, S, H, K), device="cuda", dtype=dtype)
    v = torch.randn((B, S, H, V), device="cuda", dtype=dtype)
    h0 = torch.randn((B, H, K, V), device="cuda", dtype=dtype)

    packed_q, packed_k, packed_v, cu_seqlens, _ = pack_varlen_inputs(
        q, k, v, lengths, chunk_size=None, padding_side=padding_side,
    )

    # Per-sequence reference using naive
    refs, ref_hts = [], []
    for i, L in enumerate(lengths):
        src = slice(S - L, S) if padding_side == "left" else slice(0, L)
        ref, ref_ht = naive_recurrent_linear_attn(
            q=q[i : i + 1, src],
            k=k[i : i + 1, src],
            v=v[i : i + 1, src],
            initial_state=h0[i],
            output_final_state=True,
            scale=scale,
        )
        refs.append(ref)
        ref_hts.append(ref_ht)

    ref_padded = torch.zeros((B, S, H, V), device="cuda", dtype=dtype)
    for i, L in enumerate(lengths):
        dst = slice(S - L, S) if padding_side == "left" else slice(0, L)
        ref_padded[i, dst] = refs[i][0]
    ref_ht = torch.cat(ref_hts, dim=0)

    tri_o, tri_ht = cutile_recurrent_linear_attn_fwd(
        packed_q, packed_k, packed_v,
        initial_state=h0, output_final_state=True, scale=scale, cu_seqlens=cu_seqlens,
    )

    tri_padded = unpack_packed_outputs(tri_o, lengths, cu_seqlens, max_len=S, padding_side=padding_side)
    assert_close("varlen:naive:o", ref_padded, tri_padded, 0.001)
    assert_close("varlen:naive:ht", ref_ht, tri_ht, 0.001)

    # Triton packed reference
    triton_o, triton_ht = triton_fused_recurrent_fwd(
        q=packed_q,
        k=packed_k,
        v=packed_v,
        g=None,
        g_gamma=None,
        gk=None,
        gv=None,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        reverse=False,
        cu_seqlens=cu_seqlens,
    )
    triton_padded = unpack_packed_outputs(triton_o, lengths, cu_seqlens, max_len=S, padding_side=padding_side)
    assert_close("varlen:triton:o", triton_padded, tri_padded, 0.001)
    assert_close("varlen:triton:ht", triton_ht, tri_ht, 0.001)
