import os

import torch
import triton

from fla.ops.cutile import HAS_CUTILE
from fla.ops.cutile.linear_attn.chunk import cutile_fused_chunk_fwd
from fla.ops.common.fused_chunk import fused_chunk_fwd as triton_fused_chunk_fwd


def _require_cuda() -> bool:
    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark.")
        return False
    return True


def _require_cutile() -> bool:
    if not HAS_CUTILE:
        print("cuTile is not available in this environment.")
        return False
    return True


def _make_inputs(B: int, T: int, H: int, K: int, V: int, dtype: torch.dtype):
    torch.manual_seed(42)
    device = "cuda"
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    return q, k, v


def _bench(fn, *args, **kwargs):
    return triton.testing.do_bench(lambda: fn(*args, **kwargs), warmup=20, rep=100)


def run():
    """Benchmark cuTile chunk vs Triton fused_chunk_fwd (table format)."""
    if not _require_cuda() or not _require_cutile():
        return
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    configs = [
        (2, 128, 4, 32, 32),
        (2, 512, 4, 64, 64),
        (4, 1024, 8, 64, 64),
        (4, 2048, 8, 128, 128),
        (4, 4096, 8, 128, 128),
        (4, 4096, 8, 256, 256),
        (4, 8192, 8, 128, 128),
        (4, 8192, 8, 256, 256),
        (4, 16384, 8, 128, 128),
        (4, 16384, 8, 256, 256),
    ]

    print("\n" + "=" * 70)
    print("Benchmark: cuTile chunk vs Triton fused_chunk_fwd")
    print("=" * 70)
    print(f"{'Config':<30} {'Triton (ms)':<15} {'cuTile (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    for B, T, H, K, V in configs:
        q, k, v = _make_inputs(B, T, H, K, V, torch.float32)
        scale = K ** -0.5

        with torch.no_grad():
            triton_chunk_size = min(64, max(16, triton.next_power_of_2(T)))
            t_triton = _bench(
                triton_fused_chunk_fwd,
                q, k, v, None, None, scale, None, False, None, triton_chunk_size,
            )
            t_cutile = _bench(
                cutile_fused_chunk_fwd,
                q, k, v,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=None,
                chunk_size=None,
            )

        speedup = t_triton / t_cutile if t_cutile > 0 else float("inf")
        config_str = f"B={B}, T={T}, H={H}, K={K}, V={V}"
        print(f"{config_str:<30} {t_triton:<15.3f} {t_cutile:<15.3f} {speedup:<10.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    run()
