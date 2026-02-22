"""
Benchmark & cross-validate: Triton Lightning Indexer vs TileLang FP8 Indexer.

Verifies numerical alignment and compares latency across configs.

Usage:
    python benchmarks/ops/benchmark_indexer_triton_vs_tilelang.py [--markdown]
"""

import argparse
import os
import sys
import torch

# ---- Triton indexer ----
from fla.ops.dsa.indexer import mqa_attn_return_logits_interface as triton_indexer

# ---- TileLang indexer ----
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TL_DIR = os.path.join(REPO_ROOT, "tilelang", "examples", "deepseek_v32")
if not os.path.isdir(TL_DIR):
    raise FileNotFoundError(f"TileLang examples not found at {TL_DIR}")
sys.path.insert(0, TL_DIR)

from fp8_lighting_indexer import mqa_attn_return_logits_interface  # type: ignore
from utils import per_custom_dims_cast_to_fp8  # type: ignore


def bench_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def compute_similarity(a: torch.Tensor, b: torch.Tensor):
    """Correlation-based similarity (same metric as TileLang's validate_tensor_match)."""
    a, b = a.double(), b.double()
    # Mask out non-finite values
    finite = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[finite], b[finite]
    norm_sum = (a * a + b * b).sum()
    if norm_sum == 0:
        return 1.0
    return (2 * (a * b).sum() / norm_sum).item()


# ---- Configs ----
# (S, SKV, H, D) — S=query seq len, SKV=kv seq len, H=indexer heads, D=indexer dim
CONFIGS = [
    (1024, 1024, 32, 64),
    (2048, 2048, 32, 64),
    (4096, 4096, 32, 64),
    (4096, 8192, 32, 64),
    (8192, 8192, 32, 64),
    (4096, 4096, 64, 128),
    (8192, 8192, 64, 128),
]


def run_one(S, SKV, H, D, warmup=10, iters=50):
    """Run both indexers on the same data, return (similarity, triton_ms, tilelang_ms)."""
    torch.manual_seed(42)

    # Shared inputs (bf16, as in TileLang test)
    q = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(SKV, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(S, H, device="cuda", dtype=torch.float32)

    # ---- FP8 quantization (shared path) ----
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    # ---- TileLang ----
    # TileLang uses simple causal: cu_seqlen_ks=0, cu_seqlen_ke=SKV for all tokens
    cu_ks = torch.zeros(S, dtype=torch.int32, device="cuda")
    cu_ke = torch.full((S,), SKV, dtype=torch.int32, device="cuda")

    def tl_fn():
        return mqa_attn_return_logits_interface(
            q=q_fp8, kv=kv_fp8, kv_scales=kv_scales,
            weights=w, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        )

    # ---- Triton ----
    # Pre-quantize FP8 outside timing (same as TileLang, reuse same quantization path)
    q_fp8_tri = q.to(torch.float8_e4m3fn)
    kv_fp8_tri, kv_scale_tri = per_custom_dims_cast_to_fp8(kv, (0,), False)

    def tri_fn():
        return triton_indexer(q=q_fp8_tri, kv=kv_fp8_tri, kv_scales=kv_scale_tri,
                              weights=w, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke)

    # ---- Correctness ----
    tl_out = tl_fn()                # [S, SKV]
    tri_out = tri_fn()              # [S, SKV]
    sim = compute_similarity(tl_out, tri_out)

    # ---- Benchmark ----
    tl_ms = bench_ms(tl_fn, warmup=warmup, iters=iters)
    tri_ms = bench_ms(tri_fn, warmup=warmup, iters=iters)

    return sim, tri_ms, tl_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true", help="Output markdown table")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    gpu_name = torch.cuda.get_device_name(0)
    results = []

    for S, SKV, H, D in CONFIGS:
        tag = f"S={S},SKV={SKV},H={H},D={D}"
        try:
            sim, tri_ms, tl_ms = run_one(S, SKV, H, D, warmup=args.warmup, iters=args.iters)
            ratio = tri_ms / tl_ms if tl_ms > 0 else float("inf")
            results.append((tag, S, SKV, H, D, sim, tri_ms, tl_ms, ratio))
        except Exception as exc:
            print(f"SKIP {tag}: {exc}")

    if args.markdown:
        print(f"## Lightning Indexer: Triton vs TileLang ({gpu_name})\n")
        print("| Config | Similarity | Triton (ms) | TileLang (ms) | Triton/TileLang |")
        print("|--------|-----------|-------------|---------------|-----------------|")
        for tag, S, SKV, H, D, sim, tri_ms, tl_ms, ratio in results:
            sim_str = f"{sim:.6f}" if sim >= 0.999 else f"**{sim:.6f}**"
            print(f"| {tag} | {sim_str} | {tri_ms:.3f} | {tl_ms:.3f} | {ratio:.2f}x |")
    else:
        print(f"GPU: {gpu_name}")
        print("=" * 100)
        print(f"{'Config':>35s} | {'Similarity':>10s} | {'Triton (ms)':>11s} | {'TileLang (ms)':>13s} | {'Tri/TL':>8s}")
        print("-" * 100)
        for tag, S, SKV, H, D, sim, tri_ms, tl_ms, ratio in results:
            print(f"{tag:>35s} | {sim:10.6f} | {tri_ms:11.3f} | {tl_ms:13.3f} | {ratio:7.2f}x")

    # Summary
    all_sim = [r[5] for r in results]
    if all_sim:
        min_sim = min(all_sim)
        status = "PASS" if min_sim >= 0.999 else "WARN"
        print(f"\nMin similarity: {min_sim:.6f} [{status}]")


if __name__ == "__main__":
    main()
