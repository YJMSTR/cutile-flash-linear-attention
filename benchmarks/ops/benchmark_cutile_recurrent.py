import torch
import triton
from fla.ops.linear_attn import fused_recurrent_linear_attn
from fla.ops.linear_attn.naive import naive_recurrent_linear_attn
from fla.ops.cutile.linear_attn.recurrent import cutile_recurrent_linear_attn_fwd

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[512 * 2 ** i for i in range(0, 7)], # 512 to 32768
        line_arg='provider',
        line_vals=['naive', 'triton', 'cutile'],
        line_names=['Naive (PyTorch)', 'Triton (Fused)', 'cuTile (Recurrent)'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="Execution Time (ms)",
        plot_name="Recurrent Linear Attention Forward Performance",
        args={},
    )
)
def benchmark(T, provider):
    # VRAM friendly settings
    B, H, K, V = 4, 8, 64, 64
    dtype = torch.float32 
    device = 'cuda'

    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    scale = K ** -0.5

    # Benchmark forward pass only
    # Warmup
    if provider == 'naive':
        ms = triton.testing.do_bench(lambda: naive_recurrent_linear_attn(q, k, v, scale=scale))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: fused_recurrent_linear_attn(q, k, v, scale=scale))
    elif provider == 'cutile':
        ms = triton.testing.do_bench(lambda: cutile_recurrent_linear_attn_fwd(q, k, v, scale=scale))
    
    return ms

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)
