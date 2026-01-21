import pytest
import torch
from fla.ops.linear_attn.naive import naive_recurrent_linear_attn
from fla.ops.cutile.linear_attn.recurrent import cutile_recurrent_linear_attn_fwd

@pytest.mark.parametrize("B, T, H, K, V", [
    (2, 64, 4, 32, 32),
    (1, 128, 8, 64, 64),
    (4, 32, 16, 16, 32),
])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cutile_recurrent_fwd(B, T, H, K, V, dtype):
    torch.manual_seed(42)
    device = 'cuda'
    
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    h0 = torch.randn(B, H, K, V, device=device, dtype=dtype)
    scale = K ** -0.5

    ref_o, ref_ht = naive_recurrent_linear_attn(
        q, k, v, 
        initial_state=h0, 
        output_final_state=True, 
        scale=scale
    )

    tri_o, tri_ht = cutile_recurrent_linear_attn_fwd(
        q, k, v, 
        initial_state=h0, 
        output_final_state=True, 
        scale=scale
    )

    torch.testing.assert_close(tri_o, ref_o, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tri_ht, ref_ht, atol=1e-4, rtol=1e-4)
    print(f"\n✓ Case B={B}, T={T}, H={H}, K={K}, V={V} passed!")

if __name__ == "__main__":
    test_cutile_recurrent_fwd(2, 64, 4, 32, 32, torch.float32)
