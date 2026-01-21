import torch
import cupy as cp
import numpy as np
import cuda.tile as ct
from fla.ops.cutile import HAS_CUTILE

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    result = a_tile + b_tile
    ct.store(c, index=(pid, ), tile=result)

def test():
    if not HAS_CUTILE:
        print("cuTile not found, skipping test.")
        return

    vector_size = 2**12
    tile_size = 2**4
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    # Use Torch tensors
    a = torch.randn(vector_size, device='cuda')
    b = torch.randn(vector_size, device='cuda')
    c = torch.zeros_like(a)

    print(f"Launching cuTile kernel with Torch stream: {torch.cuda.current_stream()}")
    
    ct.launch(torch.cuda.current_stream(),
              grid,
              vector_add,
              (a, b, c, tile_size))

    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    c_np = c.cpu().numpy()

    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected, decimal=5)
    print("✓ vector_add with Torch stream passed!")

if __name__ == "__main__":
    test()
