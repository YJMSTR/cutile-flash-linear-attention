import torch
from fla.ops.cutile import HAS_CUTILE

def cutile_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """
    cuTile implementation of Recurrent Linear Attention.
    """
    if not HAS_CUTILE:
        raise RuntimeError("CuTile is not installed. Please install CuTile to use this function.")

    raise NotImplementedError("CuTile recurrent linear attention is not implemented yet.")
