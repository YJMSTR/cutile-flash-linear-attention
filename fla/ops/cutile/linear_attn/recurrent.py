# based on fla/ops/common/fused_recurrent.py, but using cuTile
# S_t = S_{t-1} + k_t * v_t
# O_t = q_t * S_t

import torch
import cuda.tile as ct
import cupy as cp
from einops import rearrange
from fla.ops.cutile import HAS_CUTILE
from fla.ops.cutile.utils import next_power_of_2

@ct.kernel
def cutile_recurrent_linear_attn_fwd_kernel(
    q, # [B(Batch), T(Time), H(Heads), K(Head dim for query/key)] 
    k, # [B(Batch), T(Time), H(Heads), K(Head dim for query/key)]
    v, # [B(Batch), T(Time), H(Heads), V(Head dim for value/output)]
    o, # [B(Batch), T(Time), H(Heads), V(Head dim for value/output)]
    h0, ht, # [N(Batch), H(Heads), K(Head dim for query/key), V(Head dim for value/output)]
    cu_seqlens,
    scale,
    B, T, H: ct.Constant[int], K: ct.Constant[int], V: ct.Constant[int],
    BK: ct.Constant[int], BV: ct.Constant[int],
    USE_INITIAL_STATE: ct.Constant[bool],
    STORE_FINAL_STATE: ct.Constant[bool],
    IS_VARLEN: ct.Constant[bool],
):
    i_v, i_k, i_nh = ct.bid(0), ct.bid(1), ct.bid(2)
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = ct.load(cu_seqlens, i_n, shape=()), ct.load(cu_seqlens, i_n + 1, shape=())
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    s = ct.zeros((BK, BV), ct.float32)

    if USE_INITIAL_STATE:
        s += ct.reshape(ct.load(h0, (i_n, i_h, i_k * BK, i_v * BV), (1, 1, BK, BV)), (BK, BV))

    for t in range(0, T):
        curr_q = ct.reshape(ct.load(q, (i_n, t, i_h, i_k * BK), (1, 1, 1, BK)), (BK,)) * scale
        curr_k = ct.reshape(ct.load(k, (i_n, t, i_h, i_k * BK), (1, 1, 1, BK)), (BK,))
        curr_v = ct.reshape(ct.load(v, (i_n, t, i_h, i_v * BV), (1, 1, 1, BV)), (BV,))

        s += curr_k[:, None] * curr_v[None, :]
        curr_o = s * curr_q[:, None]
        curr_o = ct.sum(curr_o, axis=0)
        ct.store(o, (i_k, i_n, t, i_h, i_v * BV), ct.reshape(curr_o, (1, 1, 1, 1, BV)))

    if STORE_FINAL_STATE:
        ct.store(ht, (i_n, i_h, i_k * BK, i_v * BV), ct.reshape(s, (1, 1, BK, BV)))


def cutile_recurrent_linear_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):

    B, T, H, K, V = *q.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = min(next_power_of_2(K), 64), min(next_power_of_2(V), 64)
    NK, NV = ct.cdiv(K, BK), ct.cdiv(V, BV)

    if scale is None:
        scale = K ** -0.5

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    use_initial_state = h0 is not None
    use_final_state = ht is not None
    is_varlen = cu_seqlens is not None
    dummy_tensor = torch.empty(0, device=q.device)
    cu_seqlen_args = cu_seqlens if is_varlen else dummy_tensor
    h0_args = h0 if use_initial_state else dummy_tensor
    ht_args = ht if use_final_state else dummy_tensor

    grid = (NV, NK, N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        cutile_recurrent_linear_attn_fwd_kernel,
        (
            q, k, v, o, 
            h0_args, ht_args, 
            cu_seqlen_args, scale, 
            B, T, H, K, V, 
            BK, BV, 
            use_initial_state, 
            use_final_state, 
            is_varlen,
        ),
    )
    o = o.sum(0)
    return o, ht
    
    
    
    
