# based on fla/ops/common/fused_recurrent.py, but using cuTile
# S_t = S_{t-1} + k_t^T @ v_t
# O_t = q_t @ S_t

import os
import torch
import cuda.tile as ct
from fla.ops.cutile import HAS_CUTILE
from fla.ops.cutile.utils import next_power_of_2

_AUTOTUNE_ENABLED = os.getenv("CUTILE_AUTOTUNE", "1") != "0"
_AUTOTUNE_WARMUP = int(os.getenv("CUTILE_AUTOTUNE_WARMUP", "2"))
_AUTOTUNE_ITERS = int(os.getenv("CUTILE_AUTOTUNE_ITERS", "5"))
_AUTOTUNE_CACHE: dict[tuple, tuple[int, int]] = {}


def _select_block_kv(K: int, V: int, device: torch.device) -> tuple[int, int]:
    # Default behavior: match original behavior but allow larger BK
    BK = max(next_power_of_2(K), 16)
    BV = min(next_power_of_2(V), 64) 

    if BK >= 128 and V >= 64:
        BV = min(BV, 16)
    
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
        except Exception:
            return BK, BV
            
        if props.major >= 12:
            # SM120 Strategy: allow larger NK with smaller BV to reduce state size.
            # Favor BK=128 and BV=16 when K is large.
            req_BK = max(next_power_of_2(K), 16)
            BK = min(req_BK, 128)
            if BK >= 128 and V >= 64:
                BV = min(BV, 16)
            # Keep state size reasonable.
            if BK * BV > 2048:
                BV = min(BV, 16)
            if BK * BV > 2048:
                BK = min(BK, 64)
    return BK, BV


def _candidate_block_kv(K: int, V: int) -> list[tuple[int, int]]:
    base_BK = max(next_power_of_2(K), 16)
    base_BV = min(next_power_of_2(V), 64)
    candidates = {(base_BK, base_BV)}
    
    # Add High BK, Low BV candidates for SM120
    if base_BK >= 256:
        candidates.add((256, 8))
        candidates.add((256, 16))
        candidates.add((128, 32))
        candidates.add((128, 16))
    if base_BK >= 128:
        candidates.add((base_BK, 16))
    if base_BK >= 64:
        candidates.add((base_BK, 32))
        
    # Also add standard safe candidates
    candidates.add((64, 32))
    candidates.add((32, 64))
    candidates.add((32, 32))

    return sorted([c for c in candidates if c[0] >= 16 and c[1] >= 16])


def _cutile_recurrent_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    BK: int,
    BV: int,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    NK, NV = ct.cdiv(K, BK), ct.cdiv(V, BV)

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
    if NK > 1:
        o = o.sum(0).to(v)
    else:
        o = o.squeeze(0).to(v)
    return o, ht


def _bench_recurrent_config(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    BK: int,
    BV: int,
) -> float:
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(_AUTOTUNE_WARMUP):
            _cutile_recurrent_fwd_impl(
                q,
                k,
                v,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                BK=BK,
                BV=BV,
            )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_AUTOTUNE_ITERS):
            _cutile_recurrent_fwd_impl(
                q,
                k,
                v,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                BK=BK,
                BV=BV,
            )
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / max(_AUTOTUNE_ITERS, 1)


def _autotune_block_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[int, int]:
    key = (
        q.device.type,
        q.device.index,
        q.dtype,
        q.shape[1],
        q.shape[2],
        q.shape[3],
        v.shape[-1],
        cu_seqlens is not None,
        initial_state is not None,
        output_final_state,
    )
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached
    if (not _AUTOTUNE_ENABLED) or (not torch.cuda.is_available()) or (not q.is_cuda):
        return _select_block_kv(q.shape[3], v.shape[-1], q.device)

    candidates = _candidate_block_kv(q.shape[3], v.shape[-1])
    best = None
    best_ms = float("inf")
    for BK, BV in candidates:
        try:
            ms = _bench_recurrent_config(
                q,
                k,
                v,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                BK=BK,
                BV=BV,
            )
        except Exception:
            continue
        if ms < best_ms:
            best_ms = ms
            best = (BK, BV)
    if best is None:
        best = _select_block_kv(q.shape[3], v.shape[-1], q.device)
    _AUTOTUNE_CACHE[key] = best
    return best

@ct.kernel(
    occupancy=ct.ByTarget(sm_120=2, default=None),
    opt_level=ct.ByTarget(sm_120=2, default=3),
)
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

    if IS_VARLEN:
        bos, eos = ct.load(cu_seqlens, i_n, shape=(), allow_tma=True), ct.load(cu_seqlens, i_n + 1, shape=(), allow_tma=True)
        T = eos - bos
        base_b = 0
        t_offset = bos
    else:
        bos, eos = i_n * T, i_n * T + T
        base_b = i_n
        t_offset = 0

    s = ct.zeros((BK, BV), ct.float32)

    if USE_INITIAL_STATE:
        s += ct.reshape(ct.load(h0, (i_n, i_h, i_k, i_v), (1, 1, BK, BV), allow_tma=True), (BK, BV))

    for t in range(0, T):
        t_idx = t_offset + t
        curr_q = ct.reshape(
            ct.load(q, (base_b, t_idx, i_h, i_k), (1, 1, 1, BK), padding_mode=ct.PaddingMode.ZERO, allow_tma=True),
            (BK,),
        ) * scale
        curr_k = ct.reshape(
            ct.load(k, (base_b, t_idx, i_h, i_k), (1, 1, 1, BK), padding_mode=ct.PaddingMode.ZERO, allow_tma=True),
            (BK,),
        )
        curr_v = ct.reshape(
            ct.load(v, (base_b, t_idx, i_h, i_v), (1, 1, 1, BV), padding_mode=ct.PaddingMode.ZERO, allow_tma=True),
            (BV,),
        )


        s = s + ct.matmul(
            ct.reshape(curr_k, (BK, 1)),
            ct.reshape(curr_v, (1, BV)),
        )
        curr_o = ct.matmul(ct.reshape(curr_q, (1, BK)), s)
        curr_o = ct.reshape(curr_o, (BV,))
        ct.store(o, (i_k, base_b, t_idx, i_h, i_v), ct.reshape(curr_o, (1, 1, 1, 1, BV)), allow_tma=True)

    if STORE_FINAL_STATE:
        ct.store(ht, (i_n, i_h, i_k, i_v), ct.reshape(s, (1, 1, BK, BV)), allow_tma=True)


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
    if cu_seqlens is not None and B != 1:
        raise ValueError("Packed varlen requires B == 1 for cutile_recurrent_linear_attn_fwd.")
    if scale is None:
        scale = K ** -0.5

    BK, BV = _autotune_block_kv(
        q,
        k,
        v,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return _cutile_recurrent_fwd_impl(
        q,
        k,
        v,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        BK=BK,
        BV=BV,
    )
    
    
    
    
