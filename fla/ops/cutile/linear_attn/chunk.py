# based on fla/ops/common/fused_chunk.py, but using cuTile

import os
import torch
import cuda.tile as ct
from fla.ops.cutile import HAS_CUTILE
from fla.ops.cutile.utils import next_power_of_2

_AUTOTUNE_ENABLED = os.getenv("CUTILE_AUTOTUNE", "1") != "0"
_AUTOTUNE_WARMUP = int(os.getenv("CUTILE_AUTOTUNE_WARMUP", "2"))
_AUTOTUNE_ITERS = int(os.getenv("CUTILE_AUTOTUNE_ITERS", "5"))
_AUTOTUNE_CACHE: dict[tuple, tuple[int, int, int]] = {}



def _select_chunk_size(
    T: int,
    device: torch.device,
    chunk_size: int | None,
) -> int:
    if chunk_size is not None:
        return int(chunk_size)
    # Match Triton baseline, then prefer smaller tiles on SM120.
    candidate = min(64, max(16, next_power_of_2(T)))
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
        except Exception:
            return candidate
        # SM120: smaller chunk wins in current profiles.
        if props.major >= 12:
            # Only use small chunk if T is large enough to amortize state overhead
            if T >= 1024:
                return 16
            else:
                return 32
    return candidate


def _select_block_kv(
    K: int,
    V: int,
    device: torch.device,
) -> tuple[int, int]:
    BK = min(max(next_power_of_2(K), 16), 128)
    if K < 128:
        BK = min(BK, 64)
    BV = min(max(next_power_of_2(V), 16), 64)
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
        except Exception:
            return BK, BV
        # SM120 register pressure: prefer smaller K/V tiles for common head sizes.
        if props.major >= 12:
            if BK >= 128:
                BV = min(BV, 16)
            # Prioritize reducing BV over BK to avoid NK>1.
            if BK * BV > 2048:
                BV = min(BV, 32)
            if BK * BV > 2048:
                BV = min(BV, 16)
            if BK * BV > 2048:
                BK = min(BK, 64)
            if BK * BV > 2048:
                BV = min(BV, 16)
                BK = min(BK, 64)
    return BK, BV


def _candidate_chunk_sizes(T: int, cu_seqlens: torch.Tensor | None) -> list[int]:
    candidates = [16, 32, 64]
    base = min(64, max(16, next_power_of_2(T)))
    if cu_seqlens is not None:
        candidates = [c for c in candidates if (cu_seqlens % c).any().item() == 0]
        if base not in candidates and (cu_seqlens % base).any().item() == 0:
            candidates.append(base)
    else:
        if base not in candidates:
            candidates.append(base)
    return sorted(set(candidates))


def _candidate_block_kv(K: int, V: int) -> list[tuple[int, int]]:
    base_BK = min(max(next_power_of_2(K), 16), 128)
    if K < 128:
        base_BK = min(base_BK, 64)
    base_BV = min(max(next_power_of_2(V), 16), 64)
    candidates = {(base_BK, base_BV)}

    # Try sacrificing BV first (to keep NK=1)
    for cand_bv in (32, 16):
        candidates.add((base_BK, min(base_BV, cand_bv)))

    # Then try sacrificing BK
    for cand_bk in (64, 32):
        if cand_bk <= base_BK:
            candidates.add((cand_bk, base_BV))
            candidates.add((cand_bk, min(base_BV, 32)))
            candidates.add((cand_bk, min(base_BV, 16)))

    if base_BK >= 128:
        candidates.add((128, 32))
        candidates.add((128, 16))

    return sorted(candidates)


def _cutile_fused_chunk_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    BT: int,
    BK: int,
    BV: int,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    NK, NV = ct.cdiv(K, BK), ct.cdiv(V, BV)

    o = v.new_empty(NK, *v.shape, dtype=torch.float32)
    ht = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    use_initial_state = initial_state is not None
    use_final_state = output_final_state
    is_varlen = cu_seqlens is not None
    dummy_tensor = torch.empty(0, device=q.device)
    cu_seqlen_args = cu_seqlens if is_varlen else dummy_tensor
    h0_args = initial_state if use_initial_state else dummy_tensor
    ht_args = ht if use_final_state else dummy_tensor

    grid = (NV, NK, N * H)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        cutile_fused_chunk_fwd_kernel,
        (
            q,
            k,
            v,
            o,
            h0_args,
            ht_args,
            cu_seqlen_args,
            scale,
            T,
            B,
            H,
            K,
            V,
            BT,
            BK,
            BV,
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


def _bench_chunk_config(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    BT: int,
    BK: int,
    BV: int,
) -> float:
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(_AUTOTUNE_WARMUP):
            _cutile_fused_chunk_fwd_impl(
                q,
                k,
                v,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                BT=BT,
                BK=BK,
                BV=BV,
            )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_AUTOTUNE_ITERS):
            _cutile_fused_chunk_fwd_impl(
                q,
                k,
                v,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                BT=BT,
                BK=BK,
                BV=BV,
            )
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / max(_AUTOTUNE_ITERS, 1)


def _autotune_chunk_config(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[int, int, int]:
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
        BT = _select_chunk_size(q.shape[1], q.device, None)
        BK, BV = _select_block_kv(q.shape[3], v.shape[-1], q.device)
        return BT, BK, BV

    candidates_bt = _candidate_chunk_sizes(q.shape[1], cu_seqlens)
    candidates_kv = _candidate_block_kv(q.shape[3], v.shape[-1])
    if not candidates_bt:
        BT = _select_chunk_size(q.shape[1], q.device, None)
        BK, BV = _select_block_kv(q.shape[3], v.shape[-1], q.device)
        return BT, BK, BV

    best = None
    best_ms = float("inf")
    for BT in candidates_bt:
        for BK, BV in candidates_kv:
            try:
                ms = _bench_chunk_config(
                    q,
                    k,
                    v,
                    scale=scale,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens,
                    BT=BT,
                    BK=BK,
                    BV=BV,
                )
            except Exception:
                continue
            if ms < best_ms:
                best_ms = ms
                best = (BT, BK, BV)
                # print(f"New best: {best} -> {best_ms:.3f} ms")
            # print(f"Candidate: BT={BT}, BK={BK}, BV={BV} -> {ms:.3f} ms")
    if best is None:
        best = (
            _select_chunk_size(q.shape[1], q.device, None),
            *_select_block_kv(q.shape[3], v.shape[-1], q.device),
        )
    _AUTOTUNE_CACHE[key] = best
    return best


# Do 3 things in one kernel:
# 1. Intra: Compute b_s = mask(b_q @ b_k^T)     [BT, BT]
# 2. Inter: Compute b_o = b_q @ b_h + b_s @ b_v, where b_h is [BK, BV], result shape [BT, BV]
# 3. Compute b_h = b_h + b_k^T @ b_v,           [BK, BV], update state
 

@ct.kernel
def cutile_fused_chunk_fwd_kernel(
    q,                      # [B, T, H, K]
    k,                      # [B, T, H, K]
    v,                      # [B, T, H, V]
    o,                      # [B, T, H, V]
    h0,                     # [N, H, K, V]
    ht,                     # [N, H, K, V]
    cu_seqlens,
    scale,
    T,
    B: ct.Constant[int],
    H: ct.Constant[int],
    K: ct.Constant[int],
    V: ct.Constant[int],
    BT: ct.Constant[int],
    BK: ct.Constant[int],
    BV: ct.Constant[int],
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
        base_t = bos // BT
    else:
        bos, eos = i_n * T, i_n * T + T
        base_b = i_n
        base_t = 0
    NT = ct.cdiv(T, BT)

    o_i = ct.arange(BT, dtype=ct.int32)
    # causal mask
    m_s = o_i[:, None] >= o_i[None, :]
    b_h = ct.zeros((BK, BV), dtype=ct.float32)
    


    if USE_INITIAL_STATE:
        b_h = ct.reshape(ct.load(h0, (i_n, i_h, i_k, i_v), (1, 1, BK, BV), allow_tma=True), (BK, BV))
    
    for i_t in range(0, NT):
        t_idx = base_t + i_t
        b_q = ct.reshape(
            ct.load(q, (base_b, t_idx, i_h, i_k), (1, BT, 1, BK), padding_mode=ct.PaddingMode.ZERO, allow_tma=True),
            (BT, BK),
        )
        b_q = b_q * scale
        b_k_t = ct.reshape(
            ct.load(
                k,
                (base_b, i_k, i_h, t_idx),
                (1, BK, 1, BT),
                order=(0, 3, 2, 1),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=True,
            ),
            (BK, BT),
        )
        b_v = ct.reshape(
            ct.load(v, (base_b, t_idx, i_h, i_v), (1, BT, 1, BV), padding_mode=ct.PaddingMode.ZERO, allow_tma=True),
            (BT, BV),
        )

        # step 1: intra chunk
        b_s = ct.matmul(b_q, b_k_t)
        b_s = ct.where(m_s, b_s, 0)
        # step 2: inter chunk
        b_o = ct.matmul(b_s, b_v) + ct.matmul(b_q, b_h)
        ct.store(o, (i_k, base_b, t_idx, i_h, i_v), ct.reshape(b_o, (1, 1, BT, 1, BV)), allow_tma=True)
        # step 3: update state
        b_h = b_h + ct.matmul(b_k_t, b_v)


    if STORE_FINAL_STATE:
        ct.store(ht, (i_n, i_h, i_k, i_v), ct.reshape(b_h, (1, 1, BK, BV)), allow_tma=True)
    





def cutile_fused_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int | None = None,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("Packed varlen requires B == 1 for cutile_fused_chunk_fwd.")
    if chunk_size is None:
        BT, BK, BV = _autotune_chunk_config(
            q,
            k,
            v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    else:
        BT = _select_chunk_size(T, q.device, chunk_size)
        BK, BV = _select_block_kv(K, V, q.device)
    if cu_seqlens is not None and (cu_seqlens % BT).any().item():
        raise ValueError("cu_seqlens must be multiples of chunk_size for packed varlen.")
    # Aligned packed inputs are zero-padded, so per-chunk m_t masking is unnecessary.
    return _cutile_fused_chunk_fwd_impl(
        q,
        k,
        v,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        BT=BT,
        BK=BK,
        BV=BV,
    )