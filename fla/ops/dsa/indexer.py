import torch
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs

@triton.jit
def clean_logits_kernel(
    logits,
    cu_seqlen_ks,
    cu_seqlen_ke,
    S,
    BN: tl.constexpr,
):
    i_t = tl.program_id(0)

    cu_k_s = tl.load(cu_seqlen_ks + i_t).to(tl.int32)
    cu_k_e = tl.load(cu_seqlen_ke + i_t).to(tl.int32)

    for i_n in range (tl.cdiv(S, BN)):
        o_n = i_n * BN + tl.arange(0, BN)
        m_n = o_n < S
        m_out = (o_n < cu_k_s) | (o_n >= cu_k_e)
        p_logits = logits + i_t * S + o_n
        b_logits = tl.load(p_logits, mask = m_n, other=0.0)
        b_logits = tl.where(m_out, float('-inf'), b_logits)
        tl.store(p_logits, b_logits, mask=m_n)

@triton.autotune(
    configs=[
        triton.Config({'BN': BN, 'BK': BK, 'BQ': BQ}, num_warps=nw, num_stages=ns)
        for BN in [64, 128, 256]
        for BK in [64, 128]
        for BQ in [1, 2, 4]
        for nw in [4, 8, 16]
        for ns in [2, 3]
        # BN=256 with nw=16 exceeds 128-register budget on sm_90a (needs ~154 regs)
        if not (BN == 256 and nw == 16)
    ],
    key=['H', 'K', 'S', 'T'],
    **autotune_cache_kwargs,
)
@triton.jit
def mqa_attn_return_logits_kernel(
    q,
    k,
    k_scale,
    w,
    cu_seqlen_ks,
    cu_seqlen_ke,
    logits,
    T,
    S,
    H: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BQ: tl.constexpr,
):
    # grid = (cdiv(T, BQ), )
    # get the union set of BQ tokens' KV range
    # i_t: start index of the current BQ tokens
    i_t = tl.program_id(0) * BQ

    cu_k_s_min = S
    cu_k_e_max = 0

    for i_q in tl.static_range(BQ):
        if i_t + i_q < T:
            cu_k_s = tl.load(cu_seqlen_ks + i_t + i_q).to(tl.int32)
            cu_k_e = tl.load(cu_seqlen_ke + i_t + i_q).to(tl.int32)
            cu_k_s_min = tl.minimum(cu_k_s_min, cu_k_s)
            cu_k_e_max = tl.maximum(cu_k_e_max, cu_k_e)

    # load weights
    o_qh = tl.arange(0, BQ * H)
    p_w = w + i_t * H + o_qh
    m_w = (i_t + o_qh // H) < T
    b_w = tl.load(p_w, mask=m_w, other=0.0).to(tl.float32)

    # only iterate over the valid KV range [cu_k_s_min, cu_k_e_max)
    kv_start_aligned = (cu_k_s_min // BN) * BN
    n_tiles = tl.cdiv(cu_k_e_max - kv_start_aligned, BN)    # kv tiles 
    for i_n in range(n_tiles):
        o_n = kv_start_aligned + i_n * BN + tl.arange(0, BN)
        m_n = o_n < S

        # K scale
        b_ks = tl.load(k_scale + o_n, mask=m_n, other=0.0).to(tl.float32)
        # GEMM accumulator
        b_s = tl.zeros([BN, BQ * H], dtype=tl.float32)
        # GEMM loop over dimension K
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K

            # load K [BN, BK]
            p_k = k + o_n[:, None] * K + o_k[None, :]
            b_k = tl.load(p_k, mask=m_n[:, None] & m_k[None, :], other=0.0)

            # load Q [BQ*H, BK]
            o_qh_row = i_t * H + o_qh
            p_q = q + o_qh_row[:, None] * K + o_k[None, :]
            m_q = ((i_t + o_qh // H) < T)[:, None] & m_k[None, :]
            b_q = tl.load(p_q, mask=m_q, other=0.0)

            # K @ Q^T
            b_s += tl.dot(b_k, tl.trans(b_q))

        # ReLU
        b_s = tl.maximum(b_s, 0.0)
        # weight
        b_s = b_s * b_w[None, :]
        # K scale
        b_s = b_s * b_ks[:, None]

        # reduce: [BN, BQ*H] -> [BN] for each query
        for i_q in tl.static_range(BQ):
            if i_t + i_q < T:
                # extract the H heads for the current query
                m_h = (i_q * H <= o_qh)  & (o_qh < (i_q + 1) * H)
                b_logits = tl.sum(tl.where(m_h[None, :], b_s, 0.0), axis=1)
                p_logits = logits + (i_t + i_q) * S + o_n
                tl.store(p_logits, b_logits.to(p_logits.dtype.element_ty), mask=m_n)


def mqa_attn_return_logits_interface(q, kv, kv_scales, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
    T, H, K = q.shape
    S = kv.shape[0]

    logits = torch.full((T, S), float('-inf'), device=q.device, dtype=torch.float32)
    
    def grid(META):
        return (triton.cdiv(T, META['BQ']),)

    mqa_attn_return_logits_kernel[grid](
        q=q, k=kv, k_scale=kv_scales, w=weights,
        cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke,
        logits=logits,
        T=T, S=S, H=H, K=K,
    )

    if clean_logits:
        BN_clean = min(triton.next_power_of_2(S), 4096)
        clean_logits_kernel[(T,)](
            logits=logits,
            cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke,
            S=S, BN=BN_clean,
        )

    return logits
