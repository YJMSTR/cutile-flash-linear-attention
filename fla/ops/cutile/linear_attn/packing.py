from __future__ import annotations

from typing import Iterable

import torch


def _normalize_lengths(lengths: Iterable[int] | torch.Tensor) -> list[int]:
    if isinstance(lengths, torch.Tensor):
        lengths_list = [int(x) for x in lengths.flatten().tolist()]
    else:
        lengths_list = [int(x) for x in lengths]
    return lengths_list


def _aligned_lengths(lengths: list[int], chunk_size: int | None) -> list[int]:
    if chunk_size is None:
        return lengths
    return [((L + chunk_size - 1) // chunk_size) * chunk_size for L in lengths]


def build_cu_seqlens(
    lengths: Iterable[int] | torch.Tensor,
    *,
    chunk_size: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    lengths_list = _normalize_lengths(lengths)
    aligned = _aligned_lengths(lengths_list, chunk_size)
    cu_seqlens = torch.zeros(len(aligned) + 1, dtype=torch.int32, device=device)
    offset = 0
    for i, L in enumerate(aligned):
        offset += L
        cu_seqlens[i + 1] = offset
    return cu_seqlens


def pack_varlen_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: Iterable[int] | torch.Tensor,
    *,
    chunk_size: int | None = None,
    padding_side: str = "right",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    Pack padded sequences into a single batch (B=1) with optional chunk alignment.
    Returns packed (q, k, v), cu_seqlens (int32), and aligned lengths.
    """
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'.")
    if q.shape[:3] != k.shape[:3] or q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, v must share the same (B, T, H) dimensions.")

    lengths_list = _normalize_lengths(lengths)
    B, S = q.shape[:2]
    if len(lengths_list) != B:
        raise ValueError("lengths must have the same size as batch dimension.")
    if any(L > S for L in lengths_list):
        raise ValueError("All lengths must be <= sequence length S.")

    aligned = _aligned_lengths(lengths_list, chunk_size)
    cu_seqlens = build_cu_seqlens(aligned, device=q.device)
    total = int(cu_seqlens[-1].item())

    packed_q = q.new_zeros((1, total, *q.shape[2:]))
    packed_k = k.new_zeros((1, total, *k.shape[2:]))
    packed_v = v.new_zeros((1, total, *v.shape[2:]))

    offset = 0
    for i, L in enumerate(lengths_list):
        if padding_side == "left":
            src = slice(S - L, S)
        else:
            src = slice(0, L)
        dst = slice(offset, offset + L)
        packed_q[0, dst] = q[i, src]
        packed_k[0, dst] = k[i, src]
        packed_v[0, dst] = v[i, src]
        offset += aligned[i]

    return packed_q, packed_k, packed_v, cu_seqlens, aligned


def unpack_packed_outputs(
    packed: torch.Tensor,
    lengths: Iterable[int] | torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    max_len: int,
    padding_side: str = "right",
) -> torch.Tensor:
    """
    Unpack packed output back to padded [B, S, ...] using original lengths.
    """
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'.")

    lengths_list = _normalize_lengths(lengths)
    B = len(lengths_list)
    out = packed.new_zeros((B, max_len, *packed.shape[2:]))
    cu_seqlens_list = [int(x) for x in cu_seqlens.flatten().tolist()]

    for i, L in enumerate(lengths_list):
        start = cu_seqlens_list[i]
        src = slice(start, start + L)
        if padding_side == "left":
            dst = slice(max_len - L, max_len)
        else:
            dst = slice(0, L)
        out[i, dst] = packed[0, src]

    return out
