"""Triton GEMV / small-batch linear kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_id = tl.program_id(0)
    out_tile_id = tl.program_id(1)

    offs_n = out_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in tl.range(0, in_features, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < in_features
        n_mask = offs_n < out_features

        x = tl.load(
            x_ptr + token_id * in_features + offs_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + offs_k[:, None] * out_features + offs_n[None, :],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x[:, None] * weight, axis=0)

    tl.store(
        y_ptr + token_id * out_features + offs_n,
        acc,
        mask=offs_n < out_features,
    )


def _select_block_n(out_features: int) -> int:
    if out_features >= 8192:
        return 64
    return 32


def triton_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    block_k: int = 64,
    block_n: int | None = None,
) -> torch.Tensor:
    """Run y = x @ weight with a simple Triton GEMV kernel."""
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not weight.is_cuda:
        raise ValueError("weight must be a CUDA tensor")
    if x.dim() != 2:
        raise ValueError("x must be 2D: [tokens, in_features]")
    if weight.dim() != 2:
        raise ValueError("weight must be 2D: [in_features, out_features]")
    if x.shape[1] != weight.shape[0]:
        raise ValueError("weight input size must match x in_features")
    if x.dtype != weight.dtype:
        raise ValueError("x and weight dtype must match")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("triton_gemv supports float16, bfloat16, and float32")

    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    tokens, in_features = x_contig.shape
    _, out_features = weight_contig.shape
    y = torch.empty((tokens, out_features), device=x.device, dtype=x.dtype)

    block_n_value = block_n if block_n is not None else _select_block_n(out_features)
    if block_k <= 0 or block_n_value <= 0:
        raise ValueError("block_k and block_n must be positive")

    grid = (tokens, triton.cdiv(out_features, block_n_value))
    _gemv_kernel[grid](
        x_contig,
        weight_contig,
        y,
        in_features,
        out_features,
        BLOCK_K=block_k,
        BLOCK_N=block_n_value,
        num_warps=4,
    )
    return y
