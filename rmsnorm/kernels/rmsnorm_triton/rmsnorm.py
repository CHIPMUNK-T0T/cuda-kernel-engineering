"""Triton RMSNorm forward kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    hidden: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden

    row_start = row * hidden
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / hidden
    inv_rms = tl.rsqrt(variance + eps)
    y = x * inv_rms * weight

    tl.store(y_ptr + row_start + offsets, y, mask=mask)


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Run RMSNorm with a single-row-per-program Triton kernel."""
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not weight.is_cuda:
        raise ValueError("weight must be a CUDA tensor")
    if x.dim() != 2:
        raise ValueError("x must be 2D: [tokens, hidden]")
    if weight.dim() != 1:
        raise ValueError("weight must be 1D: [hidden]")
    if x.shape[1] != weight.shape[0]:
        raise ValueError("weight size must match x hidden size")
    if x.dtype != weight.dtype:
        raise ValueError("x and weight dtype must match")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("triton_rmsnorm supports float16, bfloat16, and float32")

    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    y = torch.empty_like(x_contig)

    tokens, hidden = x_contig.shape
    block_size = _next_power_of_2(hidden)
    if block_size > 131072:
        raise ValueError(f"hidden={hidden} is too large for this Triton RMSNorm kernel")

    _rmsnorm_kernel[(tokens,)](
        x_contig,
        weight_contig,
        y,
        hidden,
        float(eps),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return y
