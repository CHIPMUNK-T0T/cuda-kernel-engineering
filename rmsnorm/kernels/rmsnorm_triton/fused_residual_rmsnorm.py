"""Triton fused residual RMSNorm forward kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rmsnorm.kernels.rmsnorm_triton.rmsnorm import _next_power_of_2


@triton.jit
def _fused_residual_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
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
    residual = tl.load(residual_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    z = (x + residual).to(y_ptr.dtype.element_ty).to(tl.float32)
    variance = tl.sum(z * z, axis=0) / hidden
    inv_rms = tl.rsqrt(variance + eps)
    y = z * inv_rms * weight

    tl.store(y_ptr + row_start + offsets, y, mask=mask)


def triton_fused_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Run residual add + RMSNorm with a single-row-per-program Triton kernel."""
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not residual.is_cuda:
        raise ValueError("residual must be a CUDA tensor")
    if not weight.is_cuda:
        raise ValueError("weight must be a CUDA tensor")
    if x.dim() != 2:
        raise ValueError("x must be 2D: [tokens, hidden]")
    if residual.dim() != 2:
        raise ValueError("residual must be 2D: [tokens, hidden]")
    if weight.dim() != 1:
        raise ValueError("weight must be 1D: [hidden]")
    if residual.shape != x.shape:
        raise ValueError("residual shape must match x shape")
    if x.shape[1] != weight.shape[0]:
        raise ValueError("weight size must match x hidden size")
    if x.dtype != residual.dtype or x.dtype != weight.dtype:
        raise ValueError("x, residual, and weight dtype must match")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "triton_fused_residual_rmsnorm supports float16, bfloat16, and float32"
        )

    x_contig = x.contiguous()
    residual_contig = residual.contiguous()
    weight_contig = weight.contiguous()
    y = torch.empty_like(x_contig)

    tokens, hidden = x_contig.shape
    block_size = _next_power_of_2(hidden)
    if block_size > 131072:
        raise ValueError(
            f"hidden={hidden} is too large for this Triton fused residual RMSNorm kernel"
        )

    _fused_residual_rmsnorm_kernel[(tokens,)](
        x_contig,
        residual_contig,
        weight_contig,
        y,
        hidden,
        float(eps),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return y
