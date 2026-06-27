"""Small elementwise fusion kernels for decode-side copy/add/mul patterns."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_mul_kernel(
    x_ptr,
    residual_ptr,
    scale_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = (x + residual) * scale
    tl.store(y_ptr + offsets, y, mask=mask)


def _check_inputs(x: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not residual.is_cuda or not scale.is_cuda:
        raise ValueError("residual and scale must be CUDA tensors")
    if x.shape != residual.shape or x.shape != scale.shape:
        raise ValueError("x, residual, and scale must have the same shape")
    if x.dtype != residual.dtype or x.dtype != scale.dtype:
        raise ValueError("x, residual, and scale must have the same dtype")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are float16, bfloat16, and float32")
    return torch.empty_like(x)


def triton_add_mul(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int = 1024,
) -> torch.Tensor:
    """Run y = (x + residual) * scale as a single Triton kernel."""
    y = _check_inputs(x, residual, scale)
    x_contig = x.contiguous()
    residual_contig = residual.contiguous()
    scale_contig = scale.contiguous()
    n_elements = y.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    _add_mul_kernel[grid](
        x_contig,
        residual_contig,
        scale_contig,
        y,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return y


def triton_copy_add_mul(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int = 1024,
) -> torch.Tensor:
    """Run the clone/copy + add + mul pattern without materializing the copy."""
    return triton_add_mul(x, residual, scale, block_size=block_size)
