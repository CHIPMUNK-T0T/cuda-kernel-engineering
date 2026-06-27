"""Python loader for the CUDA RMSNorm extension."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; rmsnorm_cuda cannot be loaded.")

    this_dir = Path(__file__).resolve().parent
    build_dir = this_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    _EXTENSION = load(
        name="rmsnorm_cuda_ext",
        sources=[
            str(this_dir / "binding.cpp"),
            str(this_dir / "rmsnorm_naive.cu"),
            str(this_dir / "rmsnorm_optimized.cu"),
            str(this_dir / "fused_residual_rmsnorm.cu"),
        ],
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _EXTENSION


def rmsnorm_naive(x: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return _load_extension().forward(x, weight, float(eps))


def rmsnorm_optimized(x: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return _load_extension().forward_optimized(x, weight, float(eps))


def fused_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    return _load_extension().forward_fused_residual(x, residual, weight, float(eps))
