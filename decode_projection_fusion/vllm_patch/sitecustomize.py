"""Optional vLLM GemmaRMSNorm monkey patch.

This file is loaded automatically by Python when its directory is on
PYTHONPATH. The patch is opt-in via VLLM_GEMMA_RMSNORM_PATCH=1.
"""

from __future__ import annotations

import os
import sys


def _log(message: str) -> None:
    if os.environ.get("VLLM_GEMMA_RMSNORM_PATCH_VERBOSE", "1") != "0":
        print(f"[gemma_rmsnorm_patch] {message}", file=sys.stderr, flush=True)


def _install() -> None:
    if os.environ.get("VLLM_GEMMA_RMSNORM_PATCH", "0") != "1":
        return

    try:
        import torch
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    except Exception as exc:  # pragma: no cover - defensive startup path
        _log(f"install_failed={type(exc).__name__}: {exc}")
        if os.environ.get("VLLM_GEMMA_RMSNORM_PATCH_STRICT", "0") == "1":
            raise
        return

    backend = os.environ.get("VLLM_GEMMA_RMSNORM_PATCH_BACKEND", "triton")
    try:
        if backend == "cuda":
            from decode_projection_fusion.kernels.gemma_rmsnorm_cuda import (
                cuda_gemma_fused_add_rmsnorm as fused_add_rmsnorm,
                cuda_gemma_rmsnorm as rmsnorm,
            )
        elif backend == "triton":
            from decode_projection_fusion.kernels.gemma_rmsnorm_triton import (
                triton_gemma_fused_add_rmsnorm as fused_add_rmsnorm,
                triton_gemma_rmsnorm as rmsnorm,
            )
        else:
            raise ValueError(f"unknown backend: {backend}")
    except Exception as exc:  # pragma: no cover - defensive startup path
        _log(f"backend_import_failed={type(exc).__name__}: {exc}")
        if os.environ.get("VLLM_GEMMA_RMSNORM_PATCH_STRICT", "0") == "1":
            raise
        return

    original_forward_native = GemmaRMSNorm.forward_native
    original_forward_cuda = GemmaRMSNorm.forward_cuda

    def patched_forward_native(self, x, residual=None):
        if (
            x.is_cuda
            and x.dim() == 2
            and self.weight.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and self.weight.dtype == x.dtype
        ):
            try:
                if residual is None:
                    return rmsnorm(x, self.weight, self.variance_epsilon)
                if (
                    residual.is_cuda
                    and residual.shape == x.shape
                    and residual.dtype == x.dtype
                    and x.dtype != torch.float16
                ):
                    return fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
            except Exception as exc:
                _log(f"fallback_after_kernel_error={type(exc).__name__}: {exc}")
                if os.environ.get("VLLM_GEMMA_RMSNORM_PATCH_STRICT", "0") == "1":
                    raise
        return original_forward_native(self, x, residual)

    def patched_forward_cuda(self, x, residual=None):
        return patched_forward_native(self, x, residual)

    GemmaRMSNorm.forward_native = patched_forward_native
    GemmaRMSNorm.forward_cuda = patched_forward_cuda
    GemmaRMSNorm._cuda_kernel_patch_original_forward_native = original_forward_native
    GemmaRMSNorm._cuda_kernel_patch_original_forward_cuda = original_forward_cuda
    _log(f"installed backend={backend}")


_install()
