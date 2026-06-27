"""Python loader for the CUDA Gemma-style RMSNorm extension."""

from __future__ import annotations

import os
import shutil
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None


def _ensure_ninja_on_path() -> str | None:
    ninja = shutil.which("ninja")
    if ninja is not None:
        return ninja

    venv_bin = Path(sys.executable).parent
    venv_ninja = venv_bin / "ninja"
    if venv_ninja.exists():
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        return str(venv_ninja)

    return None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; gemma_rmsnorm_cuda cannot be loaded.")

    this_dir = Path(__file__).resolve().parent
    build_dir = Path(os.environ.get("GEMMA_RMSNORM_CUDA_BUILD_DIR", this_dir / "build"))
    build_dir.mkdir(parents=True, exist_ok=True)
    prebuilt = Path(
        os.environ.get(
            "GEMMA_RMSNORM_CUDA_PREBUILT",
            build_dir / "gemma_rmsnorm_cuda_ext.so",
        )
    )

    use_prebuilt = os.environ.get("GEMMA_RMSNORM_CUDA_USE_PREBUILT", "0") == "1"
    if prebuilt.exists() and (use_prebuilt or _ensure_ninja_on_path() is None):
        spec = importlib.util.spec_from_file_location("gemma_rmsnorm_cuda_ext", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load prebuilt extension: {prebuilt}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXTENSION = module
        return _EXTENSION

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    _EXTENSION = load(
        name="gemma_rmsnorm_cuda_ext",
        sources=[
            str(this_dir / "binding.cpp"),
            str(this_dir / "gemma_rmsnorm.cu"),
        ],
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _EXTENSION


def cuda_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    return _load_extension().forward(x, weight, float(eps))


def cuda_gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    y, residual_out = _load_extension().forward_fused_add(x, residual, weight, float(eps))
    return y, residual_out
