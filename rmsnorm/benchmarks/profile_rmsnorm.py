#!/usr/bin/env python3
"""Small RMSNorm runner for Nsight Compute profiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import torch

from rmsnorm.benchmarks.bench_rmsnorm import (
    DEFAULT_EPS,
    residual_rmsnorm_cuda_fused,
    residual_rmsnorm_pytorch,
    residual_rmsnorm_triton_fused,
    rmsnorm_cuda_naive,
    rmsnorm_cuda_optimized,
    rmsnorm_pytorch,
    rmsnorm_triton,
)


IMPLEMENTATIONS = {
    "cuda_naive": ("rmsnorm", rmsnorm_cuda_naive),
    "cuda_optimized": ("rmsnorm", rmsnorm_cuda_optimized),
    "triton_rmsnorm": ("rmsnorm", rmsnorm_triton),
    "cuda_residual_fused": ("residual_rmsnorm", residual_rmsnorm_cuda_fused),
    "triton_residual_fused": ("residual_rmsnorm", residual_rmsnorm_triton_fused),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", required=True, choices=sorted(IMPLEMENTATIONS))
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    x = torch.randn((args.tokens, args.hidden), device=device, dtype=dtype)
    residual = torch.randn((args.tokens, args.hidden), device=device, dtype=dtype)
    weight = torch.randn((args.hidden,), device=device, dtype=dtype)

    operation, fn = IMPLEMENTATIONS[args.implementation]
    with torch.no_grad():
        if operation == "rmsnorm":
            reference = rmsnorm_pytorch(x, weight, args.eps)
            run = lambda: fn(x, weight, args.eps)
        elif operation == "residual_rmsnorm":
            reference = residual_rmsnorm_pytorch(x, residual, weight, args.eps)
            run = lambda: fn(x, residual, weight, args.eps)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Build / JIT and warm up before Nsight Compute starts collecting.
        for _ in range(args.warmup):
            y = run()
        torch.cuda.synchronize()

        torch.cuda.nvtx.range_push("profile_rmsnorm")
        for _ in range(args.iters):
            y = run()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

        diff = (y.float() - reference.float()).abs()
        max_abs_error = diff.max().item()
        print(
            f"implementation={args.implementation} "
            f"tokens={args.tokens} hidden={args.hidden} "
            f"iters={args.iters} max_abs_error={max_abs_error:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
