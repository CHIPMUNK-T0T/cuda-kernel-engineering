#!/usr/bin/env python3
"""Small mini-block runner for Nsight Systems profiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import torch

from mini_transformer_block.benchmarks.bench_block import (
    DEFAULT_EPS,
    block_cuda_residual_fused,
    block_pytorch_unfused,
    block_triton_residual_fused,
)


IMPLEMENTATIONS = {
    "pytorch_unfused": block_pytorch_unfused,
    "cuda_residual_fused": block_cuda_residual_fused,
    "triton_residual_fused": block_triton_residual_fused,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", required=True, choices=sorted(IMPLEMENTATIONS))
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--out-features", type=int, default=None)
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
    out_features = args.out_features if args.out_features is not None else args.hidden

    x = torch.randn((args.tokens, args.hidden), device=device, dtype=dtype)
    residual = torch.randn((args.tokens, args.hidden), device=device, dtype=dtype)
    norm_weight = torch.randn((args.hidden,), device=device, dtype=dtype)
    projection_weight = torch.randn((args.hidden, out_features), device=device, dtype=dtype)

    fn = IMPLEMENTATIONS[args.implementation]
    with torch.no_grad():
        reference = block_pytorch_unfused(
            x,
            residual,
            norm_weight,
            projection_weight,
            args.eps,
        )
        run = lambda: fn(x, residual, norm_weight, projection_weight, args.eps)

        # Build / JIT and warm up before Nsight Systems starts collecting.
        for _ in range(args.warmup):
            y = run()
        torch.cuda.synchronize()

        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("profile_block")
        for _ in range(args.iters):
            y = run()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

        diff = (y.float() - reference.float()).abs()
        max_abs_error = diff.max().item()
        relative_l2_error = (
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-8)
        ).item()
        print(
            f"implementation={args.implementation} "
            f"tokens={args.tokens} hidden={args.hidden} out_features={out_features} "
            f"iters={args.iters} max_abs_error={max_abs_error:.6g} "
            f"relative_l2_error={relative_l2_error:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
