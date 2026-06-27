#!/usr/bin/env python3
"""Small GEMV runner for Nsight Compute profiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import torch
import torch.nn.functional as F

from decode_gemv.kernels.gemv_triton import triton_gemv


IMPLEMENTATIONS = ("torch_linear", "triton_gemv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", required=True, choices=IMPLEMENTATIONS)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--in-features", type=int, required=True)
    parser.add_argument("--out-features", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=None)
    return parser.parse_args()


def parse_dtype(value: str) -> torch.dtype:
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = parse_dtype(args.dtype)
    scale = args.in_features**0.5
    x = torch.randn((args.tokens, args.in_features), device=device, dtype=dtype)
    weight = torch.randn(
        (args.in_features, args.out_features),
        device=device,
        dtype=dtype,
    ) / scale
    linear_weight = weight.t().contiguous()

    with torch.no_grad():
        reference = x @ weight
        if args.implementation == "torch_linear":
            run = lambda: F.linear(x, linear_weight)
        elif args.implementation == "triton_gemv":
            run = lambda: triton_gemv(
                x,
                weight,
                block_k=args.block_k,
                block_n=args.block_n,
            )
        else:
            raise ValueError(f"Unknown implementation: {args.implementation}")

        # Build / JIT and warm up before the measured launch.
        for _ in range(args.warmup):
            y = run()
        torch.cuda.synchronize()

        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("profile_gemv")
        for _ in range(args.iters):
            y = run()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        diff = (y.float() - reference.float()).abs()
        print(
            f"implementation={args.implementation} "
            f"tokens={args.tokens} in_features={args.in_features} "
            f"out_features={args.out_features} dtype={args.dtype} "
            f"block_k={args.block_k} block_n={args.block_n} "
            f"iters={args.iters} max_abs_error={diff.max().item():.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
