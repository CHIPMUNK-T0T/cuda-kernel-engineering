#!/usr/bin/env python3
"""Mini decode runner for Nsight Systems profiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import torch

from mini_llm_decode.benchmarks.bench_decode import (
    DEFAULT_EPS,
    decode_cuda_residual_fused,
    decode_pytorch_unfused,
    decode_triton_residual_fused,
    make_inputs,
)


IMPLEMENTATIONS = {
    "pytorch_unfused": decode_pytorch_unfused,
    "cuda_residual_fused": decode_cuda_residual_fused,
    "triton_residual_fused": decode_triton_residual_fused,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", required=True, choices=sorted(IMPLEMENTATIONS))
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--distinct-projection-weights", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    device = torch.device("cuda")
    dtype = torch.float16
    x, residual, norm_weights, projection_weights = make_inputs(
        tokens=args.tokens,
        hidden=args.hidden,
        layers=args.layers,
        dtype=dtype,
        device=device,
        distinct_projection_weights=args.distinct_projection_weights,
    )

    fn = IMPLEMENTATIONS[args.implementation]
    with torch.no_grad():
        reference = decode_pytorch_unfused(
            x,
            residual,
            norm_weights,
            projection_weights,
            args.eps,
        )
        run = lambda: fn(x, residual, norm_weights, projection_weights, args.eps)

        # Build / JIT and warm up before Nsight Systems starts collecting.
        for _ in range(args.warmup):
            y = run()
        torch.cuda.synchronize()

        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("profile_decode")
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
        projection_weight_mode = "distinct" if args.distinct_projection_weights else "shared"
        print(
            f"implementation={args.implementation} "
            f"tokens={args.tokens} hidden={args.hidden} layers={args.layers} "
            f"projection_weights={projection_weight_mode} iters={args.iters} "
            f"max_abs_error={max_abs_error:.6g} "
            f"relative_l2_error={relative_l2_error:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
