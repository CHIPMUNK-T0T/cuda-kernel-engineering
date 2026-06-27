#!/usr/bin/env python3
"""Tune Triton GEMV block sizes for decode-like shapes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import torch
import torch.nn.functional as F

from decode_gemv.kernels.gemv_triton import triton_gemv


@dataclass(frozen=True)
class TuneResult:
    implementation: str
    tokens: int
    in_features: int
    out_features: int
    dtype: str
    block_k: int
    block_n: int
    latency_us: float
    effective_bandwidth_gb_s: float
    effective_tflops: float
    speedup_vs_torch_linear: float
    max_abs_error: float
    relative_l2_error: float


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_dtype(value: str) -> torch.dtype:
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def dtype_size_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


def estimate_bytes(tokens: int, in_features: int, out_features: int, dtype: torch.dtype) -> int:
    dtype_bytes = dtype_size_bytes(dtype)
    return (
        tokens * in_features * dtype_bytes
        + in_features * out_features * dtype_bytes
        + tokens * out_features * dtype_bytes
    )


def estimate_flops(tokens: int, in_features: int, out_features: int) -> int:
    return 2 * tokens * in_features * out_features


def cuda_time_us(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def make_inputs(
    tokens: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    device = torch.device("cuda")
    scale = in_features**0.5
    x = torch.randn((tokens, in_features), device=device, dtype=dtype)
    weight = torch.randn((in_features, out_features), device=device, dtype=dtype) / scale
    linear_weight = weight.t().contiguous()
    return x, weight, linear_weight


def benchmark_shape(
    tokens: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    block_k_values: list[int],
    block_n_values: list[int],
    warmup: int,
    runs: int,
) -> list[TuneResult]:
    x, weight, linear_weight = make_inputs(tokens, in_features, out_features, dtype)
    with torch.no_grad():
        reference = x @ weight
        torch_linear_fn = lambda: F.linear(x, linear_weight)
        torch_linear_fn()
        torch_linear_latency = cuda_time_us(torch_linear_fn, warmup, runs)

        rows: list[TuneResult] = []
        for block_k in block_k_values:
            for block_n in block_n_values:
                triton_fn = lambda: triton_gemv(
                    x,
                    weight,
                    block_k=block_k,
                    block_n=block_n,
                )
                triton_fn()
                latency_us = cuda_time_us(triton_fn, warmup, runs)
                output = triton_fn()
                diff = (output.float() - reference.float()).abs()
                relative_l2_error = (
                    torch.linalg.vector_norm(diff)
                    / torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-8)
                ).item()
                bandwidth = estimate_bytes(tokens, in_features, out_features, dtype) / latency_us / 1.0e3
                tflops = estimate_flops(tokens, in_features, out_features) / latency_us / 1.0e6
                rows.append(
                    TuneResult(
                        implementation="triton_gemv",
                        tokens=tokens,
                        in_features=in_features,
                        out_features=out_features,
                        dtype=str(dtype).replace("torch.", ""),
                        block_k=block_k,
                        block_n=block_n,
                        latency_us=latency_us,
                        effective_bandwidth_gb_s=bandwidth,
                        effective_tflops=tflops,
                        speedup_vs_torch_linear=torch_linear_latency / latency_us,
                        max_abs_error=diff.max().item(),
                        relative_l2_error=relative_l2_error,
                    )
                )
        return rows


def write_csv(path: Path, rows: list[TuneResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TuneResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_markdown(path: Path, rows: list[TuneResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Triton GEMV Tuning Summary",
        "",
        "| shape | block_k | block_n | latency us | GB/s | TFLOP/s | speedup vs torch_linear | max abs error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r.tokens, r.in_features, r.out_features, r.latency_us)):
        lines.append(
            f"| {row.tokens}x{row.in_features}x{row.out_features} | {row.block_k} | {row.block_n} | "
            f"{row.latency_us:.3f} | {row.effective_bandwidth_gb_s:.3f} | "
            f"{row.effective_tflops:.3f} | {row.speedup_vs_torch_linear:.3f} | "
            f"{row.max_abs_error:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value).strip("-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1")
    parser.add_argument("--in-features", default="2048,4096")
    parser.add_argument("--out-features", default="8192,11008")
    parser.add_argument("--block-k", default="32,64,128")
    parser.add_argument("--block-n", default="32,64,128")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--out-dir", type=Path, default=Path("decode_gemv/results/rtx4070/triton_tuning"))
    parser.add_argument("--run-name", default="triton-gemv-tuning")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    dtype = parse_dtype(args.dtype)
    tokens_values = parse_int_list(args.tokens)
    in_feature_values = parse_int_list(args.in_features)
    out_feature_values = parse_int_list(args.out_features)
    block_k_values = parse_int_list(args.block_k)
    block_n_values = parse_int_list(args.block_n)

    rows: list[TuneResult] = []
    for tokens in tokens_values:
        for in_features in in_feature_values:
            for out_features in out_feature_values:
                shape_rows = benchmark_shape(
                    tokens=tokens,
                    in_features=in_features,
                    out_features=out_features,
                    dtype=dtype,
                    block_k_values=block_k_values,
                    block_n_values=block_n_values,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                rows.extend(shape_rows)
                best = min(shape_rows, key=lambda row: row.latency_us)
                print(
                    f"best tokens={tokens} in_features={in_features} out_features={out_features} "
                    f"block_k={best.block_k} block_n={best.block_n} "
                    f"latency_us={best.latency_us:.3f} "
                    f"speedup_vs_torch_linear={best.speedup_vs_torch_linear:.3f} "
                    f"max_abs_error={best.max_abs_error:.6g}"
                )

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / "runs" / f"{timestamp}-{safe_name(args.run_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.out_dir / "summary.csv", rows)
    write_markdown(args.out_dir / "summary.md", rows)
    write_csv(run_dir / "summary.csv", rows)
    write_markdown(run_dir / "summary.md", rows)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "command": " ".join([sys.executable, *sys.argv]),
                "args": vars(args) | {"out_dir": str(args.out_dir)},
                "rows": [row.__dict__ for row in rows],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"record_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
