#!/usr/bin/env python3
"""Projection-type benchmark for decode GEMV.

This splits a projection-only decoder block into QKV, Wo, MLP gate/up, and
MLP down shapes so we can see which projection type makes Triton win or lose.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is not installed. Use the repo .venv.") from exc


DEFAULT_IMPLEMENTATIONS = ["torch_linear", "triton_tuned"]
DEFAULT_PROJECTIONS = ["qkv", "wo", "mlp_up", "mlp_down"]
DEFAULT_TOKENS = [1]
DEFAULT_HIDDEN = [2048, 4096]
DEFAULT_INTERMEDIATE = [8192, 11008]


@dataclass(frozen=True)
class BenchResult:
    projection: str
    implementation: str
    tokens: int
    in_features: int
    out_features: int
    hidden: int
    intermediate: int
    dtype: str
    device: str
    latency_us: float
    effective_bandwidth_gb_s: float
    effective_tflops: float
    max_abs_error: float
    relative_l2_error: float


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_dtype(value: str) -> torch.dtype:
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def dtype_size_bytes(dtype: torch.dtype) -> int:
    if dtype in {torch.float16, torch.bfloat16}:
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


def projection_shape(projection: str, hidden: int, intermediate: int) -> tuple[int, int]:
    if projection == "qkv":
        return hidden, 3 * hidden
    if projection == "wo":
        return hidden, hidden
    if projection == "mlp_up":
        return hidden, 2 * intermediate
    if projection == "mlp_down":
        return intermediate, hidden
    raise ValueError(f"Unknown projection: {projection}")


def make_inputs(
    tokens: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn((tokens, in_features), device=device, dtype=dtype)
    weight = torch.randn((in_features, out_features), device=device, dtype=dtype) / (
        in_features**0.5
    )
    linear_weight = weight.t().contiguous()
    return x, weight, linear_weight


def run_torch_linear(x: torch.Tensor, weight: torch.Tensor, linear_weight: torch.Tensor) -> torch.Tensor:
    del weight
    return F.linear(x, linear_weight)


def run_triton_tuned(x: torch.Tensor, weight: torch.Tensor, linear_weight: torch.Tensor) -> torch.Tensor:
    del linear_weight
    from decode_gemv.kernels.gemv_triton import triton_gemv

    return triton_gemv(x, weight, block_k=128, block_n=32)


def select_run(
    implementation: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    linear_weight: torch.Tensor,
):
    if implementation == "torch_linear":
        return lambda: run_torch_linear(x, weight, linear_weight)
    if implementation == "triton_tuned":
        if not x.is_cuda:
            raise ValueError("triton_tuned requires a CUDA device")
        return lambda: run_triton_tuned(x, weight, linear_weight)
    raise ValueError(f"Unknown implementation: {implementation}")


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


def cpu_time_us(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1_000_000.0)
    return statistics.median(samples)


def estimate_bytes(tokens: int, in_features: int, out_features: int, dtype: torch.dtype) -> int:
    dtype_bytes = dtype_size_bytes(dtype)
    return (
        tokens * in_features * dtype_bytes
        + in_features * out_features * dtype_bytes
        + tokens * out_features * dtype_bytes
    )


def estimate_flops(tokens: int, in_features: int, out_features: int) -> int:
    return 2 * tokens * in_features * out_features


def errors(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    diff = (output.float() - reference.float()).abs()
    max_abs_error = diff.max().item()
    relative_l2_error = (
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-8)
    ).item()
    return max_abs_error, relative_l2_error


def benchmark_one(
    projection: str,
    implementation: str,
    tokens: int,
    hidden: int,
    intermediate: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    runs: int,
) -> BenchResult:
    in_features, out_features = projection_shape(projection, hidden, intermediate)
    x, weight, linear_weight = make_inputs(tokens, in_features, out_features, dtype, device)
    with torch.no_grad():
        reference = run_torch_linear(x, weight, linear_weight)
        run = select_run(implementation, x, weight, linear_weight)
        run()

        if device.type == "cuda":
            latency_us = cuda_time_us(run, warmup, runs)
            torch.cuda.synchronize()
        else:
            latency_us = cpu_time_us(run, warmup, runs)

        output = run()
        max_abs_error, relative_l2_error = errors(output, reference)

    bandwidth = estimate_bytes(tokens, in_features, out_features, dtype) / latency_us / 1.0e3
    tflops = estimate_flops(tokens, in_features, out_features) / latency_us / 1.0e6
    return BenchResult(
        projection=projection,
        implementation=implementation,
        tokens=tokens,
        in_features=in_features,
        out_features=out_features,
        hidden=hidden,
        intermediate=intermediate,
        dtype=str(dtype).replace("torch.", ""),
        device=str(device),
        latency_us=latency_us,
        effective_bandwidth_gb_s=bandwidth,
        effective_tflops=tflops,
        max_abs_error=max_abs_error,
        relative_l2_error=relative_l2_error,
    )


def write_csv(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_markdown(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Projection Type Benchmark Summary",
        "",
        "| projection | implementation | tokens | in | out | hidden | intermediate | dtype | device | latency us | effective GB/s | effective TFLOP/s | max abs error | relative L2 error |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {projection} | {implementation} | {tokens} | {in_features} | {out_features} | "
            "{hidden} | {intermediate} | {dtype} | {device} | {latency_us:.3f} | "
            "{effective_bandwidth_gb_s:.3f} | {effective_tflops:.3f} | "
            "{max_abs_error:.6g} | {relative_l2_error:.6g} |".format(**row.__dict__)
        )
    path.write_text("\n".join(lines) + "\n")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value).strip("-")


def make_run_dir(out_dir: Path, run_name: str | None) -> Path:
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{safe_name(run_name)}" if run_name else ""
    run_dir = out_dir / "runs" / f"{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collect_environment(device: torch.device) -> dict[str, object]:
    env: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        env.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
                "gpu_total_memory_bytes": props.total_memory,
                "gpu_multiprocessor_count": props.multi_processor_count,
            }
        )
    return env


def write_run_record(
    run_dir: Path,
    args: argparse.Namespace,
    rows: list[BenchResult],
    console_lines: list[str],
    started_at: str,
    finished_at: str,
    environment: dict[str, object],
) -> None:
    metadata = {
        "started_at": started_at,
        "finished_at": finished_at,
        "command": " ".join([sys.executable, *sys.argv]),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": environment,
        "rows": [row.__dict__ for row in rows],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "console.txt").write_text("\n".join(console_lines) + "\n")
    write_csv(run_dir / "summary.csv", rows)
    write_markdown(run_dir / "summary.md", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default=",".join(str(v) for v in DEFAULT_TOKENS))
    parser.add_argument("--hidden", default=",".join(str(v) for v in DEFAULT_HIDDEN))
    parser.add_argument(
        "--intermediate",
        default=",".join(str(v) for v in DEFAULT_INTERMEDIATE),
    )
    parser.add_argument(
        "--projections",
        default=",".join(DEFAULT_PROJECTIONS),
        help="Comma-separated projection types: qkv,wo,mlp_up,mlp_down",
    )
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cpu", action="store_true", help="Run on CPU for harness debugging.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("decode_gemv/results/rtx4070/projection_types"),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-record-run", action="store_true")
    parser.add_argument(
        "--dedupe-shapes",
        action="store_true",
        help="Skip repeated projection/in/out shapes, useful because QKV/Wo do not depend on intermediate.",
    )
    parser.add_argument(
        "--implementations",
        default=",".join(DEFAULT_IMPLEMENTATIONS),
        help="Comma-separated implementations.",
    )
    return parser.parse_args()


def main() -> int:
    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    args = parse_args()
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise SystemExit("CUDA is not available. Use --cpu only for harness debugging.")

    dtype = parse_dtype(args.dtype)
    tokens_values = parse_int_list(args.tokens)
    hidden_values = parse_int_list(args.hidden)
    intermediate_values = parse_int_list(args.intermediate)
    projections = parse_str_list(args.projections)
    implementations = parse_str_list(args.implementations)
    environment = collect_environment(device)

    rows: list[BenchResult] = []
    console_lines: list[str] = []
    seen_shapes: set[tuple[str, int, int]] = set()
    for tokens in tokens_values:
        for hidden in hidden_values:
            for intermediate in intermediate_values:
                for projection in projections:
                    in_features, out_features = projection_shape(projection, hidden, intermediate)
                    shape_key = (projection, in_features, out_features)
                    if args.dedupe_shapes and shape_key in seen_shapes:
                        continue
                    seen_shapes.add(shape_key)
                    for implementation in implementations:
                        result = benchmark_one(
                            projection=projection,
                            implementation=implementation,
                            tokens=tokens,
                            hidden=hidden,
                            intermediate=intermediate,
                            dtype=dtype,
                            device=device,
                            warmup=args.warmup,
                            runs=args.runs,
                        )
                        rows.append(result)
                        line = (
                            f"{result.projection} {result.implementation} tokens={tokens} "
                            f"in={result.in_features} out={result.out_features} "
                            f"hidden={hidden} intermediate={intermediate} dtype={result.dtype} "
                            f"latency_us={result.latency_us:.3f} "
                            f"effective_tflops={result.effective_tflops:.3f} "
                            f"effective_bandwidth_gb_s={result.effective_bandwidth_gb_s:.3f} "
                            f"max_abs_error={result.max_abs_error:.6g}"
                        )
                        print(line)
                        console_lines.append(line)

    write_csv(args.out_dir / "summary.csv", rows)
    write_markdown(args.out_dir / "summary.md", rows)
    if not args.no_record_run:
        finished_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        run_dir = make_run_dir(args.out_dir, args.run_name)
        write_run_record(
            run_dir=run_dir,
            args=args,
            rows=rows,
            console_lines=console_lines,
            started_at=started_at,
            finished_at=finished_at,
            environment=environment,
        )
        print(f"record_dir={run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
