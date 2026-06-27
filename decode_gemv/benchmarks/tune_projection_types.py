#!/usr/bin/env python3
"""Tune Triton GEMV block sizes per decoder projection type."""

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

from decode_gemv.kernels.gemv_triton import triton_gemv


DEFAULT_PROJECTIONS = ["wo", "mlp_down"]
DEFAULT_TOKENS = [1]
DEFAULT_HIDDEN = [2048, 4096]
DEFAULT_INTERMEDIATE = [8192, 11008]


@dataclass(frozen=True)
class TuneResult:
    projection: str
    tokens: int
    hidden: int
    intermediate: int
    in_features: int
    out_features: int
    dtype: str
    device: str
    block_k: int
    block_n: int
    torch_linear_latency_us: float
    triton_latency_us: float
    triton_over_torch: float
    speedup_vs_torch_linear: float
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


def benchmark_shape(
    projection: str,
    tokens: int,
    hidden: int,
    intermediate: int,
    dtype: torch.dtype,
    device: torch.device,
    block_k_values: list[int],
    block_n_values: list[int],
    warmup: int,
    runs: int,
) -> list[TuneResult]:
    in_features, out_features = projection_shape(projection, hidden, intermediate)
    x, weight, linear_weight = make_inputs(tokens, in_features, out_features, dtype, device)
    with torch.no_grad():
        reference = F.linear(x, linear_weight)
        torch_linear_fn = lambda: F.linear(x, linear_weight)
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
                triton_latency = cuda_time_us(triton_fn, warmup, runs)
                output = triton_fn()
                max_abs_error, relative_l2_error = errors(output, reference)
                bandwidth = estimate_bytes(
                    tokens,
                    in_features,
                    out_features,
                    dtype,
                ) / triton_latency / 1.0e3
                tflops = estimate_flops(tokens, in_features, out_features) / triton_latency / 1.0e6
                rows.append(
                    TuneResult(
                        projection=projection,
                        tokens=tokens,
                        hidden=hidden,
                        intermediate=intermediate,
                        in_features=in_features,
                        out_features=out_features,
                        dtype=str(dtype).replace("torch.", ""),
                        device=str(device),
                        block_k=block_k,
                        block_n=block_n,
                        torch_linear_latency_us=torch_linear_latency,
                        triton_latency_us=triton_latency,
                        triton_over_torch=triton_latency / torch_linear_latency,
                        speedup_vs_torch_linear=torch_linear_latency / triton_latency,
                        effective_bandwidth_gb_s=bandwidth,
                        effective_tflops=tflops,
                        max_abs_error=max_abs_error,
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


def best_rows(rows: list[TuneResult]) -> list[TuneResult]:
    grouped: dict[tuple[str, int, int, int, int], list[TuneResult]] = {}
    for row in rows:
        key = (
            row.projection,
            row.tokens,
            row.hidden,
            row.intermediate,
            row.out_features,
        )
        grouped.setdefault(key, []).append(row)
    return [min(items, key=lambda row: row.triton_latency_us) for items in grouped.values()]


def write_markdown(path: Path, rows: list[TuneResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Projection Type Triton Tuning Summary",
        "",
        "## Best Per Shape",
        "",
        "| projection | shape | block_k | block_n | torch us | triton us | triton/torch | speedup | GB/s | TFLOP/s | max abs error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        best_rows(rows),
        key=lambda r: (r.projection, r.hidden, r.intermediate, r.in_features, r.out_features),
    ):
        lines.append(
            f"| {row.projection} | {row.tokens}x{row.in_features}x{row.out_features} | "
            f"{row.block_k} | {row.block_n} | {row.torch_linear_latency_us:.3f} | "
            f"{row.triton_latency_us:.3f} | {row.triton_over_torch:.3f} | "
            f"{row.speedup_vs_torch_linear:.3f} | {row.effective_bandwidth_gb_s:.3f} | "
            f"{row.effective_tflops:.3f} | {row.max_abs_error:.6g} |"
        )

    lines.extend(
        [
            "",
            "## All Tuning Rows",
            "",
            "| projection | shape | block_k | block_n | torch us | triton us | triton/torch | speedup | GB/s | TFLOP/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda r: (
            r.projection,
            r.hidden,
            r.intermediate,
            r.in_features,
            r.out_features,
            r.triton_latency_us,
        ),
    ):
        lines.append(
            f"| {row.projection} | {row.tokens}x{row.in_features}x{row.out_features} | "
            f"{row.block_k} | {row.block_n} | {row.torch_linear_latency_us:.3f} | "
            f"{row.triton_latency_us:.3f} | {row.triton_over_torch:.3f} | "
            f"{row.speedup_vs_torch_linear:.3f} | {row.effective_bandwidth_gb_s:.3f} | "
            f"{row.effective_tflops:.3f} |"
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
    rows: list[TuneResult],
    console_lines: list[str],
    environment: dict[str, object],
) -> None:
    metadata = {
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
    parser.add_argument("--block-k", default="32,64,128,256")
    parser.add_argument("--block-n", default="16,32,64,128")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("decode_gemv/results/rtx4070/projection_type_tuning"),
    )
    parser.add_argument("--run-name", default="projection-type-tuning")
    parser.add_argument("--no-record-run", action="store_true")
    parser.add_argument(
        "--dedupe-shapes",
        action="store_true",
        help="Skip repeated projection/in/out shapes, useful because QKV/Wo do not depend on intermediate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    device = torch.device("cuda")
    dtype = parse_dtype(args.dtype)
    tokens_values = parse_int_list(args.tokens)
    hidden_values = parse_int_list(args.hidden)
    intermediate_values = parse_int_list(args.intermediate)
    projections = parse_str_list(args.projections)
    block_k_values = parse_int_list(args.block_k)
    block_n_values = parse_int_list(args.block_n)
    environment = collect_environment(device)

    rows: list[TuneResult] = []
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

                    shape_rows = benchmark_shape(
                        projection=projection,
                        tokens=tokens,
                        hidden=hidden,
                        intermediate=intermediate,
                        dtype=dtype,
                        device=device,
                        block_k_values=block_k_values,
                        block_n_values=block_n_values,
                        warmup=args.warmup,
                        runs=args.runs,
                    )
                    rows.extend(shape_rows)
                    best = min(shape_rows, key=lambda row: row.triton_latency_us)
                    line = (
                        f"best projection={projection} tokens={tokens} "
                        f"in={in_features} out={out_features} hidden={hidden} "
                        f"intermediate={intermediate} block_k={best.block_k} "
                        f"block_n={best.block_n} torch_us={best.torch_linear_latency_us:.3f} "
                        f"triton_us={best.triton_latency_us:.3f} "
                        f"triton_over_torch={best.triton_over_torch:.3f} "
                        f"max_abs_error={best.max_abs_error:.6g}"
                    )
                    print(line)
                    console_lines.append(line)

    write_csv(args.out_dir / "summary.csv", rows)
    write_markdown(args.out_dir / "summary.md", rows)
    write_markdown(REPO_ROOT / "decode_gemv/results/rtx4070/projection_type_tuning_summary.md", rows)
    if not args.no_record_run:
        run_dir = make_run_dir(args.out_dir, args.run_name)
        write_run_record(
            run_dir=run_dir,
            args=args,
            rows=rows,
            console_lines=console_lines,
            environment=environment,
        )
        print(f"record_dir={run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
