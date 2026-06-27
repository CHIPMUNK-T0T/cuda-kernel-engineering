#!/usr/bin/env python3
"""Benchmark tiny copy/add/mul patterns and Triton fused replacements."""

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
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is not installed. Use the repo .venv.") from exc


DEFAULT_IMPLEMENTATIONS = [
    "torch_add_mul",
    "torch_clone_add_mul",
    "triton_add_mul",
    "triton_copy_add_mul",
]
DEFAULT_TOKENS = [1, 8, 128]
DEFAULT_FEATURES = [2048, 4096, 8192, 11008, 16384]


@dataclass(frozen=True)
class BenchResult:
    implementation: str
    tokens: int
    features: int
    dtype: str
    device: str
    latency_us: float
    estimated_bytes: int
    effective_bandwidth_gb_s: float
    max_abs_error: float
    mean_abs_error: float
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
    if dtype in {torch.float16, torch.bfloat16}:
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


def make_inputs(
    tokens: int,
    features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn((tokens, features), device=device, dtype=dtype)
    residual = torch.randn((tokens, features), device=device, dtype=dtype)
    scale = torch.randn((tokens, features), device=device, dtype=dtype) * 0.1
    return x, residual, scale


def run_torch_add_mul(x: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    z = x + residual
    return z * scale


def run_torch_clone_add_mul(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    tmp = x.clone()
    z = tmp + residual
    return z * scale


def run_triton_add_mul(x: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from decode_projection_fusion.kernels.elementwise_triton import triton_add_mul

    return triton_add_mul(x, residual, scale)


def run_triton_copy_add_mul(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    from decode_projection_fusion.kernels.elementwise_triton import triton_copy_add_mul

    return triton_copy_add_mul(x, residual, scale)


def select_run(
    implementation: str,
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
):
    if implementation == "torch_add_mul":
        return lambda: run_torch_add_mul(x, residual, scale)
    if implementation == "torch_clone_add_mul":
        return lambda: run_torch_clone_add_mul(x, residual, scale)
    if implementation == "triton_add_mul":
        return lambda: run_triton_add_mul(x, residual, scale)
    if implementation == "triton_copy_add_mul":
        return lambda: run_triton_copy_add_mul(x, residual, scale)
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


def estimate_bytes(
    implementation: str,
    tokens: int,
    features: int,
    dtype: torch.dtype,
) -> int:
    element_bytes = dtype_size_bytes(dtype)
    elements = tokens * features
    if implementation == "torch_add_mul":
        # add: read x/residual, write z. mul: read z/scale, write y.
        accesses_per_element = 6
    elif implementation == "torch_clone_add_mul":
        # clone plus the add/mul sequence above.
        accesses_per_element = 8
    elif implementation in {"triton_add_mul", "triton_copy_add_mul"}:
        # read x/residual/scale, write y.
        accesses_per_element = 4
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    return elements * element_bytes * accesses_per_element


def benchmark_one(
    implementation: str,
    tokens: int,
    features: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    runs: int,
) -> BenchResult:
    x, residual, scale = make_inputs(tokens, features, dtype, device)

    if implementation.startswith("triton") and device.type != "cuda":
        raise ValueError(f"{implementation} requires CUDA")

    with torch.no_grad():
        reference = run_torch_clone_add_mul(x, residual, scale)
        run = select_run(implementation, x, residual, scale)
        run()

        if device.type == "cuda":
            latency_us = cuda_time_us(run, warmup, runs)
            torch.cuda.synchronize()
        else:
            latency_us = cpu_time_us(run, warmup, runs)

        output = run()
        diff = (output.float() - reference.float()).abs()
        max_abs_error = diff.max().item()
        mean_abs_error = diff.mean().item()
        relative_l2_error = (
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-8)
        ).item()

    estimated = estimate_bytes(implementation, tokens, features, dtype)
    bandwidth = estimated / latency_us / 1.0e3
    return BenchResult(
        implementation=implementation,
        tokens=tokens,
        features=features,
        dtype=str(dtype).replace("torch.", ""),
        device=str(device),
        latency_us=latency_us,
        estimated_bytes=estimated,
        effective_bandwidth_gb_s=bandwidth,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        relative_l2_error=relative_l2_error,
    )


def write_csv(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def best_by_shape(rows: list[BenchResult]) -> dict[tuple[int, int], BenchResult]:
    best: dict[tuple[int, int], BenchResult] = {}
    for row in rows:
        key = (row.tokens, row.features)
        current = best.get(key)
        if current is None or row.latency_us < current.latency_us:
            best[key] = row
    return best


def baseline_lookup(rows: list[BenchResult], implementation: str) -> dict[tuple[int, int], BenchResult]:
    return {
        (row.tokens, row.features): row
        for row in rows
        if row.implementation == implementation
    }


def write_markdown(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baselines = baseline_lookup(rows, "torch_clone_add_mul")
    best = best_by_shape(rows)
    lines = [
        "# Decode Projection Fusion Copy/Add/Mul Benchmark",
        "",
        "| implementation | tokens | features | dtype | latency us | est. GB/s | max abs error | vs torch clone+add+mul |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        baseline = baselines.get((row.tokens, row.features))
        ratio = row.latency_us / baseline.latency_us if baseline else float("nan")
        lines.append(
            "| {implementation} | {tokens} | {features} | {dtype} | {latency_us:.3f} | "
            "{effective_bandwidth_gb_s:.3f} | {max_abs_error:.6g} | {ratio:.3f} |".format(
                ratio=ratio,
                **row.__dict__,
            )
        )

    lines.extend(
        [
            "",
            "## Best By Shape",
            "",
            "| tokens | features | best implementation | latency us |",
            "|---:|---:|---|---:|",
        ]
    )
    for (tokens, features), row in sorted(best.items()):
        lines.append(
            f"| {tokens} | {features} | {row.implementation} | {row.latency_us:.3f} |"
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
    command = " ".join([sys.executable, *sys.argv])
    metadata = {
        "started_at": started_at,
        "finished_at": finished_at,
        "command": command,
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
    parser.add_argument("--features", default=",".join(str(v) for v in DEFAULT_FEATURES))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cpu", action="store_true", help="Run PyTorch baselines on CPU.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("decode_projection_fusion/results/rtx4070/copy_add_mul"),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-record-run", action="store_true")
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
    feature_values = parse_int_list(args.features)
    implementations = [item.strip() for item in args.implementations.split(",") if item.strip()]
    environment = collect_environment(device)

    rows: list[BenchResult] = []
    console_lines: list[str] = []
    for tokens in tokens_values:
        for features in feature_values:
            for implementation in implementations:
                result = benchmark_one(
                    implementation=implementation,
                    tokens=tokens,
                    features=features,
                    dtype=dtype,
                    device=device,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                rows.append(result)
                line = (
                    f"{result.implementation} tokens={tokens} features={features} "
                    f"dtype={result.dtype} latency_us={result.latency_us:.3f} "
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
