#!/usr/bin/env python3
"""Benchmark Gemma-style RMSNorm native decomposition vs fused kernels."""

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
    "torch_gemma_native",
    "triton_gemma_fused",
    "cuda_gemma_fused",
]
DEFAULT_TOKENS = [1, 8, 128]
DEFAULT_HIDDEN = [2048, 4096, 8192]
DEFAULT_EPS = 1.0e-6


@dataclass(frozen=True)
class BenchResult:
    implementation: str
    tokens: int
    hidden: int
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
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn((tokens, hidden), device=device, dtype=dtype)
    # Gemma-style weights are offset by +1.0 in forward, so keep the learned
    # weight small to avoid making error analysis mostly about extreme scaling.
    weight = torch.randn((hidden,), device=device, dtype=dtype) * 0.1
    return x, weight


def torch_gemma_native(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Mirror the PyTorch-native GemmaRMSNorm decomposition used by vLLM."""
    orig_dtype = x.dtype
    weight_fp32 = weight.float() + 1.0
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    y = x_fp32 * torch.rsqrt(variance + eps)
    y = y.to(weight_fp32.dtype) * weight_fp32
    return y.to(orig_dtype)


def triton_gemma_fused(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from decode_projection_fusion.kernels.gemma_rmsnorm_triton import triton_gemma_rmsnorm

    return triton_gemma_rmsnorm(x, weight, eps)


def cuda_gemma_fused(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from decode_projection_fusion.kernels.gemma_rmsnorm_cuda import cuda_gemma_rmsnorm

    return cuda_gemma_rmsnorm(x, weight, eps)


def select_run(
    implementation: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    if implementation == "torch_gemma_native":
        return lambda: torch_gemma_native(x, weight, eps)
    if implementation == "triton_gemma_fused":
        return lambda: triton_gemma_fused(x, weight, eps)
    if implementation == "cuda_gemma_fused":
        return lambda: cuda_gemma_fused(x, weight, eps)
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


def estimate_bytes(implementation: str, tokens: int, hidden: int, dtype: torch.dtype) -> int:
    elements = tokens * hidden
    dtype_bytes = dtype_size_bytes(dtype)
    if implementation == "torch_gemma_native":
        # Approximate global traffic for native decomposition with fp32 temps.
        # This is intentionally a model, not a profiler-derived exact value.
        return elements * (3 * dtype_bytes + 24)
    if implementation in {"triton_gemma_fused", "cuda_gemma_fused"}:
        # x read + weight read + output write.
        return elements * dtype_bytes * 3
    raise ValueError(f"Unknown implementation: {implementation}")


def benchmark_one(
    implementation: str,
    tokens: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
    warmup: int,
    runs: int,
) -> BenchResult:
    x, weight = make_inputs(tokens, hidden, dtype, device)

    if implementation != "torch_gemma_native" and device.type != "cuda":
        raise ValueError(f"{implementation} requires CUDA")

    with torch.no_grad():
        reference = torch_gemma_native(x, weight, eps)
        run = select_run(implementation, x, weight, eps)
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

    estimated = estimate_bytes(implementation, tokens, hidden, dtype)
    bandwidth = estimated / latency_us / 1.0e3
    return BenchResult(
        implementation=implementation,
        tokens=tokens,
        hidden=hidden,
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


def baseline_lookup(rows: list[BenchResult]) -> dict[tuple[int, int], BenchResult]:
    return {
        (row.tokens, row.hidden): row
        for row in rows
        if row.implementation == "torch_gemma_native"
    }


def best_by_shape(rows: list[BenchResult]) -> dict[tuple[int, int], BenchResult]:
    best: dict[tuple[int, int], BenchResult] = {}
    for row in rows:
        key = (row.tokens, row.hidden)
        current = best.get(key)
        if current is None or row.latency_us < current.latency_us:
            best[key] = row
    return best


def write_markdown(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baselines = baseline_lookup(rows)
    best = best_by_shape(rows)
    lines = [
        "# Gemma-style RMSNorm Mini Benchmark",
        "",
        "| implementation | tokens | hidden | dtype | latency us | est. GB/s | max abs error | vs torch native |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        baseline = baselines.get((row.tokens, row.hidden))
        ratio = row.latency_us / baseline.latency_us if baseline else float("nan")
        lines.append(
            "| {implementation} | {tokens} | {hidden} | {dtype} | {latency_us:.3f} | "
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
            "| tokens | hidden | best implementation | latency us | speedup vs torch native |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for (tokens, hidden), row in sorted(best.items()):
        baseline = baselines.get((tokens, hidden))
        speedup = baseline.latency_us / row.latency_us if baseline else float("nan")
        lines.append(
            f"| {tokens} | {hidden} | {row.implementation} | {row.latency_us:.3f} | {speedup:.3f} |"
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
    parser.add_argument("--hidden", default=",".join(str(v) for v in DEFAULT_HIDDEN))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cpu", action="store_true", help="Run PyTorch baseline on CPU.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("decode_projection_fusion/results/rtx4070/gemma_rmsnorm"),
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
    hidden_values = parse_int_list(args.hidden)
    implementations = [item.strip() for item in args.implementations.split(",") if item.strip()]
    environment = collect_environment(device)

    rows: list[BenchResult] = []
    console_lines: list[str] = []
    for tokens in tokens_values:
        for hidden in hidden_values:
            for implementation in implementations:
                result = benchmark_one(
                    implementation=implementation,
                    tokens=tokens,
                    hidden=hidden,
                    dtype=dtype,
                    device=device,
                    eps=args.eps,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                rows.append(result)
                line = (
                    f"{result.implementation} tokens={tokens} hidden={hidden} "
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
