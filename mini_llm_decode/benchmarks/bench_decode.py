#!/usr/bin/env python3
"""Mini LLM decode-style benchmark.

This benchmark repeats a small decoder-like layer stack. Projection stays as
torch.matmul for every implementation; only the residual RMSNorm path changes.
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
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is not installed. Use the repo .venv.") from exc


from mini_transformer_block.benchmarks.bench_block import (
    residual_rmsnorm_cuda_fused,
    residual_rmsnorm_pytorch,
    residual_rmsnorm_triton_fused,
)


DEFAULT_IMPLEMENTATIONS = [
    "pytorch_unfused",
    "cuda_residual_fused",
    "triton_residual_fused",
]
DEFAULT_EPS = 1.0e-6


@dataclass(frozen=True)
class BenchResult:
    implementation: str
    tokens: int
    hidden: int
    layers: int
    projection_weight_mode: str
    dtype: str
    device: str
    latency_us: float
    tokens_per_second: float
    max_abs_error: float
    mean_abs_error: float
    relative_l2_error: float
    max_rel_error: float


def decode_pytorch_unfused(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weights: torch.Tensor,
    projection_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    h = x
    r = residual
    for layer in range(norm_weights.shape[0]):
        y = residual_rmsnorm_pytorch(h, r, norm_weights[layer], eps)
        out = y @ projection_weight_for_layer(projection_weights, layer)
        r = h
        h = out
    return h


def decode_cuda_residual_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weights: torch.Tensor,
    projection_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    h = x
    r = residual
    for layer in range(norm_weights.shape[0]):
        y = residual_rmsnorm_cuda_fused(h, r, norm_weights[layer], eps)
        out = y @ projection_weight_for_layer(projection_weights, layer)
        r = h
        h = out
    return h


def decode_triton_residual_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weights: torch.Tensor,
    projection_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    h = x
    r = residual
    for layer in range(norm_weights.shape[0]):
        y = residual_rmsnorm_triton_fused(h, r, norm_weights[layer], eps)
        out = y @ projection_weight_for_layer(projection_weights, layer)
        r = h
        h = out
    return h


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


def make_inputs(
    tokens: int,
    hidden: int,
    layers: int,
    dtype: torch.dtype,
    device: torch.device,
    distinct_projection_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn((tokens, hidden), device=device, dtype=dtype)
    residual = torch.randn((tokens, hidden), device=device, dtype=dtype)
    norm_weights = torch.randn((layers, hidden), device=device, dtype=dtype)
    if distinct_projection_weights:
        projection_weights = torch.randn((layers, hidden, hidden), device=device, dtype=dtype)
    else:
        projection_weights = torch.randn((hidden, hidden), device=device, dtype=dtype)
    projection_weights = projection_weights / (hidden**0.5)
    return x, residual, norm_weights, projection_weights


def projection_weight_for_layer(projection_weights: torch.Tensor, layer: int) -> torch.Tensor:
    if projection_weights.ndim == 2:
        return projection_weights
    return projection_weights[layer]


def select_run(
    implementation: str,
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weights: torch.Tensor,
    projection_weights: torch.Tensor,
    eps: float,
):
    if implementation == "pytorch_unfused":
        return lambda: decode_pytorch_unfused(x, residual, norm_weights, projection_weights, eps)
    if implementation == "cuda_residual_fused":
        return lambda: decode_cuda_residual_fused(x, residual, norm_weights, projection_weights, eps)
    if implementation == "triton_residual_fused":
        return lambda: decode_triton_residual_fused(x, residual, norm_weights, projection_weights, eps)
    raise ValueError(f"Unknown implementation: {implementation}")


def benchmark_one(
    implementation: str,
    tokens: int,
    hidden: int,
    layers: int,
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
    warmup: int,
    runs: int,
    distinct_projection_weights: bool,
) -> BenchResult:
    if implementation in {"cuda_residual_fused", "triton_residual_fused"} and device.type != "cuda":
        raise ValueError(f"{implementation} requires a CUDA device")

    x, residual, norm_weights, projection_weights = make_inputs(
        tokens=tokens,
        hidden=hidden,
        layers=layers,
        dtype=dtype,
        device=device,
        distinct_projection_weights=distinct_projection_weights,
    )

    with torch.no_grad():
        reference = decode_pytorch_unfused(x, residual, norm_weights, projection_weights, eps)
        run = select_run(implementation, x, residual, norm_weights, projection_weights, eps)
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
        denom = reference.float().abs().clamp_min(1.0e-8)
        max_rel_error = (diff / denom).max().item()

    return BenchResult(
        implementation=implementation,
        tokens=tokens,
        hidden=hidden,
        layers=layers,
        projection_weight_mode="distinct" if distinct_projection_weights else "shared",
        dtype=str(dtype).replace("torch.", ""),
        device=str(device),
        latency_us=latency_us,
        tokens_per_second=1_000_000.0 / latency_us,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        relative_l2_error=relative_l2_error,
        max_rel_error=max_rel_error,
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
        "# Mini LLM Decode Benchmark Summary",
        "",
        "| implementation | tokens | hidden | layers | projection weights | dtype | device | latency us | tokens/s | max abs error | mean abs error | relative L2 error | max rel error |",
        "|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {implementation} | {tokens} | {hidden} | {layers} | {projection_weight_mode} | {dtype} | {device} | "
            "{latency_us:.3f} | {tokens_per_second:.3f} | {max_abs_error:.6g} | "
            "{mean_abs_error:.6g} | {relative_l2_error:.6g} | {max_rel_error:.6g} |".format(
                **row.__dict__
            )
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
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument(
        "--distinct-projection-weights",
        action="store_true",
        help="Use one hidden x hidden projection matrix per layer. Default reuses one matrix to reduce VRAM.",
    )
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU for harness debugging.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mini_llm_decode/results/rtx4070"),
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

    dtype = torch.float16
    implementations = [item.strip() for item in args.implementations.split(",") if item.strip()]
    environment = collect_environment(device)

    rows = []
    console_lines = []
    for implementation in implementations:
        result = benchmark_one(
            implementation=implementation,
            tokens=args.tokens,
            hidden=args.hidden,
            layers=args.layers,
            dtype=dtype,
            device=device,
            eps=args.eps,
            warmup=args.warmup,
            runs=args.runs,
            distinct_projection_weights=args.distinct_projection_weights,
        )
        rows.append(result)
        line = (
            f"{result.implementation} tokens={args.tokens} hidden={args.hidden} "
            f"layers={args.layers} projection_weights={result.projection_weight_mode} "
            f"latency_us={result.latency_us:.3f} "
            f"tokens_per_second={result.tokens_per_second:.3f} "
            f"max_abs_error={result.max_abs_error:.6g} "
            f"relative_l2_error={result.relative_l2_error:.6g}"
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
