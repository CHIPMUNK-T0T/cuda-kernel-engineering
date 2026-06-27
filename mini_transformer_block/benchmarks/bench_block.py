#!/usr/bin/env python3
"""Mini transformer block benchmark.

This benchmark keeps the projection as torch.matmul for every implementation
and only swaps the residual RMSNorm path.
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


DEFAULT_HIDDEN_SIZES = [4096, 8192]
DEFAULT_NUM_TOKENS = [1, 512]
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
    out_features: int
    dtype: str
    device: str
    latency_us: float
    max_abs_error: float
    mean_abs_error: float
    relative_l2_error: float
    max_rel_error: float


def residual_rmsnorm_pytorch(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    z = x + residual
    variance = z.float().pow(2).mean(dim=-1, keepdim=True)
    y = z.float() * torch.rsqrt(variance + eps) * weight.float()
    return y.to(dtype=x.dtype)


def residual_rmsnorm_cuda_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from rmsnorm.kernels.rmsnorm_cuda import fused_residual_rmsnorm

    return fused_residual_rmsnorm(x, residual, weight, eps)


def residual_rmsnorm_triton_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from rmsnorm.kernels.rmsnorm_triton import triton_fused_residual_rmsnorm

    return triton_fused_residual_rmsnorm(x, residual, weight, eps)


def block_pytorch_unfused(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y = residual_rmsnorm_pytorch(x, residual, norm_weight, eps)
    return y @ projection_weight


def block_cuda_residual_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y = residual_rmsnorm_cuda_fused(x, residual, norm_weight, eps)
    return y @ projection_weight


def block_triton_residual_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y = residual_rmsnorm_triton_fused(x, residual, norm_weight, eps)
    return y @ projection_weight


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


def benchmark_one(
    implementation: str,
    tokens: int,
    hidden: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
    warmup: int,
    runs: int,
) -> BenchResult:
    torch.manual_seed(0)
    x = torch.randn((tokens, hidden), device=device, dtype=dtype)
    residual = torch.randn((tokens, hidden), device=device, dtype=dtype)
    norm_weight = torch.randn((hidden,), device=device, dtype=dtype)
    projection_weight = torch.randn((hidden, out_features), device=device, dtype=dtype)

    with torch.no_grad():
        reference = block_pytorch_unfused(
            x,
            residual,
            norm_weight,
            projection_weight,
            eps,
        )
        if implementation == "pytorch_unfused":
            run = lambda: block_pytorch_unfused(
                x,
                residual,
                norm_weight,
                projection_weight,
                eps,
            )
        elif implementation == "cuda_residual_fused":
            if device.type != "cuda":
                raise ValueError("cuda_residual_fused requires a CUDA device")
            block_cuda_residual_fused(x, residual, norm_weight, projection_weight, eps)
            run = lambda: block_cuda_residual_fused(
                x,
                residual,
                norm_weight,
                projection_weight,
                eps,
            )
        elif implementation == "triton_residual_fused":
            if device.type != "cuda":
                raise ValueError("triton_residual_fused requires a CUDA device")
            block_triton_residual_fused(x, residual, norm_weight, projection_weight, eps)
            run = lambda: block_triton_residual_fused(
                x,
                residual,
                norm_weight,
                projection_weight,
                eps,
            )
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

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
        out_features=out_features,
        dtype=str(dtype).replace("torch.", ""),
        device=str(device),
        latency_us=latency_us,
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
        "# Mini Transformer Block Benchmark Summary",
        "",
        "| implementation | tokens | hidden | out features | dtype | device | latency us | max abs error | mean abs error | relative L2 error | max rel error |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {implementation} | {tokens} | {hidden} | {out_features} | {dtype} | {device} | "
            "{latency_us:.3f} | {max_abs_error:.6g} | {mean_abs_error:.6g} | "
            "{relative_l2_error:.6g} | {max_rel_error:.6g} |".format(
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
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument(
        "--out-features",
        type=int,
        default=None,
        help="Projection output size. Defaults to hidden.",
    )
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU for harness debugging.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mini_transformer_block/results/rtx4070"),
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
    tokens_list = [args.tokens] if args.tokens is not None else DEFAULT_NUM_TOKENS
    hidden_list = [args.hidden] if args.hidden is not None else DEFAULT_HIDDEN_SIZES
    implementations = [item.strip() for item in args.implementations.split(",") if item.strip()]
    environment = collect_environment(device)

    rows = []
    console_lines = []
    for tokens in tokens_list:
        for hidden in hidden_list:
            out_features = args.out_features if args.out_features is not None else hidden
            for implementation in implementations:
                result = benchmark_one(
                    implementation=implementation,
                    tokens=tokens,
                    hidden=hidden,
                    out_features=out_features,
                    dtype=dtype,
                    device=device,
                    eps=args.eps,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                rows.append(result)
                line = (
                    f"{result.implementation} tokens={tokens} hidden={hidden} "
                    f"out_features={out_features} latency_us={result.latency_us:.3f} "
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
