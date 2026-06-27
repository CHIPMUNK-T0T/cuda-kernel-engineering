#!/usr/bin/env python3
"""RMSNorm benchmark harness.

The PyTorch implementation is used as the correctness reference. CUDA / Triton
implementations are measured through the same output format.
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
    raise SystemExit(
        "PyTorch is not installed. Install a CUDA-enabled PyTorch build before "
        "running this benchmark."
    ) from exc


DEFAULT_HIDDEN_SIZES = [2048, 3072, 4096, 8192]
DEFAULT_NUM_TOKENS = [1, 8, 32, 128, 512]
DEFAULT_EPS = 1.0e-6
DEFAULT_RMSNORM_IMPLEMENTATIONS = [
    "pytorch_eager",
    "cuda_naive",
    "triton_rmsnorm",
    "cuda_optimized",
]
DEFAULT_RESIDUAL_RMSNORM_IMPLEMENTATIONS = [
    "pytorch_residual_unfused",
    "cuda_residual_unfused",
    "cuda_residual_fused",
    "triton_residual_fused",
]


@dataclass(frozen=True)
class BenchResult:
    operation: str
    implementation: str
    tokens: int
    hidden: int
    dtype: str
    device: str
    latency_us: float
    effective_bandwidth_gb_s: float
    max_abs_error: float
    max_rel_error: float


def rmsnorm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps) * weight.float()
    return y.to(dtype=x.dtype)


def residual_rmsnorm_pytorch(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    z = x + residual
    return rmsnorm_pytorch(z, weight, eps)


def rmsnorm_cuda_naive(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from rmsnorm.kernels.rmsnorm_cuda import rmsnorm_naive

    return rmsnorm_naive(x, weight, eps)


def rmsnorm_cuda_optimized(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from rmsnorm.kernels.rmsnorm_cuda import rmsnorm_optimized

    return rmsnorm_optimized(x, weight, eps)


def residual_rmsnorm_cuda_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from rmsnorm.kernels.rmsnorm_cuda import fused_residual_rmsnorm

    return fused_residual_rmsnorm(x, residual, weight, eps)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from rmsnorm.kernels.rmsnorm_triton import triton_rmsnorm

    return triton_rmsnorm(x, weight, eps)


def residual_rmsnorm_triton_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from rmsnorm.kernels.rmsnorm_triton import triton_fused_residual_rmsnorm

    return triton_fused_residual_rmsnorm(x, residual, weight, eps)


def estimate_rmsnorm_bytes(tokens: int, hidden: int, dtype_bytes: int) -> int:
    # Minimal model for eager-style RMSNorm output traffic:
    # x read + weight read + output write. Temporary framework traffic is not
    # included, so this is a lower-bound useful for comparing later kernels.
    elements = tokens * hidden
    return elements * dtype_bytes * 3


def estimate_residual_rmsnorm_bytes(
    implementation: str,
    tokens: int,
    hidden: int,
    dtype_bytes: int,
) -> int:
    elements = tokens * hidden
    if implementation.endswith("_unfused"):
        # x read + residual read + z write + two z reads + weight read + output write.
        return elements * dtype_bytes * 7
    # x and residual are read once for reduction and once for output, with no z temporary:
    # two x reads + two residual reads + weight read + output write.
    return elements * dtype_bytes * 6


def estimate_effective_bytes(
    operation: str,
    implementation: str,
    tokens: int,
    hidden: int,
    dtype_bytes: int,
) -> int:
    if operation == "rmsnorm":
        return estimate_rmsnorm_bytes(tokens, hidden, dtype_bytes)
    if operation == "residual_rmsnorm":
        return estimate_residual_rmsnorm_bytes(implementation, tokens, hidden, dtype_bytes)
    raise ValueError(f"Unknown operation: {operation}")


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
    operation: str,
    implementation: str,
    tokens: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
    warmup: int,
    runs: int,
) -> BenchResult:
    torch.manual_seed(0)
    x = torch.randn((tokens, hidden), device=device, dtype=dtype)
    residual = torch.randn((tokens, hidden), device=device, dtype=dtype)
    weight = torch.randn((hidden,), device=device, dtype=dtype)

    with torch.no_grad():
        if operation == "rmsnorm":
            reference = rmsnorm_pytorch(x, weight, eps)
            if implementation == "pytorch_eager":
                run = lambda: rmsnorm_pytorch(x, weight, eps)
            elif implementation == "cuda_naive":
                if device.type != "cuda":
                    raise ValueError("cuda_naive requires a CUDA device")
                # Build/load the extension before timing.
                rmsnorm_cuda_naive(x, weight, eps)
                run = lambda: rmsnorm_cuda_naive(x, weight, eps)
            elif implementation == "cuda_optimized":
                if device.type != "cuda":
                    raise ValueError("cuda_optimized requires a CUDA device")
                # Build/load the extension before timing.
                rmsnorm_cuda_optimized(x, weight, eps)
                run = lambda: rmsnorm_cuda_optimized(x, weight, eps)
            elif implementation == "triton_rmsnorm":
                if device.type != "cuda":
                    raise ValueError("triton_rmsnorm requires a CUDA device")
                # JIT compile before timing.
                rmsnorm_triton(x, weight, eps)
                run = lambda: rmsnorm_triton(x, weight, eps)
            else:
                raise ValueError(f"Unknown implementation for rmsnorm: {implementation}")
        elif operation == "residual_rmsnorm":
            reference = residual_rmsnorm_pytorch(x, residual, weight, eps)
            if implementation == "pytorch_residual_unfused":
                run = lambda: residual_rmsnorm_pytorch(x, residual, weight, eps)
            elif implementation == "cuda_residual_unfused":
                if device.type != "cuda":
                    raise ValueError("cuda_residual_unfused requires a CUDA device")
                # Build/load the RMSNorm extension before timing the add + kernel path.
                rmsnorm_cuda_optimized(x + residual, weight, eps)
                run = lambda: rmsnorm_cuda_optimized(x + residual, weight, eps)
            elif implementation == "cuda_residual_fused":
                if device.type != "cuda":
                    raise ValueError("cuda_residual_fused requires a CUDA device")
                # Build/load the fused extension before timing.
                residual_rmsnorm_cuda_fused(x, residual, weight, eps)
                run = lambda: residual_rmsnorm_cuda_fused(x, residual, weight, eps)
            elif implementation == "triton_residual_fused":
                if device.type != "cuda":
                    raise ValueError("triton_residual_fused requires a CUDA device")
                # JIT compile before timing.
                residual_rmsnorm_triton_fused(x, residual, weight, eps)
                run = lambda: residual_rmsnorm_triton_fused(x, residual, weight, eps)
            else:
                raise ValueError(
                    f"Unknown implementation for residual_rmsnorm: {implementation}"
                )
        else:
            raise ValueError(f"Unknown operation: {operation}")

        if device.type == "cuda":
            latency_us = cuda_time_us(run, warmup, runs)
            torch.cuda.synchronize()
        else:
            latency_us = cpu_time_us(run, warmup, runs)

        output = run()
        diff = (output.float() - reference.float()).abs()
        max_abs_error = diff.max().item()
        denom = reference.float().abs().clamp_min(1.0e-8)
        max_rel_error = (diff / denom).max().item()

    bytes_moved = estimate_effective_bytes(
        operation,
        implementation,
        tokens,
        hidden,
        torch.finfo(dtype).bits // 8,
    )
    bandwidth = bytes_moved / (latency_us * 1.0e-6) / 1.0e9

    return BenchResult(
        operation=operation,
        implementation=implementation,
        tokens=tokens,
        hidden=hidden,
        dtype=str(dtype).replace("torch.", ""),
        device=str(device),
        latency_us=latency_us,
        effective_bandwidth_gb_s=bandwidth,
        max_abs_error=max_abs_error,
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
        "# RMSNorm Benchmark Summary",
        "",
        "| operation | implementation | tokens | hidden | dtype | device | latency us | effective GB/s | max abs error | max rel error |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {operation} | {implementation} | {tokens} | {hidden} | {dtype} | {device} | "
            "{latency_us:.3f} | {effective_bandwidth_gb_s:.3f} | "
            "{max_abs_error:.6g} | {max_rel_error:.6g} |".format(**row.__dict__)
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
    args_dict = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    metadata = {
        "started_at": started_at,
        "finished_at": finished_at,
        "command": command,
        "args": args_dict,
        "environment": environment,
        "rows": [row.__dict__ for row in rows],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "console.txt").write_text("\n".join(console_lines) + "\n")
    write_csv(run_dir / "summary.csv", rows)
    write_markdown(run_dir / "summary.md", rows)

    lines = [
        "# RMSNorm Benchmark Run",
        "",
        f"- started_at: `{started_at}`",
        f"- finished_at: `{finished_at}`",
        f"- command: `{command}`",
        f"- torch: `{environment.get('torch')}`",
        f"- torch_cuda: `{environment.get('torch_cuda')}`",
        f"- device: `{environment.get('device')}`",
    ]
    if "gpu_name" in environment:
        lines.extend(
            [
                f"- gpu_name: `{environment.get('gpu_name')}`",
                f"- gpu_compute_capability: `{environment.get('gpu_compute_capability')}`",
            ]
        )
    lines.extend(["", "## Console", "", "```text", *console_lines, "```", ""])
    (run_dir / "metadata.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation",
        choices=["rmsnorm", "residual_rmsnorm"],
        default="rmsnorm",
        help="Benchmark target operation.",
    )
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU even when CUDA is available.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/rtx4070"))
    parser.add_argument("--run-name", default=None, help="Optional suffix for the recorded run directory.")
    parser.add_argument("--no-record-run", action="store_true", help="Do not create a timestamped run record.")
    parser.add_argument(
        "--implementations",
        default=None,
        help=(
            "Comma-separated implementations. rmsnorm defaults to "
            "pytorch_eager,cuda_naive,triton_rmsnorm,cuda_optimized; "
            "residual_rmsnorm defaults to "
            "pytorch_residual_unfused,cuda_residual_unfused,"
            "cuda_residual_fused,triton_residual_fused."
        ),
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
    if args.implementations is None:
        defaults = (
            DEFAULT_RMSNORM_IMPLEMENTATIONS
            if args.operation == "rmsnorm"
            else DEFAULT_RESIDUAL_RMSNORM_IMPLEMENTATIONS
        )
        implementations = list(defaults)
    else:
        implementations = [item.strip() for item in args.implementations.split(",") if item.strip()]
    environment = collect_environment(device)

    rows = []
    console_lines = []
    for tokens in tokens_list:
        for hidden in hidden_list:
            for implementation in implementations:
                result = benchmark_one(
                    operation=args.operation,
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
                    f"{result.operation} {result.implementation} "
                    f"tokens={tokens} hidden={hidden} "
                    f"latency_us={result.latency_us:.3f} "
                    f"effective_gb_s={result.effective_bandwidth_gb_s:.3f} "
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
