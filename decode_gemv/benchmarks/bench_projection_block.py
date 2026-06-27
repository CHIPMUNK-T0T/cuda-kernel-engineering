#!/usr/bin/env python3
"""Projection-block benchmark for decode GEMV.

This benchmark groups several tokens=1 linear projections that appear in a
decoder layer:

- QKV: hidden -> 3 * hidden
- Wo: hidden -> hidden
- MLP gate/up: hidden -> 2 * intermediate
- MLP down: intermediate -> hidden

It is intentionally still a projection-only benchmark. Attention, KV cache, and
activation kernels are excluded so the effect of GEMV implementation choice is
visible.
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


DEFAULT_IMPLEMENTATIONS = ["torch_linear", "triton_tuned", "triton_projection_tuned"]
DEFAULT_TOKENS = [1]
DEFAULT_HIDDEN = [2048, 4096]
DEFAULT_INTERMEDIATE = [8192, 11008]


@dataclass(frozen=True)
class ProjectionWeights:
    qkv: torch.Tensor
    wo: torch.Tensor
    mlp_up: torch.Tensor
    mlp_down: torch.Tensor


@dataclass(frozen=True)
class LinearWeights:
    qkv: torch.Tensor
    wo: torch.Tensor
    mlp_up: torch.Tensor
    mlp_down: torch.Tensor


@dataclass(frozen=True)
class TritonConfig:
    block_k: int
    block_n: int
    triton_latency_us: float


@dataclass(frozen=True)
class BenchResult:
    implementation: str
    tokens: int
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


def make_weight(
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    scale = in_features**0.5
    return torch.randn((in_features, out_features), device=device, dtype=dtype) / scale


def make_inputs(
    tokens: int,
    hidden: int,
    intermediate: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, ProjectionWeights, LinearWeights]:
    torch.manual_seed(0)
    x_hidden = torch.randn((tokens, hidden), device=device, dtype=dtype)
    x_intermediate = torch.randn((tokens, intermediate), device=device, dtype=dtype)
    weights = ProjectionWeights(
        qkv=make_weight(hidden, 3 * hidden, dtype, device),
        wo=make_weight(hidden, hidden, dtype, device),
        mlp_up=make_weight(hidden, 2 * intermediate, dtype, device),
        mlp_down=make_weight(intermediate, hidden, dtype, device),
    )
    linear_weights = LinearWeights(
        qkv=weights.qkv.t().contiguous(),
        wo=weights.wo.t().contiguous(),
        mlp_up=weights.mlp_up.t().contiguous(),
        mlp_down=weights.mlp_down.t().contiguous(),
    )
    return x_hidden, x_intermediate, weights, linear_weights


def run_torch_linear(
    x_hidden: torch.Tensor,
    x_intermediate: torch.Tensor,
    weights: ProjectionWeights,
    linear_weights: LinearWeights,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del weights
    return (
        F.linear(x_hidden, linear_weights.qkv),
        F.linear(x_hidden, linear_weights.wo),
        F.linear(x_hidden, linear_weights.mlp_up),
        F.linear(x_intermediate, linear_weights.mlp_down),
    )


def run_triton_tuned(
    x_hidden: torch.Tensor,
    x_intermediate: torch.Tensor,
    weights: ProjectionWeights,
    linear_weights: LinearWeights,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del linear_weights
    from decode_gemv.kernels.gemv_triton import triton_gemv

    return (
        triton_gemv(x_hidden, weights.qkv, block_k=128, block_n=32),
        triton_gemv(x_hidden, weights.wo, block_k=128, block_n=32),
        triton_gemv(x_hidden, weights.mlp_up, block_k=128, block_n=32),
        triton_gemv(x_intermediate, weights.mlp_down, block_k=128, block_n=32),
    )


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


def load_projection_tuning_configs(
    tuning_dir: Path,
) -> dict[tuple[str, int, int, int], TritonConfig]:
    configs: dict[tuple[str, int, int, int], TritonConfig] = {}
    for summary_path in sorted((tuning_dir / "runs").glob("*/summary.csv")):
        with summary_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row["projection"],
                    int(row["tokens"]),
                    int(row["in_features"]),
                    int(row["out_features"]),
                )
                config = TritonConfig(
                    block_k=int(row["block_k"]),
                    block_n=int(row["block_n"]),
                    triton_latency_us=float(row["triton_latency_us"]),
                )
                current = configs.get(key)
                if current is None or config.triton_latency_us < current.triton_latency_us:
                    configs[key] = config
    return configs


def get_projection_config(
    configs: dict[tuple[str, int, int, int], TritonConfig],
    projection: str,
    tokens: int,
    hidden: int,
    intermediate: int,
) -> TritonConfig:
    in_features, out_features = projection_shape(projection, hidden, intermediate)
    key = (projection, tokens, in_features, out_features)
    try:
        return configs[key]
    except KeyError as exc:
        raise ValueError(
            "Missing Triton tuning config for "
            f"projection={projection} tokens={tokens} in={in_features} out={out_features}. "
            "Run decode_gemv/scripts/tune_projection_types.sh for this shape first."
        ) from exc


def run_triton_projection_tuned(
    x_hidden: torch.Tensor,
    x_intermediate: torch.Tensor,
    weights: ProjectionWeights,
    linear_weights: LinearWeights,
    configs: dict[tuple[str, int, int, int], TritonConfig],
    tokens: int,
    hidden: int,
    intermediate: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del linear_weights
    from decode_gemv.kernels.gemv_triton import triton_gemv

    qkv = get_projection_config(configs, "qkv", tokens, hidden, intermediate)
    wo = get_projection_config(configs, "wo", tokens, hidden, intermediate)
    mlp_up = get_projection_config(configs, "mlp_up", tokens, hidden, intermediate)
    mlp_down = get_projection_config(configs, "mlp_down", tokens, hidden, intermediate)
    return (
        triton_gemv(x_hidden, weights.qkv, block_k=qkv.block_k, block_n=qkv.block_n),
        triton_gemv(x_hidden, weights.wo, block_k=wo.block_k, block_n=wo.block_n),
        triton_gemv(x_hidden, weights.mlp_up, block_k=mlp_up.block_k, block_n=mlp_up.block_n),
        triton_gemv(
            x_intermediate,
            weights.mlp_down,
            block_k=mlp_down.block_k,
            block_n=mlp_down.block_n,
        ),
    )


def select_run(
    implementation: str,
    x_hidden: torch.Tensor,
    x_intermediate: torch.Tensor,
    weights: ProjectionWeights,
    linear_weights: LinearWeights,
    configs: dict[tuple[str, int, int, int], TritonConfig],
    tokens: int,
    hidden: int,
    intermediate: int,
):
    if implementation == "torch_linear":
        return lambda: run_torch_linear(x_hidden, x_intermediate, weights, linear_weights)
    if implementation == "triton_tuned":
        if not x_hidden.is_cuda:
            raise ValueError("triton_tuned requires a CUDA device")
        return lambda: run_triton_tuned(x_hidden, x_intermediate, weights, linear_weights)
    if implementation == "triton_projection_tuned":
        if not x_hidden.is_cuda:
            raise ValueError("triton_projection_tuned requires a CUDA device")
        return lambda: run_triton_projection_tuned(
            x_hidden,
            x_intermediate,
            weights,
            linear_weights,
            configs,
            tokens,
            hidden,
            intermediate,
        )
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


def projection_shapes(hidden: int, intermediate: int) -> list[tuple[int, int]]:
    return [
        (hidden, 3 * hidden),
        (hidden, hidden),
        (hidden, 2 * intermediate),
        (intermediate, hidden),
    ]


def estimate_bytes(tokens: int, hidden: int, intermediate: int, dtype: torch.dtype) -> int:
    dtype_bytes = dtype_size_bytes(dtype)
    total = 0
    for in_features, out_features in projection_shapes(hidden, intermediate):
        total += tokens * in_features * dtype_bytes
        total += in_features * out_features * dtype_bytes
        total += tokens * out_features * dtype_bytes
    return total


def estimate_flops(tokens: int, hidden: int, intermediate: int) -> int:
    total = 0
    for in_features, out_features in projection_shapes(hidden, intermediate):
        total += 2 * tokens * in_features * out_features
    return total


def max_errors(
    output: tuple[torch.Tensor, ...],
    reference: tuple[torch.Tensor, ...],
) -> tuple[float, float]:
    max_abs_error = 0.0
    diff_norm = torch.zeros((), device=reference[0].device, dtype=torch.float32)
    ref_norm = torch.zeros((), device=reference[0].device, dtype=torch.float32)
    for actual, expected in zip(output, reference):
        diff = (actual.float() - expected.float()).abs()
        max_abs_error = max(max_abs_error, diff.max().item())
        diff_norm = diff_norm + torch.linalg.vector_norm(diff).square()
        ref_norm = ref_norm + torch.linalg.vector_norm(expected.float()).square()
    relative_l2_error = (diff_norm.sqrt() / ref_norm.sqrt().clamp_min(1.0e-8)).item()
    return max_abs_error, relative_l2_error


def benchmark_one(
    implementation: str,
    tokens: int,
    hidden: int,
    intermediate: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    runs: int,
    configs: dict[tuple[str, int, int, int], TritonConfig],
) -> BenchResult:
    x_hidden, x_intermediate, weights, linear_weights = make_inputs(
        tokens, hidden, intermediate, dtype, device
    )
    with torch.no_grad():
        reference = run_torch_linear(x_hidden, x_intermediate, weights, linear_weights)
        run = select_run(
            implementation,
            x_hidden,
            x_intermediate,
            weights,
            linear_weights,
            configs,
            tokens,
            hidden,
            intermediate,
        )
        run()

        if device.type == "cuda":
            latency_us = cuda_time_us(run, warmup, runs)
            torch.cuda.synchronize()
        else:
            latency_us = cpu_time_us(run, warmup, runs)

        output = run()
        max_abs_error, relative_l2_error = max_errors(output, reference)

    bandwidth = estimate_bytes(tokens, hidden, intermediate, dtype) / latency_us / 1.0e3
    tflops = estimate_flops(tokens, hidden, intermediate) / latency_us / 1.0e6
    return BenchResult(
        implementation=implementation,
        tokens=tokens,
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
        "# Projection Block Benchmark Summary",
        "",
        "| implementation | tokens | hidden | intermediate | dtype | device | latency us | effective GB/s | effective TFLOP/s | max abs error | relative L2 error |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {implementation} | {tokens} | {hidden} | {intermediate} | {dtype} | {device} | "
            "{latency_us:.3f} | {effective_bandwidth_gb_s:.3f} | {effective_tflops:.3f} | "
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
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cpu", action="store_true", help="Run on CPU for harness debugging.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("decode_gemv/results/rtx4070/projection_block"),
    )
    parser.add_argument(
        "--projection-tuning-dir",
        type=Path,
        default=Path("decode_gemv/results/rtx4070/projection_type_tuning"),
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
    intermediate_values = parse_int_list(args.intermediate)
    implementations = [item.strip() for item in args.implementations.split(",") if item.strip()]
    environment = collect_environment(device)
    configs = load_projection_tuning_configs(args.projection_tuning_dir)

    rows: list[BenchResult] = []
    console_lines: list[str] = []
    for tokens in tokens_values:
        for hidden in hidden_values:
            for intermediate in intermediate_values:
                for implementation in implementations:
                    result = benchmark_one(
                        implementation=implementation,
                        tokens=tokens,
                        hidden=hidden,
                        intermediate=intermediate,
                        dtype=dtype,
                        device=device,
                        warmup=args.warmup,
                        runs=args.runs,
                        configs=configs,
                    )
                    rows.append(result)
                    line = (
                        f"{result.implementation} tokens={tokens} hidden={hidden} "
                        f"intermediate={intermediate} dtype={result.dtype} "
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
