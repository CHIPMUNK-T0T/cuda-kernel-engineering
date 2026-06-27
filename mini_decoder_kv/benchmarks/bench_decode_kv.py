#!/usr/bin/env python3
"""Mini decoder benchmark with attention and KV cache.

This is a synthetic Qwen-like decode workload. QKV projection, attention, KV
cache reads, and output projection stay in PyTorch; only residual RMSNorm is
swapped between PyTorch, CUDA, and Triton implementations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
PRESETS = {
    "qwen_like_2b": {
        "hidden": 2048,
        "layers": 16,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 128,
    },
    "qwen_like_4b": {
        "hidden": 2560,
        "layers": 24,
        "num_heads": 20,
        "num_kv_heads": 4,
        "head_dim": 128,
    },
}


@dataclass(frozen=True)
class ModelConfig:
    preset: str
    hidden: int
    layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class BenchResult:
    implementation: str
    preset: str
    context_len: int
    hidden: int
    layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    device: str
    latency_us: float
    tokens_per_second: float
    max_abs_error: float
    mean_abs_error: float
    relative_l2_error: float
    max_rel_error: float


def resolve_config(args: argparse.Namespace) -> ModelConfig:
    if args.preset not in PRESETS:
        raise ValueError(f"Unknown preset: {args.preset}")
    preset = PRESETS[args.preset]
    hidden = args.hidden if args.hidden is not None else preset["hidden"]
    layers = args.layers if args.layers is not None else preset["layers"]
    num_heads = args.num_heads if args.num_heads is not None else preset["num_heads"]
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else preset["num_kv_heads"]
    head_dim = args.head_dim if args.head_dim is not None else preset["head_dim"]
    if hidden != num_heads * head_dim:
        raise ValueError("hidden must equal num_heads * head_dim")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    return ModelConfig(
        preset=args.preset,
        hidden=hidden,
        layers=layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def residual_rmsnorm(
    implementation: str,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if implementation == "pytorch_unfused":
        return residual_rmsnorm_pytorch(x, residual, weight, eps)
    if implementation == "cuda_residual_fused":
        return residual_rmsnorm_cuda_fused(x, residual, weight, eps)
    if implementation == "triton_residual_fused":
        return residual_rmsnorm_triton_fused(x, residual, weight, eps)
    raise ValueError(f"Unknown implementation: {implementation}")


def make_inputs(
    config: ModelConfig,
    context_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    hidden = config.hidden
    kv_hidden = config.num_kv_heads * config.head_dim
    scale = hidden**0.5
    return {
        "x": torch.randn((1, hidden), device=device, dtype=dtype),
        "residual": torch.randn((1, hidden), device=device, dtype=dtype),
        "norm_weights": torch.randn((config.layers, hidden), device=device, dtype=dtype),
        "wq": torch.randn((config.layers, hidden, hidden), device=device, dtype=dtype) / scale,
        "wk": torch.randn((config.layers, hidden, kv_hidden), device=device, dtype=dtype) / scale,
        "wv": torch.randn((config.layers, hidden, kv_hidden), device=device, dtype=dtype) / scale,
        "wo": torch.randn((config.layers, hidden, hidden), device=device, dtype=dtype) / scale,
        "k_cache": torch.randn(
            (config.layers, context_len, config.num_kv_heads, config.head_dim),
            device=device,
            dtype=dtype,
        ),
        "v_cache": torch.randn(
            (config.layers, context_len, config.num_kv_heads, config.head_dim),
            device=device,
            dtype=dtype,
        ),
    }


def attention_one_layer(
    y: torch.Tensor,
    layer: int,
    inputs: dict[str, torch.Tensor],
    config: ModelConfig,
) -> torch.Tensor:
    q = y @ inputs["wq"][layer]
    k_new = y @ inputs["wk"][layer]
    v_new = y @ inputs["wv"][layer]

    q = q.view(config.num_heads, config.head_dim)
    k_new = k_new.view(1, config.num_kv_heads, config.head_dim)
    v_new = v_new.view(1, config.num_kv_heads, config.head_dim)

    k = torch.cat((inputs["k_cache"][layer], k_new), dim=0)
    v = torch.cat((inputs["v_cache"][layer], v_new), dim=0)

    repeat = config.num_heads // config.num_kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)

    scores = torch.einsum("hd,shd->hs", q, k) / math.sqrt(config.head_dim)
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=y.dtype)
    context = torch.einsum("hs,shd->hd", probs, v).reshape(1, config.hidden)
    return context @ inputs["wo"][layer]


def decode_once(
    implementation: str,
    inputs: dict[str, torch.Tensor],
    config: ModelConfig,
    eps: float,
) -> torch.Tensor:
    h = inputs["x"]
    residual = inputs["residual"]
    for layer in range(config.layers):
        y = residual_rmsnorm(
            implementation,
            h,
            residual,
            inputs["norm_weights"][layer],
            eps,
        )
        out = attention_one_layer(y, layer, inputs, config)
        residual = h
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


def benchmark_one(
    implementation: str,
    config: ModelConfig,
    context_len: int,
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
    warmup: int,
    runs: int,
) -> BenchResult:
    if implementation in {"cuda_residual_fused", "triton_residual_fused"} and device.type != "cuda":
        raise ValueError(f"{implementation} requires a CUDA device")

    inputs = make_inputs(config, context_len, dtype, device)
    with torch.no_grad():
        reference = decode_once("pytorch_unfused", inputs, config, eps)
        run = lambda: decode_once(implementation, inputs, config, eps)
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
        preset=config.preset,
        context_len=context_len,
        hidden=config.hidden,
        layers=config.layers,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
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
        "# Mini Decoder KV Benchmark Summary",
        "",
        "| implementation | preset | context | hidden | layers | heads | kv heads | head dim | dtype | device | latency us | tokens/s | max abs error | relative L2 error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {implementation} | {preset} | {context_len} | {hidden} | {layers} | "
            "{num_heads} | {num_kv_heads} | {head_dim} | {dtype} | {device} | "
            "{latency_us:.3f} | {tokens_per_second:.3f} | {max_abs_error:.6g} | "
            "{relative_l2_error:.6g} |".format(**row.__dict__)
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
    parser.add_argument("--preset", default="qwen_like_2b", choices=sorted(PRESETS))
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU for harness debugging.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mini_decoder_kv/results/rtx4070"),
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
    config = resolve_config(args)
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
            config=config,
            context_len=args.context_len,
            dtype=dtype,
            device=device,
            eps=args.eps,
            warmup=args.warmup,
            runs=args.runs,
        )
        rows.append(result)
        line = (
            f"{result.implementation} preset={result.preset} context_len={result.context_len} "
            f"hidden={result.hidden} layers={result.layers} heads={result.num_heads} "
            f"kv_heads={result.num_kv_heads} latency_us={result.latency_us:.3f} "
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
