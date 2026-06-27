#!/usr/bin/env python3
"""Nsight Systems runner for mini decoder KV profiling."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import torch

from mini_decoder_kv.benchmarks.bench_decode_kv import (
    DEFAULT_EPS,
    PRESETS,
    ModelConfig,
    decode_once,
    make_inputs,
    residual_rmsnorm,
    resolve_config,
)


PROFILE_SECTIONS = [
    "rmsnorm",
    "qkv_projection",
    "kv_cache",
    "attention_scores",
    "attention_softmax",
    "attention_value",
    "output_projection",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        required=True,
        choices=["pytorch_unfused", "cuda_residual_fused", "triton_residual_fused"],
    )
    parser.add_argument("--preset", default="qwen_like_2b", choices=sorted(PRESETS))
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    return parser.parse_args()


def record_section(
    section: str,
    event_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
):
    class SectionTimer:
        def __enter__(self):
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            torch.cuda.nvtx.range_push(section)
            return self

        def __exit__(self, exc_type, exc, tb):
            torch.cuda.nvtx.range_pop()
            self.end.record()
            event_pairs[section].append((self.start, self.end))
            return False

    return SectionTimer()


def decode_once_profiled(
    implementation: str,
    inputs: dict[str, torch.Tensor],
    config: ModelConfig,
    eps: float,
    event_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> torch.Tensor:
    h = inputs["x"]
    residual = inputs["residual"]
    scale = math.sqrt(config.head_dim)
    repeat = config.num_heads // config.num_kv_heads

    for layer in range(config.layers):
        torch.cuda.nvtx.range_push(f"layer_{layer:02d}")
        with record_section("rmsnorm", event_pairs):
            y = residual_rmsnorm(
                implementation,
                h,
                residual,
                inputs["norm_weights"][layer],
                eps,
            )

        with record_section("qkv_projection", event_pairs):
            q = y @ inputs["wq"][layer]
            k_new = y @ inputs["wk"][layer]
            v_new = y @ inputs["wv"][layer]
            q = q.view(config.num_heads, config.head_dim)
            k_new = k_new.view(1, config.num_kv_heads, config.head_dim)
            v_new = v_new.view(1, config.num_kv_heads, config.head_dim)

        with record_section("kv_cache", event_pairs):
            k = torch.cat((inputs["k_cache"][layer], k_new), dim=0)
            v = torch.cat((inputs["v_cache"][layer], v_new), dim=0)
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        with record_section("attention_scores", event_pairs):
            scores = torch.einsum("hd,shd->hs", q, k) / scale

        with record_section("attention_softmax", event_pairs):
            probs = torch.softmax(scores.float(), dim=-1).to(dtype=y.dtype)

        with record_section("attention_value", event_pairs):
            context = torch.einsum("hs,shd->hd", probs, v).reshape(1, config.hidden)

        with record_section("output_projection", event_pairs):
            out = context @ inputs["wo"][layer]

        residual = h
        h = out
        torch.cuda.nvtx.range_pop()
    return h


def summarize_events(
    event_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> dict[str, float]:
    totals = {}
    for section in PROFILE_SECTIONS:
        pairs = event_pairs.get(section, [])
        totals[section] = sum(start.elapsed_time(end) * 1000.0 for start, end in pairs)
    return totals


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    config = resolve_config(args)
    device = torch.device("cuda")
    dtype = torch.float16
    inputs = make_inputs(config, args.context_len, dtype, device)

    with torch.no_grad():
        reference = decode_once("pytorch_unfused", inputs, config, args.eps)
        for _ in range(args.warmup):
            y = decode_once(args.implementation, inputs, config, args.eps)
        torch.cuda.synchronize()

        event_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("profile_decode_kv")
        for _ in range(args.iters):
            y = decode_once_profiled(args.implementation, inputs, config, args.eps, event_pairs)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

        diff = (y.float() - reference.float()).abs()
        relative_l2_error = (
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-8)
        ).item()
        totals = summarize_events(event_pairs)
        total_section_us = sum(totals.values())

        print(
            f"implementation={args.implementation} preset={args.preset} "
            f"context_len={args.context_len} hidden={config.hidden} layers={config.layers} "
            f"heads={config.num_heads} kv_heads={config.num_kv_heads} iters={args.iters} "
            f"max_abs_error={diff.max().item():.6g} "
            f"relative_l2_error={relative_l2_error:.6g}"
        )
        print("section,total_us,share")
        for section in PROFILE_SECTIONS:
            total_us = totals[section]
            share = total_us / total_section_us if total_section_us > 0 else 0.0
            print(f"{section},{total_us:.3f},{share:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
