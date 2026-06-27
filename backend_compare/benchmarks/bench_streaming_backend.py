#!/usr/bin/env python3
"""Streaming benchmark for OpenAI-compatible LLM backends."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = Path("backend_compare/prompts/decode_japanese.txt")


@dataclass(frozen=True)
class StreamBenchResult:
    backend: str
    model: str
    base_url: str
    prompt_path: str
    max_tokens: int
    run_index: int
    total_latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    itl_mean_ms: float | None
    itl_p50_ms: float | None
    itl_p95_ms: float | None
    prompt_tokens: int | None
    generated_tokens: int
    chunk_count: int
    tokens_per_second: float | None
    decode_tokens_per_second: float | None
    finish_reason: str | None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def format_optional(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def parse_max_tokens_list(value: str) -> list[int]:
    tokens = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        tokens.append(int(stripped))
    if not tokens:
        raise argparse.ArgumentTypeError("max token list must not be empty")
    return tokens


def iter_sse_json(response: Any):
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        yield json.loads(payload)


def bench_openai_streaming(
    model: str,
    base_url: str,
    prompt: str,
    prompt_path: Path,
    max_tokens: int,
    run_index: int,
    timeout: float,
) -> StreamBenchResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    token_event_times: list[float] = []
    text_chunks = 0
    prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    finish_reason: str | None = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for event in iter_sse_json(response):
                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    usage_completion_tokens = usage.get(
                        "completion_tokens", usage_completion_tokens
                    )

                choices = event.get("choices") or []
                for choice in choices:
                    finish_reason = choice.get("finish_reason") or finish_reason
                    text = choice.get("text") or ""
                    if text == "":
                        continue
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    token_event_times.append(now)
                    text_chunks += 1
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from streaming request: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to {base_url}: {exc}") from exc

    finished = time.perf_counter()
    total_latency_ms = (finished - started) * 1000.0
    ttft_ms = ((first_token_at - started) * 1000.0) if first_token_at else None

    generated_tokens = usage_completion_tokens or text_chunks
    intervals_ms = [
        (right - left) * 1000.0
        for left, right in zip(token_event_times, token_event_times[1:])
    ]
    itl_mean_ms = statistics.mean(intervals_ms) if intervals_ms else None
    itl_p50_ms = percentile(intervals_ms, 0.50)
    itl_p95_ms = percentile(intervals_ms, 0.95)

    tpot_ms: float | None = None
    decode_tokens_per_second: float | None = None
    if first_token_at and generated_tokens > 1:
        decode_elapsed_s = max(finished - first_token_at, 0.0)
        if decode_elapsed_s > 0:
            tpot_ms = decode_elapsed_s * 1000.0 / (generated_tokens - 1)
            decode_tokens_per_second = (generated_tokens - 1) / decode_elapsed_s

    tokens_per_second = None
    if total_latency_ms > 0 and generated_tokens > 0:
        tokens_per_second = generated_tokens / (total_latency_ms / 1000.0)

    return StreamBenchResult(
        backend="openai_compatible_stream",
        model=model,
        base_url=base_url,
        prompt_path=str(prompt_path),
        max_tokens=max_tokens,
        run_index=run_index,
        total_latency_ms=total_latency_ms,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        itl_mean_ms=itl_mean_ms,
        itl_p50_ms=itl_p50_ms,
        itl_p95_ms=itl_p95_ms,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        chunk_count=text_chunks,
        tokens_per_second=tokens_per_second,
        decode_tokens_per_second=decode_tokens_per_second,
        finish_reason=finish_reason,
    )


def write_csv(path: Path, rows: list[StreamBenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(StreamBenchResult.__dataclass_fields__.keys())
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def aggregate_rows(rows: list[StreamBenchResult]) -> list[dict[str, object]]:
    grouped: dict[int, list[StreamBenchResult]] = {}
    for row in rows:
        grouped.setdefault(row.max_tokens, []).append(row)

    out = []
    for max_tokens, group in sorted(grouped.items()):
        def values(name: str) -> list[float]:
            return [
                float(value)
                for row in group
                if (value := getattr(row, name)) is not None
            ]

        out.append(
            {
                "max_tokens": max_tokens,
                "runs": len(group),
                "mean_total_latency_ms": statistics.mean(values("total_latency_ms")),
                "median_total_latency_ms": statistics.median(values("total_latency_ms")),
                "mean_ttft_ms": statistics.mean(values("ttft_ms")),
                "median_ttft_ms": statistics.median(values("ttft_ms")),
                "mean_tpot_ms": statistics.mean(values("tpot_ms")),
                "median_tpot_ms": statistics.median(values("tpot_ms")),
                "mean_itl_p50_ms": statistics.mean(values("itl_p50_ms")),
                "mean_itl_p95_ms": statistics.mean(values("itl_p95_ms")),
                "mean_tokens_per_second": statistics.mean(values("tokens_per_second")),
                "median_tokens_per_second": statistics.median(values("tokens_per_second")),
                "mean_decode_tokens_per_second": statistics.mean(
                    values("decode_tokens_per_second")
                ),
                "mean_generated_tokens": statistics.mean(values("generated_tokens")),
                "min_generated_tokens": min(values("generated_tokens")),
            }
        )
    return out


def write_aggregate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[StreamBenchResult]) -> None:
    aggregate = aggregate_rows(rows)
    lines = [
        "# Streaming Backend Benchmark Summary",
        "",
        "## Aggregate",
        "",
        "| max tokens | runs | mean TTFT ms | mean TPOT ms | mean ITL p50 ms | mean ITL p95 ms | mean total latency ms | mean tokens/s | mean decode tokens/s | mean generated tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['max_tokens']} | {row['runs']} | "
            f"{format_optional(row['mean_ttft_ms'])} | "
            f"{format_optional(row['mean_tpot_ms'])} | "
            f"{format_optional(row['mean_itl_p50_ms'])} | "
            f"{format_optional(row['mean_itl_p95_ms'])} | "
            f"{format_optional(row['mean_total_latency_ms'])} | "
            f"{format_optional(row['mean_tokens_per_second'])} | "
            f"{format_optional(row['mean_decode_tokens_per_second'])} | "
            f"{format_optional(row['mean_generated_tokens'])} |"
        )

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| max tokens | run | TTFT ms | TPOT ms | ITL p50 ms | ITL p95 ms | total latency ms | generated tokens | chunks | tokens/s | decode tokens/s | finish |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.max_tokens} | {row.run_index} | "
            f"{format_optional(row.ttft_ms)} | "
            f"{format_optional(row.tpot_ms)} | "
            f"{format_optional(row.itl_p50_ms)} | "
            f"{format_optional(row.itl_p95_ms)} | "
            f"{format_optional(row.total_latency_ms)} | "
            f"{row.generated_tokens} | {row.chunk_count} | "
            f"{format_optional(row.tokens_per_second)} | "
            f"{format_optional(row.decode_tokens_per_second)} | "
            f"{row.finish_reason or 'n/a'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_run_dir(out_dir: Path, backend: str, model: str) -> Path:
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    safe_model = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in model)
    run_dir = out_dir / "runs" / f"{timestamp}-{backend}-{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens-list", type=parse_max_tokens_list, default=[128])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("backend_compare/results/rtx4070/stream_requests"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    rows: list[StreamBenchResult] = []
    run_dir = make_run_dir(args.out_dir, "openai_compatible_stream", args.model)

    for max_tokens in args.max_tokens_list:
        for warmup_index in range(1, args.warmup + 1):
            row = bench_openai_streaming(
                model=args.model,
                base_url=args.base_url,
                prompt=prompt,
                prompt_path=args.prompt,
                max_tokens=max_tokens,
                run_index=warmup_index,
                timeout=args.timeout,
            )
            print(
                f"warmup={warmup_index} model={row.model} max_tokens={max_tokens} "
                f"total_latency_ms={row.total_latency_ms:.3f} "
                f"ttft_ms={format_optional(row.ttft_ms)} "
                f"tpot_ms={format_optional(row.tpot_ms)} "
                f"generated_tokens={row.generated_tokens} "
                f"tokens_per_second={format_optional(row.tokens_per_second)}"
            )

        for run_index in range(1, args.runs + 1):
            row = bench_openai_streaming(
                model=args.model,
                base_url=args.base_url,
                prompt=prompt,
                prompt_path=args.prompt,
                max_tokens=max_tokens,
                run_index=run_index,
                timeout=args.timeout,
            )
            rows.append(row)
            print(
                f"model={row.model} run={run_index} max_tokens={max_tokens} "
                f"total_latency_ms={row.total_latency_ms:.3f} "
                f"ttft_ms={format_optional(row.ttft_ms)} "
                f"tpot_ms={format_optional(row.tpot_ms)} "
                f"itl_p50_ms={format_optional(row.itl_p50_ms)} "
                f"itl_p95_ms={format_optional(row.itl_p95_ms)} "
                f"generated_tokens={row.generated_tokens} "
                f"tokens_per_second={format_optional(row.tokens_per_second)} "
                f"decode_tokens_per_second={format_optional(row.decode_tokens_per_second)}"
            )

    metadata = {
        "command": " ".join([sys.executable, *sys.argv]),
        "args": vars(args) | {"prompt": str(args.prompt), "out_dir": str(args.out_dir)},
        "rows": [row.__dict__ for row in rows],
        "aggregate": aggregate_rows(rows),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_csv(run_dir / "summary.csv", rows)
    write_aggregate_csv(run_dir / "aggregate.csv", aggregate_rows(rows))
    write_markdown(run_dir / "summary.md", rows)
    write_csv(args.out_dir / "summary.csv", rows)
    write_aggregate_csv(args.out_dir / "aggregate.csv", aggregate_rows(rows))
    write_markdown(args.out_dir / "summary.md", rows)
    print(f"record_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
