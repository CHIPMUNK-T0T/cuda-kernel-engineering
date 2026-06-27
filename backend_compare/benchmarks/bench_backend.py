#!/usr/bin/env python3
"""Benchmark Ollama and OpenAI-compatible LLM backends."""

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
class BenchResult:
    backend: str
    model: str
    base_url: str
    prompt_path: str
    max_tokens: int
    run_index: int
    latency_ms: float
    prompt_tokens: int | None
    generated_tokens: int | None
    tokens_per_second: float | None
    backend_reported_eval_tps: float | None
    load_ms: float | None


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc}") from exc


def bench_ollama(
    model: str,
    base_url: str,
    prompt: str,
    prompt_path: Path,
    max_tokens: int,
    num_ctx: int,
    run_index: int,
    timeout: float,
) -> BenchResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    started = time.perf_counter()
    response = post_json(f"{base_url.rstrip('/')}/api/generate", payload, timeout)
    latency_ms = (time.perf_counter() - started) * 1000.0

    prompt_tokens = response.get("prompt_eval_count")
    generated_tokens = response.get("eval_count")
    eval_duration_ns = response.get("eval_duration")
    load_duration_ns = response.get("load_duration")
    backend_tps = None
    if generated_tokens is not None and eval_duration_ns:
        backend_tps = generated_tokens / (eval_duration_ns / 1_000_000_000.0)
    wall_tps = None
    if generated_tokens is not None and latency_ms > 0:
        wall_tps = generated_tokens / (latency_ms / 1000.0)

    return BenchResult(
        backend="ollama",
        model=model,
        base_url=base_url,
        prompt_path=str(prompt_path),
        max_tokens=max_tokens,
        run_index=run_index,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        tokens_per_second=wall_tps,
        backend_reported_eval_tps=backend_tps,
        load_ms=(load_duration_ns / 1_000_000.0) if load_duration_ns else None,
    )


def bench_openai_completions(
    model: str,
    base_url: str,
    prompt: str,
    prompt_path: Path,
    max_tokens: int,
    run_index: int,
    timeout: float,
) -> BenchResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    started = time.perf_counter()
    response = post_json(f"{base_url.rstrip('/')}/v1/completions", payload, timeout)
    latency_ms = (time.perf_counter() - started) * 1000.0

    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    generated_tokens = usage.get("completion_tokens")
    wall_tps = None
    if generated_tokens is not None and latency_ms > 0:
        wall_tps = generated_tokens / (latency_ms / 1000.0)

    return BenchResult(
        backend="openai_compatible",
        model=model,
        base_url=base_url,
        prompt_path=str(prompt_path),
        max_tokens=max_tokens,
        run_index=run_index,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        tokens_per_second=wall_tps,
        backend_reported_eval_tps=None,
        load_ms=None,
    )


def write_csv(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def format_optional(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_markdown(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Backend Benchmark Summary",
        "",
        "| backend | model | run | latency ms | prompt tokens | generated tokens | wall tokens/s | backend eval tokens/s | load ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.backend} | `{row.model}` | {row.run_index} | "
            f"{row.latency_ms:.3f} | {format_optional(row.prompt_tokens)} | "
            f"{format_optional(row.generated_tokens)} | "
            f"{format_optional(row.tokens_per_second)} | "
            f"{format_optional(row.backend_reported_eval_tps)} | "
            f"{format_optional(row.load_ms)} |"
        )

    generated = [row.tokens_per_second for row in rows if row.tokens_per_second is not None]
    if generated:
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- median wall tokens/s: `{statistics.median(generated):.3f}`",
                f"- min wall tokens/s: `{min(generated):.3f}`",
                f"- max wall tokens/s: `{max(generated):.3f}`",
            ]
        )
        if len(generated) >= 2:
            lines.append(f"- mean wall tokens/s: `{statistics.mean(generated):.3f}`")
    path.write_text("\n".join(lines) + "\n")


def make_run_dir(out_dir: Path, backend: str, model: str) -> Path:
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    safe_model = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in model)
    run_dir = out_dir / "runs" / f"{timestamp}-{backend}-{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["ollama", "openai_compatible"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out-dir", type=Path, default=Path("backend_compare/results/rtx4070"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = args.prompt.read_text().strip()
    rows = []
    run_dir = make_run_dir(args.out_dir, args.backend, args.model)

    for warmup_index in range(1, args.warmup + 1):
        if args.backend == "ollama":
            row = bench_ollama(
                model=args.model,
                base_url=args.base_url,
                prompt=prompt,
                prompt_path=args.prompt,
                max_tokens=args.max_tokens,
                num_ctx=args.num_ctx,
                run_index=warmup_index,
                timeout=args.timeout,
            )
        else:
            row = bench_openai_completions(
                model=args.model,
                base_url=args.base_url,
                prompt=prompt,
                prompt_path=args.prompt,
                max_tokens=args.max_tokens,
                run_index=warmup_index,
                timeout=args.timeout,
            )
        print(
            f"warmup={warmup_index} backend={row.backend} model={row.model} "
            f"latency_ms={row.latency_ms:.3f} "
            f"generated_tokens={format_optional(row.generated_tokens)} "
            f"tokens_per_second={format_optional(row.tokens_per_second)}"
        )

    for run_index in range(1, args.runs + 1):
        if args.backend == "ollama":
            row = bench_ollama(
                model=args.model,
                base_url=args.base_url,
                prompt=prompt,
                prompt_path=args.prompt,
                max_tokens=args.max_tokens,
                num_ctx=args.num_ctx,
                run_index=run_index,
                timeout=args.timeout,
            )
        else:
            row = bench_openai_completions(
                model=args.model,
                base_url=args.base_url,
                prompt=prompt,
                prompt_path=args.prompt,
                max_tokens=args.max_tokens,
                run_index=run_index,
                timeout=args.timeout,
            )
        rows.append(row)
        print(
            f"backend={row.backend} model={row.model} run={row.run_index} "
            f"latency_ms={row.latency_ms:.3f} "
            f"prompt_tokens={format_optional(row.prompt_tokens)} "
            f"generated_tokens={format_optional(row.generated_tokens)} "
            f"tokens_per_second={format_optional(row.tokens_per_second)} "
            f"backend_eval_tokens_per_second={format_optional(row.backend_reported_eval_tps)}"
        )

    metadata = {
        "command": " ".join([sys.executable, *sys.argv]),
        "args": vars(args) | {"prompt": str(args.prompt), "out_dir": str(args.out_dir)},
        "warmup_runs": args.warmup,
        "rows": [row.__dict__ for row in rows],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_csv(run_dir / "summary.csv", rows)
    write_markdown(run_dir / "summary.md", rows)
    write_csv(args.out_dir / "summary.csv", rows)
    write_markdown(args.out_dir / "summary.md", rows)
    print(f"record_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
