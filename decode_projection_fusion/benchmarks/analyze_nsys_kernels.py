from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KernelRow:
    start_ns: int
    duration_ns: int
    name: str


def parse_int(value: str) -> int:
    return int(float(value.strip()))


def read_trace(path: Path) -> list[KernelRow]:
    rows: list[KernelRow] = []
    with path.open(newline="") as f:
        header: list[str] | None = None
        for raw_line in f:
            if raw_line.startswith("Start (ns),"):
                header = next(csv.reader([raw_line]))
                break
        if header is None:
            raise ValueError(f"missing Nsight trace header: {path}")

        reader = csv.DictReader(f, fieldnames=header)
        for row in reader:
            if not row or not row.get("Start (ns)") or not row.get("Duration (ns)"):
                continue
            try:
                rows.append(
                    KernelRow(
                        start_ns=parse_int(row["Start (ns)"]),
                        duration_ns=parse_int(row["Duration (ns)"]),
                        name=(row.get("Name") or "").strip(),
                    )
                )
            except ValueError:
                continue
    return rows


def classify_kernel(name: str) -> str:
    lower = name.lower()
    lower_without_nocast = lower.replace("nocast", "")

    if "gemvx" in lower or "cublasgemv" in lower:
        return "cuBLAS GEMV"
    if "cutlass" in lower or "gemm" in lower or "cublas" in lower:
        return "GEMM / cuBLAS / CUTLASS"
    if "flash" in lower or "attention" in lower:
        return "attention"
    if "act_and_mul" in lower or "silu" in lower or "swiglu" in lower:
        return "activation / SwiGLU"
    if "rms" in lower or "norm" in lower or "rsqrt" in lower or "mean" in lower or "pow" in lower:
        return "norm / reduce"
    if (
        "direct_copy" in lower
        or "copy_kernel" in lower
        or "memcpy" in lower
        or "cast_kernel" in lower_without_nocast
        or "to_copy" in lower
    ):
        return "copy / cast"
    if "fill" in lower:
        return "fill"
    if "vectorized_elementwise" in lower or "unrolled_elementwise" in lower:
        return "elementwise"
    if "rope" in lower or "rotary" in lower:
        return "RoPE"
    if "cache" in lower or "reshape_and_cache" in lower:
        return "KV cache / layout"
    if "chunk_" in lower or "mamba" in lower or "selective" in lower or "delta_rule" in lower:
        return "Qwen hybrid / state-space"
    if "softmax" in lower or "sampling" in lower or "topk" in lower:
        return "sampling / softmax"
    return "other"


def short_name(name: str, limit: int = 140) -> str:
    compact = " ".join(name.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--start-s", type=float, default=45.0)
    parser.add_argument("--end-s", type=float, default=70.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("decode_projection_fusion/results/rtx4070/request_window"),
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    start_ns = int(args.start_s * 1_000_000_000)
    end_ns = int(args.end_s * 1_000_000_000)

    trace_rows = [
        row
        for row in read_trace(args.trace)
        if start_ns <= row.start_ns < end_ns
    ]
    total_ns = sum(row.duration_ns for row in trace_rows)

    by_family: dict[str, dict[str, object]] = defaultdict(
        lambda: {"total_time_ns": 0, "instances": 0}
    )
    by_name: dict[str, dict[str, object]] = defaultdict(
        lambda: {"total_time_ns": 0, "instances": 0, "family": ""}
    )

    for row in trace_rows:
        family = classify_kernel(row.name)
        by_family[family]["total_time_ns"] = int(by_family[family]["total_time_ns"]) + row.duration_ns
        by_family[family]["instances"] = int(by_family[family]["instances"]) + 1

        by_name[row.name]["total_time_ns"] = int(by_name[row.name]["total_time_ns"]) + row.duration_ns
        by_name[row.name]["instances"] = int(by_name[row.name]["instances"]) + 1
        by_name[row.name]["family"] = family

    family_rows = []
    for family, value in by_family.items():
        family_total = int(value["total_time_ns"])
        instances = int(value["instances"])
        family_rows.append(
            {
                "family": family,
                "total_time_ns": family_total,
                "share_pct": f"{(family_total / total_ns * 100) if total_ns else 0:.3f}",
                "instances": instances,
                "avg_ns": f"{(family_total / instances) if instances else 0:.1f}",
            }
        )
    family_rows.sort(key=lambda row: int(row["total_time_ns"]), reverse=True)

    name_rows = []
    for name, value in by_name.items():
        name_total = int(value["total_time_ns"])
        instances = int(value["instances"])
        name_rows.append(
            {
                "family": value["family"],
                "total_time_ns": name_total,
                "share_pct": f"{(name_total / total_ns * 100) if total_ns else 0:.3f}",
                "instances": instances,
                "avg_ns": f"{(name_total / instances) if instances else 0:.1f}",
                "name": short_name(name),
            }
        )
    name_rows.sort(key=lambda row: int(row["total_time_ns"]), reverse=True)

    candidate_rows = [
        row
        for row in name_rows
        if row["family"] in {"copy / cast", "elementwise", "fill", "KV cache / layout", "other"}
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "family_summary.csv", family_rows)
    write_csv(args.out_dir / "kernel_summary.csv", name_rows[: args.top])
    write_csv(args.out_dir / "candidate_kernels.csv", candidate_rows[: args.top])

    markdown = [
        "# Decode Projection Fusion Request-Window Analysis",
        "",
        "## Source",
        "",
        f"- trace: `{args.trace}`",
        f"- window: `{args.start_s:.1f}s-{args.end_s:.1f}s`",
        f"- kernels in window: `{len(trace_rows)}`",
        f"- total GPU kernel time: `{total_ns}` ns",
        "",
        "## Family Summary",
        "",
        "| family | total time ns | share | instances | avg ns |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in family_rows:
        markdown.append(
            f"| {row['family']} | {row['total_time_ns']} | {row['share_pct']}% | {row['instances']} | {row['avg_ns']} |"
        )

    markdown.extend(
        [
            "",
            "## Candidate Kernels",
            "",
            "GEMV/GEMM/attention/norm を除いた、fusion 候補になりうる上位 kernel。",
            "",
            "| family | total time ns | share | instances | avg ns | name |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in candidate_rows[: args.top]:
        markdown.append(
            f"| {row['family']} | {row['total_time_ns']} | {row['share_pct']}% | {row['instances']} | {row['avg_ns']} | `{row['name']}` |"
        )

    markdown.extend(
        [
            "",
            "## Initial Read",
            "",
            "- cuBLAS GEMV 本体は別テーマ `decode_gemv/` で扱ったため、ここでは主対象にしない。",
            "- 上位 candidate が PyTorch native の copy/cast/elementwise/fill に偏るなら、次は mini reproduction を作る。",
            "- candidate が pre-ready/warmup 由来に見える場合は、window を狭めて再集計する。",
            "",
        ]
    )
    (args.out_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")

    print(f"wrote={args.out_dir}")
    for row in family_rows[:8]:
        print(
            f"family={row['family']} share={row['share_pct']}% "
            f"total_ns={row['total_time_ns']} instances={row['instances']}"
        )


if __name__ == "__main__":
    main()
