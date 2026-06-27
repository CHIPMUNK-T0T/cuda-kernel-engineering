from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelRow:
    start_ns: int
    duration_ns: int
    stream: str
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
                        stream=(row.get("Strm") or "").strip(),
                        name=(row.get("Name") or "").strip(),
                    )
                )
            except ValueError:
                continue
    rows.sort(key=lambda row: (row.start_ns, row.stream, row.name))
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


def increment_bucket(
    bucket: dict[tuple[str, ...], dict[str, object]],
    key: tuple[str, ...],
    duration_ns: int,
) -> None:
    if key not in bucket:
        bucket[key] = {"instances": 0, "total_target_time_ns": 0}
    bucket[key]["instances"] = int(bucket[key]["instances"]) + 1
    bucket[key]["total_target_time_ns"] = int(bucket[key]["total_target_time_ns"]) + duration_ns


def sorted_bucket_rows(
    bucket: dict[tuple[str, ...], dict[str, object]],
    field_names: tuple[str, ...],
    total_target_ns: int,
    top: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, value in bucket.items():
        instances = int(value["instances"])
        total_ns = int(value["total_target_time_ns"])
        row: dict[str, object] = {
            field: key[index]
            for index, field in enumerate(field_names)
        }
        row.update(
            {
                "instances": instances,
                "total_target_time_ns": total_ns,
                "share_of_target_pct": f"{(total_ns / total_target_ns * 100) if total_target_ns else 0:.3f}",
                "avg_target_ns": f"{(total_ns / instances) if instances else 0:.1f}",
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: int(row["total_target_time_ns"]), reverse=True)
    return rows[:top]


def build_context_rows(
    trace_rows: list[KernelRow],
    target_indices: list[int],
    radius: int,
) -> list[dict[str, object]]:
    context_rows: list[dict[str, object]] = []
    for target_index in target_indices:
        target = trace_rows[target_index]
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            context_index = target_index + offset
            if context_index < 0 or context_index >= len(trace_rows):
                continue
            context = trace_rows[context_index]
            context_rows.append(
                {
                    "target_start_ns": target.start_ns,
                    "target_duration_ns": target.duration_ns,
                    "target_stream": target.stream,
                    "target_family": classify_kernel(target.name),
                    "target_name": short_name(target.name),
                    "offset": offset,
                    "context_start_ns": context.start_ns,
                    "context_duration_ns": context.duration_ns,
                    "context_stream": context.stream,
                    "context_family": classify_kernel(context.name),
                    "context_name": short_name(context.name),
                }
            )
    return context_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--start-s", type=float, default=45.0)
    parser.add_argument("--end-s", type=float, default=70.0)
    parser.add_argument("--target-family", default="copy / cast")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("decode_projection_fusion/results/rtx4070/copy_context"),
    )
    args = parser.parse_args()

    if args.radius < 1:
        raise ValueError("--radius must be >= 1")

    start_ns = int(args.start_s * 1_000_000_000)
    end_ns = int(args.end_s * 1_000_000_000)
    trace_rows = [
        row
        for row in read_trace(args.trace)
        if start_ns <= row.start_ns < end_ns
    ]
    target_indices = [
        index
        for index, row in enumerate(trace_rows)
        if classify_kernel(row.name) == args.target_family
    ]
    total_target_ns = sum(trace_rows[index].duration_ns for index in target_indices)

    target_names: dict[tuple[str, ...], dict[str, object]] = {}
    prev_family: dict[tuple[str, ...], dict[str, object]] = {}
    next_family: dict[tuple[str, ...], dict[str, object]] = {}
    family_triplets: dict[tuple[str, ...], dict[str, object]] = {}
    name_triplets: dict[tuple[str, ...], dict[str, object]] = {}

    for index in target_indices:
        target = trace_rows[index]
        target_family = classify_kernel(target.name)
        target_name = short_name(target.name)
        increment_bucket(target_names, (target_family, target_name), target.duration_ns)

        prev = trace_rows[index - 1] if index > 0 else None
        nxt = trace_rows[index + 1] if index + 1 < len(trace_rows) else None
        prev_family_name = classify_kernel(prev.name) if prev else "<none>"
        next_family_name = classify_kernel(nxt.name) if nxt else "<none>"
        prev_short_name = short_name(prev.name) if prev else "<none>"
        next_short_name = short_name(nxt.name) if nxt else "<none>"

        increment_bucket(prev_family, (prev_family_name,), target.duration_ns)
        increment_bucket(next_family, (next_family_name,), target.duration_ns)
        increment_bucket(
            family_triplets,
            (prev_family_name, target_family, next_family_name),
            target.duration_ns,
        )
        increment_bucket(
            name_triplets,
            (prev_short_name, target_name, next_short_name),
            target.duration_ns,
        )

    target_name_rows = sorted_bucket_rows(
        target_names,
        ("target_family", "target_name"),
        total_target_ns,
        args.top,
    )
    prev_family_rows = sorted_bucket_rows(
        prev_family,
        ("previous_family",),
        total_target_ns,
        args.top,
    )
    next_family_rows = sorted_bucket_rows(
        next_family,
        ("next_family",),
        total_target_ns,
        args.top,
    )
    family_triplet_rows = sorted_bucket_rows(
        family_triplets,
        ("previous_family", "target_family", "next_family"),
        total_target_ns,
        args.top,
    )
    name_triplet_rows = sorted_bucket_rows(
        name_triplets,
        ("previous_name", "target_name", "next_name"),
        total_target_ns,
        args.top,
    )
    context_rows = build_context_rows(trace_rows, target_indices[: args.top], args.radius)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "target_kernel_summary.csv", target_name_rows)
    write_csv(args.out_dir / "previous_family_summary.csv", prev_family_rows)
    write_csv(args.out_dir / "next_family_summary.csv", next_family_rows)
    write_csv(args.out_dir / "family_triplet_summary.csv", family_triplet_rows)
    write_csv(args.out_dir / "name_triplet_summary.csv", name_triplet_rows)
    write_csv(args.out_dir / "sample_context_rows.csv", context_rows)

    markdown = [
        "# Copy/Cast Context Analysis",
        "",
        "## Source",
        "",
        f"- trace: `{args.trace}`",
        f"- window: `{args.start_s:.1f}s-{args.end_s:.1f}s`",
        f"- target family: `{args.target_family}`",
        f"- context radius: `{args.radius}`",
        f"- kernels in window: `{len(trace_rows)}`",
        f"- target kernels: `{len(target_indices)}`",
        f"- target total time: `{total_target_ns}` ns",
        "",
        "Note: context is based on `Start (ns)` order. It is a practical adjacency signal, not a strict dependency graph.",
        "",
        "## Target Kernel Summary",
        "",
        "| target | total target time ns | share | instances | avg ns |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in target_name_rows[:10]:
        markdown.append(
            f"| `{row['target_name']}` | {row['total_target_time_ns']} | {row['share_of_target_pct']}% | {row['instances']} | {row['avg_target_ns']} |"
        )

    markdown.extend(
        [
            "",
            "## Previous Family",
            "",
            "| previous family | total target time ns | share | instances | avg target ns |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in prev_family_rows[:10]:
        markdown.append(
            f"| {row['previous_family']} | {row['total_target_time_ns']} | {row['share_of_target_pct']}% | {row['instances']} | {row['avg_target_ns']} |"
        )

    markdown.extend(
        [
            "",
            "## Next Family",
            "",
            "| next family | total target time ns | share | instances | avg target ns |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in next_family_rows[:10]:
        markdown.append(
            f"| {row['next_family']} | {row['total_target_time_ns']} | {row['share_of_target_pct']}% | {row['instances']} | {row['avg_target_ns']} |"
        )

    markdown.extend(
        [
            "",
            "## Family Triplets",
            "",
            "| previous | target | next | total target time ns | share | instances | avg target ns |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in family_triplet_rows[:15]:
        markdown.append(
            f"| {row['previous_family']} | {row['target_family']} | {row['next_family']} | "
            f"{row['total_target_time_ns']} | {row['share_of_target_pct']}% | {row['instances']} | {row['avg_target_ns']} |"
        )

    markdown.extend(
        [
            "",
            "## Initial Read Guide",
            "",
            "- `cuBLAS GEMV -> copy/cast -> elementwise` が多ければ、projection 後処理の layout/cast が候補。",
            "- `copy/cast -> copy/cast` が多ければ、連続 copy/cast の削減候補。",
            "- `Qwen hybrid/state-space` 周辺に偏るなら、Qwen3.5 固有 path の調査が必要。",
            "- `sampling / softmax` 周辺に偏るなら、decode 後段の logits/sampling 側を調査する。",
            "",
        ]
    )
    (args.out_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")

    print(f"wrote={args.out_dir}")
    print(f"target_family={args.target_family} target_kernels={len(target_indices)} total_target_ns={total_target_ns}")
    for row in family_triplet_rows[:8]:
        print(
            f"triplet={row['previous_family']} -> {row['target_family']} -> {row['next_family']} "
            f"share={row['share_of_target_pct']}% instances={row['instances']}"
        )


if __name__ == "__main__":
    main()
