#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 <implementation> <layers> <hidden> [iters] [projection_weight_mode]" >&2
  echo "projection_weight_mode: distinct or shared. default: distinct" >&2
  exit 2
fi

implementation="$1"
layers="$2"
hidden="$3"
iters="${4:-1}"
projection_weight_mode="${5:-distinct}"
tokens=1

case "$implementation" in
  pytorch_unfused|cuda_residual_fused|triton_residual_fused)
    ;;
  *)
    echo "Unknown implementation: $implementation" >&2
    exit 2
    ;;
esac

case "$projection_weight_mode" in
  distinct)
    projection_weight_args=(--distinct-projection-weights)
    ;;
  shared)
    projection_weight_args=()
    ;;
  *)
    echo "Unknown projection_weight_mode: $projection_weight_mode" >&2
    exit 2
    ;;
esac

export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:/usr/local/cuda-12.8/bin:$PATH"
export TMPDIR="${TMPDIR:-$PWD/.nsys-tmp}"
mkdir -p "$TMPDIR"

if ! command -v nsys >/dev/null 2>&1; then
  echo "Missing Nsight Systems CLI: nsys" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
out_dir="mini_llm_decode/results/rtx4070/nsys/${timestamp}-${implementation}-layers${layers}-hidden${hidden}-${projection_weight_mode}"
mkdir -p "$out_dir"

cmd=(
  nsys profile
  --force-overwrite=true
  --trace=cuda,nvtx,cublas
  --capture-range=cudaProfilerApi
  --capture-range-end=stop
  --sample=none
  --cpuctxsw=none
  --stats=false
  --output "$out_dir/profile"
  .venv/bin/python mini_llm_decode/benchmarks/profile_decode.py
  --implementation "$implementation"
  --tokens "$tokens"
  --hidden "$hidden"
  --layers "$layers"
  --iters "$iters"
  "${projection_weight_args[@]}"
)

{
  echo "# Nsight Systems mini decode run"
  echo
  echo "- implementation: \`$implementation\`"
  echo "- tokens: \`$tokens\`"
  echo "- hidden: \`$hidden\`"
  echo "- layers: \`$layers\`"
  echo "- projection_weight_mode: \`$projection_weight_mode\`"
  echo "- iters: \`$iters\`"
  echo "- command: \`${cmd[*]}\`"
} > "$out_dir/metadata.md"

"${cmd[@]}" 2>&1 | tee "$out_dir/nsys.log"

report="$out_dir/profile.nsys-rep"
if [[ -f "$report" ]]; then
  nsys stats --force-export true --report cuda_gpu_kern_sum --format csv "$report" \
    > "$out_dir/cuda_gpu_kern_sum.csv"
  nsys stats --force-export true --report cuda_gpu_trace --format csv "$report" \
    > "$out_dir/cuda_gpu_trace.csv"
fi

echo "record_dir=$out_dir"
