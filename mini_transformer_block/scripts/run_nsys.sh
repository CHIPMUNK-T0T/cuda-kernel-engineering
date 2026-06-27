#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 <implementation> <tokens> <hidden> [iters] [out_features]" >&2
  exit 2
fi

implementation="$1"
tokens="$2"
hidden="$3"
iters="${4:-1}"
out_features="${5:-$hidden}"

case "$implementation" in
  pytorch_unfused|cuda_residual_fused|triton_residual_fused)
    ;;
  *)
    echo "Unknown implementation: $implementation" >&2
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
out_dir="mini_transformer_block/results/rtx4070/nsys/${timestamp}-${implementation}-tokens${tokens}-hidden${hidden}-out${out_features}"
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
  .venv/bin/python mini_transformer_block/benchmarks/profile_block.py
  --implementation "$implementation"
  --tokens "$tokens"
  --hidden "$hidden"
  --out-features "$out_features"
  --iters "$iters"
)

{
  echo "# Nsight Systems mini block run"
  echo
  echo "- implementation: \`$implementation\`"
  echo "- tokens: \`$tokens\`"
  echo "- hidden: \`$hidden\`"
  echo "- out_features: \`$out_features\`"
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
