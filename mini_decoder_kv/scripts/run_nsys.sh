#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <implementation> <preset> <context_len> [iters]" >&2
  exit 2
fi

implementation="$1"
preset="$2"
context_len="$3"
iters="${4:-1}"

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
out_dir="mini_decoder_kv/results/rtx4070/nsys/${timestamp}-${implementation}-${preset}-ctx${context_len}"
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
  .venv/bin/python mini_decoder_kv/benchmarks/profile_decode_kv.py
  --implementation "$implementation"
  --preset "$preset"
  --context-len "$context_len"
  --iters "$iters"
)

{
  echo "# Nsight Systems mini decoder KV run"
  echo
  echo "- implementation: \`$implementation\`"
  echo "- preset: \`$preset\`"
  echo "- context_len: \`$context_len\`"
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
  nsys stats --force-export true --report nvtx_sum --format csv "$report" \
    > "$out_dir/nvtx_sum.csv" || true
fi

echo "record_dir=$out_dir"
