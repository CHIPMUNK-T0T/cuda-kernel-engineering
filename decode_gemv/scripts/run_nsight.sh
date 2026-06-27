#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export PATH="$PWD/.venv/bin:$PATH"
export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.8/bin:$PATH"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run: UV_CACHE_DIR=.uv-cache uv venv .venv" >&2
  exit 1
fi

if ! command -v ncu >/dev/null 2>&1; then
  echo "Missing Nsight Compute CLI: ncu" >&2
  exit 1
fi

implementation="${1:-torch_linear}"
tokens="${2:-1}"
in_features="${3:-2048}"
out_features="${4:-8192}"
iters="${5:-1}"
warmup="${GEMV_WARMUP:-10}"
dtype="${GEMV_DTYPE:-bfloat16}"
block_k="${GEMV_BLOCK_K:-64}"
block_n="${GEMV_BLOCK_N:-}"
ncu_set="${NCU_SET:-basic}"
ncu_target_processes="${NCU_TARGET_PROCESSES:-application-only}"
ncu_launch_skip="${NCU_LAUNCH_SKIP:-0}"

case "$implementation" in
  torch_linear)
    kernel_regex="${NCU_KERNEL_REGEX:-regex:.*}"
    ;;
  triton_gemv)
    kernel_regex="${NCU_KERNEL_REGEX:-regex:.*gemv_kernel.*}"
    ;;
  *)
    echo "Unknown implementation: $implementation" >&2
    echo "Expected: torch_linear, triton_gemv" >&2
    exit 1
    ;;
esac

timestamp="$(date +%Y%m%d-%H%M%S)"
run_name="${timestamp}-${implementation}-tokens${tokens}-in${in_features}-out${out_features}"
out_dir="decode_gemv/results/rtx4070/nsight/${run_name}"
mkdir -p "$out_dir"

command=(
  ncu
  --force-overwrite
  --target-processes "$ncu_target_processes"
  --profile-from-start off
  --kernel-name-base demangled
  --kernel-name "$kernel_regex"
  --launch-skip "$ncu_launch_skip"
  --launch-count "$iters"
  --set "$ncu_set"
  --export "$out_dir/profile"
  --log-file "$out_dir/ncu.log"
  .venv/bin/python decode_gemv/benchmarks/profile_gemv.py
  --implementation "$implementation"
  --tokens "$tokens"
  --in-features "$in_features"
  --out-features "$out_features"
  --iters "$iters"
  --warmup "$warmup"
  --dtype "$dtype"
  --block-k "$block_k"
)

if [[ -n "$block_n" ]]; then
  command+=(--block-n "$block_n")
fi

{
  echo "# Nsight Compute Run"
  echo
  echo "- implementation: \`$implementation\`"
  echo "- tokens: \`$tokens\`"
  echo "- in_features: \`$in_features\`"
  echo "- out_features: \`$out_features\`"
  echo "- dtype: \`$dtype\`"
  echo "- block_k: \`$block_k\`"
  echo "- block_n: \`${block_n:-auto}\`"
  echo "- iters: \`$iters\`"
  echo "- warmup: \`$warmup\`"
  echo "- ncu_set: \`$ncu_set\`"
  echo "- ncu_target_processes: \`$ncu_target_processes\`"
  echo "- ncu_launch_skip: \`$ncu_launch_skip\`"
  echo "- kernel_regex: \`$kernel_regex\`"
  echo "- command: \`${command[*]}\`"
  echo
} > "$out_dir/metadata.md"

"${command[@]}" | tee "$out_dir/console.txt"

if grep -q "ERR_NVGPUCTRPERM" "$out_dir/ncu.log"; then
  {
    echo
    echo "Nsight Compute could not access NVIDIA GPU performance counters."
    echo "Try running the same command via sudo with PATH preserved, or enable profiling permissions."
    echo "record_dir=$out_dir"
  } >&2
  exit 2
fi

echo "record_dir=$out_dir"
