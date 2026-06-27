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

implementation="${1:-cuda_optimized}"
tokens="${2:-1}"
hidden="${3:-4096}"
iters="${4:-1}"
ncu_set="${NCU_SET:-full}"
ncu_target_processes="${NCU_TARGET_PROCESSES:-all}"
ncu_launch_skip="${NCU_LAUNCH_SKIP:-10}"

case "$implementation" in
  cuda_naive)
    kernel_regex="regex:.*rmsnorm_naive_kernel.*"
    ;;
  cuda_optimized)
    kernel_regex="regex:.*rmsnorm_optimized_kernel.*"
    ;;
  triton_rmsnorm)
    kernel_regex="regex:.*rmsnorm_kernel.*"
    ;;
  cuda_residual_fused)
    kernel_regex="regex:.*fused_residual_rmsnorm_kernel.*"
    ;;
  triton_residual_fused)
    kernel_regex="regex:.*fused_residual_rmsnorm_kernel.*"
    ;;
  *)
    echo "Unknown implementation: $implementation" >&2
    echo "Expected: cuda_naive, cuda_optimized, triton_rmsnorm, cuda_residual_fused, triton_residual_fused" >&2
    exit 1
    ;;
esac

timestamp="$(date +%Y%m%d-%H%M%S)"
run_name="${timestamp}-${implementation}-tokens${tokens}-hidden${hidden}"
out_dir="rmsnorm/results/rtx4070/nsight/${run_name}"
mkdir -p "$out_dir"

command=(
  ncu
  --force-overwrite
  --target-processes "$ncu_target_processes"
  --kernel-name-base demangled
  --kernel-name "$kernel_regex"
  --launch-skip "$ncu_launch_skip"
  --launch-count "$iters"
  --set "$ncu_set"
  --export "$out_dir/profile"
  --log-file "$out_dir/ncu.log"
  .venv/bin/python rmsnorm/benchmarks/profile_rmsnorm.py
  --implementation "$implementation"
  --tokens "$tokens"
  --hidden "$hidden"
  --iters "$iters"
)

{
  echo "# Nsight Compute Run"
  echo
  echo "- implementation: \`$implementation\`"
  echo "- tokens: \`$tokens\`"
  echo "- hidden: \`$hidden\`"
  echo "- iters: \`$iters\`"
  echo "- ncu_set: \`$ncu_set\`"
  echo "- ncu_target_processes: \`$ncu_target_processes\`"
  echo "- ncu_launch_skip: \`$ncu_launch_skip\`"
  echo "- kernel_regex: \`$kernel_regex\`"
  echo "- command: \`${command[*]}\`"
  echo
} > "$out_dir/metadata.md"

"${command[@]}" | tee "$out_dir/console.txt"

echo "record_dir=$out_dir"
