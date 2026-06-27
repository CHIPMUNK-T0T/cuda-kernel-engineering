#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [2b|4b] [port]" >&2
  exit 2
fi

size="${1:-2b}"
port="${2:-8000}"
max_model_len="${MAX_MODEL_LEN:-4096}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-4096}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
image="${VLLM_IMAGE:-vllm/vllm-openai:nightly}"
duration="${NSYS_DURATION:-90}"
delay="${NSYS_DELAY:-0}"
profile_scope="${NSYS_PROFILE_SCOPE:-whole_session}"
nsys_host="${NSYS_HOST:-/opt/nvidia/nsight-systems/2024.6.2}"
nsys_bin="${NSYS_BIN:-/usr/local/bin/nsys}"

case "$size" in
  2b)
    model="Qwen/Qwen3.5-2B"
    ;;
  4b)
    model="Qwen/Qwen3.5-4B"
    ;;
  *)
    echo "Usage: $0 [2b|4b] [port]" >&2
    exit 2
    ;;
esac

if [[ ! -x "$nsys_bin" ]]; then
  echo "Missing host Nsight Systems CLI: $nsys_bin" >&2
  exit 1
fi

if [[ ! -x "${nsys_host}/target-linux-x64/nsys" ]]; then
  echo "Missing Nsight Systems target binary: ${nsys_host}/target-linux-x64/nsys" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
out_suffix="vllm-qwen35-${size}"
if [[ "$profile_scope" != "whole_session" ]]; then
  out_suffix="${out_suffix}-${profile_scope}"
fi
out_dir="backend_compare/results/rtx4070/nsys/${timestamp}-${out_suffix}"
mkdir -p "$out_dir"
abs_out_dir="$(realpath "$out_dir")"

cat > "$out_dir/metadata.md" <<EOF
# vLLM Nsight Systems run

- model: \`$model\`
- port: \`$port\`
- image: \`$image\`
- profile_scope: \`$profile_scope\`
- delay: \`${delay}s\`
- duration: \`${duration}s\`
- max_model_len: \`$max_model_len\`
- max_num_batched_tokens: \`$max_num_batched_tokens\`
- gpu_memory_utilization: \`$gpu_memory_utilization\`
- nsys_host: \`$nsys_host\`
- nsys_bin: \`$nsys_bin\`
EOF

echo "record_dir=$out_dir"
if [[ "$delay" != "0" ]]; then
  echo "Nsight capture starts ${delay}s after container launch and runs for ${duration}s."
  echo "Start the request from another terminal after the server is ready and inside that capture window:"
else
  echo "Start a request from another terminal after the server is ready:"
fi
echo "  bash backend_compare/scripts/request_vllm_qwen35_profile.sh $size http://127.0.0.1:$port"

docker run --rm --gpus all \
  --name "vllm-qwen35-${size}-nsys" \
  -p "${port}:8000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${nsys_host}:/opt/nsight-systems:ro" \
  -v "${abs_out_dir}:/profile-out" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --ipc=host \
  --shm-size 1g \
  --cap-add SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --entrypoint bash \
  "$image" \
  -lc "/opt/nsight-systems/target-linux-x64/nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,cublas,cudnn \
    --sample=none \
    --cpuctxsw=none \
    --delay=${delay} \
    --duration=${duration} \
    --stats=false \
    --output=/profile-out/profile \
    vllm serve '$model' \
      --dtype auto \
      --max-model-len '$max_model_len' \
      --max-num-batched-tokens '$max_num_batched_tokens' \
      --gpu-memory-utilization '$gpu_memory_utilization'" \
  2>&1 | tee "$out_dir/nsys_server.log"

report="$out_dir/profile.nsys-rep"
if [[ -f "$report" ]]; then
  "$nsys_bin" stats --force-export true --report cuda_gpu_kern_sum --format csv "$report" \
    > "$out_dir/cuda_gpu_kern_sum.csv" || true
  "$nsys_bin" stats --force-export true --report cuda_gpu_trace --format csv "$report" \
    > "$out_dir/cuda_gpu_trace.csv" || true
  "$nsys_bin" stats --force-export true --report nvtx_sum --format csv "$report" \
    > "$out_dir/nvtx_sum.csv" || true
fi

echo "record_dir=$out_dir"
