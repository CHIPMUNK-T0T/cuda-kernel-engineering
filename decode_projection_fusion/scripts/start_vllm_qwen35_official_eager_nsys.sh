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
max_num_seqs="${MAX_NUM_SEQS:-1}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
image="${VLLM_IMAGE:-vllm/vllm-openai:nightly}"
duration="${NSYS_DURATION:-180}"
delay="${NSYS_DELAY:-120}"
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
out_dir="decode_projection_fusion/results/rtx4070/qwen35_eager_nsys/${timestamp}-official-nightly-qwen35-${size}"
mkdir -p "$out_dir"
abs_out_dir="$(realpath "$out_dir")"

cat > "$out_dir/metadata.md" <<EOF
# Qwen3.5 official vLLM eager Nsight Systems run

- variant: \`official nightly\`
- model: \`$model\`
- port: \`$port\`
- image: \`$image\`
- server option: \`--enforce-eager\`
- delay: \`${delay}s\`
- duration: \`${duration}s\`
- max_model_len: \`$max_model_len\`
- max_num_batched_tokens: \`$max_num_batched_tokens\`
- max_num_seqs: \`$max_num_seqs\`
- gpu_memory_utilization: \`$gpu_memory_utilization\`
EOF

echo "record_dir=$out_dir"
echo "After the server is ready, run:"
echo "  bash backend_compare/scripts/request_vllm_qwen35_profile.sh $size http://127.0.0.1:$port 5 128 3"

docker run --rm --gpus all \
  --name "vllm-qwen35-${size}-official-eager-nsys" \
  -p "${port}:8000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${nsys_host}:/opt/nsight-systems:ro" \
  -v "${abs_out_dir}:/profile-out" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
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
      --max-num-seqs '$max_num_seqs' \
      --gpu-memory-utilization '$gpu_memory_utilization' \
      --enforce-eager" \
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
