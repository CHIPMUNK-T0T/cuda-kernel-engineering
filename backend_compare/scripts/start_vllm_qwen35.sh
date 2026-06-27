#!/usr/bin/env bash
set -euo pipefail

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

case "$size" in
  2b)
    model="Qwen/Qwen3.5-2B"
    ;;
  4b)
    model="Qwen/Qwen3.5-4B"
    ;;
  *)
    echo "Usage: $0 <2b|4b> [port]" >&2
    exit 2
    ;;
esac

docker run --rm --gpus all \
  --name "vllm-qwen35-${size}" \
  -p "${port}:8000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --ipc=host \
  --shm-size 1g \
  "$image" \
  --model "$model" \
  --dtype auto \
  --max-model-len "$max_model_len" \
  --max-num-batched-tokens "$max_num_batched_tokens" \
  --gpu-memory-utilization "$gpu_memory_utilization"
