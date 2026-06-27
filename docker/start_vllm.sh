#!/bin/bash
# Qwen3.5-4B FP8 — RAG/CAG benchmark 用
# モデル: RedHatAI/Qwen3.5-4B-FP8-dynamic
# RTX 4070 12GB では CUDA graph が OOM になるため --enforce-eager 必須。
# CAG で全文コンテキストを載せるため max-model-len を 32768 に設定。

MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

docker run -d \
  --name vllm-server \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 1g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash \
  vllm/vllm-openai:nightly \
  -c "pip install pytest -q && vllm serve RedHatAI/Qwen3.5-4B-FP8-dynamic \
    --dtype auto \
    --attention-backend flash_attn \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --enforce-eager \
    --enable-prefix-caching"

echo "vLLM 起動中 → http://localhost:8000"
echo "ログ確認: docker logs -f vllm-server"
