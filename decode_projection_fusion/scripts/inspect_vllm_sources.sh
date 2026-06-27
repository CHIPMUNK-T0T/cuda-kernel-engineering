#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-vllm/vllm-openai:nightly}
OUT_DIR=${OUT_DIR:-decode_projection_fusion/results/rtx4070/source_inspection}

mkdir -p "$OUT_DIR"

docker run --rm --entrypoint python3 "$IMAGE" \
  -c "import vllm; print(vllm.__file__)" \
  > "$OUT_DIR/vllm_path.txt"

docker run --rm --entrypoint bash "$IMAGE" -lc \
  "grep -R -n 'GemmaRMSNorm as Qwen3_5RMSNorm\\|Qwen3_5RMSNorm' \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py" \
  > "$OUT_DIR/qwen35_rmsnorm_refs.txt"

docker run --rm --entrypoint bash "$IMAGE" -lc \
  "sed -n '130,180p' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/layernorm.py" \
  > "$OUT_DIR/gemma_rmsnorm_excerpt.txt"

docker run --rm --entrypoint bash "$IMAGE" -lc \
  "sed -n '1,55p' /usr/local/lib/python3.12/dist-packages/vllm/ir/ops/layernorm.py" \
  > "$OUT_DIR/ir_layernorm_excerpt.txt"

docker run --rm --entrypoint bash "$IMAGE" -lc \
  "sed -n '1,55p' /usr/local/lib/python3.12/dist-packages/vllm/kernels/vllm_c.py" \
  > "$OUT_DIR/vllm_c_rmsnorm_excerpt.txt"

docker run --rm --entrypoint bash "$IMAGE" -lc \
  "sed -n '280,320p' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py" \
  > "$OUT_DIR/qwen3_next_attention_excerpt.txt"

docker run --rm --entrypoint bash "$IMAGE" -lc \
  "sed -n '650,790p' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py" \
  > "$OUT_DIR/qwen_gdn_copy_layout_excerpt.txt"

cat > "$OUT_DIR/summary.md" <<'MD'
# vLLM Source Inspection For Copy/Cast Candidates

## Source

- image: `vllm/vllm-openai:nightly`
- vLLM path: see `vllm_path.txt`

## Strongest Candidate

The strongest source candidate is Qwen3.5 `GemmaRMSNorm`.

Why:

- `qwen3_5.py` aliases `GemmaRMSNorm` as `Qwen3_5RMSNorm`.
- Qwen3.5 decoder layers use `Qwen3_5RMSNorm` for input, post-attention, and final norm.
- `GemmaRMSNorm.forward_cuda()` calls `forward_native()`.
- `GemmaRMSNorm.forward_native()` builds `weight = self.weight.data.float() + 1.0`.
- `vllm_c` RMSNorm only supports the fast C kernel when input and weight dtype match.
- With bf16 activations and fp32 Gemma-style weight, `ir.ops.rms_norm` can fall back to PyTorch-native decomposition.
- The native decomposition contains `to(float32)`, `pow`, `mean`, `rsqrt`, multiply, and `to(orig_dtype)`, matching the trace pattern around `elementwise -> copy/cast -> norm/reduce`.

## Secondary Candidate

Qwen GatedDeltaNet layout cleanup is the next candidate.

Why:

- The GDN path explicitly discusses non-contiguous split views.
- It uses `torch.cat` to force contiguous buffers and reduce several `contiguous()` copies into one.
- For Qwen3.5 non-interleaved path, `b = b.contiguous()` and `a = a.contiguous()` remain.
- This matches the trace shape where copy/cast appears around Qwen hybrid/state-space kernels, although its total share is lower than RMSNorm-adjacent copies.

## Lower Priority Candidate

Full-attention Q/K norm reshape path is worth tracking but not first.

Why:

- `Qwen3NextAttention.forward()` does QKV projection, split, q/k view, q/k norm, RoPE, attention.
- The view operations themselves should not copy if the layout is compatible.
- The stronger observed copy/cast pattern currently points to native GemmaRMSNorm dtype/copy behavior.

## Next Implementation Direction

First mini reproduction should model Gemma-style RMSNorm:

```text
weight_fp32 = weight.float() + 1.0
x_fp32 = x.to(float32)
variance = mean(x_fp32 ** 2)
y = x_fp32 * rsqrt(variance + eps) * weight_fp32
y = y.to(bfloat16)
```

Compare:

- PyTorch native GemmaRMSNorm-style baseline
- Triton/CUDA fused GemmaRMSNorm
- optional fused residual GemmaRMSNorm

This is better aligned with the trace than simple add/mul fusion.
MD

echo "wrote=$OUT_DIR"
