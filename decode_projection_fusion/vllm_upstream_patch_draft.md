# vLLM Upstream Patch Draft

This file is a design draft, not a ready-to-apply patch.

The current experiment uses `decode_projection_fusion/vllm_patch/sitecustomize.py`
to monkey patch `GemmaRMSNorm.forward_native/forward_cuda`.
For OSS, the same idea should be implemented inside vLLM's layer code.

## Target File

```text
vllm/model_executor/layers/layernorm.py
```

## Target Class

```python
@CustomOp.register("gemma_rms_norm")
class GemmaRMSNorm(CustomOp):
    ...
```

## Current Shape

```python
def forward_cuda(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return self.forward_native(x, residual)
```

## Proposed Shape

```python
def _can_use_fused_gemma_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
) -> bool:
    if not x.is_cuda or not weight.is_cuda:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if x.dim() != 2:
        return False
    if weight.dim() != 1 or weight.shape[0] != x.shape[-1]:
        return False
    if residual is not None:
        if not residual.is_cuda:
            return False
        if residual.shape != x.shape or residual.dtype != x.dtype:
            return False
    return True
```

```python
def forward_cuda(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if _can_use_fused_gemma_rmsnorm(x, residual, self.weight):
        try:
            if residual is None:
                return fused_gemma_rmsnorm(
                    x,
                    self.weight,
                    self.variance_epsilon,
                )
            return fused_gemma_add_rmsnorm(
                x,
                residual,
                self.weight,
                self.variance_epsilon,
            )
        except Exception:
            # Upstream version should likely avoid broad exceptions unless vLLM
            # maintainers prefer defensive fallback. This is only a draft.
            pass
    return self.forward_native(x, residual)
```

## Kernel API Needed

The upstream kernel should expose two functions:

```python
fused_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor
```

```python
fused_gemma_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]
```

The second function should match vLLM's residual contract:

```text
return normalized_output, updated_residual
```

## Weight Dtype Requirement

The upstream source inspection showed the native implementation explicitly uses:

```python
weight = self.weight.data.float() + 1.0
```

So the upstream investigation must confirm the actual runtime dtype of
`self.weight` for Qwen3.5 and Gemma-family models.

The fork patch should therefore support:

```text
x: bf16/fp16
weight: fp32
output: x dtype
```

This is more general and closer to the current native semantics than requiring
`x.dtype == weight.dtype`.

## Correctness Tests Needed

Compare fused output against `GemmaRMSNorm.forward_native()` for:

- `residual=None`
- `residual` present
- dtype `bfloat16`
- dtype `float16`
- dtype `float32`
- hidden sizes `2048`, `4096`, `8192`
- tokens `1`, `8`, `128`

Suggested tolerances:

```text
bf16: max_abs_error <= 0.01
fp16: max_abs_error <= 0.01
fp32: max_abs_error <= 1e-4
```

## Performance Tests Needed

Minimum PR-facing performance matrix:

| model | condition | batch | max tokens |
|---|---|---:|---:|
| Qwen3.5-2B | decode-heavy | 1 | 128 |
| Qwen3.5-2B | decode-heavy | 1 | 512 |
| Gemma-family small model | decode-heavy | 1 | 128 |
| Gemma-family small model | decode-heavy | 1 | 512 |

Metrics:

- TTFT
- TPOT
- ITL p50
- decode tokens/s
- correctness smoke output

## PR Message Shape

Possible title:

```text
Add fused Triton path for GemmaRMSNorm
```

Claim:

```text
This patch reduces native decomposition overhead in Gemma-style RMSNorm by
fusing fp32 RMS reduction, offset weight multiply, and output cast into one
kernel for supported CUDA inputs. Unsupported cases keep the existing native
fallback.
```

Avoid claiming:

```text
Speeds up all vLLM workloads.
Speeds up all Qwen/Gemma models.
Replaces GEMV/GEMM bottlenecks.
```
