# Mini Decoder KV Nsight Systems Summary

## Conditions

- GPU: RTX 4070
- workload: `mini_decoder_kv`
- preset: `qwen_like_2b`
- hidden: `2048`
- layers: `16`
- heads: `16`
- kv heads: `4`
- head dim: `128`
- dtype: `float16`
- iters: `1`

The section timing below is measured by the profiling runner with NVTX ranges and CUDA events. It is useful for component ratios, but the absolute latency should be read together with the normal benchmark summary because profiling ranges add overhead.

## Context 128

| implementation | RMSNorm | QKV | KV cache | attn score | softmax | attn value | Wo |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch unfused | 29.2% | 13.9% | 14.2% | 14.3% | 8.9% | 9.9% | 9.6% |
| CUDA fused | 8.6% | 15.9% | 19.6% | 21.9% | 12.1% | 12.5% | 9.5% |
| Triton fused | 13.8% | 17.8% | 17.8% | 12.6% | 12.8% | 10.9% | 14.2% |

## Context 2048

| implementation | RMSNorm | QKV | KV cache | attn score | softmax | attn value | Wo |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch unfused | 23.1% | 19.6% | 21.3% | 10.0% | 8.3% | 10.3% | 7.5% |
| CUDA fused | 9.0% | 15.4% | 26.2% | 15.0% | 16.2% | 8.6% | 9.5% |
| Triton fused | 10.7% | 19.9% | 28.0% | 9.8% | 7.9% | 7.9% | 15.9% |

## Read

- PyTorch unfused spends about 23-29% of the profiled section time on residual RMSNorm-side work.
- CUDA fused reduces the RMSNorm-side share to about 9% in both short and long context settings.
- After fusion, the visible bottleneck moves to KV cache, attention, QKV, and Wo rather than RMSNorm itself.
- Context 2048 does not make the fused RMSNorm irrelevant. The benchmark still showed CUDA fused at 1.17x and Triton fused at 1.29x versus PyTorch unfused, but the remaining optimization target is now the decoder body around KV cache and attention.

## Records

| context | implementation | record dir |
|---:|---|---|
| 128 | PyTorch unfused | `mini_decoder_kv/results/rtx4070/nsys/20260620-125512-pytorch_unfused-qwen_like_2b-ctx128` |
| 128 | CUDA fused | `mini_decoder_kv/results/rtx4070/nsys/20260620-125527-cuda_residual_fused-qwen_like_2b-ctx128` |
| 128 | Triton fused | `mini_decoder_kv/results/rtx4070/nsys/20260620-125538-triton_residual_fused-qwen_like_2b-ctx128` |
| 2048 | PyTorch unfused | `mini_decoder_kv/results/rtx4070/nsys/20260620-124629-pytorch_unfused-qwen_like_2b-ctx2048` |
| 2048 | CUDA fused | `mini_decoder_kv/results/rtx4070/nsys/20260620-124643-cuda_residual_fused-qwen_like_2b-ctx2048` |
| 2048 | Triton fused | `mini_decoder_kv/results/rtx4070/nsys/20260620-125457-triton_residual_fused-qwen_like_2b-ctx2048` |
