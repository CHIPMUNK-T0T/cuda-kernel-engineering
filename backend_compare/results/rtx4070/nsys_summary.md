# vLLM Qwen3.5 2B Nsight Systems Summary

## Conditions

- backend: vLLM OpenAI-compatible server
- model: `Qwen/Qwen3.5-2B`
- GPU: RTX 4070
- Nsight duration: `300s`
- request during profile: warmup `1`, measured runs `3`, max tokens `128`
- record: `backend_compare/results/rtx4070/nsys/20260620-135208-vllm-qwen35-2b`
- request record: `backend_compare/results/rtx4070/profile_requests/runs/20260620-135423-openai_compatible-Qwen-Qwen3-5-2B`

## Request During Profiling

| run | latency ms | generated tokens | wall tokens/s |
|---:|---:|---:|---:|
| 1 | 1735.971 | 128 | 73.734 |
| 2 | 1760.221 | 128 | 72.718 |
| 3 | 1731.603 | 128 | 73.920 |

| metric | value |
|---|---:|
| median wall tokens/s | 73.734 |
| mean wall tokens/s | 73.457 |
| min wall tokens/s | 72.718 |
| max wall tokens/s | 73.920 |

The non-profiled steady-state baseline was about `92 tok/s`, so this request run includes profiler overhead.

## Kernel Summary

This profile includes server startup, model load, vLLM warmup, and the benchmark requests. It is not a clean request-only profile yet.

| category | total time ns | share | instances |
|---|---:|---:|---:|
| elementwise / copy / misc | 10,130,400,251 | 44.536% | 315,568 |
| FlashAttention | 7,070,155,576 | 31.082% | 5,904 |
| GEMM / GEMV | 5,020,122,203 | 22.070% | 59,299 |
| Qwen hybrid / Mamba-like kernels | 220,839,151 | 0.971% | 37,316 |
| norm-related | 177,438,008 | 0.780% | 103,883 |
| sampling / softmax | 43,494,047 | 0.191% | 10 |
| RoPE | 40,960,986 | 0.180% | 3,108 |
| other | 35,016,238 | 0.154% | 3,340 |
| KV cache | 8,324,598 | 0.037% | 5,130 |

## Norm-Related Kernels Observed

Representative rows:

| kernel family | total time ns | instances | note |
|---|---:|---:|---|
| PyTorch mean reduce | 65,229,898 | 31,354 | RMSNorm-like reduction path |
| PyTorch pow | 35,824,017 | 31,354 | RMSNorm-like square path |
| PyTorch rsqrt | 30,159,103 | 31,354 | RMSNorm-like inverse sqrt path |
| PyTorch layer norm | 28,356,108 | 49 | layer norm path |
| Triton `layer_norm_fwd_kernel` | 11,416,056 | 9,252 | vLLM / Triton norm kernel |

The vLLM log also reports:

```text
IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
```

So vLLM does have RMSNorm / fused add RMSNorm kernel paths, but in this whole-session profile the norm-related GPU time is much smaller than attention and GEMM/GEMV.

## Read

- In the real vLLM backend, attention and GEMM/GEMV dominate the captured GPU kernel time.
- Norm-related kernels are visible, but only about `0.78%` of this whole-session profile.
- This does not prove custom RMSNorm cannot help, because the profile includes startup and warmup. It does suggest that end-to-end vLLM tokens/sec is more likely to be dominated by attention, GEMV/GEMM, and framework/runtime overhead.
- A cleaner request-only profile would require a narrower capture window after the server is ready.

## Request-Only Profile

Conditions:

- Nsight delay: `90s`
- Nsight duration: `180s`
- request during profile: warmup `1`, measured runs `3`, max tokens `128`
- record: `backend_compare/results/rtx4070/nsys/20260620-152158-vllm-qwen35-2b-request_only`
- request record: `backend_compare/results/rtx4070/profile_requests/runs/20260620-152413-openai_compatible-Qwen-Qwen3-5-2B`

Request result:

| run | latency ms | generated tokens | wall tokens/s |
|---:|---:|---:|---:|
| 1 | 1749.611 | 128 | 73.159 |
| 2 | 1778.298 | 128 | 71.979 |
| 3 | 1784.253 | 128 | 71.739 |

| metric | value |
|---|---:|
| median wall tokens/s | 71.979 |
| mean wall tokens/s | 72.292 |
| min wall tokens/s | 71.739 |
| max wall tokens/s | 73.159 |

Kernel summary:

Note: this direct `cuda_gpu_kern_sum.csv` aggregate still includes some pre-ready / warmup work because capture started before the server was fully ready.

| category | total time ns | share | instances |
|---|---:|---:|---:|
| GEMM / GEMV | 4,381,179,044 | 49.943% | 59,198 |
| elementwise / copy / misc | 4,011,510,193 | 45.729% | 304,502 |
| norm-related | 147,114,635 | 1.677% | 103,829 |
| Qwen hybrid / Mamba-like kernels | 146,415,042 | 1.669% | 26,601 |
| sampling / softmax | 43,588,478 | 0.497% | 10 |
| FlashAttention | 29,064,742 | 0.331% | 5,880 |
| KV cache | 8,348,040 | 0.095% | 5,130 |
| RoPE | 3,296,315 | 0.038% | 3,084 |
| other | 1,913,690 | 0.022% | 726 |

Read:

- Request-only でも norm-related kernel は見えるが、share は `1.677%` に留まる。
- この条件では `GEMM/GEMV` と `elementwise / copy / misc` が支配的。
- whole-session より FlashAttention share が小さいのは、今回の capture が decode request 中心で、prefill/attention より GEMV/elementwise が前面に出ているためと考えられる。
- vLLM end-to-end で「RMSNorm custom kernel だけで大きく速くなる」とはまだ言えない。mini decoder では効果が見えたが、production backend では他の kernel と runtime overhead に薄まる。

## Request Window Re-aggregation

The trace was re-aggregated from `cuda_gpu_trace.csv` around the measured request window, excluding the pre-ready / warmup-heavy segment.

Details: `backend_compare/results/rtx4070/vllm_request_window_breakdown.md`

Approximate request window: `45s-70s` from capture start.

| family | share |
|---|---:|
| cuBLAS GEMV | 86.788% |
| PyTorch copy / cast | 3.314% |
| PyTorch elementwise math | 3.284% |
| norm-related | 1.429% |
| PyTorch RMSNorm reduce | 1.279% |
| GEMM / CUTLASS / cuBLAS | 1.213% |
| Qwen hybrid / Mamba-like | 1.185% |
| vLLM SwiGLU `act_and_mul_kernel` | 0.407% |

Read:

- The largest true request bottleneck is decode GEMV / small-batch linear work, not SwiGLU.
- `SwiGLU` is not a strong vLLM-driven next target because vLLM already uses `vllm::act_and_mul_kernel` and its request-window share is small.
- The realistic next choices are either a hard but meaningful `decode_gemv/` theme, or a more implementable `elementwise_fusion/` theme targeting residual PyTorch native copy/cast/math kernels.
