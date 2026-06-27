# Copy/Cast Context Analysis

## Source

- trace: `decode_projection_fusion/results/rtx4070/vllm_gemma_patch_nsys/20260624-230057-vllm-qwen35-2b-request_only_gemma_patch/cuda_gpu_trace.csv`
- window: `35.0s-60.0s`
- target family: `copy / cast`
- context radius: `2`
- kernels in window: `266328`
- target kernels: `57816`
- target total time: `64518748` ns

Note: context is based on `Start (ns)` order. It is a practical adjacency signal, not a strict dependency graph.

## Target Kernel Summary

| target | total target time ns | share | instances | avg ns |
|---|---:|---:|---:|---:|
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 23209479 | 35.973% | 14980 | 1549.4 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 16893689 | 26.184% | 12504 | 1351.1 |
| `[CUDA memcpy Device-to-Device]` | 14902682 | 23.098% | 17452 | 853.9 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 5678596 | 8.801% | 6144 | 924.3 |
| `[CUDA memcpy Host-to-Device]` | 2822419 | 4.375% | 5688 | 496.2 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 562086 | 0.871% | 512 | 1097.8 |
| `[CUDA memcpy Device-to-Host]` | 386116 | 0.598% | 512 | 754.1 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 63681 | 0.099% | 24 | 2653.4 |

## Previous Family

| previous family | total target time ns | share | instances | avg target ns |
|---|---:|---:|---:|---:|
| copy / cast | 20602100 | 31.932% | 18615 | 1106.7 |
| cuBLAS GEMV | 16661383 | 25.824% | 15752 | 1057.7 |
| elementwise | 12210570 | 18.926% | 7688 | 1588.3 |
| other | 11321160 | 17.547% | 11412 | 992.0 |
| GEMM / cuBLAS / CUTLASS | 3491053 | 5.411% | 3752 | 930.5 |
| fill | 146626 | 0.227% | 524 | 279.8 |
| Qwen hybrid / state-space | 85248 | 0.132% | 72 | 1184.0 |
| <none> | 608 | 0.001% | 1 | 608.0 |

## Next Family

| next family | total target time ns | share | instances | avg target ns |
|---|---:|---:|---:|---:|
| copy / cast | 19655989 | 30.466% | 18615 | 1055.9 |
| elementwise | 12934965 | 20.048% | 10748 | 1203.5 |
| norm / reduce | 10945245 | 16.964% | 6216 | 1760.8 |
| GEMM / cuBLAS / CUTLASS | 10915429 | 16.918% | 12800 | 852.8 |
| other | 5833458 | 9.041% | 5268 | 1107.3 |
| RoPE | 3364593 | 5.215% | 3072 | 1095.2 |
| fill | 868621 | 1.346% | 1096 | 792.5 |
| <none> | 448 | 0.001% | 1 | 448.0 |

## Family Triplets

| previous | target | next | total target time ns | share | instances | avg target ns |
|---|---|---|---:|---:|---:|---:|
| elementwise | copy / cast | norm / reduce | 10807514 | 16.751% | 6144 | 1759.0 |
| cuBLAS GEMV | copy / cast | GEMM / cuBLAS / CUTLASS | 10382458 | 16.092% | 12192 | 851.6 |
| copy / cast | copy / cast | elementwise | 9276972 | 14.379% | 6676 | 1389.6 |
| other | copy / cast | copy / cast | 7294450 | 11.306% | 7244 | 1007.0 |
| copy / cast | copy / cast | copy / cast | 7049604 | 10.926% | 7691 | 916.6 |
| cuBLAS GEMV | copy / cast | copy / cast | 4990524 | 7.735% | 3048 | 1637.3 |
| copy / cast | copy / cast | RoPE | 3364593 | 5.215% | 3072 | 1095.2 |
| other | copy / cast | other | 2998184 | 4.647% | 3144 | 953.6 |
| GEMM / cuBLAS / CUTLASS | copy / cast | elementwise | 2622752 | 4.065% | 3048 | 860.5 |
| cuBLAS GEMV | copy / cast | other | 1288401 | 1.997% | 512 | 2516.4 |
| other | copy / cast | elementwise | 597029 | 0.925% | 512 | 1166.1 |
| GEMM / cuBLAS / CUTLASS | copy / cast | other | 596554 | 0.925% | 512 | 1165.1 |
| elementwise | copy / cast | other | 534374 | 0.828% | 508 | 1051.9 |
| copy / cast | copy / cast | fill | 442663 | 0.686% | 584 | 758.0 |
| elementwise | copy / cast | elementwise | 438212 | 0.679% | 512 | 855.9 |

## Initial Read Guide

- `cuBLAS GEMV -> copy/cast -> elementwise` が多ければ、projection 後処理の layout/cast が候補。
- `copy/cast -> copy/cast` が多ければ、連続 copy/cast の削減候補。
- `Qwen hybrid/state-space` 周辺に偏るなら、Qwen3.5 固有 path の調査が必要。
- `sampling / softmax` 周辺に偏るなら、decode 後段の logits/sampling 側を調査する。
