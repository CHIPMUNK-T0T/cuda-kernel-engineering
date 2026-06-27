# Copy/Cast Context Analysis

## Source

- trace: `backend_compare/results/rtx4070/nsys/20260620-152158-vllm-qwen35-2b-request_only/cuda_gpu_trace.csv`
- window: `45.0s-70.0s`
- target family: `copy / cast`
- context radius: `2`
- kernels in window: `516446`
- target kernels: `132994`
- target total time: `176649452` ns

Note: context is based on `Start (ns)` order. It is a practical adjacency signal, not a strict dependency graph.

## Target Kernel Summary

| target | total target time ns | share | instances | avg ns |
|---|---:|---:|---:|---:|
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` | 113263481 | 64.118% | 65128 | 1739.1 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` | 27904587 | 15.797% | 31223 | 893.7 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 16667165 | 9.435% | 12491 | 1334.3 |
| `[CUDA memcpy Device-to-Device]` | 14864509 | 8.415% | 17443 | 852.2 |
| `[CUDA memcpy Host-to-Device]` | 2913234 | 1.649% | 5663 | 514.4 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` | 556224 | 0.315% | 511 | 1088.5 |
| `[CUDA memcpy Device-to-Host]` | 418749 | 0.237% | 512 | 817.9 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` | 61503 | 0.035% | 23 | 2674.0 |

## Previous Family

| previous family | total target time ns | share | instances | avg target ns |
|---|---:|---:|---:|---:|
| elementwise | 75050563 | 42.486% | 57654 | 1301.7 |
| copy / cast | 47176836 | 26.706% | 31893 | 1479.2 |
| cuBLAS GEMV | 39753516 | 22.504% | 27944 | 1422.6 |
| other | 11341507 | 6.420% | 11588 | 978.7 |
| GEMM / cuBLAS / CUTLASS | 3094839 | 1.752% | 3326 | 930.5 |
| fill | 152127 | 0.086% | 520 | 292.6 |
| Qwen hybrid / state-space | 80064 | 0.045% | 69 | 1160.3 |

## Next Family

| next family | total target time ns | share | instances | avg target ns |
|---|---:|---:|---:|---:|
| elementwise | 62265673 | 35.248% | 35819 | 1738.3 |
| norm / reduce | 51680903 | 29.256% | 31292 | 1651.6 |
| copy / cast | 30829052 | 17.452% | 31893 | 966.6 |
| cuBLAS GEMV | 21643644 | 12.252% | 24384 | 887.6 |
| other | 5782926 | 3.274% | 5258 | 1099.8 |
| RoPE | 3348760 | 1.896% | 3071 | 1090.4 |
| fill | 878622 | 0.497% | 1091 | 805.3 |
| GEMM / cuBLAS / CUTLASS | 219168 | 0.124% | 185 | 1184.7 |
| <none> | 704 | 0.000% | 1 | 704.0 |

## Family Triplets

| previous | target | next | total target time ns | share | instances | avg target ns |
|---|---|---|---:|---:|---:|---:|
| elementwise | copy / cast | norm / reduce | 51553193 | 29.184% | 31223 | 1651.1 |
| copy / cast | copy / cast | elementwise | 35190119 | 19.921% | 19464 | 1808.0 |
| cuBLAS GEMV | copy / cast | elementwise | 23212942 | 13.141% | 12192 | 1903.9 |
| elementwise | copy / cast | cuBLAS GEMV | 21643644 | 12.252% | 24384 | 887.6 |
| cuBLAS GEMV | copy / cast | copy / cast | 15312064 | 8.668% | 15240 | 1004.7 |
| other | copy / cast | copy / cast | 7543444 | 4.270% | 7752 | 973.1 |
| copy / cast | copy / cast | copy / cast | 7111690 | 4.026% | 7679 | 926.1 |
| copy / cast | copy / cast | RoPE | 3348760 | 1.896% | 3071 | 1090.4 |
| other | copy / cast | other | 2987281 | 1.691% | 3140 | 951.4 |
| GEMM / cuBLAS / CUTLASS | copy / cast | elementwise | 2833975 | 1.604% | 3141 | 902.3 |
| cuBLAS GEMV | copy / cast | other | 1228510 | 0.695% | 512 | 2399.4 |
| copy / cast | copy / cast | other | 945471 | 0.535% | 1029 | 918.8 |
| other | copy / cast | elementwise | 591614 | 0.335% | 511 | 1157.8 |
| elementwise | copy / cast | other | 541600 | 0.307% | 508 | 1066.1 |
| copy / cast | copy / cast | fill | 452382 | 0.256% | 580 | 780.0 |

## Initial Read Guide

- `cuBLAS GEMV -> copy/cast -> elementwise` が多ければ、projection 後処理の layout/cast が候補。
- `copy/cast -> copy/cast` が多ければ、連続 copy/cast の削減候補。
- `Qwen hybrid/state-space` 周辺に偏るなら、Qwen3.5 固有 path の調査が必要。
- `sampling / softmax` 周辺に偏るなら、decode 後段の logits/sampling 側を調査する。
