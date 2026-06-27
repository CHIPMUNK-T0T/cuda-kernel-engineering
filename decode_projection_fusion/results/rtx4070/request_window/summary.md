# Decode Projection Fusion Request-Window Analysis

## Source

- trace: `backend_compare/results/rtx4070/nsys/20260620-152158-vllm-qwen35-2b-request_only/cuda_gpu_trace.csv`
- window: `45.0s-70.0s`
- kernels in window: `516446`
- total GPU kernel time: `4780790768` ns

## Family Summary

| family | total time ns | share | instances | avg ns |
|---|---:|---:|---:|---:|
| cuBLAS GEMV | 4149161560 | 86.788% | 58424 | 71018.1 |
| copy / cast | 176649452 | 3.695% | 132994 | 1328.3 |
| norm / reduce | 129467248 | 2.708% | 102882 | 1258.4 |
| elementwise | 111974317 | 2.342% | 122729 | 912.4 |
| other | 76070240 | 1.591% | 54788 | 1388.4 |
| GEMM / cuBLAS / CUTLASS | 58065114 | 1.215% | 6389 | 9088.3 |
| Qwen hybrid / state-space | 43972340 | 0.920% | 9420 | 4668.0 |
| activation / SwiGLU | 19479816 | 0.407% | 12285 | 1585.7 |
| fill | 7953803 | 0.166% | 10391 | 765.5 |
| attention | 4801523 | 0.100% | 3072 | 1563.0 |
| RoPE | 3195355 | 0.067% | 3072 | 1040.2 |

## Candidate Kernels

GEMV/GEMM/attention/norm を除いた、fusion 候補になりうる上位 kernel。

| family | total time ns | share | instances | avg ns | name |
|---|---:|---:|---:|---:|---|
| copy / cast | 110787389 | 2.317% | 62935 | 1760.3 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` |
| elementwise | 55478577 | 1.160% | 62446 | 888.4 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(in...` |
| other | 45125682 | 0.944% | 37554 | 1201.6 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` |
| copy / cast | 27904587 | 0.584% | 31223 | 893.7 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` |
| elementwise | 23678151 | 0.495% | 24892 | 951.2 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFun...` |
| elementwise | 22144299 | 0.463% | 24570 | 901.3 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(...` |
| copy / cast | 16667165 | 0.349% | 12491 | 1334.3 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` |
| copy / cast | 14864509 | 0.311% | 17443 | 852.2 | `[CUDA memcpy Device-to-Device]` |
| other | 12712506 | 0.266% | 9144 | 1390.3 | `_causal_conv1d_update_kernel` |
| other | 7881838 | 0.165% | 3583 | 2199.8 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::nati...` |
| fill | 6975662 | 0.146% | 9282 | 751.5 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<c10::BFloat16>, std::array<char *, (unsigned long)1>>(int,...` |
| elementwise | 3729462 | 0.078% | 3072 | 1214.0 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]...` |
| other | 3495382 | 0.073% | 2044 | 1710.1 | `_compute_slot_mapping_kernel` |
| copy / cast | 2913234 | 0.061% | 5663 | 514.4 | `[CUDA memcpy Host-to-Device]` |
| other | 2796348 | 0.058% | 512 | 5461.6 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, (int)4, (i...` |
| elementwise | 2791291 | 0.058% | 3072 | 908.6 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native:...` |
| copy / cast | 1920062 | 0.040% | 1672 | 1148.4 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` |
| elementwise | 1832028 | 0.038% | 2044 | 896.3 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<int>, std::array<char *, (unsigned long)3>>(int, T2, T3)` |
| other | 1296669 | 0.027% | 580 | 2235.6 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::nati...` |
| elementwise | 1280510 | 0.027% | 1533 | 835.3 | `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char *, (unsigned long)3>, (int)4, TrivialOffse...` |
| other | 857566 | 0.018% | 508 | 1688.1 | `void at::native::<unnamed>::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, (int)2, (int)2, (int)-2>(at::cuda::detail::TensorInf...` |
| other | 629374 | 0.013% | 69 | 9121.4 | `merge_16x16_to_64x64_inverse_kernel` |
| copy / cast | 556224 | 0.012% | 511 | 1088.5 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` |
| copy / cast | 556030 | 0.012% | 521 | 1067.2 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` |
| elementwise | 524127 | 0.011% | 511 | 1025.7 | `void at::native::vectorized_elementwise_kernel<(int)2, at::native::CUDAFunctor_add<long>, std::array<char *, (unsigned long)3>>(int, T2, T3)` |
| elementwise | 435744 | 0.009% | 511 | 852.7 | `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_add<int>, std::array<char *, (unsigned long)2>, (int)4, Trivia...` |
| fill | 422878 | 0.009% | 529 | 799.4 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>>(int, T2, T3)` |
| copy / cast | 418749 | 0.009% | 512 | 817.9 | `[CUDA memcpy Device-to-Host]` |
| fill | 389535 | 0.008% | 511 | 762.3 | `void at::native::unrolled_elementwise_kernel<at::native::FillFunctor<int>, std::array<char *, (unsigned long)1>, (int)4, TrivialOffsetCal...` |
| other | 304767 | 0.006% | 69 | 4416.9 | `recompute_w_u_fwd_kernel` |
| other | 291807 | 0.006% | 512 | 569.9 | `[CUDA memset]` |
| other | 263070 | 0.006% | 69 | 3812.6 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::...` |
| other | 258176 | 0.005% | 69 | 3741.7 | `_causal_conv1d_fwd_kernel` |
| fill | 165728 | 0.003% | 69 | 2401.9 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::<unnamed>::masked_fill_kernel(a...` |
| other | 130271 | 0.003% | 69 | 1888.0 | `_fused_post_conv_kernel` |
| elementwise | 70944 | 0.001% | 69 | 1028.2 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase &)::[lambda(bool) (inst...` |
| copy / cast | 61503 | 0.001% | 23 | 2674.0 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIter...` |
| other | 20224 | 0.000% | 3 | 6741.3 | `_zero_kv_blocks_kernel` |
| elementwise | 9184 | 0.000% | 9 | 1020.4 | `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<int>(at::TensorIteratorBase &, at::native::...` |
| other | 6560 | 0.000% | 3 | 2186.7 | `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, long, long, bool)` |

## Initial Read

- cuBLAS GEMV 本体は別テーマ `decode_gemv/` で扱ったため、ここでは主対象にしない。
- 上位 candidate が PyTorch native の copy/cast/elementwise/fill に偏るなら、次は mini reproduction を作る。
- candidate が pre-ready/warmup 由来に見える場合は、window を狭めて再集計する。
