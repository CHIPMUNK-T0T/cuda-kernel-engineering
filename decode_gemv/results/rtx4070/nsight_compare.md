# Nsight Compute Comparison: torch_linear vs triton_gemv

## Runs

```text
decode_gemv/results/rtx4070/nsight/20260620-163742-torch_linear-tokens1-in2048-out8192/
decode_gemv/results/rtx4070/nsight/20260620-164453-triton_gemv-tokens1-in2048-out8192/
decode_gemv/results/rtx4070/nsight/20260620-165732-triton_gemv-tokens1-in2048-out8192/
decode_gemv/results/rtx4070/nsight/20260620-164455-torch_linear-tokens1-in4096-out11008/
decode_gemv/results/rtx4070/nsight/20260620-164457-triton_gemv-tokens1-in4096-out11008/
```

## Summary

| shape | implementation | kernel | duration us | DRAM throughput | L2 throughput | compute throughput | achieved occupancy | registers/thread | grid | block |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1x2048x8192` | `torch_linear` | cuBLAS GEMV | 76.93 | 90.68% | 62.65% | 34.34% | 23.94% | 168 | 2048 | 64 |
| `1x2048x8192` | `triton_gemv` default | `_gemv_kernel` | 81.54 | 84.47% | 41.81% | 22.34% | 23.20% | 40 | 128 | 128 |
| `1x2048x8192` | `triton_gemv` tuned `BK=128 BN=32` | `_gemv_kernel` | 78.08 | 90.78% | 43.71% | 25.79% | 46.29% | 40 | 256 | 128 |
| `1x4096x11008` | `torch_linear` | cuBLAS GEMV | 203.68 | 92.01% | 45.07% | 35.31% | 24.34% | 162 | 2752 | 128 |
| `1x4096x11008` | `triton_gemv` | `_gemv_kernel` | 206.56 | 90.28% | 44.34% | 23.89% | 30.99% | 40 | 172 | 128 |

## Read

- `torch_linear` maps to cuBLAS GEMV in both shapes.
- Both cuBLAS and Triton are DRAM-throughput heavy. This supports the decode GEMV hypothesis: the bottleneck is mainly weight streaming from memory, not pure math throughput.
- Tuning `1x2048x8192` from `BLOCK_N=64` to `BLOCK_N=32` improved Nsight duration from `81.54 us` to `78.08 us`.
- The tuned Triton run raises DRAM throughput from `84.47%` to `90.78%` and achieved occupancy from `23.20%` to `46.29%`.
- cuBLAS still has higher compute throughput, but the tuned Triton kernel reaches similar DRAM throughput for this shape.
- Triton uses far fewer registers per thread and has higher theoretical occupancy, but this does not directly make it faster.
- Triton launches far fewer blocks because one program covers an output tile. This simple mapping reduces scheduling overhead, but it also exposes less fine-grained parallelism than cuBLAS.
- For the large shape, Triton is close to cuBLAS in duration: `206.56 us` vs `203.68 us`.

## Interpretation

The tuned Triton kernel improves the `1x2048x8192` case by making the output tile narrower.
That increases the number of programs, raises achieved occupancy, and gets closer to cuBLAS-level DRAM throughput.

The current Triton kernel is already near cuBLAS for the large output-width case, but it still does not use the GPU as effectively in every metric:

- lower compute throughput
- lower or similar memory throughput
- lower L2 throughput in the `2048x8192` case
- much less block-level parallelism

This means the next optimization target is not simply "increase occupancy".
The next target should be improving memory movement and parallel decomposition:

- tune `BLOCK_N` / `BLOCK_K`
- split reduction across more programs for larger `in_features`
- improve reuse of `x`
- consider CUDA C++ optimized GEMV for finer control
- consider fusion only after GEMV mechanics are understood

## Claim Boundary

At this point, it is reasonable to say:

- vLLM request profiling pointed to decode GEMV as the dominant backend cost.
- Standalone GEMV confirms `tokens=1` is memory-throughput limited.
- A simple Triton kernel can get close to cuBLAS on large output-width decode shapes, but cuBLAS still uses the hardware more effectively.
- Triton tuning found at least one representative decode GEMV shape where the standalone benchmark beats `torch_linear`, and Nsight shows that this comes with higher DRAM throughput and occupancy than the initial Triton mapping.

It is not yet reasonable to say:

- this improves vLLM tokens/sec
- the custom GEMV is generally faster than cuBLAS
- Triton is the final implementation route
