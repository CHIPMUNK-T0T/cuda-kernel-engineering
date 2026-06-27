# Nsight Compute Status

Latest attempt:

```text
decode_gemv/results/rtx4070/nsight/20260620-163535-torch_linear-tokens1-in2048-out8192/
decode_gemv/results/rtx4070/nsight/20260620-163537-triton_gemv-tokens1-in2048-out8192/
decode_gemv/results/rtx4070/nsight/20260620-163539-torch_linear-tokens1-in4096-out11008/
decode_gemv/results/rtx4070/nsight/20260620-163541-triton_gemv-tokens1-in4096-out11008/
```

Status:

- Python runner executed.
- Correctness printed `max_abs_error`.
- Nsight Compute metrics were not collected.

Reason:

```text
ERR_NVGPUCTRPERM
```

The user does not currently have permission to access NVIDIA GPU performance counters.

Next action:

Run `decode_gemv/scripts/run_nsight.sh` with sudo and an explicit PATH, or enable profiling permissions at the driver level.

## Partial Success

The sudo rerun succeeded for:

```text
decode_gemv/results/rtx4070/nsight/20260620-163742-torch_linear-tokens1-in2048-out8192/
```

Captured kernel:

```text
cuBLAS GEMV
```

Key metrics:

| metric | value |
|---|---:|
| Duration | 76.93 us |
| Memory Throughput | 90.68% |
| DRAM Throughput | 90.68% |
| L2 Cache Throughput | 62.65% |
| Compute Throughput | 34.34% |
| Achieved Occupancy | 23.94% |
| Registers / Thread | 168 |

Initial read:

- `torch_linear` maps to cuBLAS GEMV for this shape.
- DRAM throughput is already very high.
- Compute throughput is much lower than memory throughput.
- This supports the hypothesis that decode GEMV is memory-throughput limited rather than pure compute limited.

Remaining:

```bash
sudo env "PATH=$PWD/.venv/bin:/usr/local/cuda/bin:/usr/local/cuda-12.8/bin:$PATH" \
  UV_CACHE_DIR=.uv-cache \
  NCU_SET=basic \
  NCU_TARGET_PROCESSES=application-only \
  bash decode_gemv/scripts/run_nsight.sh triton_gemv 1 2048 8192

sudo env "PATH=$PWD/.venv/bin:/usr/local/cuda/bin:/usr/local/cuda-12.8/bin:$PATH" \
  UV_CACHE_DIR=.uv-cache \
  NCU_SET=basic \
  NCU_TARGET_PROCESSES=application-only \
  bash decode_gemv/scripts/run_nsight.sh torch_linear 1 4096 11008

sudo env "PATH=$PWD/.venv/bin:/usr/local/cuda/bin:/usr/local/cuda-12.8/bin:$PATH" \
  UV_CACHE_DIR=.uv-cache \
  NCU_SET=basic \
  NCU_TARGET_PROCESSES=application-only \
  bash decode_gemv/scripts/run_nsight.sh triton_gemv 1 4096 11008
```

## Complete Comparison

The four-run comparison is now available:

```text
decode_gemv/results/rtx4070/nsight_compare.md
```
