# Nsight Systems for vLLM Qwen3.5 2B

## Purpose

Profile the real vLLM backend and check whether RMSNorm-related kernels are still visible, or whether attention / GEMM / KV cache dominates.

This is not yet a custom RMSNorm speedup measurement. It is a backend visibility check.

## Recommended Flow: Request-Only Window

Use this when the goal is to exclude server startup, model load, and most warmup work from the capture.

Stop the normal vLLM container first if it is using the same GPU or port:

```bash
docker stop vllm-qwen35-2b
```

Terminal 1:

```bash
NSYS_DELAY=90 NSYS_DURATION=180 bash backend_compare/scripts/start_vllm_qwen35_nsys_request_only.sh 2b 8000
```

`NSYS_DELAY` is counted from container launch. `90s` assumes Qwen3.5 2B becomes ready around 50-70s and starts capture after the server is likely ready. `NSYS_DURATION=180` keeps the server alive long enough for warmup and measured requests.

Terminal 2:

```bash
bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8000 3 128 1
```

Start Terminal 2 immediately after Terminal 1. The request script waits for `/v1/models` before running the benchmark, so it avoids sending requests before the server is ready.

After `NSYS_DELAY + NSYS_DURATION` seconds, Terminal 1 exits and writes:

```text
backend_compare/results/rtx4070/nsys/<timestamp>-vllm-qwen35-2b-request_only/
```

## Whole-Session Flow

Stop the normal vLLM container first if it is using the same GPU or port:

```bash
docker stop vllm-qwen35-2b
```

Terminal 1:

```bash
NSYS_DURATION=90 bash backend_compare/scripts/start_vllm_qwen35_nsys.sh 2b 8000
```

Wait until the vLLM server prints that it is ready.

Terminal 2:

```bash
bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8000 3 128 1
```

After `NSYS_DURATION` seconds, Terminal 1 exits and writes:

```text
backend_compare/results/rtx4070/nsys/<timestamp>-vllm-qwen35-2b/
```

Expected files:

- `profile.nsys-rep`
- `nsys_server.log`
- `cuda_gpu_kern_sum.csv`
- `cuda_gpu_trace.csv`
- `nvtx_sum.csv`
- `metadata.md`

## If It Fails

The script mounts the host Nsight Systems installation from:

```text
/opt/nvidia/nsight-systems/2024.6.2
```

Override it if needed:

```bash
NSYS_HOST=/opt/nvidia/nsight-systems/2024.6.2 bash backend_compare/scripts/start_vllm_qwen35_nsys.sh
```

If profiling fails due to permissions, keep the error log and rerun after enabling NVIDIA profiler access on the host.

If the port is already used, either stop the normal vLLM container or use another port:

```bash
NSYS_DURATION=90 bash backend_compare/scripts/start_vllm_qwen35_nsys.sh 2b 8001
bash backend_compare/scripts/request_vllm_qwen35_profile.sh 2b http://127.0.0.1:8001 3 128 1
```

## What To Inspect

After profiling, inspect `cuda_gpu_kern_sum.csv` and group kernels roughly into:

- RMSNorm / layernorm / normalization
- attention
- GEMM / matmul / cublas
- KV cache / copy / reshape
- sampling / logits

The important question is whether normalization still has a visible share in the real backend. If it is already tiny, custom RMSNorm is less likely to move end-to-end tokens/sec. If it remains visible, a backend integration path is worth investigating.
