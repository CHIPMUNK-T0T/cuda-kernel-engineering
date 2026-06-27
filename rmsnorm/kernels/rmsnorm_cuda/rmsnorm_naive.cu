#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t>
__global__ void rmsnorm_naive_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    int hidden,
    float eps) {
  extern __shared__ float shared_sum[];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = row * hidden;

  float local_sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float value = static_cast<float>(x[row_offset + col]);
    local_sum += value * value;
  }

  shared_sum[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(hidden) + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float normalized = static_cast<float>(x[row_offset + col]) * inv_rms;
    const float scaled = normalized * static_cast<float>(weight[col]);
    y[row_offset + col] = static_cast<scalar_t>(scaled);
  }
}

torch::Tensor rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  auto x_contig = x.contiguous();
  auto weight_contig = weight.contiguous();
  auto y = torch::empty_like(x_contig);

  const int tokens = static_cast<int>(x_contig.size(0));
  const int hidden = static_cast<int>(x_contig.size(1));
  const int threads = 256;
  const int shared_bytes = threads * static_cast<int>(sizeof(float));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_contig.scalar_type(), "rmsnorm_forward_cuda", [&] {
    rmsnorm_naive_kernel<scalar_t><<<tokens, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x_contig.data_ptr<scalar_t>(),
        weight_contig.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        hidden,
        static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}
