#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;

__inline__ __device__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__inline__ __device__ float block_reduce_sum(float value) {
  __shared__ float warp_sums[kBlockSize / kWarpSize];
  __shared__ float block_sum;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp_id] = value;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int i = 0; i < kBlockSize / kWarpSize; ++i) {
      sum += warp_sums[i];
    }
    block_sum = sum;
  }
  __syncthreads();
  return block_sum;
}

template <typename scalar_t>
__global__ __launch_bounds__(kBlockSize) void gemma_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = row * hidden;

  float local_sum = 0.0f;
  for (int col = tid; col < hidden; col += kBlockSize) {
    const float value = static_cast<float>(x[row_offset + col]);
    local_sum += value * value;
  }

  const float sum = block_reduce_sum(local_sum);
  const float inv_rms = rsqrtf(sum / static_cast<float>(hidden) + eps);

  for (int col = tid; col < hidden; col += kBlockSize) {
    const float value = static_cast<float>(x[row_offset + col]);
    const float gemma_weight = static_cast<float>(weight[col]) + 1.0f;
    y[row_offset + col] = static_cast<scalar_t>(value * inv_rms * gemma_weight);
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(kBlockSize) void gemma_fused_add_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    scalar_t* __restrict__ residual_out,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = row * hidden;

  float local_sum = 0.0f;
  for (int col = tid; col < hidden; col += kBlockSize) {
    const float z_float =
        static_cast<float>(x[row_offset + col]) + static_cast<float>(residual[row_offset + col]);
    const scalar_t z_scalar = static_cast<scalar_t>(z_float);
    const float z = static_cast<float>(z_scalar);
    local_sum += z * z;
  }

  const float sum = block_reduce_sum(local_sum);
  const float inv_rms = rsqrtf(sum / static_cast<float>(hidden) + eps);

  for (int col = tid; col < hidden; col += kBlockSize) {
    const float z_float =
        static_cast<float>(x[row_offset + col]) + static_cast<float>(residual[row_offset + col]);
    const scalar_t z_scalar = static_cast<scalar_t>(z_float);
    const float z = static_cast<float>(z_scalar);
    const float gemma_weight = static_cast<float>(weight[col]) + 1.0f;
    residual_out[row_offset + col] = z_scalar;
    y[row_offset + col] = static_cast<scalar_t>(z * inv_rms * gemma_weight);
  }
}

}  // namespace

torch::Tensor gemma_rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  auto x_contig = x.contiguous();
  auto weight_contig = weight.contiguous();
  auto y = torch::empty_like(x_contig);

  const int tokens = static_cast<int>(x_contig.size(0));
  const int hidden = static_cast<int>(x_contig.size(1));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      x_contig.scalar_type(),
      "gemma_rmsnorm_forward_cuda",
      [&] {
        gemma_rmsnorm_kernel<scalar_t>
            <<<tokens, kBlockSize, 0, c10::cuda::getCurrentCUDAStream()>>>(
                x_contig.data_ptr<scalar_t>(),
                weight_contig.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                hidden,
                static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}

std::vector<torch::Tensor> gemma_fused_add_rmsnorm_forward_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
  auto x_contig = x.contiguous();
  auto residual_contig = residual.contiguous();
  auto weight_contig = weight.contiguous();
  auto y = torch::empty_like(x_contig);
  auto residual_out = torch::empty_like(x_contig);

  const int tokens = static_cast<int>(x_contig.size(0));
  const int hidden = static_cast<int>(x_contig.size(1));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      x_contig.scalar_type(),
      "gemma_fused_add_rmsnorm_forward_cuda",
      [&] {
        gemma_fused_add_rmsnorm_kernel<scalar_t>
            <<<tokens, kBlockSize, 0, c10::cuda::getCurrentCUDAStream()>>>(
                x_contig.data_ptr<scalar_t>(),
                residual_contig.data_ptr<scalar_t>(),
                weight_contig.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                residual_out.data_ptr<scalar_t>(),
                hidden,
                static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {y, residual_out};
}
