#include <torch/extension.h>

torch::Tensor gemma_rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, double eps);
std::vector<torch::Tensor> gemma_fused_add_rmsnorm_forward_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps);

void check_inputs(torch::Tensor x, torch::Tensor weight) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "x must be 2D: [tokens, hidden]");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D: [hidden]");
  TORCH_CHECK(x.size(1) == weight.size(0), "weight size must match x hidden size");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight dtype must match");
}

torch::Tensor gemma_rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  check_inputs(x, weight);
  return gemma_rmsnorm_forward_cuda(x, weight, eps);
}

std::vector<torch::Tensor> gemma_fused_add_rmsnorm_forward(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
  check_inputs(x, weight);
  TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
  TORCH_CHECK(residual.dim() == 2, "residual must be 2D: [tokens, hidden]");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape must match x shape");
  TORCH_CHECK(residual.scalar_type() == x.scalar_type(), "residual and x dtype must match");
  return gemma_fused_add_rmsnorm_forward_cuda(x, residual, weight, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gemma_rmsnorm_forward, "Gemma-style RMSNorm forward (CUDA)");
  m.def(
      "forward_fused_add",
      &gemma_fused_add_rmsnorm_forward,
      "Gemma-style fused add RMSNorm forward (CUDA)");
}
