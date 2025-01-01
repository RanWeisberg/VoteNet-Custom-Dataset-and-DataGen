#include "interpolate.h"
#include "utils.h"

void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx);

void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *features, const int *idx,
                                      const float *weight, float *out);

void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_features);

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows) {
  CHECK_CONTIGUOUS(unknowns);
  CHECK_CONTIGUOUS(knows);
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);

  if (unknowns.is_cuda()) {
    CHECK_CUDA(knows);
  }

  auto B = unknowns.size(0);
  auto N = unknowns.size(1);
  auto M = knows.size(1);

  auto dist2 = torch::zeros({B, N, 3}, unknowns.options());
  auto idx = torch::zeros({B, N, 3}, unknowns.options().dtype(at::kInt));

  if (unknowns.is_cuda()) {
    three_nn_kernel_wrapper(B, N, M, unknowns.data_ptr<float>(),
                            knows.data_ptr<float>(), dist2.data_ptr<float>(),
                            idx.data_ptr<int>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return {dist2, idx};
}

at::Tensor three_interpolate(at::Tensor features, at::Tensor idx, at::Tensor weight) {
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(features);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (features.is_cuda()) {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
  }

  auto B = features.size(0);
  auto C = features.size(1);
  auto N = idx.size(1);
  auto M = features.size(2);

  auto output = torch::zeros({B, C, N}, features.options());

  if (features.is_cuda()) {
    three_interpolate_kernel_wrapper(B, C, M, N, features.data_ptr<float>(),
                                     idx.data_ptr<int>(), weight.data_ptr<float>(),
                                     output.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return output;
}

at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx, at::Tensor weight, const int M) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (grad_out.is_cuda()) {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
  }

  auto B = grad_out.size(0);
  auto C = grad_out.size(1);
  auto N = grad_out.size(2);

  auto grad_features = torch::zeros({B, C, M}, grad_out.options());

  if (grad_out.is_cuda()) {
    three_interpolate_grad_kernel_wrapper(B, C, N, M, grad_out.data_ptr<float>(),
                                          idx.data_ptr<int>(), weight.data_ptr<float>(),
                                          grad_features.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return grad_features;
}
