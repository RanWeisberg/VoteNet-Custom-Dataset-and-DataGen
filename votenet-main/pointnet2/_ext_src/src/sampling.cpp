#include "sampling.h"
#include "utils.h"

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);

void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int npoints,
                                            const float *dataset, float *temp,
                                            int *idxs);

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.is_cuda()) {
    CHECK_CUDA(idx);
  }

  auto B = points.size(0);
  auto C = points.size(1);
  auto N = points.size(2);
  auto S = idx.size(1);

  auto output = torch::zeros({B, C, S}, points.options());

  if (points.is_cuda()) {
    gather_points_kernel_wrapper(B, C, N, S, points.data_ptr<float>(), idx.data_ptr<int>(),
                                 output.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int N) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.is_cuda()) {
    CHECK_CUDA(idx);
  }

  auto B = grad_out.size(0);
  auto C = grad_out.size(1);
  auto S = grad_out.size(2);

  auto grad_points = torch::zeros({B, C, N}, grad_out.options());

  if (grad_out.is_cuda()) {
    gather_points_grad_kernel_wrapper(B, C, N, S, grad_out.data_ptr<float>(), idx.data_ptr<int>(),
                                      grad_points.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return grad_points;
}

at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  if (points.is_cuda()) {
    auto B = points.size(0);
    auto N = points.size(1);

    auto idxs = torch::zeros({B, nsamples}, points.options().dtype(at::kInt));
    auto temp = torch::full({B, N}, 1e10, points.options());

    furthest_point_sampling_kernel_wrapper(B, N, nsamples, points.data_ptr<float>(),
                                           temp.data_ptr<float>(), idxs.data_ptr<int>());

    return idxs;
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }
}
