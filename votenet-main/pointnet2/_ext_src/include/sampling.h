#pragma once
#include <torch/extension.h>
#include <vector>

at::Tensor gather_points(at::Tensor points, at::Tensor idx);
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int N);
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples);
