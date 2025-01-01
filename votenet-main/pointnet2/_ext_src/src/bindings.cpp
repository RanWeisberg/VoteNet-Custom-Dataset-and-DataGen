// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include "ball_query.h"
#include "group_points.h"
#include "interpolate.h"
#include "sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_points", &gather_points, "Point gathering operation");
  m.def("gather_points_grad", &gather_points_grad, "Gradient of point gathering operation");
  m.def("furthest_point_sampling", &furthest_point_sampling, "Furthest point sampling operation");

  m.def("three_nn", &three_nn, "Three nearest neighbors operation");
  m.def("three_interpolate", &three_interpolate, "Three-dimensional interpolation operation");
  m.def("three_interpolate_grad", &three_interpolate_grad, "Gradient of three-dimensional interpolation operation");

  m.def("ball_query", &ball_query, "Ball query operation");

  m.def("group_points", &group_points, "Point grouping operation");
  m.def("group_points_grad", &group_points_grad, "Gradient of point grouping operation");
}
