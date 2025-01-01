// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// CUDA kernel to find three nearest neighbors
__global__ void three_nn_kernel(int b, int n, int m,
                                const float *__restrict__ unknown,
                                const float *__restrict__ known,
                                float *__restrict__ dist2,
                                int *__restrict__ idx) {
  // Implementation details omitted for brevity
}

void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx) {
  // CUDA stream management and kernel launch
}

// CUDA kernel for three-interpolation
__global__ void three_interpolate_kernel(int b, int c, int m, int n,
                                         const float *__restrict__ points,
                                         const int *__restrict__ idx,
                                         const float *__restrict__ weight,
                                         float *__restrict__ out) {
  // Implementation details omitted for brevity
}

void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out) {
  // CUDA stream management and kernel launch
}

// CUDA kernel for gradient computation in three-interpolation
__global__ void three_interpolate_grad_kernel(
    int b, int c, int n, int m, const float *__restrict__ grad_out,
    const int *__restrict__ idx, const float *__restrict__ weight,
    float *__restrict__ grad_points) {
  // Implementation details omitted for brevity
}

void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points) {
  // CUDA stream management and kernel launch
}

