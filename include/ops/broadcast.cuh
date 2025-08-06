#pragma once
#include <core/tensor.cuh>

__global__ void broadcast_add_rowwise_kernel(const float* input, const float* bias, float* output, int rows, int cols);

void broadcast_add_rowwise(const Tensor& input, const Tensor& bias, Tensor& output);

void broadcast_add_rowwise_inplace(Tensor& input_output, const Tensor& bias);