/*
activation_function.cuh
*/
#pragma once

__global__ void step_kernel(const float* input, float* output, int size);

__global__ void sigmoid_kernel(const float* input, float* output, int size);
void sigmoid(const Tensor& input, Tensor& output);
Tensor sigmoid(const Tensor& input);

__global__ void relu_kernel(const float* input, float* output, int size);
void relu(const Tensor& input, Tensor& output);
Tensor relu(const Tensor& input);