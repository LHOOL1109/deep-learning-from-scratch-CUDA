/*
activation_function.cuh
*/
#pragma once

__global__ void step_function(const float* input, float* output, int size);
__global__ void sigmoid_function(const float* input, float* output, int size);
__global__ void relu_function(const float* input, float* output, int size);
