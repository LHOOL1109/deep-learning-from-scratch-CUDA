#include <cuda_runtime.h>
#include "../../include/ops/activation_function.cuh"
#include "../../include/utils/cuda_utils.cuh"

__global__ void step_function(const float* input, float* output, int size)
{
    int idx = CUDA_1D_IDX;
    if (idx >= size) return;
    output[idx] = (input[idx] > 0) ? 1 : 0;
}

__global__ void sigmoid_function(const float* input, float* output, int size)
{
    int idx = CUDA_1D_IDX;
    if (idx >= size) return;
    output[idx] = 1 / (1 + expf(-input[idx]));
}

__global__ void relu_function(const float* input, float* output, int size)
{
    int idx = CUDA_1D_IDX;
    if (idx >= size) return;
    output[idx] = (input[idx] > 0) ? input[idx] : 0;
}