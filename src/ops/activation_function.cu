#include <cuda_runtime.h>
#include <core/tensor.cuh>
#include "ops/activation_function.cuh"
#include "utils/cuda_utils.cuh"

__global__ void step_kernel(const float* input, float* output, int size)
{
    int idx = CUDA_1D_IDX;
    if (idx >= size) return;
    output[idx] = (input[idx] > 0) ? 1 : 0;
}

__global__ void sigmoid_kernel(const float* input, float* output, int size)
{
    int idx = CUDA_1D_IDX;
    if (idx >= size) return;
    output[idx] = 1 / (1 + expf(-input[idx]));
}

void sigmoid(const Tensor& input, Tensor& output)
{
    if (input.size() != output.size())
        throw std::runtime_error("relu: size mismatch");

    int size = input.size();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<grid_size, block_size>>>(
        input.device_ptr(),
        output.device_ptr(),
        size
    );
    cudaDeviceSynchronize();
}

Tensor sigmoid(const Tensor& input)
{
    Tensor output(input.batchs(), input.channels(), input.height(), input.width());
    sigmoid(input, output);
    return output;
}


__global__ void relu_kernel(const float* input, float* output, int size)
{
    int idx = CUDA_1D_IDX;
    if (idx >= size) return;
    output[idx] = (input[idx] > 0) ? input[idx] : 0;
}


void relu(const Tensor& input, Tensor& output)
{
    if (input.size() != output.size())
        throw std::runtime_error("relu: size mismatch");

    int size = input.size();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    relu_kernel<<<grid_size, block_size>>>(
        input.device_ptr(),
        output.device_ptr(),
        size
    );
    cudaDeviceSynchronize();
}

Tensor relu(const Tensor& input)
{
    Tensor output(input.batchs(), input.channels(), input.height(), input.width());
    relu(input, output);
    return output;
}