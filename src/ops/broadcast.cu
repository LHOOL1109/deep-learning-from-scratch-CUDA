#include <core/tensor.cuh>
#include <stdexcept>

__global__ void broadcast_add_rowwise_kernel(const float* input, const float* bias, float* output, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int col = idx % cols;
    output[idx] = input[idx] + bias[col];
}

void broadcast_add_rowwise(const Tensor& input, const Tensor& bias, Tensor& output)
{
    int rows = input.height();
    int cols = input.width();

    int total = rows * cols;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    broadcast_add_rowwise_kernel<<<grid_size, block_size>>>
    (
        input.device_ptr(),
        bias.device_ptr(),
        output.device_ptr(),
        rows,
        cols
    );
}

void broadcast_add_rowwise_inplace(Tensor& input_output, const Tensor& bias)
{
    int rows = input_output.height();
    int cols = input_output.width();
    int total = rows * cols;


    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    broadcast_add_rowwise_kernel<<<grid_size, block_size>>>(
        input_output.device_ptr(),
        bias.device_ptr(),
        input_output.device_ptr(),
        rows,
        cols
    );
    cudaDeviceSynchronize();
}