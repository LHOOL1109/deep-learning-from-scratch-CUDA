#include <cuda_runtime.h>
#include <iostream>

// AND gate
__global__ void and_gate(const float* x1, const float* x2, int* output, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    float w1 = 0.5f, w2 = 0.5f, b = -0.7f;
    float tmp = x1[idx] * w1 + x2[idx] * w2 + b;
    output[idx] = (tmp <= 0) ? 0 : 1;
}

__global__ void nand_gate(const float* x1, const float* x2, int* output, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >=n) return;
    float w1 = -0.5f, w2 = -0.5f, b = 0.7f;
    float tmp = x1[idx] * w1 + x2[idx] * w2 + b;
    output[idx] = (tmp <= 0) ? 0 : 1;
}

__global__ void or_gate(const float* x1, const float* x2, int* output, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >=n) return;
    float w1 = 0.5f, w2 = 0.5f, b = -0.2f;
    float tmp = x1[idx] * w1 + x2[idx] * w2 + b;
    output[idx] = (tmp <= 0) ? 0 : 1;
}

__global__ void xor_gate(const float* x1, const float* x2, int* output, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >=n) return;
    float nand_tmp = x1[idx] * -0.5f + x2[idx] * -0.5f + 0.7f;
    int s1 = (nand_tmp <= 0) ? 0 : 1;

    float or_tmp = x1[idx] * 0.5f + x2[idx] * 0.5f - 0.2f;
    int s2 = (or_tmp <= 0) ? 0 : 1;

    float and_tmp = s1 * 0.5f + s2 * 0.5f - 0.7f;
    output[idx] = (and_tmp <= 0) ? 0 : 1;
}

int main()
{
    const int N = 4;
    // host
    float h_x1[N] = {0, 1, 0, 1};
    float h_x2[N] = {0, 0, 1, 1};
    int h_and_output[N];
    int h_nand_output[N];
    int h_or_output[N];
    int h_xor_output[N];
    
    // device
    float *d_x1, *d_x2;
    int *d_and_output;
    int *d_nand_output;
    int *d_or_output;
    int *d_xor_output;

    cudaMalloc(&d_x1, N * sizeof(float));
    cudaMalloc(&d_x2, N * sizeof(float));
    cudaMalloc(&d_and_output, N * sizeof(int));
    cudaMalloc(&d_nand_output, N * sizeof(int));
    cudaMalloc(&d_or_output, N * sizeof(int));
    cudaMalloc(&d_xor_output, N * sizeof(int));

    cudaMemcpy(d_x1, h_x1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, N * sizeof(float), cudaMemcpyHostToDevice);
    
    and_gate<<<1, N>>>(d_x1, d_x2, d_and_output, N);
    nand_gate<<<1, N>>>(d_x1, d_x2, d_nand_output, N);
    or_gate<<<1, N>>>(d_x1, d_x2, d_or_output, N);
    xor_gate<<<1, N>>>(d_x1, d_x2, d_xor_output, N);
    
    cudaMemcpy(h_and_output, d_and_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nand_output, d_nand_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_or_output, d_or_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xor_output, d_xor_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
    {
        std::cout << 
        "x1: "<< h_x1[i] << 
        ", x2: " << h_x2[i] << 
        ", AND gate result: " << h_and_output[i] << 
        ", NAND gate result: " << h_nand_output[i] << 
        ", OR gate result: " << h_or_output[i] <<
        ", XOR gate result: " << h_xor_output[i] <<
        std::endl;
    }
    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_and_output);
    cudaFree(d_nand_output);
    cudaFree(d_or_output);
    cudaFree(d_xor_output);

    return 0;
}