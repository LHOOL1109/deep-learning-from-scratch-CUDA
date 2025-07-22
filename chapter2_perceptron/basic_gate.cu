#include <cuda_runtime.h>
#include <iostream>
#include "../include/cuda_utils.cuh"
#include "../include/test_utils.hpp"


__device__ int logic_gate(float x1, float x2, float w1, float w2, float b) 
{
    float tmp = x1 * w1 + x2 * w2 + b;
    return (tmp <= 0.0f) ? 0 : 1;
}

// AND gate
__global__ void and_gate(const float* x1, const float* x2, int* output, int n)
{
    int idx = IDX_1D;
    if (idx >= n) return;
    output[idx] = logic_gate(x1[idx], x2[idx], 0.5f, 0.5f, -0.7f);
}
// NAND gate
__global__ void nand_gate(const float* x1, const float* x2, int* output, int n)
{
    int idx = IDX_1D;
    if (idx >= n) return;
    output[idx] = logic_gate(x1[idx], x2[idx], -0.5f, -0.5f, 0.7f);
}
// OR gate
__global__ void or_gate(const float* x1, const float* x2, int* output, int n)
{
    int idx = IDX_1D;
    if (idx >= n) return;
    output[idx] = logic_gate(x1[idx], x2[idx], 0.5f, 0.5f, -0.2f);
}
// XOR gate
__global__ void xor_gate(const float* x1, const float* x2, int* output, int n)
{
    int idx = IDX_1D;
    if (idx >= n) return;
    // nand
    int s1 = logic_gate(x1[idx], x2[idx], -0.5f, -0.5f, 0.7f);
    // or
    int s2 = logic_gate(x1[idx], x2[idx], 0.5f, 0.5f, -0.2f);
    // and
    output[idx] = logic_gate((float) s1, (float) s2, 0.5f, 0.5f, -0.7f);
}

int main()
{
    const int N = 2046;
    int num_threads = 1024;
    int num_blocks = (N + num_threads - 1) / num_threads;
    // host
    float h_x1[N];
    float h_x2[N];
    get_random_list(h_x1, N);
    get_random_list(h_x2, N, 43);

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
    
    and_gate<<<num_blocks, num_threads>>>(d_x1, d_x2, d_and_output, N);
    nand_gate<<<num_blocks, num_threads>>>(d_x1, d_x2, d_nand_output, N);
    or_gate<<<num_blocks, num_threads>>>(d_x1, d_x2, d_or_output, N);
    xor_gate<<<num_blocks, num_threads>>>(d_x1, d_x2, d_xor_output, N);
    
    cudaMemcpy(h_and_output, d_and_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nand_output, d_nand_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_or_output, d_or_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xor_output, d_xor_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
    {
        std::cout << 
        "idx: " << i <<
        ", x1: "<< h_x1[i] << 
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