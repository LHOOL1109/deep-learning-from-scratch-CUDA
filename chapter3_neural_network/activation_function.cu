#include <cuda_runtime.h>
#include <iostream>
#include "../include/cuda_utils.cuh"
#include "../include/test_utils.hpp"

__global__ void step_function(const float* X, int* output, int n)
{
    int idx = IDX_1D;
    if (idx >= n) return;
    output[idx] = (X[idx] > 0) ? 1 : 0;
}

__global__ void sigmoid(const float* X, float* output, int n)
{
    int idx = IDX_1D;
    if (idx >= n) return;
    output[idx] = 1.f / (1.f + expf(-X[idx]));
}

__global__ void relu(const float* X, float* output, int n)
{
    int idx = IDX_1D;
    if (idx >= n) return;
    output[idx] = (X[idx] > 0) ? X[idx] : 0;
}

int main()
{
    const int N = 30124;
    int num_threads = 1024;
    int num_blocks = (N + num_threads - 1) / num_threads;
    float h_x1[N];
    get_random_list(h_x1, N);
    int h_step_output[N];
    float h_sigmoid_output[N];
    float h_relu_output[N];

    float* d_x1;
    int* d_step_output;
    float* d_sigmoid_output;
    float* d_relu_output;

    cudaMalloc(&d_x1, N * sizeof(float));
    cudaMalloc(&d_step_output, N * sizeof(int));
    cudaMalloc(&d_sigmoid_output, N * sizeof(float));
    cudaMalloc(&d_relu_output, N * sizeof(float));

    cudaMemcpy(d_x1, h_x1, N * sizeof(float), cudaMemcpyHostToDevice);
    
    step_function<<<num_blocks, num_threads>>>(d_x1, d_step_output, N);
    sigmoid<<<num_blocks, num_threads>>>(d_x1, d_sigmoid_output, N);
    relu<<<num_blocks, num_threads>>>(d_x1, d_relu_output, N);
    cudaMemcpy(h_step_output, d_step_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sigmoid_output, d_sigmoid_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_output, d_relu_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
    {
        std::cout <<
        "idx: " << i <<
        ",  x: " << h_x1[i] <<
        ",  step_function result: " << h_step_output[i] <<
        ",  sigmoid result: " << h_sigmoid_output[i] <<
        ",  relu result: " << h_relu_output[i] <<
        std::endl;
    }
    cudaFree(d_x1);
    cudaFree(d_step_output);
    return 0;
}