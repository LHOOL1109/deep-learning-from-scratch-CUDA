#pragma once

#define CUDA_1D_IDX (blockIdx.x * blockDim.x + threadIdx.x)