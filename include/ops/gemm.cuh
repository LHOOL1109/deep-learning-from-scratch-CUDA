#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <core/tensor.cuh>


Tensor matmul(const Tensor& A, const Tensor& B, cublasHandle_t handle);

