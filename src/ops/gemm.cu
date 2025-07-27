#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <core/tensor.cuh>

Tensor matmul(const Tensor& A, const Tensor& B, cublasHandle_t handle)
{
    // A: (M, K), B: (K, N)
    int K = A.width();
    int M = A.size() / K;
    int N = B.width();

    Tensor C(M, N);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.device_ptr(), N,
        A.device_ptr(), K,
        &beta,
        C.device_ptr(), N
    );
    return C;
}
