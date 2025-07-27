#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cassert>

#include <core/tensor.cuh>
#include <ops/gemm.cu>


void matmul_host(const Tensor& A, const Tensor& B, Tensor& C) 
{
    int M = A.size() / A.width();
    int K = A.width();
    int N = B.width();

    const float* a = A.host_ptr();
    const float* b = B.host_ptr();
    float* c = C.host_ptr();

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += a[i * K + k] * b[k * N + j];
            c[i * N + j] = sum;
        }
}


void compare_tensor(const Tensor& a, const Tensor& b, float epsilon = 1e-5f) 
{
    assert(a.size() == b.size());

    const float* ha = a.host_ptr();
    const float* hb = b.host_ptr();

    int mismatch = 0;
    for (int i = 0; i < a.size(); ++i) {
        float diff = std::abs(ha[i] - hb[i]);
        if (diff > epsilon) {
            if (++mismatch <= 10)
                std::cout << "Mismatch at " << i << ": " << ha[i] << " vs " << hb[i] << "\n";
        }
    }

    if (mismatch == 0)
        std::cout << "[✔] MatMul test passed.\n";
    else
        std::cout << "[✘] " << mismatch << " mismatches found.\n";
}

// ------------------------ Main ------------------------

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // A: (2, 3), B: (3, 4)
    Tensor A(2, 3);
    Tensor B(3, 4);

    A.fill_host({1, 2, 3,
                 4, 5, 6});
    B.fill_host({1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 10, 11, 12});

    A.to_device();
    B.to_device();

    // GPU matmul
    Tensor C = matmul(A, B, handle);
    C.to_host();

    // CPU 정답
    Tensor C_ref(2, 4);
    matmul_host(A, B, C_ref);

    // 비교
    compare_tensor(C_ref, C);

    cublasDestroy(handle);
    for (int i = 0; i < C_ref.size(); i++)
    {
        std::cout << "C ref:" << C_ref.host_ptr()[i] 
        << "C(cublas): " << C.host_ptr()[i] << std::endl;
    }
    return 0;
}
