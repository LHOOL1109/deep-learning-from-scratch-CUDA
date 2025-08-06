#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cassert>

#include <core/tensor.cuh>
#include <utils/host_utils.hpp>
#include <ops/activation_function.cuh>

void relu_host(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i)
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}

void compare_arrays(const float* a, const float* b, int size, float epsilon = 1e-5f) {
    int mismatch = 0;
    for (int i = 0; i < size; ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > epsilon) {
            if (++mismatch <= 10)
                std::cout << "Mismatch at " << i << ": " << a[i] << " vs " << b[i] << "\n";
        }
    }
    if (mismatch == 0)
        std::cout << "[✔] ReLU test passed.\n";
    else
        std::cout << "[✘] " << mismatch << " mismatches found.\n";
}

int main() {
    const int size = 1024;

    Tensor input(size);
    Tensor output_host(size);
    Tensor output_device(size);

    generate_random(input.host_ptr(), size, -1.0f, 1.0f);

    relu_host(input.host_ptr(), output_host.host_ptr(), size);

    input.to_device();

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input.device_ptr(), output_device.device_ptr(), size);
    cudaDeviceSynchronize();

    output_device.to_host();

    compare_arrays(output_host.host_ptr(), output_device.host_ptr(), size);

    return 0;
}
