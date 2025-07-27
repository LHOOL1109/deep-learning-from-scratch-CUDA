/*
tensor.cuh
include/core/tensor.cuh
*/
#pragma once

#include <stdexcept>
#include <vector>


class Tensor
{
    private:
        float* host_data_ = nullptr;
        float* device_data_ = nullptr;
        int batchs_;
        int channels_;
        int height_;
        int width_;


    protected:

    public:
        Tensor(int W) : Tensor(1, 1, 1, W) {}
        Tensor(int H, int W) : Tensor(1, 1, H, W) {}
        Tensor(int C, int H, int W) : Tensor(1, C, H, W) {}
        Tensor(int B, int C, int H, int W) : batchs_(B), channels_(C), height_(H), width_(W)
        {
            int total = size();
            host_data_ = new float[total];
            cudaError_t err = cudaMalloc(&device_data_, sizeof(float) * total);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("cudaMalloc failed");
            }
        }
        ~Tensor()
        {
            delete[] host_data_;
            cudaFree(device_data_);
        }
        
        void to_host()
        {
            cudaError_t err = cudaMemcpy(host_data_, device_data_, sizeof(float) * size(), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpy (to_device) failed: " + std::string(cudaGetErrorString(err)));
            }
        }

        void to_device()
        {
            cudaError_t err = cudaMemcpy(device_data_, host_data_, sizeof(float) * size(), cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpy (to_host) failed: " + std::string(cudaGetErrorString(err)));
            }
        }

        float* host_ptr() { return host_data_; }
        const float* host_ptr() const { return host_data_; }
        float* device_ptr() { return device_data_; }
        const float* device_ptr() const { return device_data_; }

        int size() const { return batchs_ * channels_ * height_ * width_; }
        int batchs() const { return batchs_; }
        int channels() const { return channels_; }
        int height() const { return height_; }
        int width() const { return width_; }

        void fill_host(const std::vector<float>& data) {
            int total = size();
            if (data.size() != total) {
                throw std::runtime_error("fill_host: size mismatch");
            }
            std::copy(data.begin(), data.end(), host_data_);
        }
};
