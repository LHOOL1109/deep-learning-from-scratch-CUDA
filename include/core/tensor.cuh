/* 
tensor.cuh 
include/core/tensor.cuh
*/
#include <iostream>

namespace Tensor{
    // 생성은 무조건 호스트에서!
    template <typename T>
    class TensorBase
    {
        protected:
            T* host_data_ = nullptr;
            T* device_data_ = nullptr;
            
        private:
            int num_elements_;
        
        public:
            TensorBase(T* host_data, int dim) : host_data_(host_data), num_elements_(dim) {};
            ~TensorBase() 
            {
                if (device_data_) cudaFree(device_data_);
            }
            void to_device()  // host_data를 device_data로 copy
            {
                if (!device_data_) cudaMalloc(&device_data_, sizeof(T) * num_elements_);
                cudaMemcpy(device_data_, host_data_, sizeof(T) * num_elements_, cudaMemcpyHostToDevice);
            };
            void to_host() // device_data를 host_data로 copy
            {
                cudaMemcpy(host_data_, device_data_, sizeof(T) * num_elements_, cudaMemcpyDeviceToHost);
            };
            __host__ __device__ int size() const { return num_elements_; }
            T* device_ptr() const { return device_data_; }
    };

    template <typename T>
    class Tensor1D : public TensorBase<T>
    {
        private:
            int rows_;

        public:
            Tensor1D(T* host_data, int rows) : TensorBase<T>(host_data, rows), rows_(rows) {}
            T operator()(int row_index) const { return this->host_data_[row_index]; }
    };

    template <typename T>
    class Tensor2D : public TensorBase<T>
    {
        private:
            int rows_;
            int cols_;

        public:
            Tensor2D(T* host_data, int rows, int cols) : TensorBase<T>(host_data, rows * cols), rows_(rows), cols_(cols) {}
            T operator()(int row_index, int col_index) const { return this->host_data_[row_index * cols_ + col_index]; }
    };

    template <typename T>
    class Tensor3D : public TensorBase<T>
    {
        private:
            int depth_;
            int rows_;
            int cols_;

        public:
            Tensor3D(T* host_data, int depth, int rows, int cols) : 
                TensorBase<T>(host_data, rows * cols * depth), rows_(rows), cols_(cols), depth_(depth) {}
            T operator()(int depth_index, int row_index, int col_index) const 
            { 
                return this->host_data_[(depth_index * rows_ + row_index) * cols_ + col_index];
            }
    };

    template <typename T>
    class Tensor4D : public TensorBase<T>
    {
        private:
            int batch_;
            int depth_;
            int rows_;
            int cols_;

        public:
            Tensor4D(T* host_data, int batch, int depth, int rows, int cols) : 
                TensorBase<T>(host_data, batch * depth * rows * cols), batch_(batch) , depth_(depth), rows_(rows), cols_(cols) {}
            T operator()(int batch_index, int depth_index, int row_index, int col_index) const 
            { 
                return this->host_data_[((batch_index * depth_ + depth_index) * rows_ + row_index) * cols_ + col_index];
            }
    };
}