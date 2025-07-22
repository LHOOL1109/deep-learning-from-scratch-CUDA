#pragma once

#include <cuda_runtime.h>

template <typename T>
class Matrix
{
    private:
        T* data;
        int rows, cols;
    
    public:
        __host__ __device__ Matrix(T* data, int rows, int cols);
        __host__ __device__ int index(int row, int col) const;
        __host__ __device__ T& operator()(int row, int col);
        __host__ __device__ const T& operator()(int row, int col) const;
        __host__ __device__ int row_dim() const;
        __host__ __device__ int col_dim() const;
};

template <typename T>
inline __host__ __device__ Matrix<T>::Matrix(T* data, int rows, int cols)
    : data(data), rows(rows), cols(cols) {}

template <typename T>
inline __host__ __device__ int Matrix<T>::index(int row, int col) const 
{
    return col + this->cols * row;
}

template <typename T>
inline __host__ __device__ T& Matrix<T>::operator()(int row, int col)
{
    return this->data[index(row, col)];
}

template <typename T>
inline __host__ __device__ const T& Matrix<T>::operator()(int row, int col) const
{
    return this->data[index(row, col)];
}

template <typename T>
inline __host__ __device__ int Matrix<T>::row_dim() const
{
    return this->rows;
}

template <typename T>
inline __host__ __device__ int Matrix<T>::col_dim() const
{
    return this->cols;
}