#include <unordered_map>
#include <core/tensor.cuh>
#include <ops/gemm.cuh>
#include <string>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <ops/broadcast.cuh>
#include <ops/activation_function.cuh>

using Network = std::unordered_map<std::string, Tensor>;


Network init_network()
{
    Network network;
    
    Tensor W1(2, 3);
    W1.fill_host({0.1, 0.3, 0.5, 0.2, 0.4, 0.6});
    W1.to_device();
    network.emplace("W1", std::move(W1));

    Tensor b1(1, 3);
    b1.fill_host({0.1, 0.2, 0.3});
    b1.to_device();
    network.emplace("b1", std::move(b1));

    Tensor W2(3, 2);
    W2.fill_host({0.1, 0.4, 0.2, 0.5, 0.3, 0.6});
    W2.to_device();
    network.emplace("W2", std::move(W2));

    Tensor b2(1, 2);
    b2.fill_host({0.1, 0.2});
    b2.to_device();
    network.emplace("b2", std::move(b2));

    Tensor W3(2, 2);
    W3.fill_host({0.1, 0.3, 0.2, 0.4});
    W3.to_device();
    network.emplace("W3", std::move(W3));

    Tensor b3(1, 2);
    b3.fill_host({0.1, 0.2});
    b3.to_device();
    network.emplace("b3", std::move(b3));

    return network;
}
//TODO
Tensor forward(Network& network, Tensor x)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    Tensor& W1 = network.at("W1");
    Tensor& W2 = network.at("W2");
    Tensor& W3 = network.at("W3");
    
    Tensor& b1 = network.at("b1");
    Tensor& b2 = network.at("b2");
    Tensor& b3 = network.at("b3");
    
    Tensor a1 = matmul(x, W1, handle);
    broadcast_add_rowwise_inplace(a1, b1);
    Tensor z1 = sigmoid(a1);

    z1.to_host();
    std::cout << "[z1] ";
    for (int i = 0; i < z1.size(); ++i)
        std::cout << z1.host_ptr()[i] << " ";
    std::cout << std::endl;


    Tensor a2 = matmul(z1, W2, handle);
    broadcast_add_rowwise_inplace(a2, b2);
    Tensor z2 = sigmoid(a2);

    std::cout << "[z2] ";
    for (int i = 0; i < z2.size(); ++i)
        std::cout << z2.host_ptr()[i] << " ";
    std::cout << std::endl;
    
    Tensor a3 = matmul(z2, W3, handle);
    broadcast_add_rowwise_inplace(a3, b3);
    Tensor z3 = sigmoid(a3);

    z3.to_host();
    std::cout << "[z3] ";
    for (int i = 0; i < z3.size(); ++i)
        std::cout << z3.host_ptr()[i] << " ";
    std::cout << std::endl;

    cublasDestroy(handle);
    return z3;
}

int main()
{
    Network net = init_network();
    Tensor x(1, 2);
    x.fill_host({1.0, 0.5});
    x.to_device();

    Tensor y = forward(net, x);
    y.to_host();

    for (int i = 0; i < y.size(); ++i)
    {
        std::cout << y.host_ptr()[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}