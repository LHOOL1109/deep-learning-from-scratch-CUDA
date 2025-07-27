#include <unordered_map>
#include <core/tensor.cuh>
#include <string>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using Network = std::unordered_map<std::string, Tensor>;


Network init_network()
{
    Network network;
    
    Tensor W1(2, 3);
    W1.fill_host({0.1, 0.3, 0.5, 0.2, 0.4, 0.6});
    network.emplace("W1", std::move(W1));

    Tensor b1(1, 3);
    b1.fill_host({0.1, 0.2, 0.3});
    network.emplace("b1", std::move(b1));

    Tensor W2(3, 2);
    W2.fill_host({0.1, 0.4, 0.2, 0.5, 0.3, 0.6});
    network.emplace("W2", std::move(W2));

    Tensor b2(1, 2);
    b2.fill_host({0.1, 0.2});
    network.emplace("b2", std::move(b2));

    Tensor W3(2, 2);
    W3.fill_host({0.1, 0.3, 0.2, 0.4});
    network.emplace("W3", std::move(W3));

    Tensor b3(1, 2);
    b3.fill_host({0.1, 0.2});
    network.emplace("b3", std::move(b3));

    return network;
}
//TODO
Tensor forward(Network network, Tensor x)
{
    Tensor W1 = network["W1"];
    Tensor W2 = network["W2"];
    Tensor W3 = network["W3"];
    
    Tensor b1 = network["b1"];
    Tensor b2 = network["b2"];
    Tensor b3 = network["b3"];
    
}