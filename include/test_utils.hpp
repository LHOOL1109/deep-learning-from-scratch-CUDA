#pragma once
#include <random>
#include <algorithm>

inline void get_random_list(float* arr, int size, unsigned seed = 42,
                            float min = -1.f, float max = 1.f) 
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min, max);
    for (int i = 0; i < size; ++i)
        arr[i] = dist(rng);
}