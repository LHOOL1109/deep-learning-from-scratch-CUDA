#include <random>

void generate_random(float* arr, int size, float min = 0.0f, float max = 1.0f)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (int i = 0; i < size; ++i) {
        arr[i] = dist(gen);
    }
}