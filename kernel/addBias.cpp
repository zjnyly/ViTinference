#pragma once
#include <vector>
#include "../utils/tensor.hpp"

template <class T>
Tensor<T> *addBias(Tensor<T> *mat, Tensor<T> *bias)
{
    auto I = mat->getDimension()[0];
    auto J = mat->getDimension()[1];

    auto A = mat->getDataPointer();
    auto B = bias->getDataPointer();

    for (auto i = 0; i < I; i++)
    {
        for (auto j = 0; j < J; j++)
        {
            A[i * J + j] += B[j];
        }
    }
}
