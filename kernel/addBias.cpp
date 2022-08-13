#pragma once
#include <vector>
#include "../utils/tensor.hpp"

template <class T>
Tensor<T> *addBias(Tensor<T> *mat, Tensor<T> *bias)
{
    auto matDim = mat->getDimension();
    auto I = matDim[matDim.size() - 2];
    auto J = matDim[matDim.size() - 1];

    auto biasMat = new Tensor<T>(matDim);

    auto N = 1;
    auto GAP = I * J;
    for (auto i = 0; i < matDim.size() - 2; i++)
    {
        N *= matDim[i];
    }

    auto A = mat->getDataPointer();
    auto B = biasMat->getDataPointer();
    for (auto n = 0; n < N; n++)
    {
#pragma omp parallel for
        for (auto i = 0; i < I; i++)
        {
            for (auto j = 0; j < J; j++)
            {
                A[n * GAP + i * J + j] += B[n * GAP + j];
            }
        }
    }
}