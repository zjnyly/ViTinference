#pragma once
#include <vector>
#include "../utils/tensor.hpp"

template <class T>
Tensor<T> * concat(Tensor<T> *matA, Tensor<T> *matB, int dim)
{

    auto matADim = matA->getDimension();
    auto matBDim = matB->getDimension();

    std::vector<int> newMatDim = matADim;

    newMatDim[dim] += matBDim[dim];

    auto concatTensor = new Tensor<T>(newMatDim);

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
