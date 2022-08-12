#pragma once
#include <vector>
#include "../utils/tensor.hpp"

template <class T>
Tensor<T> * matdiv(Tensor<T> *matA, T divident)
{

    auto size = matA->getDataSize();
    auto dimension = matA->getDimension();


    auto matB = new Tensor<T>(dimension);

    auto A = matA->getDataPointer();
    auto B = matB->getDataPointer();

#pragma omp parallel for

    for (auto i = 0; i < size; i++)
    {
        B[i] = A[i] / divident;
    }

    // matA->showData();
    // matB->showData();
    // matC->showData();
    return matB;
}
