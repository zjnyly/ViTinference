#pragma once
#include <vector>
#include <cmath>
#include "../utils/tensor.hpp"

template <class T>
Tensor<T> * matpow(Tensor<T> *matA, T index)
{

    auto size = matA->getDataSize();
    auto dimension = matA->getDimension();


    auto matB = new Tensor<T>(dimension);

    auto A = matA->getDataPointer();
    auto B = matB->getDataPointer();

#pragma omp parallel for
    for (auto i = 0; i < size; i++)
    {
        B[i] = pow(A[i], index);
        // std::cout<<B[i]<<std::endl;
    }

    // matA->showData();
    // matB->showData();
    // matC->showData();
    // matB->showDimension();
    // matB->showRawData();

    // std::cout<<size<<std::endl;
    return matB;
}
