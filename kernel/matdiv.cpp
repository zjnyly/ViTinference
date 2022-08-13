#pragma once
#include <vector>
#include "../utils/tensor.hpp"

template <class T>
Tensor<T> *matdiv(Tensor<T> *matA, T divident, bool mult)
{

    auto size = matA->getDataSize();
    auto dimension = matA->getDimension();

    auto matB = new Tensor<T>(dimension);

    auto A = matA->getDataPointer();
    auto B = matB->getDataPointer();

    if (mult)
    {
#pragma omp parallel for

        for (auto i = 0; i < size; i++)
        {
            B[i] = A[i] * divident;
        }
    }
    else
    {
#pragma omp parallel for
    
        for (auto i = 0; i < size; i++)
        {
            B[i] = A[i] / divident;
        }
    }

    // matA->showData();
    // matB->showData();
    // matC->showData();
    return matB;
}

template <class T>
Tensor<T> *matdiv(Tensor<T> *matA, Tensor<T> *matB, bool multi = false)
{
    auto size = matA->getDataSize();
    auto matADimension = matA->getDimension();
    auto matBDimension = matB->getDimension();

    auto matC = new Tensor<T>(matADimension);

    auto A = matA->getDataPointer();
    auto B = matB->getDataPointer();
    auto C = matC->getDataPointer();

    // matA->showDimension();
    // matB->showDimension();

    // for broadcast
    auto div = getDiv(matADimension, matBDimension);
    auto mod = getMod(matADimension, matBDimension);

    // std::cout<<"ha"<<div<<std::endl;

    if (div == -1)
    {
        div = matA->getDataSize();
    }

    // std::cout<<"ha"<<div<<std::endl;

    if (multi)
    {
#pragma omp parallel for

        for (auto i = 0; i < size; i++)
        {

            C[i] = A[i] * B[i / div + i % mod];
            // std::cout<<B[i / div + i % mod]<<" ";
        }
    }
    else
    {
#pragma omp parallel for

        for (auto i = 0; i < size; i++)
        {
            C[i] = A[i] / B[i / div + i % mod];
        }
    }
    return matC;
}
