#pragma once
#include <vector>
#include "../utils/tensor.hpp"


enum OP {ADD, MINUS, PRODUCT, DIV, POW};

template <class T>
Tensor<T> * matadd(T * matA, T * matB, OP opcode)
{
    
}



template <class T>
Tensor<T> * matadd(Tensor<T> *matA, Tensor<T> *matB, bool minus = false)
{

    auto size = matA->getDataSize();
    auto dimension = matA->getDimension();


    auto matC = new Tensor<T>(dimension);

    auto A = matA->getDataPointer();
    auto B = matB->getDataPointer();
    auto C = matC->getDataPointer();



    if (minus)
    {
#pragma omp parallel for

        for (auto i = 0; i < size; i++)
        {
            C[i] = A[i] - B[i];
        }
    }
    else
    {
#pragma omp parallel for

        for (auto i = 0; i < size; i++)
        {
            C[i] = A[i] + B[i];
        }
    }
    return matC;
}


template <class T>
Tensor<T> * matadd(Tensor<T> *matA, T number, bool minus = false)
{

    auto size = matA->getDataSize();
    auto dimension = matA->getDimension();


    auto matB = new Tensor<T>(dimension);

    auto A = matA->getDataPointer();
    auto B = matB->getDataPointer();

    if (minus)
    {
#pragma omp parallel for

        for (auto i = 0; i < size; i++)
        {
            B[i] = A[i] - number;
        }
    }
    else
    {
#pragma omp parallel for

        for (auto i = 0; i < size; i++)
        {
            B[i] = A[i] + number;
        }
    }
    return matB;
}

