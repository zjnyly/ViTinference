#pragma once
#include <vector>
#include "../utils/tensor.hpp"

template <class T>
Tensor<T> *matadd(Tensor<T> *matA, Tensor<T> *matB)
{

    auto I = matA->getDimension()[0];
    auto J = matA->getDimension()[1];
    // std::cout<<i<<" " << j << " " << k << std::endl;
    std::cout<<"hi"<<std::endl;
    matA->showDimension();
    
    matB->showDimension();


    std::vector<int> dimension = {I, J};

    auto matC = new Tensor<T>(dimension);

    auto A = matA->getDataPointer();
    auto B = matB->getDataPointer();
    auto C = matC->getDataPointer();

#pragma omp parallel for

    for (auto i = 0; i < I; i++)
    {
        for (auto j = 0; j < J; j++)
        {
            auto idx = i * J + j;
            C[idx] = A[idx] + B[idx];
        }
    }

    // matA->showData();
    // matB->showData();
    // matC->showData();
    return matC;
}
