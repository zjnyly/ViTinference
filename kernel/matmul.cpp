#pragma once
#include <vector>
#include "../utils/tensor.hpp"



template<class T>
Tensor<T> * matmul(Tensor<T> * matA, Tensor<T> * matB)
{

    auto I = matA->getDimension()[0];
    auto J = matA->getDimension()[1];
    auto K = matB->getDimension()[0];
    // std::cout<<i<<" " << j << " " << k << std::endl;

    std::vector<int> dimension = {I, K};

    auto matC = new Tensor<T>(dimension);

    auto A = matA->getDataPointer();
    auto B = matB->getDataPointer();
    auto C = matC->getDataPointer();

#pragma omp parallel for
    for(auto k = 0; k < K; k++)
    {
        for(auto i = 0; i < I; i++)
        {
            for(auto j = 0; j < J; j++)
            {
                C[i * K + k] += A[i * J + j] * B[k * J + j]; 
            }
        }
    }

    // matA->showData();
    // matB->showData();
    // matC->showData();
    return matC;
}

