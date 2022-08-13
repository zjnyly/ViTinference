#pragma once
#include <vector>
#include "../utils/tensor.hpp"

template <class T>
Tensor<T> *matmul(Tensor<T> *matA, Tensor<T> *matB)
{
    auto matADim = matA->getDimension();
    auto matBDim = matB->getDimension();
    auto matCDim = matADim;

    auto I = matADim[matADim.size() - 2];
    auto J = matADim[matADim.size() - 1];
    auto K = matBDim[matBDim.size() - 1];

    // std::cout<<I<< " "<< J << " " << K << std::endl;

    auto N = 1;
    auto GAP = I * K;
    for (auto i = 0; i < matCDim.size() - 2; i++)
    {
        N *= matCDim[i];
    }
    // std::cout<<i<<" " << j << " " << k << std::endl;
    matCDim[matCDim.size() - 1] = K;
    // std::vector<int> dimension = {I, K};

    auto matC = new Tensor<T>(matCDim);

    auto A = matA->getDataPointer();
    auto B = matB->getDataPointer();
    auto C = matC->getDataPointer();

    for (auto n = 0; n < N; n++)
    {
#pragma omp parallel for
        for (auto k = 0; k < K; k++)
        {
            for (auto i = 0; i < I; i++)
            {
                for (auto j = 0; j < J; j++)
                {
                    C[n * GAP + i * K + k] += A[n * GAP + i * J + j] * B[n * GAP + k * J + j];
                    // std::cout<<A[N * GAP + i * J + j] * B[N * GAP + k * J + j]<<std::endl;
                    // std::cout<<N * GAP + i * K + k<<std::endl;
                }
            }
        }
    }

    // matA->showData();
    // matB->showData();
    // matC->showData();
    return matC;
}
