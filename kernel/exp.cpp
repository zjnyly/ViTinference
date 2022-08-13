#pragma once
#include <vector>
#include <cmath>
#include "../utils/tensor.hpp"

// y5=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))##np.tanh(x)

template <class T>
Tensor<T> * exp(Tensor<T> * data)
{

    auto size = data->getDataSize();
    auto dimension = data->getDimension();


    auto out = new Tensor<T>(dimension);

    auto A = data->getDataPointer();
    auto B = out->getDataPointer();

#pragma omp parallel for

    for (auto i = 0; i < size; i++)
    {
        B[i] = std::exp(A[i]); 
    }

    return out;
}
