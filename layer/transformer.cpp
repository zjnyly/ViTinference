#pragma once
#include "../utils/tensor.hpp"
#include "attention.cpp"
#include "layerNorm.cpp"

template<class T>
Tensor<T> * transformer(Tensor<T> * data)
{

    data = layerNorm<T>(data, -1, 0, 0);
    data = attention<T>(data, 0, 0);

    // auto ans = matmul(input, weight);
    // addBias(ans, bias);
    // return ans;
}