#pragma once
#include "../utils/tensor.hpp"

template<class T>
Tensor<T> * Transformer(Tensor<T> * weight, Tensor<T> * input, Tensor<T> * bias)
{
    // weight->showDimension();
    // input->showDimension();


    // std::vector<double>data = {1,2,3,4,5,6,7,8};
    // std::vector<int>dimension = {2, 4};
    // auto test = new Tensor<double>(data, dimension);
    // std::vector<double>data_ = {1,2};
    // std::vector<int>dimension_ = {2};
    // auto test_ = new Tensor<double>(data_, dimension_);


    // addBias(matmul(test, test), test_);


    auto ans = matmul(input, weight);
    addBias(ans, bias);
    return ans;
}