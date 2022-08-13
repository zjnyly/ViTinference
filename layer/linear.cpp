#pragma once
#include <string>

#include "../kernel/matmul.cpp"
#include "../kernel/addBias.cpp"
#include "../kernel/transpose.cpp"

#include "../utils/tensor.hpp"

template<class T>
Tensor<T> * Linear(Tensor<T> * input, std::string PATH_weight, std::string PATH_bias, std::vector<int> DIM_weight, std::vector<int> DIM_bias, bool haveBias = true)
{
    auto weight = readNpyData<double>(PATH_weight, DIM_weight);    

    // weight->showData();

    weight = transpose<double>(weight, -2, -1);
    auto ans = matmul(input, weight);
    if(haveBias)
    {
        auto bias = readNpyData<double>(PATH_bias, DIM_bias);
        // bias->showData();
        addBias(ans, bias);
    }

    return ans;
}