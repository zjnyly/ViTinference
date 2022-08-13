#pragma once
#include <string>
#include <vector>

#include "../kernel/matmul.cpp"
#include "../kernel/matdiv.cpp"
#include "../kernel/matpow.cpp"
#include "../kernel/mean.cpp"
#include "../kernel/view.cpp"
#include "../kernel/chunk.cpp"
#include "../kernel/variance.cpp"
#include "../kernel/transpose.cpp"
#include "../kernel/softmax.cpp"

#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"
#include "layerNorm.cpp"

template <class T>
Tensor<T> *mlp(Tensor<T> *data)
{
    //####################################################################################################
    // nn.LayerNorm(dim)
    //####################################################################################################
    std::string PATH_mlp_head_0_weight = "mlp_head.0.weight";
    std::string PATH_mlp_head_0_bias = "mlp_head.0.bias";
    std::vector<int> DIM_mlp_head_0_weight = {1024};
    std::vector<int> DIM_mlp_head_0_bias = {1024};
    auto out1 = layerNorm<T>(data, -1, PATH_mlp_head_0_weight, PATH_mlp_head_0_bias, DIM_mlp_head_0_weight, DIM_mlp_head_0_bias);


    //####################################################################################################
    // nn.Linear(dim, num_classes)
    //####################################################################################################
    std::string PATH_mlp_head_1_weight = "mlp_head.1.weight";
    std::string PATH_mlp_head_1_bias = "mlp_head.1.bias";
    std::vector<int> DIM_mlp_head_1_weight = {2, 1024};
    std::vector<int> DIM_mlp_head_1_bias = {2};
    auto out2 = Linear<T>(out1, PATH_mlp_head_1_weight, PATH_mlp_head_1_bias, DIM_mlp_head_1_weight, DIM_mlp_head_1_bias, true);

    return out2;
}
