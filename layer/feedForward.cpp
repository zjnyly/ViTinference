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
#include "gelu.cpp"



template<class T>
Tensor<T> * feedForward(Tensor<T> * data, int layer, int idx)
{

    std::string PATH_transformer_layers_fn_net_0_weight = "transformer.layers." + std::to_string(layer) + "." + std::to_string(idx) + ".fn.net.0.weight";
    std::string PATH_transformer_layers_fn_net_0_bias = "transformer.layers." + std::to_string(layer) + "." + std::to_string(idx) + ".fn.net.0.bias";
    std::vector<int> DIM_transformer_layers_fn_net_0_weight = {2048, 1024};
    std::vector<int> DIM_transformer_layers_fn_net_0_bias = {2048};
    auto data1 = Linear<T>(data, PATH_transformer_layers_fn_net_0_weight, PATH_transformer_layers_fn_net_0_bias, DIM_transformer_layers_fn_net_0_weight, DIM_transformer_layers_fn_net_0_bias, true);

    auto data2 = gelu<T>(data1);


    std::string PATH_transformer_layers_fn_net_3_weight = "transformer.layers." + std::to_string(layer) + "." + std::to_string(idx) + ".fn.net.3.weight";
    std::string PATH_transformer_layers_fn_net_3_bias = "transformer.layers." + std::to_string(layer) + "." + std::to_string(idx) + ".fn.net.3.bias";
    std::vector<int> DIM_transformer_layers_fn_net_3_weight = {1024, 2048};
    std::vector<int> DIM_transformer_layers_fn_net_3_bias = {1024};
    auto data3 = Linear<T>(data2, PATH_transformer_layers_fn_net_3_weight, PATH_transformer_layers_fn_net_3_bias, DIM_transformer_layers_fn_net_3_weight, DIM_transformer_layers_fn_net_3_bias, true);

    return data3;
}
