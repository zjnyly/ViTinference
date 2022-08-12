#pragma once
#include <string>
#include <vector>

#include "../kernel/matmul.cpp"
#include "../kernel/matdiv.cpp"
#include "../kernel/matpow.cpp"
#include "../kernel/mean.cpp"
#include "../kernel/view.cpp"
#include "../kernel/variance.cpp"


#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"



template<class T>
Tensor<T> * attention(Tensor<T> * data, int layer, int idx)
{

    std::string PATH_transformer_layers_fn_to_qkv_weight = "transformer.layers." + std::to_string(layer) + "." + std::to_string(idx) + ".fn.to_qkv.weight";

    std::vector<int> DIM_transformer_layers_fn_to_qkv_weight = {3072, 1024};
 
    auto DATA_transformer_layers_fn_to_qkv_weight = readNpyData<double>(PATH_transformer_layers_fn_to_qkv_weight, DIM_transformer_layers_fn_to_qkv_weight);
    auto ANS_transformer_layers_fn_to_qkv = Linear(DATA_transformer_layers_fn_to_qkv_weight, data, (Tensor<T>*)nullptr, false);

}
