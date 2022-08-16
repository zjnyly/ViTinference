#pragma once
#include "../utils/tensor.hpp"

#include "../kernel/matadd.cpp"

#include "attention.cpp"
#include "layerNorm.cpp"
#include "gelu.cpp"
#include "feedForward.cpp"


    template <class T>
    Tensor<T> *transformer(Tensor<T> *data, int layers)
    {
        Tensor<T> *out = data;
        for (auto layer = 0; layer < layers; layer++)
        {
            std::vector<int> DIM_transformer_layers_norm_weight = {1024};
            std::vector<int> DIM_transformer_layers_norm_bias = {1024};
            std::string PATH_transformer_layers_norm_weight = "transformer.layers." + std::to_string(layer) + "." + std::to_string(0) + ".norm.weight";
            std::string PATH_transformer_layers_norm_bias = "transformer.layers." + std::to_string(layer) + "." + std::to_string(0) + ".norm.bias";

            //####################################################################################################
            // x = attn(x) + x
            //####################################################################################################
            auto data1 = layerNorm<T>(out, -1, PATH_transformer_layers_norm_weight, PATH_transformer_layers_norm_bias, DIM_transformer_layers_norm_weight, DIM_transformer_layers_norm_bias);
            auto data2 = attention<T>(data1, layer, 0);
            out = matadd<T>(out, data2);
            // out->showData();
            PATH_transformer_layers_norm_weight = "transformer.layers." + std::to_string(layer) + "." + std::to_string(1) + ".norm.weight";
            PATH_transformer_layers_norm_bias = "transformer.layers." + std::to_string(layer) + "." + std::to_string(1) + ".norm.bias";

            //####################################################################################################
            // x = ff(x) + x
            //####################################################################################################
            auto data3 = layerNorm<T>(out, -1, PATH_transformer_layers_norm_weight, PATH_transformer_layers_norm_bias, DIM_transformer_layers_norm_weight, DIM_transformer_layers_norm_bias);
            auto data4 = feedForward<T>(data3, layer, 1);
            out = matadd<T>(out, data4);
            // out->showData();
        }

        return out;
    }
