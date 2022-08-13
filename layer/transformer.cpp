#pragma once
#include "../utils/tensor.hpp"
#include "attention.cpp"
#include "layerNorm.cpp"
#include "gelu.cpp"
#include "feedForward.cpp"

#include "../kernel/matadd.cpp"

// x = attn(x) + x
// x = ff(x) + x
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

        // x = attn(x) + x

        auto data1 = layerNorm<T>(out, -1, PATH_transformer_layers_norm_weight, PATH_transformer_layers_norm_bias, DIM_transformer_layers_norm_weight, DIM_transformer_layers_norm_bias);
        auto data2 = attention<T>(data1, layer, 0);
        out = matadd<T>(out, data2);

        PATH_transformer_layers_norm_weight = "transformer.layers." + std::to_string(layer) + "." + std::to_string(1) + ".norm.weight";
        PATH_transformer_layers_norm_bias = "transformer.layers." + std::to_string(layer) + "." + std::to_string(1) + ".norm.bias";

        // x = ff(x) + x

        auto data3 = layerNorm<T>(out, -1, PATH_transformer_layers_norm_weight, PATH_transformer_layers_norm_bias, DIM_transformer_layers_norm_weight, DIM_transformer_layers_norm_bias);
        auto data4 = feedForward<T>(data3, layer, 1);
        out = matadd<T>(out, data4);
    }

    return out;
    // out->showData();

    // data6->showData();

    // std::vector<int> inputDataDimension = {50, 1024};
    // auto inputData = new Tensor<double>(inputDataDimension, true);
    // inputData->showRawData();

    // data = feedForward<T>(inputData, 0, 1);

    // std::vector<int> testDim = {2, 3, 4};
    // auto testData = new Tensor<double>(testDim, true);
    // gelu<T>(testData);
    // data = gelu<T>(data);

    // auto ans = matmul(input, weight);
    // addBias(ans, bias);
    // return ans;
}