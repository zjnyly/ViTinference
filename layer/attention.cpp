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



// nn.Linear(dim, hidden_dim),
// nn.GELU(),
// nn.Dropout(dropout),
// nn.Linear(hidden_dim, dim),
// nn.Dropout(dropout)

template<class T>
Tensor<T> * attention(Tensor<T> * data, int layer, int idx)
{

    // data->showRawData();

    std::string PATH_transformer_layers_fn_to_qkv_weight = "transformer.layers." + std::to_string(layer) + "." + std::to_string(idx) + ".fn.to_qkv.weight";

    std::vector<int> DIM_transformer_layers_fn_to_qkv_weight = {3072, 1024};
 
    auto DATA_transformer_layers_fn_to_qkv_weight = readNpyData<double>(PATH_transformer_layers_fn_to_qkv_weight, DIM_transformer_layers_fn_to_qkv_weight);
    
    // DATA_transformer_layers_fn_to_qkv_weight->showData();
    data = Linear<T>(DATA_transformer_layers_fn_to_qkv_weight, data, (Tensor<T>*)nullptr, false);

    // data->showRawData();

    auto QKV = chunk<T>(data, 3, -1);

    auto Q = QKV[0], K = QKV[1], V = QKV[2]; 

    // rearrange(t, 'n (h d) -> h n d', h = self.heads)

    std::vector<std::pair<std::string, int>> originalView = {{"n", 50}, {"h", 16}, {"d", 64}};
    std::vector<int> originalDimension = {50, 16, 64};
    std::vector<std::pair<std::string, int>> rearrangedView = {{"h", 16}, {"n", 50}, {"d", 64}};
    std::vector<int> rearrangedDimension = {16, 50, 64};
    Q = rearrange<T>(Q, originalView, rearrangedView, originalDimension, rearrangedDimension);
    K = rearrange<T>(Q, originalView, rearrangedView, originalDimension, rearrangedDimension);
    V = rearrange<T>(Q, originalView, rearrangedView, originalDimension, rearrangedDimension);


    K = transpose<T>(K, -2, -1);

    // Q->showDimension();
    // K->showDimension();

    // dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

    auto dots = matmul<T>(Q, K);

    // dots->showData();

    auto scale = std::pow(1024, -0.5);
    dots = matdiv<T>(dots, scale, true); 


    



    // std::vector<int> testDim = {2, 2, 4};
    // auto testData = new Tensor<double>(testDim, true);
    // chunk<double>(testData, 4, -1); 

    auto attn = softmax<T>(dots, -1);

    // dots->showDimension();
    // attn->showDimension();
    // V->showDimension();

    auto out = matmul<T>(attn, V);



    out = rearrange<T>(out, rearrangedView, originalView, rearrangedDimension, originalDimension);


    // transformer.layers.0.0.fn.to_out.0.weight


    std::string PATH_transformer_layers_fn_to_out_0_weight = "transformer.layers." + std::to_string(layer) + ".0.fn.to_out.0.weight";
    std::string PATH_transformer_layers_fn_to_out_0_bias = "transformer.layers." + std::to_string(layer) + ".0.fn.to_out.0.bias";

    std::vector<int> DIM_transformer_layers_fn_to_out_0_weight = {1024, 1024};
    std::vector<int> DIM_transformer_layers_fn_to_out_0_bias = {1024};
 
    auto DATA_transformer_layers_fn_to_out_0_weight = readNpyData<double>(PATH_transformer_layers_fn_to_out_0_weight, DIM_transformer_layers_fn_to_out_0_weight);
    auto DATA_transformer_layers_fn_to_out_0_bias = readNpyData<double>(PATH_transformer_layers_fn_to_out_0_bias, DIM_transformer_layers_fn_to_out_0_bias);
    
    // DATA_transformer_layers_fn_to_out_0_weight->showData();
    // DATA_transformer_layers_fn_to_out_0_bias->showData();
    
    out = Linear<T>(DATA_transformer_layers_fn_to_out_0_weight, data, DATA_transformer_layers_fn_to_out_0_bias, true);
    return out;

    // out->showDimension();

    // dots->showDimension();


    // Q->showDimension();



}
