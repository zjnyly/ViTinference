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



template<class T>
Tensor<T> * attention(Tensor<T> * data, int layer, int idx)
{
    //####################################################################################################
    // qkv = self.to_qkv(x).chunk(3, dim = -1)
    //####################################################################################################
    std::string PATH_transformer_layers_fn_to_qkv_weight = "transformer.layers." + std::to_string(layer) + "." + std::to_string(idx) + ".fn.to_qkv.weight";
    std::vector<int> DIM_transformer_layers_fn_to_qkv_weight = {3072, 1024};
    auto out1 = Linear<T>(data, PATH_transformer_layers_fn_to_qkv_weight, "", DIM_transformer_layers_fn_to_qkv_weight, std::vector<int>(), false);

    
    //####################################################################################################
    // q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
    //####################################################################################################
    auto QKV = chunk<T>(out1, 3, -1);
    auto Q = QKV[0], K = QKV[1], V = QKV[2]; 
    std::vector<std::pair<std::string, int>> originalView = {{"n", 50}, {"h", 16}, {"d", 64}};
    std::vector<int> originalDimension = {50, 16, 64};
    std::vector<std::pair<std::string, int>> rearrangedView = {{"h", 16}, {"n", 50}, {"d", 64}};
    std::vector<int> rearrangedDimension = {16, 50, 64};
    Q = rearrange<T>(Q, originalView, rearrangedView, originalDimension, rearrangedDimension);
    K = rearrange<T>(Q, originalView, rearrangedView, originalDimension, rearrangedDimension);
    V = rearrange<T>(Q, originalView, rearrangedView, originalDimension, rearrangedDimension);

    
    //####################################################################################################
    // dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    //####################################################################################################
    K = transpose<T>(K, -2, -1);
    auto dim_head = 64;
    auto scale = std::pow(dim_head, -0.5);
    auto dots = matmul<T>(Q, K);
    dots = matdiv<T>(dots, scale, true); 

    
    //####################################################################################################
    // attn = self.attend(dots)
    //####################################################################################################
    auto attn = softmax<T>(dots, -1);

    
    //####################################################################################################
    // out = torch.matmul(attn, v)
    //####################################################################################################
    auto out3 = matmul<T>(attn, V);

    
    //####################################################################################################
    // out = rearrange(out, 'b h n d -> b n (h d)')
    //####################################################################################################
    auto out4 = rearrange<T>(out3, rearrangedView, originalView, rearrangedDimension, originalDimension);

    
    //####################################################################################################
    // out = self.to_out(out)
    //####################################################################################################
    std::string PATH_transformer_layers_fn_to_out_0_weight = "transformer.layers." + std::to_string(layer) + ".0.fn.to_out.0.weight";
    std::string PATH_transformer_layers_fn_to_out_0_bias = "transformer.layers." + std::to_string(layer) + ".0.fn.to_out.0.bias";
    std::vector<int> DIM_transformer_layers_fn_to_out_0_weight = {1024, 1024};
    std::vector<int> DIM_transformer_layers_fn_to_out_0_bias = {1024};
    auto out5 = Linear<T>(out4, PATH_transformer_layers_fn_to_out_0_weight, PATH_transformer_layers_fn_to_out_0_bias, DIM_transformer_layers_fn_to_out_0_weight, DIM_transformer_layers_fn_to_out_0_bias,  true);
    return out5;
}
