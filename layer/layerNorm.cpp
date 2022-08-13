#pragma once
// https://zhuanlan.zhihu.com/p/288300334
// x = torch.rand(2,3,4,5)
// layer = nn.LayerNorm(5)

// mean = x.mean(axis=3).reshape(-1,x.shape[1],x.shape[2],1)
// var = x.var(axis=3,unbiased=False).reshape(-1,x.shape[1],x.shape[2],1)
// out2 = (x-mean)/((var+1e-5)**0.5)


//https://blog.csdn.net/weixin_39228381/article/details/107939602

// elementwise_affine

#include "../kernel/matmul.cpp"
#include "../kernel/matdiv.cpp"
#include "../kernel/matpow.cpp"
#include "../kernel/mean.cpp"
#include "../kernel/view.cpp"
#include "../kernel/variance.cpp"


#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"



template<class T>
Tensor<T> * layerNorm(Tensor<T> * data, int dim, std::string PATH_layers_norm_weight, std::string PATH_layers_norm_bias, std::vector<int> DIM_norm_weight, std::vector<int> DIM_norm_bias)
{
    dim = getIdx<T>(data, dim);
    auto meanData = mean<T>(data, dim);
    auto meanDataDim = data->getDimension();
    meanDataDim[dim] = 1;
    auto meanReshape = view<T>(meanData, meanDataDim);
    auto varianceData = variance<T>(data, dim);
    auto varianceReshape = view<T>(varianceData, meanDataDim);
    auto numerator = matadd<T>(data, meanReshape, true);
    auto varPlusEps = matadd<T>(varianceReshape, 1e-5);
    auto denomenator = matpow<T>(varPlusEps, 0.5);
    auto norm = matdiv<T>(numerator, denomenator, false);
    auto DATA_transformer_layers_norm_weight = readNpyData<double>(PATH_layers_norm_weight, DIM_norm_weight);
    auto DATA_transformer_layers_norm_bias = readNpyData<double>(PATH_layers_norm_bias, DIM_norm_bias);
    auto DATA_transformer_layers_norm_weight_reshape = view<T>(DATA_transformer_layers_norm_weight, {1, DIM_norm_weight[0]});
    auto affine = matdiv<T>(norm, DATA_transformer_layers_norm_weight_reshape, true);
    auto DATA_transformer_layers_norm_bias_reshape = view<T>(DATA_transformer_layers_norm_bias, {1, DIM_norm_bias[0]});
    auto addBias = matadd<T>(affine, DATA_transformer_layers_norm_bias_reshape);
    return addBias;
}
