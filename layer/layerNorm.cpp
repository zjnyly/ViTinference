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
Tensor<T> * layerNorm(Tensor<T> * data, int dim, int layer, int idx)
{
    dim = getIdx<T>(data, dim);


    auto meanData = mean<T>(data, dim);
    auto meanDataDim = data->getDimension();
    meanDataDim[dim] = 1;
    auto meanReshape = view<T>(meanData, meanDataDim);

    auto varianceData = variance<T>(data, dim);
    auto varianceReshape = view<T>(varianceData, meanDataDim);

    // (x-mean)
    auto numerator = matadd<T>(data, meanReshape, true);
    auto varPlusEps = matadd<T>(varianceReshape, 1e-5);
    auto denomenator = matpow<T>(varPlusEps, 0.5);
    auto norm = matdiv<T>(numerator, denomenator, false);


    std::string PATH_transformer_layers_norm_weight = "transformer.layers." +  std::to_string(layer) + "." +  std::to_string(idx) + ".norm.weight";
    std::string PATH_transformer_layers_norm_bias = "transformer.layers." +  std::to_string(layer) + "." +  std::to_string(idx) + ".norm.bias";
    
    // std::cout<<PATH_transformer_layers_norm_weight<<std::endl;
    std::vector<int> DIM_transformer_layers_norm_weight = {1024};
    std::vector<int> DIM_transformer_layers_norm_bias = {1024};

    auto DATA_transformer_layers_norm_weight = readNpyData<double>(PATH_transformer_layers_norm_weight, DIM_transformer_layers_norm_weight);
    auto DATA_transformer_layers_norm_bias = readNpyData<double>(PATH_transformer_layers_norm_bias, DIM_transformer_layers_norm_bias);
 
    // DATA_transformer_layers_norm_weight->showRawData();
    // DATA_transformer_layers_norm_bias->showRawData();
    // DATA_transformer_layers_norm_weight;
    auto DATA_transformer_layers_norm_weight_reshape = view<T>(DATA_transformer_layers_norm_weight, {1, 1024});

    // for(int i = 0; i < 1024;i++)
    // {
    //     std::cout<<DATA_transformer_layers_norm_weight->getDataPointer()[i]<<" ";
    // }
    // auto affine = matdiv<T>(norm, DATA_transformer_layers_norm_weight, true);
    auto affine = matdiv<T>(norm, DATA_transformer_layers_norm_weight_reshape, true);
    // affine->showDimension();

    auto DATA_transformer_layers_norm_bias_reshape = view<T>(DATA_transformer_layers_norm_bias, {1, 1024});

    auto addBias = matadd<T>(affine, DATA_transformer_layers_norm_bias_reshape);
    // addBias->showDimension();
    
    // auto DATA_to_patch_embedding_1_weight = readNpyData<double>(PATH_to_patch_embedding_1_weight, DIM_to_patch_embedding_1_weight);
    // auto DATA_to_patch_embedding_1_bias = readNpyData<double>(PATH_to_patch_embedding_1_bias, DIM_to_patch_embedding_1_bias);
    // auto ANS_to_patch_embedding_1 = Linear(DATA_to_patch_embedding_1_weight, slicedData, DATA_to_patch_embedding_1_bias);

    return addBias;


    // meanReshape->showRawData();
    // meanReshape->showDimension();
    // numerator->showRawData();
    // numerator->showDimension();
    // varianceReshape->showRawData();
    // varianceReshape->showDimension();
    // varPlusEps->showRawData();
    // varPlusEps->showDimension();
    // denomenator->showRawData();
    // denomenator->showDimension();
    // norm->showRawData();
    // norm->showDimension();



    


    // std::vector<double>data = {1,2,3,4,5,6,7,8};
    // std::vector<int>dimension = {2, 4};
    // auto test = new Tensor<double>(data, dimension);
    // std::vector<double>data_ = {1,2};
    // std::vector<int>dimension_ = {2};
    // auto test_ = new Tensor<double>(data_, dimension_);


    // addBias(matmul(test, test), test_);


    // auto ans = matmul(input, weight);
    // addBias(ans, bias);
    // return ans;
}

// template<class T>
// Tensor<T> * layerNorm(Tensor<T> * data, Tensor<T> * dim)
// {
//     // auto mean = mean(data, -1);
    


//     // std::vector<double>data = {1,2,3,4,5,6,7,8};
//     // std::vector<int>dimension = {2, 4};
//     // auto test = new Tensor<double>(data, dimension);
//     // std::vector<double>data_ = {1,2};
//     // std::vector<int>dimension_ = {2};
//     // auto test_ = new Tensor<double>(data_, dimension_);


//     // addBias(matmul(test, test), test_);


//     // auto ans = matmul(input, weight);
//     // addBias(ans, bias);
//     // return ans;
// }
