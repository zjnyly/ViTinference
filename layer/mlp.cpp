//  self.mlp_head = nn.Sequential(
//             nn.LayerNorm(dim),
//             nn.Linear(dim, num_classes)
//         )

// -->name: mlp_head.0.weight -->grad_requirs: True  -->shape: torch.Size([1024])
// -->name: mlp_head.0.bias -->grad_requirs: True  -->shape: torch.Size([1024])
// -->name: mlp_head.1.weight -->grad_requirs: True  -->shape: torch.Size([2, 1024])
// -->name: mlp_head.1.bias -->grad_requirs: True  -->shape: torch.Size([2])

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

    // mlp_head.0.weight
    std::string PATH_mlp_head_0_weight = "mlp_head.0.weight";
    std::string PATH_mlp_head_0_bias = "mlp_head.0.bias";

    std::vector<int> DIM_mlp_head_0_weight = {1024};
    std::vector<int> DIM_mlp_head_0_bias = {1024};

    // x = attn(x) + x

    auto data1 = layerNorm<T>(data, -1, PATH_mlp_head_0_weight, PATH_mlp_head_0_bias, DIM_mlp_head_0_weight, DIM_mlp_head_0_bias);

    
    std::string PATH_mlp_head_1_weight = "mlp_head.1.weight";
    std::string PATH_mlp_head_1_bias = "mlp_head.1.bias";

    std::vector<int> DIM_mlp_head_1_weight = {2, 1024};
    std::vector<int> DIM_mlp_head_1_bias = {2};

    auto DATA_mlp_head_1_weight = readNpyData<double>(PATH_mlp_head_1_weight, DIM_mlp_head_1_weight);
    auto DATA_mlp_head_1_bias = readNpyData<double>(PATH_mlp_head_1_bias, DIM_mlp_head_1_bias);

    auto out = Linear<T>(DATA_mlp_head_1_weight, data1, DATA_mlp_head_1_bias, true);

    return out;
}
