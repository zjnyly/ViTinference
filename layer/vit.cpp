#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "../utils/tensor.hpp"
#include "../utils/readNpyData.cpp"

#include "../kernel/rearrange.cpp"
#include "../kernel/slice.cpp"
#include "../kernel/repeat.cpp"
#include "../kernel/concat.cpp"
#include "../kernel/matadd.cpp"
#include "../kernel/mean.cpp"
#include "../kernel/variance.cpp"
#include "../kernel/view.cpp"
#include "../kernel/chunk.cpp"

#include "linear.cpp"
#include "transformer.cpp"
#include "layerNorm.cpp"
#include "mlp.cpp"

template <class T>
Tensor<T> *ViT(Tensor<T> *inputData)
{
    //####################################################################################################
    // Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
    //####################################################################################################
    std::vector<std::pair<std::string, int>> originalView = {{"b", 1}, {"c", 3}, {"h", 7}, {"p1", 32}, {"w", 7}, {"p2", 32}};
    std::vector<int> originalDimension = {1, 3, 7, 32, 7, 32};
    std::vector<std::pair<std::string, int>> rearrangedView = {{"b", 1}, {"h", 7}, {"w", 7}, {"p1", 32}, {"p2", 32}, {"c", 3}};
    std::vector<int> rearrangedDimension = {1, 49, 3072};
    auto rearrangedData = rearrange<double>(inputData, originalView, rearrangedView, originalDimension, rearrangedDimension);
    std::vector<std::pair<int, int>> sliceMetric = {{0, 0}, {0, 49}, {0, 3072}};
    auto out_1 = slice(rearrangedData, sliceMetric);

    // rearrangedData->showData();
    // out_1->showDimension();
    // out_1->showData();

    //####################################################################################################
    // Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
    //####################################################################################################
    std::string PATH_to_patch_embedding_1_weight = "to_patch_embedding.1.weight";
    std::string PATH_to_patch_embedding_1_bias = "to_patch_embedding.1.bias";
    std::vector<int> DIM_to_patch_embedding_1_weight = {1024, 3072};
    std::vector<int> DIM_to_patch_embedding_1_bias = {1024};
    auto out_2 = Linear(out_1, PATH_to_patch_embedding_1_weight, PATH_to_patch_embedding_1_bias, DIM_to_patch_embedding_1_weight, DIM_to_patch_embedding_1_bias, true);

    // out_2->showData();

    //####################################################################################################
    // repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
    //####################################################################################################
    std::string PATH_cls_token = "cls_token";
    std::vector<int> DIM_cls_token = {1, 1, 1024};
    auto DATA_cls_token = readNpyData<double>(PATH_cls_token, DIM_cls_token);
    std::vector<std::pair<int, int>> SLICE_cls_token = {{0, 0}, {0, 1}, {0, 1024}};
    auto DATA_cls_token_sliced = slice(DATA_cls_token, SLICE_cls_token);

    // DATA_cls_token_sliced->showData();

    //####################################################################################################
    // torch.cat((cls_tokens, x), dim=1)
    //####################################################################################################
    auto out3 = concat<double>(DATA_cls_token_sliced, out_2, 0);

    // out3->showData();

    //####################################################################################################
    // self.pos_embedding[:, :(n + 1)]
    //####################################################################################################
    std::string PATH_pos_embedding = "pos_embedding";   
    std::vector<int> DIM_pos_embedding = {1, 50, 1024};
    auto DATA_pos_embedding = readNpyData<double>(PATH_pos_embedding, DIM_pos_embedding);
    std::vector<std::pair<int, int>> SLICE_pos_embedding = {{0, 0}, {0, 50}, {0, 1024}};
    auto DATA_pos_embedding_1 = slice(DATA_pos_embedding, SLICE_pos_embedding);
    auto out4 = matadd<double>(out3, DATA_pos_embedding_1);

    // out4->showData();

    //####################################################################################################
    // self.transformer(x)
    //####################################################################################################
    auto out5 = transformer<double>(out4, 1);

    // out5->showData();


    //####################################################################################################
    // x[:, 0]
    //####################################################################################################
    std::vector<std::pair<int, int>> SLICE_out5 = {{0, 1}, {0, 1024}};
    auto out6 = slice(out5, SLICE_out5);

    // out6->showData();

    //####################################################################################################
    // self.mlp_head(x)
    //####################################################################################################
    auto out7 = mlp<double>(out6);

    // out7->showData();

    return out7;
}