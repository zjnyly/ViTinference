#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "utils/tensor.hpp"
#include "utils/readNpyData.cpp"

#include "kernel/rearrange.cpp"
#include "kernel/slice.cpp"
#include "kernel/repeat.cpp"
#include "kernel/concat.cpp"
#include "kernel/matadd.cpp"
#include "kernel/mean.cpp"
#include "kernel/variance.cpp"
#include "kernel/view.cpp"
#include "kernel/chunk.cpp"

#include "layer/linear.cpp"
#include "layer/transformer.cpp"
#include "layer/layerNorm.cpp"
#include "layer/mlp.cpp"

int main()
{
    // Fake input image, just one image per batch
    std::vector<int> inputDataDimension = {1, 3, 224, 224};
    auto inputData = new Tensor<double>(inputDataDimension, true);




    // inputData->showRawData();
    // std::cout<<inputData->getDataSize()<<std::endl;

    // Then reshape the image from dim [1, 3, 224, 224] to [1, 1024, 3072]
    std::vector<std::pair<std::string, int>> originalView = {{"b", 1}, {"c", 3}, {"h", 7}, {"p1", 32}, {"w", 7}, {"p2", 32}};
    std::vector<int> originalDimension = {1, 3, 7, 32, 7, 32};
    std::vector<std::pair<std::string, int>> rearrangedView = {{"b", 1}, {"h", 7}, {"w", 7}, {"p1", 32}, {"p2", 32}, {"c", 3}};
    std::vector<int> rearrangedDimension = {1, 49, 3072};
    auto rearrangedData = rearrange<double>(inputData, originalView, rearrangedView, originalDimension, rearrangedDimension);
    // rearrangedData->showDimension();

    std::vector<std::pair<int, int>> sliceMetric = {{0, 0}, {0, 49}, {0, 3072}};

    // rearrangedData->showData();
    auto slicedData = slice(rearrangedData, sliceMetric);
    // slicedData->showData();


    //##################################################
    //to_patch_embedding.1
    //##################################################
    std::string PATH_to_patch_embedding_1_weight = "to_patch_embedding.1.weight";
    std::string PATH_to_patch_embedding_1_bias = "to_patch_embedding.1.bias";
    
    std::vector<int> DIM_to_patch_embedding_1_weight = {1024, 3072};
    std::vector<int> DIM_to_patch_embedding_1_bias = {1024};
 
    auto DATA_to_patch_embedding_1_weight = readNpyData<double>(PATH_to_patch_embedding_1_weight, DIM_to_patch_embedding_1_weight);
    auto DATA_to_patch_embedding_1_bias = readNpyData<double>(PATH_to_patch_embedding_1_bias, DIM_to_patch_embedding_1_bias);

 
    auto ANS_to_patch_embedding_1 = Linear(DATA_to_patch_embedding_1_weight, slicedData, DATA_to_patch_embedding_1_bias);
    // ANS_to_patch_embedding_1->showDimension();


    //##################################################
    //cls_token
    //##################################################
    std::string PATH_cls_token = "cls_token";
    std::vector<int> DIM_cls_token = {1, 1, 1024};
    auto DATA_cls_token = readNpyData<double>(PATH_cls_token, DIM_cls_token);

    // DATA_cls_token->showData();

    std::vector<std::pair<int, int>> SLICE_cls_token = {{0, 0}, {0, 1}, {0, 1024}};
    auto DATA_cls_token_1 = slice(DATA_cls_token, SLICE_cls_token);

    // DATA_cls_token_1->showDimension();
    // DATA_cls_token_1->showData();

    auto concat_x = concat<double>(DATA_cls_token_1, ANS_to_patch_embedding_1, 0);

    // concat_x->showDimension();



    // x += self.pos_embedding[:, :(n + 1)]

    std::string PATH_pos_embedding = "pos_embedding";   
    std::vector<int> DIM_pos_embedding = {1, 50, 1024};
    auto DATA_pos_embedding = readNpyData<double>(PATH_pos_embedding, DIM_pos_embedding);

    std::vector<std::pair<int, int>> SLICE_pos_embedding = {{0, 0}, {0, 50}, {0, 1024}};
    // std::cout<<"hi"<<std::endl;
    // DATA_pos_embedding->showDimension();
    auto DATA_pos_embedding_1 = slice(DATA_pos_embedding, SLICE_pos_embedding);
    // std::cout<<"hi"<<std::endl;

    // ANS_to_patch_embedding_1->showDimension();
    // x += self.pos_embedding[:, :(n + 1)]
    auto DATA_x_plus_pos_embedding = matadd<double>(concat_x, DATA_pos_embedding_1);



    // std::vector<int> testDim = {2, 3, 4};
    // auto testData = new Tensor<double>(testDim, true);
    // chunk<double>(testData, 4, -1);


    auto datan = transformer<double>(DATA_x_plus_pos_embedding, 1);

    // datan->showDimension();

    // x[:, 0]

    std::vector<std::pair<int, int>> SLICE_datan = {{0, 1}, {0, 1024}};

    auto datax = slice(datan, SLICE_datan);
    datax->showDimension();

    mlp<double>(datax)->showRawData();
    




    


    // layerNorm<double>(testData, -1);

    
    // layerNorm<double>();



    
    // testData->showRawData();
    // auto meanData = mean<double>(testData, -1);
    // // meanData->showDimension();
    // auto viewData = view<double>(meanData, {2, 3, -1});
    // viewData->showRawData();
    // // std::cout<<"test"<<std::endl;
    // matadd(testData, viewData, true)->showRawData();

    // variance<double>(testData, -1);
    // view<double>(testData, { -1, 2, 3, 1})->showDimension();


    

    



    // No need for dropout



    


    


    // std::pair<int, int> repeatAt(0, 1);
    // Repeat<double>(DATA_cls_token, repeatAt);






    // DATA_to_patch_embedding_1_weight->showData();
    // DATA_to_patch_embedding_1_bias->showData();

    



    // auto dataPtr = data->getDataPointer();

    // for(int i = 0; i < 2048; i++)
    // {
    //     std::cout<<dataPtr[i]<<" ";
    // }
    // auto data = readNpyData(path, dimension);




// Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),

    


    
}






