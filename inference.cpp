#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "utils/tensor.hpp"
#include "utils/readNpyData.cpp"
#include "kernel/rearrange.cpp"
#include "kernel/linear.cpp"
#include "kernel/slice.cpp"

int main()
{
    // Fake input image, just one image per batch
    std::vector<int> inputDataDimension = {1, 3, 224, 224};
    auto inputData = new Tensor<double>(inputDataDimension);


    

    // inputData->showDimension();

    // Then reshape the image from dim [1, 3, 224, 224] to [1, 1024, 3072]
    std::vector<std::pair<std::string, int>> originalView = {{"b", 1}, {"c", 3}, {"h", 7}, {"p1", 32}, {"w", 7}, {"p2", 32}};
    std::vector<int> originalDimension = {1, 3, 7, 32, 7, 32};
    std::vector<std::pair<std::string, int>> rearrangedView = {{"b", 1}, {"h", 7}, {"w", 7}, {"p1", 32}, {"p2", 32}, {"c", 3}};
    std::vector<int> rearrangedDimension = {1, 1024, 3072};
    auto rearrangedData = rearrange<double>(inputData, originalView, rearrangedView, originalDimension, rearrangedDimension);
    // rearrangedData->showDimension();

    std::vector<std::pair<int, int>> sliceMetric = {{0, 0}, {0, 1024}, {0, 3072}};
    auto slicedData = slice(rearrangedData, sliceMetric);

    // slicedData->showData();


    std::string PATH_to_patch_embedding_1_weight = "to_patch_embedding.1.weight";
    std::string PATH_to_patch_embedding_1_bias = "to_patch_embedding.1.bias";
    
    std::vector<int> DIM_to_patch_embedding_1_weight = {1024, 3072};
    std::vector<int> DIM_to_patch_embedding_1_bias = {1024};
 
    auto DATA_to_patch_embedding_1_weight = readNpyData<double>(PATH_to_patch_embedding_1_weight, DIM_to_patch_embedding_1_weight);
    auto DATA_to_patch_embedding_1_bias = readNpyData<double>(PATH_to_patch_embedding_1_bias, DIM_to_patch_embedding_1_bias);


    Linear(DATA_to_patch_embedding_1_weight, slicedData, DATA_to_patch_embedding_1_bias);

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






