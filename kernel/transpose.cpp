#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"

// Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),

template <class T>
Tensor<T> * transpose(Tensor<T> * originalData, int dim1, int dim2)
{
    // Then reshape the image from dim [1, 3, 224, 224] to [1, 1024, 3072]

    dim1 = getIdx(originalData, dim1);
    dim2 = getIdx(originalData, dim2);


    auto originalDimension = originalData->getDimension();
    auto rearrangedDimension = originalDimension;

    std::vector<std::pair<std::string, int>> originalView;
    std::vector<std::pair<std::string, int>> rearrangedView;


    char id = 'a';

    for(auto i = 0; i < originalDimension.size(); i++)
    {
        originalView.push_back({std::string(1, id++), originalDimension[i]});
    }

    rearrangedView = originalView;
    std::swap(rearrangedView[dim1], rearrangedView[dim2]);
    std::swap(rearrangedDimension[dim1], rearrangedDimension[dim2]);

    // for(auto i = 0; i < dataDim.size(); i++)
    // {
    //     std::cout<<rearrangedView[i].first<<" "<< rearrangedView[i].second<<std::endl;
    // }

    
    // std::vector<std::pair<std::string, int>> originalView = {{"b", 1}, {"c", 3}, {"h", 7}, {"p1", 32}, {"w", 7}, {"p2", 32}};
    // std::vector<int> originalDimension = {1, 3, 7, 32, 7, 32};
    // std::vector<std::pair<std::string, int>> rearrangedView = {{"b", 1}, {"h", 7}, {"w", 7}, {"p1", 32}, {"p2", 32}, {"c", 3}};
    // std::vector<int> rearrangedDimension = {1, 49, 3072};
    auto transposedData = rearrange<T>(originalData, originalView, rearrangedView, originalDimension, rearrangedDimension);
    return transposedData;
}