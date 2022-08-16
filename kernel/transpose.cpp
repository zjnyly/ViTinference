#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"



template <class T>
Tensor<T> * transpose(Tensor<T> * originalData, int dim1, int dim2)
{

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

    auto transposedData = rearrange<T>(originalData, originalView, rearrangedView, originalDimension, rearrangedDimension);
    return transposedData;
}