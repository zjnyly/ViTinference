#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"
#include "matdiv.cpp"


template <class T>
Tensor<T> *softmax(Tensor<T> *originalData, int dim)
{

    dim = getIdx(originalData, dim);

    auto originalDimension = originalData->getDimension();

    if (dim != originalDimension.size() - 1)
    {
        auto rearrangedDimension = originalDimension;

        std::vector<std::pair<std::string, int>> originalView;
        std::vector<std::pair<std::string, int>> rearrangedView;

        char id = 'a';

        for (auto i = 0; i < originalDimension.size(); i++)
        {
            originalView.push_back({std::string(1, id++), originalDimension[i]});
        }

        rearrangedView = originalView;
        std::swap(rearrangedView[dim], rearrangedView[rearrangedView.size() - 1]);
        std::swap(rearrangedDimension[dim], rearrangedDimension[rearrangedDimension.size() - 1]);

        originalData = rearrange<T>(originalData, originalView, rearrangedView, originalDimension, rearrangedDimension);

        originalData->showDimension();
        originalDimension = rearrangedDimension;
    }

    auto softmaxMat = new Tensor<T>(originalDimension);

    auto originalDataPointer = originalData->getDataPointer(); 
    auto softmaxMatPointer = softmaxMat->getDataPointer(); 

    auto innerSize = originalDimension[dim];
    auto outerSize = originalData->getDataSize() / innerSize;

    // https://www.cnblogs.com/ysugyl/p/12922598.html#:~:text=%E5%87%BD%E6%95%B0%E5%AE%9E%E7%8E%B0%20%E7%94%B1%E4%BA%8E%E6%8C%87%E6%95%B0%E5%87%BD%E6%95%B0%E7%9A%84%E6%94%BE%E5%A4%A7%E4%BD%9C%E7%94%A8%E8%BF%87%E4%BA%8E%E6%98%8E%E6%98%BE%EF%BC%8C%E5%A6%82%E6%9E%9C%E7%9B%B4%E6%8E%A5%E4%BD%BF%E7%94%A8softmax%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F%20s%20o%20f%20t%20m%20a,j%20e%20x%20p%20%28x%20j%29%20%E8%BF%9B%E8%A1%8C%E5%87%BD%E6%95%B0%E5%AE%9E%E7%8E%B0%EF%BC%8C%E5%AE%B9%E6%98%93%E5%AF%BC%E8%87%B4%E6%95%B0%E6%8D%AE%E6%BA%A2%E5%87%BA%20%28%E4%B8%8A%E6%BA%A2%29%E3%80%82
    // https://blog.csdn.net/fengbingchun/article/details/75220591
    for(auto i = 0; i < outerSize; i++)
    {

        auto alpha = *std::max_element(originalDataPointer + i * innerSize, originalDataPointer + (i + 1) * innerSize);
        T denominator{0};
        for(auto j = 0; j < innerSize; j++)
        {
            auto idx = i * innerSize + j;
            softmaxMatPointer[idx] = std::exp(originalDataPointer[idx] - alpha);
            denominator += softmaxMatPointer[idx];
        }

        for(auto j = 0; j < innerSize; j++)
        {
            auto idx = i * innerSize + j;
            softmaxMatPointer[idx] /= denominator;
        }
    }

    return softmaxMat;
}