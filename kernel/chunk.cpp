#pragma once
#include <vector>
#include <algorithm>
#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"
#include "slice.cpp"


template <class T>
Tensor<T> ** chunk(Tensor<T> * data, int pieces, int dim)
{  
    dim = getIdx<double>(data, dim);
    auto dataDim = data->getDimension();
    auto chunkDim = dataDim;
    auto chunkSize = chunkDim[dim] / pieces;
    chunkDim[dim] = chunkSize;

    auto chunks = new Tensor<T> * [pieces];

    std::vector<std::pair<int, int>> sliceMetric;

    for(auto i = 0; i < chunkDim.size(); i++)
    {
        sliceMetric.push_back({0, chunkDim[i]});
    }

    for(auto i = 0; i < pieces; i++)
    {
        sliceMetric[dim].first = i * chunkSize;
        sliceMetric[dim].second = (i + 1) * chunkSize;
        chunks[i] = slice<T>(data, sliceMetric);
    }

    return chunks;
}