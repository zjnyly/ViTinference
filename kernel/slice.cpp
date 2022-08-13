#pragma once
#include <vector>
#include <algorithm>
#include "../utils/tensor.hpp"

template <class T>
Tensor<T> *allocateTensor(std::vector<std::pair<int, int>> &sliceMetric)
{
    long long size = 1;
    std::vector<int> dimension;
    for (auto i = 0; i < sliceMetric.size(); i++)
    {
        auto dim = sliceMetric[i].second - sliceMetric[i].first;
        if (dim != 0)
        {
            dimension.push_back(dim);
        }
    }
    return new Tensor<T>(dimension);
}

template <class T>
void performSlice(
    T *originalData,
    T *slicedData,
    std::vector<std::pair<int, int>> &sliceMetric,
    std::vector<int> &variableInteval,
    int &loopDepth,
    int &originalIdx,
    int &slicedIdx)
{

    if (loopDepth == variableInteval.size())
    {
        slicedData[originalIdx] = originalData[slicedIdx];
        originalIdx += 1;
        return;
    }
    if (sliceMetric[loopDepth].first == sliceMetric[loopDepth].second)
    {
        auto increment = sliceMetric[loopDepth].first * variableInteval[loopDepth];
        loopDepth += 1;
        slicedIdx += increment;
        performSlice<T>(originalData, slicedData, sliceMetric, variableInteval, loopDepth, originalIdx, slicedIdx);
        loopDepth -= 1;
        slicedIdx -= increment;
    }
    else
    {
        for (int i = sliceMetric[loopDepth].first; i < sliceMetric[loopDepth].second; i++)
        {
            auto increment = i * variableInteval[loopDepth];
            loopDepth += 1;
            slicedIdx += increment;
            performSlice<T>(originalData, slicedData, sliceMetric, variableInteval, loopDepth, originalIdx, slicedIdx);
            loopDepth -= 1;
            slicedIdx -= increment;
        }
    }
}

template <class T>
Tensor<T> *slice(Tensor<T> *input, std::vector<std::pair<int, int>> &sliceMetric)
{
    auto output = allocateTensor<T>(sliceMetric);

    std::vector<int> variableInteval;

    auto inputDimension = input->getDimension();

    long long index = 1;

    for (int i = inputDimension.size() - 1; i >= 0; i--)
    {
        variableInteval.push_back(index);
        index *= inputDimension[i];
    }

    reverse(variableInteval.begin(), variableInteval.end());

    int loopDepth = 0;
    int originalIdx = 0;
    int slicedIdx = 0;

    performSlice(input->getDataPointer(), output->getDataPointer(), sliceMetric, variableInteval, loopDepth, originalIdx, slicedIdx);
    
    return output;
}