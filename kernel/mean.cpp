#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

#include "../utils/tensor.hpp"
#include "matdiv.cpp"



template <class T>
void performMean(
    T *originalData,
    T *meanMatData,
    std::vector<int> &originalDimension,
    std::vector<int> &meanMatIntevalTable,
    int &loopDepth,
    int &originalIdx,
    int &meanMatIdx,
    int &dim)
{
    if (loopDepth == originalDimension.size())
    {
        meanMatData[meanMatIdx] += originalData[originalIdx];
        originalIdx += 1;

        return;
    }

    for (int i = 0; i < originalDimension[loopDepth]; i++)
    {
        auto increment = 0;
        if (loopDepth != dim)
        {
            if (loopDepth > dim)
            {
                increment = i * meanMatIntevalTable[loopDepth - 1];
            }
            else
            {
                increment = i * meanMatIntevalTable[loopDepth];
            }
        }

        loopDepth += 1;
        meanMatIdx += increment;
        performMean<T>(originalData, meanMatData, originalDimension, meanMatIntevalTable, loopDepth, originalIdx, meanMatIdx, dim);
        loopDepth -= 1;
        meanMatIdx -= increment;
    }
}

template <class T>
Tensor<T> *mean(Tensor<T> *originalData, int dim)
{
    if (dim < 0)
    {
        dim = originalData->getDimension().size() + dim;
    }

    auto originalDimension = originalData->getDimension();

    std::vector<int> meanMatDim;

    for (auto i = 0; i < originalDimension.size(); i++)
    {
        if (i != dim)
        {
            meanMatDim.push_back(originalDimension[i]);
        }
    }

    auto meanMat = new Tensor<T>(meanMatDim);

    auto meanMatIntevalTable = getIntevalTable(meanMatDim);

    int loopDepth = 0;
    int meanMatIdx = 0;
    int originalIdx = 0;

    performMean<T>(originalData->getDataPointer(), meanMat->getDataPointer(), originalDimension, meanMatIntevalTable, loopDepth, originalIdx, meanMatIdx, dim);

    return matdiv(meanMat, (T)originalDimension[dim], false);
}