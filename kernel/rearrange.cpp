#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

#include "../utils/tensor.hpp"

long long calculateIndex();
template <class T>
void performRearrange(
    T * originalData, 
    T * rearrangedData, 
    std::vector<int> & originalDimension,
    std::vector<std::pair<std::string, int>> & variableIntevalTable,
    int & loopDepth,
    int & originalIdx,
    int & rearrangedIdx)
{
    if(loopDepth == originalDimension.size())
    {
        rearrangedData[rearrangedIdx] = originalData[originalIdx];
        originalIdx += 1;
        return;
    }

    for(int i = 0; i < originalDimension[loopDepth]; i++)
    {
        auto increment = i * variableIntevalTable[loopDepth].second;
        loopDepth += 1;
        rearrangedIdx += increment;
        performRearrange<T>(originalData, rearrangedData, originalDimension, variableIntevalTable, loopDepth, originalIdx, rearrangedIdx);
        loopDepth -= 1;
        rearrangedIdx -= increment;
    }
}


template <class T>
Tensor<T> * rearrange(
    Tensor<T> * originalData, 
    std::vector<std::pair<std::string, int>> & originalView, 
    std::vector<std::pair<std::string, int>> & rearrangedView,
    std::vector<int> & originalDimension, 
    std::vector<int> & rearrangedDimension)
{

    std::map<std::string, int> variableIntevalTable;
    long long inteval = 1;
    for(auto iter = rearrangedView.rbegin(); iter != rearrangedView.rend(); iter++)
    {
        auto varName = (*iter).first;
        variableIntevalTable[varName] = inteval;
        inteval *= (*iter).second;
    }


    for(auto i = 0; i < originalView.size(); i++)
    {
        originalView[i].second = variableIntevalTable[originalView[i].first];
    }

    auto rearrangedData = new Tensor<T>(rearrangedDimension);

    int loopDepth = 0;
    int originalIdx = 0;
    int rearrangedIdx = 0;

    performRearrange<T>(originalData->getDataPointer(), rearrangedData->getDataPointer(), originalDimension, originalView, loopDepth, originalIdx, rearrangedIdx);
    return rearrangedData;
}