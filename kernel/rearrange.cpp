#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"


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
void performRearrangeLoop(
    T * originalData, 
    T * rearrangedData, 
    std::vector<int> & originalDimension,
    std::vector<std::pair<std::string, int>> & variableIntevalTable,
    int size)
{
    auto originalIntevalTable = getIntevalTable(originalDimension);
    int dimsize = originalDimension.size() - 1;

    for(int i = 0; i < size; i++)
    {
        int  rearrangedIdx = 0;
        int  originalIdx = i;
        // #pragma UNROLL(10)
        for(int j =  dimsize; j >= 0; j--)
        {
            // std::cout<<j<<" "<<(originalIdx % originalDimension[j])<< " "<< variableIntevalTable[j].second<<std::endl;
            rearrangedIdx += (originalIdx % originalDimension[j]) * variableIntevalTable[j].second;
            originalIdx = originalIdx / originalDimension[j];
        }
        rearrangedData[rearrangedIdx] = originalData[i];
        // originalData[i];

        // std::cout<<i<< " " <<rearrangedIdx<<std::endl;
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

    // for(auto i = 0; i < originalView.size(); i++)
    // {
    //     std::cout<<originalView[i].second<<std::endl;
    // }

    auto rearrangedData = new Tensor<T>(rearrangedDimension);

    int loopDepth = 0;
    int originalIdx = 0;
    int rearrangedIdx = 0;

    
    clock_t start,end;
    start=clock();
    performRearrangeLoop<T>(originalData->getDataPointer(), rearrangedData->getDataPointer(), originalDimension, originalView, originalData->getDataSize());
    // performRearrange<T>(originalData->getDataPointer(), rearrangedData->getDataPointer(), originalDimension, originalView, loopDepth, originalIdx, rearrangedIdx);
    end = clock();
    double endtime=(double)(end-start)/CLOCKS_PER_SEC;

	// std::cout<<"Total time:"<<endtime * 1000<<"ms"<<std::endl;

    return rearrangedData;
}