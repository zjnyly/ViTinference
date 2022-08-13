#pragma once
#include <vector>
#include <cstdio>
#include <cstring>
#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"


template <class T>
void performConcat(
    T * originalData, 
    T * concatedData,
    std::vector<int> & originalDim,
    std::vector<std::pair<int, int>> & concatDim,
    std::vector<int> & concatIntevalTable,
    int & loopDepth,
    int & originalIndex,
    int & concatIndex)
{
    if(loopDepth == originalDim.size())
    {
        concatedData[concatIndex] = originalData[originalIndex];
        originalIndex++;
        return;
    }


    for(int i = concatDim[loopDepth].first; i < concatDim[loopDepth].second; i++)
    {
        auto concatIncrement = i * concatIntevalTable[loopDepth];
        loopDepth += 1;
        concatIndex += concatIncrement;
        performConcat<T>(originalData, concatedData, originalDim, concatDim, concatIntevalTable, loopDepth, originalIndex, concatIndex);
        loopDepth -= 1;
        concatIndex -= concatIncrement;
    }
}



template <class T>
void performSimpleConcat(Tensor<T> *matA, Tensor<T> *matB,  Tensor<T> *concatTensor)
{
    auto concatTensorPointer = concatTensor->getDataPointer();
    auto matAPointer = matA->getDataPointer();
    auto matBPointer = matB->getDataPointer();

    auto matASize = matA->getDataSize();
    auto matBSize = matB->getDataSize();


    memcpy((void *)concatTensorPointer, (void *)matAPointer, sizeof( T ) * matASize);
    memcpy((void *)(concatTensorPointer + matASize), (void *)matBPointer, sizeof( T ) * matBSize);
}



template <class T>
Tensor<T> * concat(Tensor<T> *matA, Tensor<T> *matB, int dim)
{
    auto matADim = matA->getDimension();
    auto matBDim = matB->getDimension();

    std::vector<int> newMatDim = matADim;

    std::vector<std::pair<int, int>> concatDimGap;

    for(int i = 0; i < matADim.size(); i++)
    {
        concatDimGap.push_back({0, matADim[i]});
    }


    newMatDim[dim] += matBDim[dim];

    auto concatTensor = new Tensor<T>(newMatDim);
    auto concatDim = concatTensor->getDimension();

    auto concatIntevalTable = getIntevalTable(concatDim);


    int loopDepth = 0;
    int originalIdx = 0;
    int concatIdx = 0;
    performConcat<double>(
        matA->getDataPointer(), 
        concatTensor->getDataPointer(), 
        matADim, 
        concatDimGap, 
        concatIntevalTable, 
        loopDepth, 
        originalIdx, 
        concatIdx);


    concatDimGap[dim].first = matADim[dim];
    concatDimGap[dim].second = newMatDim[dim];


    loopDepth = 0;
    originalIdx = 0;
    concatIdx = 0;
    performConcat<double>(
        matB->getDataPointer(), 
        concatTensor->getDataPointer(), 
        matBDim, 
        concatDimGap, 
        concatIntevalTable, 
        loopDepth, 
        originalIdx, 
        concatIdx);

    return concatTensor;


    
}
