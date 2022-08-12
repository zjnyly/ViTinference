#pragma once 
#pragma once
#include <vector>
#include <algorithm>
#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"



template <class T>
void performRepeat(
    T * originalData, 
    T * repeatedData, 
    std::vector<int> & originalDimension,
    std::vector<int> & repeatDimension,
    std::vector<int> & originalIntevalTable,
    std::vector<int> & repeatIntevalTable,
    int & loopDepth,
    int & originalIdx,
    int & repeatedIdx)
{
    if(loopDepth == repeatDimension.size())
    {
        
        repeatedData[repeatedIdx] = originalData[originalIdx];
        // repeatedData[repeatedIdx];
        // originalData[originalIdx];
        // std::cout<<originalIdx<<std::endl;
        // std::cout<<repeatedIdx<<std::endl;


        return;
    }

    for(int i = 0; i < repeatDimension[repeatDimension.size() - 1 - loopDepth]; i++)
    {
        auto repeatIncrement = i * repeatIntevalTable[loopDepth];
        // std::cout<<i << " " << originalDimension[loopDepth]<< " "<<i % originalDimension[loopDepth]<<  std::endl;
        auto originalIncrement = (i % originalDimension[repeatDimension.size() - 1 - loopDepth]) * originalIntevalTable[loopDepth];

        loopDepth += 1;
        repeatedIdx += repeatIncrement;
        originalIdx += originalIncrement;
        performRepeat<T>(originalData, repeatedData, originalDimension, repeatDimension, originalIntevalTable, repeatIntevalTable, loopDepth, originalIdx, repeatedIdx);
        loopDepth -= 1;
        repeatedIdx -= repeatIncrement;
        originalIdx -= originalIncrement;
    }
}


template<class T>
Tensor<T> * Repeat(Tensor<T> * input, std::pair<int, int> & repeatAt)
{
    std::vector<int> variableInteval;


    auto originalDimension = input->getDimension();
    auto repeatDimension = originalDimension;

    repeatDimension[repeatAt.first] *= repeatAt.second;

    auto repeatedTensor = new Tensor<T>(repeatDimension);

    auto originalIntevalTable = getIntevalTable(originalDimension);
    auto repeatIntevalTable = getIntevalTable(repeatDimension);

    int loopDepth = 0;
    int originalIdx = 0;
    int repeatedIdx = 0;

    performRepeat(input->getDataPointer(), repeatedTensor->getDataPointer(), originalDimension, repeatDimension, originalIntevalTable, repeatIntevalTable, loopDepth, originalIdx, repeatedIdx);

//     performRepeat(
//     T * originalData, 
//     T * repeatedData, 
//     std::vector<int> & originalDimension,
//     std::vector<int> & repeatDimension,
//     std::vector<int> & originalIntevalTable,
//     std::vector<int> & repeatIntevalTable,
//     int & loopDepth,
//     int & originalIdx,
//     int & repeatedIdx)
// {

}