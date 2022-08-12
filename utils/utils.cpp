#pragma once
#include <vector>
#include "tensor.hpp"


std::vector<int> getIntevalTable(std::vector<int> & dimension)
{
    std::vector<int> intevalTable;

    long long index = 1;

    for(int i = dimension.size() - 1; i >= 0; i--)
    {
        intevalTable.push_back(index);
        index *= dimension[i];
    }
    
    std::reverse(intevalTable.begin(), intevalTable.end());
    return intevalTable;
}


int getDiv(std::vector<int> & matAdim, std::vector<int> & matBdim)
{
    auto intevalTable = getIntevalTable(matAdim);


    for(auto pos = 0; pos < matAdim.size(); pos++)
    {
        if(matAdim[pos] != matBdim[pos])
        {
            if(pos == 0)
            {
                return -1;
            }
            return  intevalTable[pos - 1];
        }
    }
    // std::cout<<"here"<<std::endl;
    return 1;
}

int getMod(std::vector<int> & matAdim, std::vector<int> & matBdim)
{
    auto intevalTable = getIntevalTable(matAdim);


    for(auto pos = 0; pos < matAdim.size(); pos++)
    {
        if(matAdim[pos] != matBdim[pos])
        {
            return  intevalTable[pos];
        }
    }
    // std::cout<<"here"<<std::endl;
    return 1;
}

template <class T>
int getIdx(Tensor<T> * originalData, int dim)
{
    if(dim < 0)
    {
       return originalData->getDimension().size() + dim;
    }
    return dim;
}