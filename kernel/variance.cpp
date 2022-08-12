#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

#include "../utils/tensor.hpp"
#include "matdiv.cpp"
#include "matadd.cpp"
#include "matpow.cpp"
#include "mean.cpp"



template <class T>
Tensor<T> * variance(Tensor<T> * originalData, int dim)
{  
    auto originalDim = originalData->getDimension();
    if(dim < 0)
    {
        dim = originalDim.size() + dim;
    }
    
    // originalData->showRawData();
    auto X2 = matpow<T>(originalData, 2);
    // X2->showRawData();
    auto EX2 = mean<T>(X2, dim);
    // EX2->showRawData();
    auto EX = mean<T>(originalData, dim);
    // EX->showRawData();
    auto E2X = matpow<T>(EX, 2);
    // E2X->showRawData();
    auto biasdVar = matadd<T>(EX2 , E2X, true);
    // biasdVar->showRawData();
 
    auto unbiasedFactor = (double)(originalDim[dim] - 1)/(originalDim[dim]);

    // std::cout<<unbiasedFactor<<std::endl;
    auto unbiasedVar = matdiv<T>(biasdVar, unbiasedFactor);

    return unbiasedVar;

}