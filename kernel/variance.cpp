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
    
    auto X2 = matpow<T>(originalData, 2);
    auto EX2 = mean<T>(X2, dim);
    auto EX = mean<T>(originalData, dim);
    auto E2X = matpow<T>(EX, 2);
    auto biasdVar = matadd<T>(EX2 , E2X, true);
    auto unbiasedFactor = (double)(originalDim[dim] - 1)/(originalDim[dim]);
    auto unbiasedVar = matdiv<T>(biasdVar, unbiasedFactor, false);
    return unbiasedVar;

}