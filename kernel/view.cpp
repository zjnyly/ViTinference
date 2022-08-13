#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "../utils/tensor.hpp"
#include "matdiv.cpp"
#include "matadd.cpp"
#include "matpow.cpp"
#include "mean.cpp"


template <class T>
Tensor<T> * view(Tensor<T> * originalData, std::vector<int>view)
{  
    auto size = originalData->getDataSize();
    auto currentSize = 1;
    for(auto i = 0; i < view.size(); i++)
    {
        currentSize *= view[i];
    }
    currentSize = abs(currentSize);
    for(auto i = 0; i < view.size(); i++)
    {
        if(view[i] < 0)
        {
            view[i] = (int)size / currentSize;
        }
    }
    auto viewMat = new Tensor<T>(originalData->getDataPointer(), view);
    return viewMat;
}