#pragma once
#include <vector>
#include <cmath>
#include "../utils/tensor.hpp"

#include "exp.cpp"
#include "matdiv.cpp"
#include "matadd.cpp"

// y5=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))##np.tanh(x)

template <class T>
Tensor<T> * tanh(Tensor<T> * data)
{

    auto expX = exp<T>(data);
    // expX->showRawData();
    auto negX = matdiv<T>(data, -1, true);
    auto expnegX = exp<T>(negX);
    auto numerator = matadd<T>(expX, expnegX, true);
    auto denominaror = matadd<T>(expX, expnegX, false);
    auto ans = matdiv<T>(numerator, denominaror);
    return ans;
}
