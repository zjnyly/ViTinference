#pragma once
#include <vector>
#include <cmath>


#include "../utils/tensor.hpp"
#include "../utils/utils.cpp"

#include "../kernel/matpow.cpp"
#include "../kernel/matdiv.cpp"
#include "../kernel/matadd.cpp"
#include "../kernel/tanh.cpp"


// https://blog.csdn.net/w137093940/article/details/112756141
// https://blog.csdn.net/tianyunlinger/article/details/119728944
// https://pytorch.org/docs/stable/generated/torch.nn.GELU.html?highlight=gelu

// 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))

template <class T>
Tensor<T> * gelu(Tensor<T> * data)
{
    auto data1 = matpow<T>(data, 3);
    auto data2 = matdiv<T>(data1, 0.044715, true);
    auto data3 = matadd<T>(data2, data);
    auto coef = std::pow(2 / M_PI, 0.5);
    auto data4 = matdiv<T>(data3, coef, true);
    auto data5 = tanh<T>(data4);
    auto data6 = matadd<T>(data5, 1);
    auto data7 = matdiv<T>(data6, data, true);
    auto ans = matdiv<T>(data7, 0.5, true);
    return ans;
}