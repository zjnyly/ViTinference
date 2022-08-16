#pragma once 
#include "module.h"

template <class T>
class vit : public module<T>
{
    public:
    vit(std::string preNode) 
    {
        std::cout<<preNode<<std::endl;
    }
    virtual std::vector<Tensor<T> *> forward(Tensor<T> *input);
};