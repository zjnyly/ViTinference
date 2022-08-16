#pragma once
#include <vector>
#include <map>
#include <string>
#include "../utils/tensor.hpp"

template <class T>
class module
{
public:
    module<T>()
    {
        std::cout<<"created"<<std::endl;
    }
    virtual std::string getName(){};
    virtual std::vector<Tensor<T> *> forward(Tensor<T> *input){};

private:
    std::map<std::string, module<T> *> net;
    std::map<std::string, Tensor<T> *> data;
    std::string name;
};
