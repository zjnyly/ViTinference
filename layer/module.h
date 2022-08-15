#pragma once
#include <vector>
#include <map>
#include <string>
#include "../utils/tensor.hpp"


    template <class T>
    class module
    {
    public:
        virtual module();
        virtual vector<Tensor<T> *> forward(Tensor<T> *input) = 0;

    private:
        std::map <std::string, module<T> *> net;
        std::map <std::string, Tensor<T> *> data;
    };

