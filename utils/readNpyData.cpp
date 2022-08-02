#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "tensor.hpp"

template <class T>
Tensor<T> * readNpyData(std::string name, std::vector<int> & dimension)
{

    auto weight = new Tensor<T>(dimension);

    std::ifstream in("./weights/" + name, std::ios::in | std::ios::binary);

    in.read((char * )weight->getDataPointer(),  sizeof( T ) * weight->getDataSize());

    in.close();

    return weight;
}




