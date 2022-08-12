#pragma once
#include <vector>


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