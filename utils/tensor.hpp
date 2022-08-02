#pragma once
#include <vector>
#include <algorithm>

template <class T>
class Tensor
{
public:
    Tensor(std::vector<int> & _dimension)
    {
        // store the dimension
        this->dimension = _dimension;

        // for(int i = 0; i < _dimension.size(); i++)
        // {
        //     this->dimension[i] = _dimension[i];
        // }

        // calculate the size
        for(auto & dim : _dimension)
        {
            this->size *= dim;
        }

        // allocate the data buffer for data
        this->data = new T[this->size];
    }

    Tensor(std::vector<T> & _data, std::vector<int> & _dimension)
    {
        // store the dimension
        this->dimension = _dimension;

        // calculate the size
        for(auto & dim : _dimension)
        {
            this->size *= dim;
        }

        // allocate the data buffer for data
        this->data = new T[this->size];

        for(int i = 0; i < size; i++)
        {
            this->data[i] = _data[i];
        }
    }

    T * getDataPointer()
    {
        return this->data;
    }

    long long getDataSize()
    {
        return this->size;
    }

    std::vector<int> getDimension()
    {
        return this->dimension;
    }

    void showDimension()
    {
        std::cout<<"[";
        for(int i = 0; i < this->dimension.size() - 1; i++)
        {
            std::cout<<" "<<this->dimension[i]<<",";
        }
        std::cout<<" "<<this->dimension[this->dimension.size() - 1];
        std::cout<<"]"<<std::endl;
    }

    void showData()
    {
        std::cout<<"HEAD>>>"<<std::endl;
        for(int i = 0; i < std::min(this->size, 100ll); i++)
        {
            std::cout<<this->data[i]<< " ";
        }
        std::cout<<std::endl;
        std::cout<<"TAIL<<<"<<std::endl;
        for(int i = this->size - 1; i > std::max(this->size - 1 - 100, 0ll); i--)
        {
            std::cout<<this->data[i]<< " ";
        }
        std::cout<<std::endl;
    }


private:
    T * data = nullptr;
    std::vector<int> dimension;
    long long size = 1;
};