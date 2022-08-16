#include <iostream>
#include <string>
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>

#include "./layer/vit.cpp"
#include "./layer/vit.h"
#include "./layer/module.h"

using namespace std;

void test()
{
    std::vector<int> inputDataDimension = {1, 3, 224, 224};
    for (auto i = 0; i < 100; i++)
    {
        string path = "./test/data/" + to_string(i);
        cout<<path<<endl;
        auto inputData = loadImage<double>(path, inputDataDimension);
        auto ans = ViT<double>(inputData);

        auto lables = ans->getDataPointer();

        std::cout << "我猜是";

        if (lables[0] > lables[1])
        {
            std::cout << "猫" << std::endl;
        }
        else
        {
            std::cout << "狗" << std::endl;
        }
    }
}

void test2()
{
    std::vector<int> inputDataDimension = {1, 3, 4, 4};
    auto inputData = new Tensor<double>(inputDataDimension, true);
    std::vector<std::pair<std::string, int>> originalView = {{"b", 1}, {"c", 3}, {"h", 2}, {"p1", 2}, {"w", 2}, {"p2", 2}};
    std::vector<int> originalDimension = {1, 3, 2, 2, 2, 2};
    std::vector<std::pair<std::string, int>> rearrangedView = {{"b", 1}, {"h", 2}, {"w", 2}, {"p1", 2}, {"p2", 2}, {"c", 3}};
    std::vector<int> rearrangedDimension = {1, 4, 12};
    auto rearrangedData = rearrange<double>(inputData, originalView, rearrangedView, originalDimension, rearrangedDimension);
//     std::vector<std::pair<int, int>> sliceMetric = {{0, 0}, {0, 49}, {0, 3072}};
//     auto out_1 = slice(rearrangedData, sliceMetric);
}

int main()
{
    //  module<double>();
    // vit<double>("test");
    // test2();
    // while(1);
    // Fake input image, just one image per batch
    std::vector<int> inputDataDimension = {1, 3, 224, 224};
    auto inputData = new Tensor<double>(inputDataDimension, true);

    //////////////////////////////////
    auto ans = ViT<double>(inputData);
    //////////////////////////////////

    ans->showRawData();

    auto lables = ans->getDataPointer();

    std::cout << "我猜是";

    if (lables[0] > lables[1])
    {
        std::cout << "猫" << std::endl;
    }
    else
    {
        std::cout << "狗" << std::endl;
    }

    return 0;
}
