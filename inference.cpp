#include <iostream>
#include <string>
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>

#include "./layer/vit.cpp"

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

int main()
{
    test();
    // Fake input image, just one image per batch
    std::vector<int> inputDataDimension = {1, 3, 224, 224};
    auto inputData = new Tensor<double>(inputDataDimension, true);

    //////////////////////////////////
    auto ans = ViT<double>(inputData);
    //////////////////////////////////

    // ans->showRawData();

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
