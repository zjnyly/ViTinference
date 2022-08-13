#include <iostream>
#include <ctime>
#include "./layer/vit.cpp"

int main()
{
    // Fake input image, just one image per batch
    std::vector<int> inputDataDimension = {1, 3, 224, 224};
    auto inputData = new Tensor<double>(inputDataDimension, true);

    //////////////////////////////////
    auto ans = ViT<double>(inputData);
    //////////////////////////////////

    std::cout<<"猫  狗"<<std::endl;
    ans->showRawData();
    return 0;
}






