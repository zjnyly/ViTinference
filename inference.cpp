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
    
    // ans->showRawData();
    
    auto lables = ans->getDataPointer();

    std::cout<<"我猜是";

    if(lables[0] > lables[1])
    {
        std::cout<<"猫"<<std::endl;
    }
    else
    {
        std::cout<<"狗"<<std::endl;
    }

    
    return 0;
}






