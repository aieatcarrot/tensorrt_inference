#include <getopt.h>
#include <opencv2/opencv.hpp>
#include "resnet.h"
#include <chrono>
#include <thread>
typedef std::chrono::high_resolution_clock Clock;


void trtInference(Resnet* resnet, int numberOfIterations)
{
    ImageStruct imgStr;
    imgStr.width = 224;
    imgStr.height = 224;
    imgStr.channel = 3;
    imgStr.dataNum = 1;
    imgStr.dataType = Im_Type::INT8;
    imgStr.data.reserve(1);
    cudaMalloc(&imgStr.data[0], 224 * 224 * 3 * sizeof(char));
    for(int i = 0; i < numberOfIterations; i++)
    {
        resnet->inference(imgStr);
    }
}

int main(int argc, char **argv)
{    
    //Set configurations
    string trt_path;
    string onnx_path;
    int numberOfIterations = 1;
    int threadNum = 1;
    //Precision
    int i;
    bool conv_model = false;
    while((i = getopt(argc, argv, "p:d:s:h:n:m:t:c:o:")) != -1)
    {
        switch (i)
        {
            case 'p':
                printf("Precision\n");
                //printf("%s\n", optarg);
                //fflush(stdout);
                // set_precision(optarg, config);
                // printf("fp16: %d    |   ", config.FP16);
                // printf("INT8: %d    |   \n", config.INT8);
                break;
            case 'd':
                printf("DLA\n");
                fflush(stdout);
                // set_dla(atoi(optarg), config);
                // printf("DLA cores %d\n", config.dlaCore);
                break;
            case 's':
                printf("Workspace size");
                // config.maxWorkspaceSize = atoi(optarg);
                break;
            case 'h':
                print_help();
                break;
            case 'n':
                printf("%d\n", atoi(optarg));
                numberOfIterations = atoi(optarg);
                break;
            case 'm':
                trt_path = optarg;
                // strcpy(trt_path, optarg);
                printf("%s\n", trt_path.c_str());
                break;
            case 't':
                threadNum = atoi(optarg);
                printf("work thread %d\n", threadNum);
                break;
            case 'c':
                conv_model = atoi(optarg);
                break;
            case 'o':
                onnx_path = optarg;
                printf("%s\n", onnx_path.c_str());
        }
    }
    if(conv_model)
    {
        Model::convert_onnx_to_trt_model(onnx_path, trt_path);
    }
    
    Model model;
    model.init(trt_path, threadNum, 0);
    ImageStruct is;
    vector<thread*> threadVec;
    vector<Resnet*> resnetVec;
    for(int i = 0; i < threadNum; i++)
    {
        Resnet* resnet = new Resnet(&model);
        thread* th = new thread(trtInference, resnet, numberOfIterations);
        threadVec.emplace_back(th);
        resnetVec.emplace_back(resnet);
    }

    auto t1 = Clock::now();
    auto t2 = Clock::now();
    t1 = Clock::now();
    for(int i = 0; i < threadNum; i++)
    {
        threadVec[i]->join();
        delete resnetVec[i];
        delete threadVec[i];
    }
    t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
    cout << "Success! Average time per inference on "<< numberOfIterations * 2 <<" was " << totalTime / (numberOfIterations*2) << "ms" << endl;
}