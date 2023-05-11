#ifndef _RESNET_H
#define _RESNET_H

#include "engine.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>

enum class Im_Type : int32_t
{
    //! 8-bit integer representing a quantized floating-point value.
    INT8 = 0,

    //! Signed 32-bit integer format.
    INT32 = 1,

    //! 32-bit floating point format.
    FLOAT32 = 2,

    //! IEEE 16-bit floating-point format.
    FLOAT16 = 3,

    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
    kBOOL = 4
};

struct ImageStruct
{
    //image width
    int width;
    //image height
    int height;
    //image channel
    int channel;
    //image data type
    Im_Type dataType;
    //data point
    vector<void*> data;
    //data number
    int dataNum;
};

class Resnet
{
    private:
    Engine _engine;
    
    public:
    Resnet(Model* model);
    ~Resnet();
    void* output_buffer;
    //This function takes in the frame and resizes it to fit the Resnet50 model. It also normalizes the image
    bool cpu_resize_and_normalize(ImageStruct& ImStr);
    bool gpu_resize_and_normalize(ImageStruct& ImStr);
    // bool gpu_reszie_and_normalize(void* gpu_ptr, )

    //This function calculates the softmax to find the most probable classes for the image
    bool calculateProbability(float* gpu_output, const nvinfer1::Dims& dims, int batchSize);

    //Gets the class name from the file that is associated with the image
    vector<std::string> getClassNames(const std::string& imagenet_classes);

    bool inference(ImageStruct& ImStr);
};

#endif