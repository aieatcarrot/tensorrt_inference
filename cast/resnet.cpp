#include "resnet.h"

Resnet::Resnet(Model* model)
{
    this->_engine.init(model);
    this->_engine.malloc_cuda_memory();
    this->output_buffer = malloc(this->_engine.get_output_memory_size() * sizeof(float));
    void* gpu_input = nullptr;
    this->_engine.get_input_ptr_by_index(&gpu_input, 0);
}

Resnet::~Resnet()
{
    free(this->output_buffer);
}

bool Resnet::cpu_resize_and_normalize(ImageStruct& ImgStr)
{
    // cv::cuda::GpuMat gpu_frame;
    // cv::Mat frame(cv::Size(ImgStr.width, ImgStr.height), CV_8UC3, ImgStr.data[0]);
    // gpu_frame.upload(frame);

    // auto input_width = dims.d[3];
    // auto input_height = dims.d[2];
    // auto channels = dims.d[1];
    // auto input_size = cv::Size(input_width, input_height);
    // // resize
    // cv::cuda::GpuMat resized;
    // cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    // // normalize
    // cv::cuda::GpuMat flt_image;
    // resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    // cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    // cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    // // to tensor
    // std::vector<cv::cuda::GpuMat> chw;
    // for (size_t i = 0; i < channels; ++i)
    // {
    //     chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    // }
    // cv::cuda::split(flt_image, chw);
    // return true;
}


bool Resnet::gpu_resize_and_normalize(ImageStruct& ImgStr)
{
    auto dims = this->_engine.get_model_input_dim(0);
    auto input_width = dims.dimension[3];
    auto input_height = dims.dimension[2];
    auto channels = dims.dimension[1];
    auto input_size = cv::Size(input_width, input_height);
    for(int i = 0; i < ImgStr.dataNum; i++)
    {
        cv::cuda::GpuMat gpu_frame(cv::Size(ImgStr.width, ImgStr.height), CV_8UC3, ImgStr.data[i]);
        if(ImgStr.width != input_width or ImgStr.height != input_height)
        {
            // resize
            cv::cuda::resize(gpu_frame, gpu_frame, input_size, 0, 0, cv::INTER_NEAREST);
        }
        // normalize
        cv::cuda::GpuMat flt_image;
        gpu_frame.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
        cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
        cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
        // to tensor
        std::vector<cv::cuda::GpuMat> chw;
        void* gpu_input = nullptr;
        
        // this->_engine.get_input_ptr_by_batch(&gpu_input, i, 0);
        this->_engine.get_input_ptr_by_index(&gpu_input, 0);
        // this->_engine.get_input_ptr(&gpu_input);
        for (size_t j = 0; j < channels; ++j)
        {
            chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, (float*)gpu_input + j * input_width * input_height));
        }
        cv::cuda::split(flt_image, chw);
    }
    
    return true;
}

bool Resnet::calculateProbability(float* gpu_output, const nvinfer1::Dims& dims, int batchSize)
{

    return true;
}

bool Resnet::inference(ImageStruct& ImStr)
{   
    gpu_resize_and_normalize(ImStr);
    this->_engine.inference();
    this->_engine.output_device_copy_to_host(&this->output_buffer);
    this->_engine.synchronize();
}