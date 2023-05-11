#ifndef _MODEL_H
#define _MODEL_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvOnnxConfig.h>

#include "/home/luo/TensorRT-8.0.3.4/samples/common/buffers.h"
#include <memory>

#include "logging.h"
#include "common.hpp"
#include <cuda_runtime.h>
#include <queue>
using namespace std;


class Logger : public nvinfer1::ILogger           
{
    public:
        void log(Severity severity, const char* msg) noexcept override
        {
            // suppress info-level messages
            if (severity != Severity::kINFO)
                std::cout << msg << std::endl;
        }
};

struct Dimension
{
    std::vector<int> dimension;
};

class Model
{
    private:
        //logging
        Logger _gLogger;
        //Engine for inference
        shared_ptr<nvinfer1::ICudaEngine> _engine;
        //Context for running inference
        // shared_ptr<nvinfer1::IExecutionContext> m_context;
        queue<shared_ptr<nvinfer1::IExecutionContext>> _context_queue;
        //runtime
        shared_ptr<nvinfer1::IRuntime> _runtime;

        //NCHW
        std::vector<int> _input_indexes;
        std::vector<int> _output_indexes;
        std::vector<nvinfer1::Dims> _input_dimensions;
        std::vector<nvinfer1::Dims> _output_dimensions;

        //type


    public:
        Model();
        ~Model();
        static bool convert_onnx_to_trt_model(string in_onnx_file, string out_trt_file, bool fp16_mode=false, int dlaCore=-1);
        bool init(string trt_file, int context_num, int device_index);
        // string get_input_name();
        Dimension get_input_dimensions(int index = 0);
        // string get_output_name();
        Dimension get_output_dimensions(int index = 0);
        //get model context
        bool get_context(shared_ptr<nvinfer1::IExecutionContext>& context);
        //get dim size
        size_t get_size_by_dim(Dimension& dim);
        //get input number
        size_t get_input_num();
        size_t get_input_index(int index);
        //get output number
        size_t get_output_num();
        size_t get_output_index(int index);
        //get
        size_t get_nbs();
        //get data type
        DataType get_data_type(int index);
        //print info
        void print_info();
};

#endif