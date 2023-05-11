#ifndef _ENGINE_H
#define _ENGINE_H

#include "model.h"
using namespace std;

class Engine
{
    private:
        //Context for running inference
        shared_ptr<nvinfer1::IExecutionContext> _context;
        //Cuda stream incase asynchronous
        Model* _model;
        cudaStream_t _cuda_stream;
        // vector<void*> _buffers;
        void** _buffers;

    public:
        Engine();
        ~Engine();
        bool init(Model* model);
        bool malloc_cuda_memory();
        bool malloc_input_cuda_memory();
        bool malloc_output_cuda_memory();
        bool input_host_copy_to_device(void* input[]);
        bool input_device_copy_to_host(void* input[]);
        bool output_host_copy_to_device(void* output[]);
        bool output_device_copy_to_host(void* output[]);
        bool inference();
        bool synchronize();
        size_t get_input_memory_size();
        size_t get_output_memory_size();
        bool get_input_ptr(void** ptr[]);
        bool get_input_ptr_by_index(void** ptr, int index = 0);
        bool get_input_ptr_by_batch(void** ptr, int batch = 0, int index = 0);
        
        bool get_output_ptr(void** ptr[]);
        bool get_output_ptr_by_index(void** ptr, int index = 0);
        Dimension get_model_input_dim(int index);
};

#endif