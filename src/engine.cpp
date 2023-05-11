#include "engine.h"

// #define CHECK if()

Engine::Engine(){}

Engine::~Engine()
{
    for(int i = 0; i < this->_model->get_nbs(); i++)
    {
        try
        {
            cudaFree(this->_buffers[i]);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    }
    free(this->_buffers);

    if(this->_cuda_stream)
    {
        cudaStreamDestroy(this->_cuda_stream);
    }

}

bool Engine::init(Model* model)
{
    this->_model = model;
    // this->_buffers.reserve(model->get_nbs());
    this->_buffers = (void**)malloc(sizeof(void*) * model->get_nbs());
    if (model->get_context(this->_context))
    {
        throw std::runtime_error("context is empty\n");
    }
    auto cudaRet = cudaStreamCreate(&this->_cuda_stream);
    if(cudaRet != 0)
    {
        throw std::runtime_error("Unable to create cuda stream");
    }
    return false;
}

bool Engine::malloc_cuda_memory()
{
    //malloc input cuda memory
    for (int i = 0; i < this->_model->get_input_num(); i++)
    {
        auto dim = this->_model->get_input_dimensions(i);
        auto bindingSize = this->_model->get_size_by_dim(dim);
        auto index = this->_model->get_input_index(i);
        auto data_type = this->_model->get_data_type(index);
        switch (data_type)
        {
        case nvinfer1::DataType::kFLOAT:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(float));
            break;
        case nvinfer1::DataType::kHALF:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(short));
            break;
        case nvinfer1::DataType::kINT32:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(int));
            break;
        case nvinfer1::DataType::kINT8:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(char));
            break;
        default:
            break;
        }
    }

    //malloc output cuda memory
    for (int i = 0; i < this->_model->get_output_num(); i++)
    {
        auto dim = this->_model->get_output_dimensions(i);
        auto bindingSize = this->_model->get_size_by_dim(dim);
        auto index = this->_model->get_output_index(i);
        auto data_type = this->_model->get_data_type(index);
        switch (data_type)
        {
        case nvinfer1::DataType::kFLOAT:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(float));
            break;
        case nvinfer1::DataType::kHALF:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(short));
            break;
        case nvinfer1::DataType::kINT32:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(int));
            break;
        case nvinfer1::DataType::kINT8:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(char));
            break;
        default:
            break;
        }
    }
    return false;
}

bool Engine::malloc_input_cuda_memory()
{
    for (int i = 0; i < this->_model->get_input_num(); i++)
    {
        auto dim = this->_model->get_input_dimensions(i);
        auto bindingSize = this->_model->get_size_by_dim(dim);
        auto index = this->_model->get_input_index(i);
        auto data_type = this->_model->get_data_type(index);
        switch (data_type)
        {
        case nvinfer1::DataType::kFLOAT:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(float));
            break;
        case nvinfer1::DataType::kHALF:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(short));
            break;
        case nvinfer1::DataType::kINT32:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(int));
            break;
        case nvinfer1::DataType::kINT8:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(char));
            break;
        default:
            break;
        }
    }
    return false;
}

bool Engine::malloc_output_cuda_memory()
{
    for (int i = 0; i < this->_model->get_output_num(); i++)
    {
        auto dim = this->_model->get_output_dimensions(i);
        auto bindingSize = this->_model->get_size_by_dim(dim);
        auto index = this->_model->get_output_index(i);
        auto data_type = this->_model->get_data_type(index);
        switch (data_type)
        {
        case nvinfer1::DataType::kFLOAT:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(float));
            break;
        case nvinfer1::DataType::kHALF:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(short));
            break;
        case nvinfer1::DataType::kINT32:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(int));
            break;
        case nvinfer1::DataType::kINT8:
            cudaMalloc(&this->_buffers[index], bindingSize * sizeof(char));
            break;
        default:
            break;
        }
    }
    return false;
}

bool Engine::input_host_copy_to_device(void* input[])
{
    for(int i = 0; i < this->_model->get_input_num(); i++)
    {
        auto index = this->_model->get_input_index(i);
        auto dim = this->_model->get_input_dimensions(i);
        auto size = this->_model->get_size_by_dim(dim);
        cudaMemcpyAsync(this->_buffers[index], input[i], size, cudaMemcpyHostToDevice, this->_cuda_stream);
    }
    return false;
}

bool Engine::input_device_copy_to_host(void* input[])
{
    for(int i = 0; i < this->_model->get_input_num(); i++)
    {
        auto index = this->_model->get_output_index(i);
        auto dim = this->_model->get_input_dimensions(i);
        auto size = this->_model->get_size_by_dim(dim);
        cudaMemcpyAsync(input[i], this->_buffers[index], size, cudaMemcpyDeviceToHost, this->_cuda_stream);
    }
    return false;
}

bool Engine::output_host_copy_to_device(void* output[])
{
    for(int i = 0; i < this->_model->get_output_num(); i++)
    {
        auto index = this->_model->get_output_index(i);
        auto dim = this->_model->get_output_dimensions(i);
        auto size = this->_model->get_size_by_dim(dim);
        cudaMemcpyAsync(this->_buffers[index], output[i], size, cudaMemcpyDeviceToHost, this->_cuda_stream);
    }
    return false;
}

bool Engine::output_device_copy_to_host(void* output[])
{
    for(int i = 0; i < this->_model->get_output_num(); i++)
    {
        auto index = this->_model->get_output_index(i);
        auto dim = this->_model->get_output_dimensions(i);
        auto size = this->_model->get_size_by_dim(dim);
        cudaMemcpyAsync(output[i], this->_buffers[index], size, cudaMemcpyDeviceToHost, this->_cuda_stream);
    }
    return false;
}

size_t Engine::get_input_memory_size()
{
    size_t input_size = 0;
    for(int i = 0; i < this->_model->get_input_num(); i++)
    {
        auto index = this->_model->get_input_index(i);
        auto dim = this->_model->get_input_dimensions(i);
        auto size = this->_model->get_size_by_dim(dim);
        input_size += size;
    }
    return input_size;
}

size_t Engine::get_output_memory_size()
{
    size_t output_size = 0;
    for(int i = 0; i < this->_model->get_output_num(); i++)
    {
        auto index = this->_model->get_output_index(i);
        auto dim = this->_model->get_output_dimensions(i);
        auto size = this->_model->get_size_by_dim(dim);
        output_size += size;
    }
    return output_size;
}

bool Engine::inference()
{
    this->_context->enqueueV2(this->_buffers, this->_cuda_stream, nullptr);
    // this->_context->executeV2(this->_buffers.data());
    return false;

}

bool Engine::synchronize()
{
    auto status = cudaStreamSynchronize(this->_cuda_stream);
    if(status != 0)
    {
        std::cout << "Unable to synchronize cuda stream" << std::endl;
        return false;
    }
    return true;
}

bool Engine::get_input_ptr(void** ptr[])
{
    
    for(int i = 0; i < this->_model->get_input_num(); i++)
    {
        auto index = this->_model->get_input_index(i);
        *ptr[i] = this->_buffers[index];
    }
    return false;
}

bool Engine::get_input_ptr_by_index(void** ptr, int index)
{
    *ptr = this->_buffers[index];
    return false;
}

bool Engine::get_input_ptr_by_batch(void** ptr, int batch_index, int input_index)
{
    auto dim = this->_model->get_input_dimensions(input_index);
    auto index = this->_model->get_input_index(input_index);
    auto data_type = this->_model->get_data_type(index);
    auto batch_elem = dim.dimension[1] * dim.dimension[2] * dim.dimension[3];
    switch (data_type)
    {
        case nvinfer1::DataType::kFLOAT:
            *ptr = (float*)this->_buffers[index] + batch_elem * batch_index;
            break;
        case nvinfer1::DataType::kHALF:
            *ptr = (short*)this->_buffers[index] + batch_elem * batch_index;
            break;
        case nvinfer1::DataType::kINT32:
            *ptr = (int*)this->_buffers[index] + batch_elem * batch_index;
            break;
        case nvinfer1::DataType::kINT8:
            *ptr = (char*)this->_buffers[index] + batch_elem * batch_index;
            break;
    }
    return false;
}

bool Engine::get_output_ptr(void** ptr[])
{
    for(int i = 0; i < this->_model->get_output_num(); i++)
    {
        auto index = this->_model->get_output_index(i);
        *ptr[i] = this->_buffers[index];
    }
    return false;
}

bool Engine::get_output_ptr_by_index(void** ptr, int index)
{
    *ptr = this->_buffers[index];
    return false;
}

Dimension Engine::get_model_input_dim(int index)
{
    return this->_model->get_input_dimensions(index);
}