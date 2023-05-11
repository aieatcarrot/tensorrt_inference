#include "model.h"

Model::Model(){}

Model::~Model(){}

bool Model::convert_onnx_to_trt_model(string in_onnx_file, string out_trt_file, bool fp16_mode, int dlaCore)
{
    printf("Starting builder...\n");
    fflush(stdout);
    Logger  gLogger;
    ifstream f(in_onnx_file.c_str());
    if(!f.good())
    {
        cout << in_onnx_file <<" not exists..." << endl;
        return true;
    }

        // Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    if (!builder)
    {
        std::string msg = "Failed to create builder";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        return false; 
    }

    // Create network definition
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!network)
    {
        std::string msg = "Failed create network definition";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        builder->destroy();
        return false; 
    }

    // Create ONNX parser and parse network
    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser || !parser->parseFromFile(in_onnx_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR)))
    {
        std::string msg = "Failed to parse onnx file";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    // Create engine from parsed network
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // Maximum size that any layer in network can use
    config->setMaxWorkspaceSize(1ULL << 30);
    
    
    if(fp16_mode)
        config->setFlag(BuilderFlag::kFP16);
    
    if(dlaCore == 0 || dlaCore == 1)
    {
        config->setFlag(BuilderFlag::kGPU_FALLBACK);
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(dlaCore);
    }
        
    // Set max batch size to one image
    auto dim = network->getInput(0)->getDimensions();
    builder->setMaxBatchSize(dim.d[0]);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        std::string msg = "Failed to create engine from parsed network";
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        parser->destroy();
        network->destroy();
        config->destroy();
        builder->destroy();
        return false;
    }

    // Serialize engine and store it to file
    unique_ptr<IHostMemory> serialized_engine{builder->buildSerializedNetwork(*network, *config)};
    cout << "Writing serialized model to disk..." << endl;
    //write the engine to disk
    ofstream outfile(out_trt_file.c_str(), ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    outfile.close();
    cout << "The engine has been built and saved to disk successfully" << endl;

    // Free memory (can be deleted after creation of engine)
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    // Free memory
    engine->destroy();
    printf("%d\n", dim.d[0]);
    return true;
}


bool Model::init(string trt_file, int context_num, int device_index)
{    
    // Create runtime
    this->_runtime = shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(this->_gLogger));
    if (!this->_runtime)
    {
        std::string msg = "Failed to create runtime";
        this->_gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        return false;
    }

    auto ret = cudaSetDevice(device_index);
    //Checks if device index exists. Should only be one device for this program. Might test with more at a later time
    if(ret != 0)
    {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(device_index) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Load data from file
    std::ifstream p(trt_file);  
    p.seekg( 0, std::ios::end );  
    size_t model_size = p.tellg();  
    char *model_data = new char[model_size];  
    p.seekg(0, std::ios::beg);   
    p.read(model_data, model_size);  
    p.close();

    // Create engine
    this->_engine = shared_ptr<nvinfer1::ICudaEngine>(this->_runtime->deserializeCudaEngine(model_data, model_size, nullptr));
    if (!this->_engine)
    {
        std::string msg = "Failed to create engine";
        this->_gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        this->_runtime->destroy();
        this->_runtime = nullptr;
        delete model_data;
        return false;
    }

    // Get input and output buffer indexes and dimensions
    for (size_t i = 0; i < this->_engine->getNbBindings(); i++)
    {
        if (this->_engine->bindingIsInput(i))
        {
            this->_input_dimensions.emplace_back(this->_engine->getBindingDimensions(i));
            this->_input_indexes.emplace_back(i);
        }
        else
        {
            this->_output_dimensions.emplace_back(this->_engine->getBindingDimensions(i));
            this->_output_indexes.emplace_back(i);
        }
    }

    //Get inpuot and output name
    // this->_engine->getBindingName()
    for (int i = 0; i < context_num; i++)
    {
        this->_context_queue.push(shared_ptr<nvinfer1::IExecutionContext>(this->_engine->createExecutionContext()));
    }

    delete model_data;
    print_info();
    return true;
}

void Model::print_info()
{
    std::cout << "==================================================" << endl;
    std::cout << "input num:" << this->_input_indexes.size() << endl;
    for(int i = 0; i < this->_input_indexes.size(); i++)
    {
        string type;
        auto datatype = this->_engine->getBindingDataType(this->get_input_index(i));
        switch (datatype)
        {
        case nvinfer1::DataType::kFLOAT:
            type = "FLOAT32";
            break;
        case nvinfer1::DataType::kHALF:
            type = "FLOAT16";
            break;
        case nvinfer1::DataType::kINT32:
            type = "INT32";
            break;
        case nvinfer1::DataType::kINT8:
            type = "INT8";
            break;
        default:
            break;
        }
        
        std::cout << "input:"<< i 
                <<"  batch size:" << this->_input_dimensions[i].d[0] 
                <<"  channel:" << this->_input_dimensions[i].d[1] 
                <<"  width:" << this->_input_dimensions[i].d[2] 
                <<"  height:" << this->_input_dimensions[i].d[3]
                <<"  datatype:" <<  type << endl;
    }
    std::cout << "==================================================" << endl;
    std::cout << "output num:" << this->_output_indexes.size() << endl;
    for(int i = 0; i < this->_input_indexes.size(); i++)
    {
        string type;
        auto datatype = this->_engine->getBindingDataType(this->get_output_index(i));
        switch (datatype)
        {
        case nvinfer1::DataType::kFLOAT:
            type = "FLOAT32";
            break;
        case nvinfer1::DataType::kHALF:
            type = "FLOAT16";
            break;
        case nvinfer1::DataType::kINT32:
            type = "INT32";
            break;
        case nvinfer1::DataType::kINT8:
            type = "INT8";
            break;
        default:
            break;
        }
        std::cout << "output:"<< i 
                <<"  batch size:" << this->_output_dimensions[i].d[0] 
                <<"  channel:" << this->_output_dimensions[i].d[1] 
                <<"  width:" << this->_output_dimensions[i].d[2] 
                <<"  height:" << this->_output_dimensions[i].d[3]
                <<"  datatype:" <<  type << endl;
    }
    std::cout << "==================================================" << endl;
}

Dimension Model::get_input_dimensions(int index)
{
    Dimension input_dimension;
    for (std::size_t j = 0; j < this->_input_dimensions[index].nbDims; j++)
    {
        input_dimension.dimension.emplace_back(this->_input_dimensions[index].d[j]);
    }
    return input_dimension;
}

Dimension Model::get_output_dimensions(int index)
{
    Dimension output_dimension;
    for (std::size_t j = 0; j < this->_output_dimensions[index].nbDims; j++)
    {
        output_dimension.dimension.emplace_back(this->_output_dimensions[index].d[j]);
    }
    return output_dimension;
}

bool Model::get_context(shared_ptr<nvinfer1::IExecutionContext>& context)
{
    if(this->_context_queue.empty())
    {
        return true;
    }
    context = this->_context_queue.front();
    this->_context_queue.pop();
    return false;
}

size_t Model::get_size_by_dim(Dimension& dim)
{
    size_t size = 1;
    for (size_t i = 0; i < dim.dimension.size(); ++i)
    {
        size *= dim.dimension[i];
    }
    return size;
}

size_t Model::get_input_num()
{
    return this->_input_indexes.size();
}

size_t Model::get_input_index(int index)
{
    return this->_input_indexes[index];
}

size_t Model::get_output_num()
{
    return this->_output_indexes.size();
}

size_t Model::get_output_index(int index)
{
    return this->_output_indexes[index];
}

size_t Model::get_nbs()
{
    return this->_engine->getNbBindings();
}

DataType Model::get_data_type(int index)
{
    this->_engine->getBindingDataType(index);
}