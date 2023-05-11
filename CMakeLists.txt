# set the project name
project("Yolov engine")
cmake_minimum_required(VERSION 3.10)

#links libraries
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(TensorRT_DIR "/home/luo/TensorRT-8.0.3.4")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package( OpenCV 4.6.0 REQUIRED )
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")
message(${OpenCV_DIR})
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	# -gencode arch=compute_53,code=sm_53
	-gencode arch=compute_61,code=sm_61
)

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}; 
		-gencode arch_72,code=sm_72
		#-gencode arch_86,code=sm_86 Uncomment for 8.6 architectures
	)
endif()

# specify the executable and corresponding source file
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)  #This seems to have added the cuda_runtime api.
include_directories(${TensorRT_DIR}/include)
include_directories(${TensorRT_DIR}/samples/common)
include_directories(${OpenCV_INCLUDE_DIRS})		
include_directories(${PROJECT_SOURCE_DIR}/include)	
include_directories(${PROJECT_SOURCE_DIR}/cast)	

link_directories(${TensorRT_DIR}/lib)
#add source file
file(GLOB_RECURSE SRC_DIR_LIST "src/*.cpp" "src/*.c" "cast/*.cpp")	#遍历获取source/路径下所有的*.cpp和*.c文件列表
add_executable(engine main.cpp ${SRC_DIR_LIST})

target_link_libraries(engine pthread)
target_link_libraries(engine cudart)
target_link_libraries(engine nvparsers)
target_link_libraries(engine nvonnxparser)
target_link_libraries(engine nvinfer)
target_link_libraries(engine ${OpenCV_LIBS})
