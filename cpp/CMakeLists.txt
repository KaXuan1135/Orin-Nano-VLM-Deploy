cmake_minimum_required(VERSION 3.10)

project(cpp_multimodal_runner 
        VERSION 1.0.0 
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_library(NVINFER_LIB nvinfer)

set(TRT_LLM_ROOT "/home/pi/kx/TensorRT-LLM")
find_library(TRT_LLM_LIB
    NAMES tensorrt_llm
    PATHS "${TRT_LLM_ROOT}/tensorrt_llm/libs"
    REQUIRED
)

set(TRT_LLM_INC_DIR "${TRT_LLM_ROOT}/cpp/include")
include_directories(${TRT_LLM_INC_DIR})
link_directories(/usr/local/cuda/lib64)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu") # Customized Kernel

add_library(${PROJECT_NAME}Lib STATIC ${SOURCES} ${CUDA_SOURCES})

target_include_directories(${PROJECT_NAME}Lib
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CUDA_INCLUDE_DIRS}
        ${TRT_LLM_INC_DIR}
        /home/pi/kx/ffvlm/external/tokenizerscpp/include
)

target_link_libraries(${PROJECT_NAME}Lib 
    PUBLIC 
        ${OpenCV_LIBS}
        ${TRT_LLM_LIB}
        nvinfer
        cudart
        /home/pi/kx/TensorRT-LLM/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so
                
        # Temporary
        /home/pi/kx/ffvlm/build/external/tokenizerscpp/libtokenizers_cpp.a
        /home/pi/kx/ffvlm/build/external/tokenizerscpp/libtokenizers_c.a  # 加上这个作为底层支撑

)

add_executable(test_vlm_sync tests/test_vlm_sync.cpp)
target_link_libraries(test_vlm_sync PRIVATE ${PROJECT_NAME}Lib)

add_executable(test_vlm_async tests/test_vlm_async.cpp)
target_link_libraries(test_vlm_async PRIVATE ${PROJECT_NAME}Lib)