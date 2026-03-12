cmake_minimum_required(VERSION 3.10)
project(my_engine_binding)

set(TRT_LLM_ROOT "/home/pi/kx/TensorRT-LLM" CACHE PATH "Path to TensorRT-LLM root directory")
message(STATUS "[Internal] Using TRT_LLM_ROOT: ${TRT_LLM_ROOT}")


# 1. 自动寻找 Python 和 Pybind11
find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

find_library(TRT_LLM_LIB
    NAMES tensorrt_llm
    PATHS "${TRT_LLM_ROOT}/tensorrt_llm/libs"
    REQUIRED
)
find_library(TRT_LLM_PLUGIN_LIB 
    NAMES nvinfer_plugin_tensorrt_llm
    PATHS "${TRT_LLM_ROOT}/cpp/build/tensorrt_llm/plugins"
    NO_DEFAULT_PATH
)

link_directories(
    /usr/local/cuda/lib64
)



# 2. 设置包含路径 (让编译器找到你的头文件)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include 
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/tokenizers-cpp/include
    ${TRT_LLM_ROOT}/cpp/include    
)

# 3. 定义编译目标 (pybind11_add_module 会自动帮你生成 .so)
pybind11_add_module(my_engine_binding bindings.cpp)
# 4. 关键点：链接你的核心推理库和 TensorRT-LLM 依赖
target_link_libraries(my_engine_binding PRIVATE 
    trtcppinference  # 你写好的 C++ 核心静态/动态库
    ${OpenCV_LIBS}
    tokenizers_cpp
    nvinfer
    cudart
    ${TRT_LLM_LIB}
    ${TRT_LLM_PLUGIN_LIB}
)


# 设置输出名，确保它符合 Python 的导入规范
set_target_properties(my_engine_binding PROPERTIES PREFIX "")