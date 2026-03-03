cmake_minimum_required(VERSION 3.10)
project(trtcppinference VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(TRT_LLM_ROOT "/home/pi/kx/TensorRT-LLM" CACHE PATH "Path to TensorRT-LLM root directory")
message(STATUS "[Internal] Using TRT_LLM_ROOT: ${TRT_LLM_ROOT}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "[Internal] Setting up Rust/Cargo toolchain for aarch64...")
    set(ENV{CC} "gcc")
    set(ENV{CXX} "g++")
    set(ENV{AR} "ar")
    set(ENV{CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER} "gcc")
endif()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

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

add_subdirectory(external/tokenizers-cpp)

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp" "src/*.cu")
add_library(${PROJECT_NAME} STATIC ${LIB_SOURCES})

target_link_directories(${PROJECT_NAME} PUBLIC /usr/local/cuda/lib64)

target_include_directories(${PROJECT_NAME}
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${CUDA_INCLUDE_DIRS}
        ${TRT_LLM_ROOT}/cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/tokenizers-cpp/include
)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC 
        nvinfer cudart tokenizers_cpp
        ${OpenCV_LIBS}
        ${TRT_LLM_LIB}
        ${TRT_LLM_PLUGIN_LIB}
)

add_executable(test_vlm_sync tests/test_vlm_sync.cpp)
target_link_libraries(test_vlm_sync PRIVATE ${PROJECT_NAME})

add_executable(test_vlm_async tests/test_vlm_async.cpp)
target_link_libraries(test_vlm_async PRIVATE ${PROJECT_NAME})