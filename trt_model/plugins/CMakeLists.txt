cmake_minimum_required(VERSION 3.18)
project(requant_plugin LANGUAGES CXX CUDA)

# ---- Paths (edit if needed) ----
set(TENSORRT_ROOT "E:/NVIDIA GPU Computing Toolkit/TensorRT-10.8.0.43" CACHE PATH "TensorRT root")
set(CUDA_ROOT "E:/NVIDIA GPU Computing Toolkit/CUDA/v12.8" CACHE PATH "CUDA root")

# ---- Build options ----
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set your GPU arch if needed (e.g. 86 for RTX 30xx, 89 for RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA arch")

add_library(requant_plugin SHARED
    custom_op.cpp
    custom_op.cu
)

set_target_properties(requant_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "requant_plugin"
)

target_include_directories(requant_plugin PRIVATE
    "${TENSORRT_ROOT}/include"
    "${CUDA_ROOT}/include"
)

# Link directories
set(TRT_LIB_DIR "${TENSORRT_ROOT}/lib")
if (EXISTS "${TENSORRT_ROOT}/lib/x64")
    set(TRT_LIB_DIR "${TENSORRT_ROOT}/lib/x64")
endif()

target_link_directories(requant_plugin PRIVATE
    "${TRT_LIB_DIR}"
    "${CUDA_ROOT}/lib/x64"
)

# Link libraries
target_link_libraries(requant_plugin PRIVATE
    nvinfer_10
    nvinfer_plugin_10
    cudart
)
