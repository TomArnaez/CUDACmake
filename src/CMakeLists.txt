cmake_minimum_required(VERSION 3.29)

set(project_name "cudaLib")
project(${project_name} LANGUAGES CXX C CUDA)

###############################################################################
# CXX compiler setup
###############################################################################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")

###############################################################################
# Dependencies
###############################################################################

find_package(tl-expected REQUIRED)
find_package(CUDAToolkit REQUIRED)

###############################################################################
# Program
###############################################################################

add_library(cudaLib SHARED)

target_compile_options(cudaLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

target_sources(cudaLib
    PUBLIC
        externalcuda.hpp
        PRIVATE
        core.hpp
        cudahelpers.hpp
        core.cu
        corrections.hpp
        corrections.cu
        cudahelpers.cpp
        externalcuda.cu
        histogrameq.cu
)

target_include_directories(cudaLib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cudaLib tl::expected CUDA::cudart)
set_target_properties(cudaLib PROPERTIES CUDA_ARCHITECTURES native)