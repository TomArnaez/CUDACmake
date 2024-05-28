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

###############################################################################
# Dependencies
###############################################################################

find_package(tl-expected REQUIRED)
find_package(CUDAToolkit REQUIRED)

###############################################################################
# Program
###############################################################################

add_library(cudaLib SHARED)

target_sources(cudaLib
    PUBLIC
        core.hpp
        cudahelpers.hpp
        externalcuda.hpp
    PRIVATE
        core.cpp
        corrections.hpp
        corrections.cpp
        cudahelpers.cpp
        externalcuda.cpp
)

target_include_directories(cudaLib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cudaLib PRIVATE tl::expected CUDA::cudart)
set_target_properties(cudaLib PROPERTIES CUDA_ARCHITECTURES native)