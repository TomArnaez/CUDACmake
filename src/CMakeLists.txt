cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")

find_package(tl-expected REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(cudaLib)

target_sources(cudaLib
    PRIVATE
        cudahelpers.cpp

        core.cu
        corrections.cu
        externalcuda.cu
        histogrameq.cu
)

target_include_directories(cudaLib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cudaLib 
    tl::expected
    CUDA::cudart
)

target_compile_options(cudaLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

set_target_properties(cudaLib PROPERTIES CUDA_ARCHITECTURES native)