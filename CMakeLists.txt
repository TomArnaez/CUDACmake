cmake_minimum_required(VERSION 3.29)

project(CUDALIB LANGUAGES CXX C CUDA)

add_subdirectory(src)
enable_testing()
add_subdirectory(test)