cmake_minimum_required(VERSION 3.10)
project(LeNet5_CUDA_Inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This tells the compiler where to look for #include "utils/file_io.h"
include_directories(include)

# Compile BOTH cpp files into the final executable
add_executable(baseline_test 
    src/baseline/main.cpp
    src/utils/file_io.cpp
)