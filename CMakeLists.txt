CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

include(CTest)

set(CXX_STANDARD 11)

add_library(TensorScript src/tensor.cpp)

add_subdirectory(test)