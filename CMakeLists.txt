CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

include(CTest)

set(CXX_STANDARD 17)

add_library(TensorScript src/tensor.cpp src/math.cpp)

include_directories(include)
add_subdirectory(test)