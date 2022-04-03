cmake_minimum_required(VERSION 3.21)
project(OpenCL_Demo)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCL REQUIRED)

add_executable(VectorAdd vector_add.cpp)
target_link_libraries(VectorAdd OpenCL::OpenCL)
