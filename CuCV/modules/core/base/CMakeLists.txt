cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

cuda_add_library(CuCVBase
BaseDefine.h
VectorHelper.h
Mat2.h
Mat3.h
Mat4.h
Base.h)

target_link_libraries(CuCVBase
cuda)
