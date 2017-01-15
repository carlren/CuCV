cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(IMGPROC_SRC
BasicProc.h
Resize.cu)

cuda_add_library(CuCvImgProc
    ${IMGPROC_SRC})

target_link_libraries(CuCvImgProc
    CuCvCore
    cuda)
