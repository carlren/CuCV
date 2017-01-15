cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(GSLICR_SRC
gSLICr_kernel.h
gSLICr.h
gSLICr.cu)

cuda_add_library(CuCvSuperPixel
    ${GSLICR_SRC})

target_link_libraries(CuCvSuperPixel
    CuCvCore
    cuda)
