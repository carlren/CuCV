cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

add_subdirectory(base)

set(CORE_SRC
LinearMemory.h
TextureImage.h)

cuda_add_library(CuCvCore
    ${CORE_SRC})

target_compile_options(CuCvCore PUBLIC -std=c++11)

target_include_directories(CuCvCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(CuCvCore
    CuCVBase
    cuda)
