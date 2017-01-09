cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

#find CUDA
find_package(CUDA)
if (CUDA_FOUND)
    message(STATUS "VISLIB: CUDA ${CUDA_VERSION_STRING} found!")
    add_library(cuda INTERFACE)
    target_include_directories(cuda INTERFACE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(cuda INTERFACE ${CUDA_LIBRARIES})
endif()

#Detect CUDA GPU architectore and select the correct one to build
message(STATUS "Detecting CUDA GPU architectore versions:")
try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_compute_capability.c
    CMAKE_FLAGS
    -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
    -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
    RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
message(STATUS "Compile: ${RUN_OUTPUT_VAR}")
if (COMPILE_RESULT_VAR)
    message(STATUS "compiled -> " ${RUN_RESULT_VAR})
else()
    message(STATUS "didn't compile")
endif()

if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
    message(STATUS "worked")
    set(CUDA_HAVE_GPU TRUE CACHE BOOL "Whether CUDA-capable GPU is present")
    set(CUDA_COMPUTE_CAPABILITY ${RUN_OUTPUT_VAR} CACHE STRING "Compute capability of CUDA-capable GPU present")
    mark_as_advanced(CUDA_COMPUTE_CAPABILITY)

    set(CUDA_ARCH_OPTIONS "${CUDA_ARCH_OPTIONS} --generate-code arch=compute_${CUDA_COMPUTE_CAPABILITY},code=compute_${CUDA_COMPUTE_CAPABILITY}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math ${CUDA_ARCH_OPTIONS}")
else()
    message(STATUS "didn't work")
    set(CUDA_HAVE_GPU FALSE CACHE BOOL "Whether CUDA-capable GPU is present")
endif()

list(APPEND "-std=c++11 -ftree-vectorize")

# Eigen
set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Eigen")
include_directories(${EIGEN_INCLUDE_DIR})

# modules
add_subdirectory(modules)


# make a dummy file for library
cuda_add_library(CuCv
    dummy.h)

target_include_directories(CuCv PUBLIC
    ${EIGEN_INCLUDE_DIR})

target_link_libraries(CuCv
    CuCvCore)
