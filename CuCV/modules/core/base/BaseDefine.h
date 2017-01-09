#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cstdio>
#include <stdexcept>

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _DEVICE_AND_HOST_ __device__	// for CUDA device code
#else
#define _DEVICE_AND_HOST_ 
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _DEVICE_AND_HOST_CONSTANT_ __constant__	// for CUDA device code
#else
#define _DEVICE_AND_HOST_CONSTANT_
#endif

#ifndef CuCvSafeCall
#define CuCvSafeCall(err) CuCv::__cudaSafeCall(err, __FILE__, __LINE__)
#endif

#ifndef BLOCK_DIM
#define BLOCK_DIM 16
#endif

namespace CuCv {

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
                file, line, cudaGetErrorString(err) );
        exit(-1);
    }
}

template <typename T> 
_DEVICE_AND_HOST_ T & getElement2D(T* data, const int x, const int y, const size_t pitch)
{
        return ((T*)((char*) data + y * pitch))[x];
}

enum MemoryCopyDirection 
{ 
    CPU_TO_CPU, 
    CPU_TO_GPU, 
    GPU_TO_CPU, 
    GPU_TO_GPU 
};

}
