#pragma once

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
