#pragma once
#include <stdlib.h>
#include <cstring>

#include "base/BaseDefine.h"

namespace CuCv {

    template <typename T>
    class LinearMemory
    {
    private:
        
        T * data_cpu_;
        T * data_gpu_;

        size_t data_count_;
        
        bool gpu_allocated_;
        bool cpu_allocated_;
                
        /** @brief free all memory */
        void free()
        {
            if (gpu_allocated_ && data_gpu_!=NULL) CuCvSafeCall(cudaFree(data_gpu_));
            if (cpu_allocated_ && data_cpu_!=NULL) CuCvSafeCall(cudaFreeHost(data_cpu_));
            
            gpu_allocated_ = false;
            cpu_allocated_ = false;
        }
        
        
        /** @brief allocate memory */
        void
        allocate(  
                const size_t data_count, 
                const bool allocate_CPU = true, 
                const bool allocate_GPU = true)
        {
            data_count_ = data_count;
            
            if (allocate_GPU) CuCvSafeCall(cudaMalloc((void**)&data_gpu_, sizeof(T)*data_count_));
            if (allocate_CPU) CuCvSafeCall(cudaMallocHost((void**)&data_cpu_, sizeof(T)*data_count_));
            
            gpu_allocated_ = allocate_GPU;
            cpu_allocated_ = allocate_CPU;
        }
        
    public:
        
        /** @brief constructor  */
        explicit 
        LinearMemory(  
                const int data_count_,
                const bool allocate_CPU = true, 
                const bool allocate_GPU = true)
        {
            allocate(
                    data_count_,
                    allocate_CPU,
                    allocate_GPU);
        }
        
        /** @brief destructor */
        ~LinearMemory()
        {
            free();
        }
                
        /**
         * @brief copy data from a pointer
         * assuming the data in the pointer has the same size as the allocated size
         */
        void
        loadData(T * data, MemoryCopyDirection mcd)
        {
            switch (mcd)
            {
            case CPU_TO_CPU:
                CuCvSafeCall(cudaMemcpy(data_cpu_, data, sizeof(T)*data_count_, cudaMemcpyHostToHost));
                break;
            case GPU_TO_CPU:
                CuCvSafeCall(cudaMemcpy(data_cpu_, data, sizeof(T)*data_count_, cudaMemcpyDeviceToHost));
                break;
            case CPU_TO_GPU:
                CuCvSafeCall(cudaMemcpy(data_gpu_, data, sizeof(T)*data_count_, cudaMemcpyHostToDevice));
                break;
            case GPU_TO_GPU:
                CuCvSafeCall(cudaMemcpy(data_gpu_, data, sizeof(T)*data_count_, cudaMemcpyDeviceToDevice));
                break;
            default:
                break;
            }
        }
        
        /** @brief copy from a another instance */
        void
        copyFrom(const LinearMemory<T> & source)
        {
            if ((source.gpuAllocated() != gpu_allocated_) || 
                (source.cpuAllocated() != cpu_allocated_) ||
                (source.size() != data_count_))
            {
                free();
                allocate(source.size(),
                         source.cpuAllocated(),
                         source.gpuAllocated());
                
                if (source.cpuAllocated()) loadData(source.getPtrCPU(),CPU_TO_CPU);
                if (source.gpuAllocated()) loadData(source.getPtrGPU(),GPU_TO_GPU);
            }            
        }
        
        
        /** @brief set values to zero */
        void 
        setZero()
        {
            if (gpu_allocated_) CuCvSafeCall(cudaMemset(data_gpu_,0,data_count_ * sizeof(T)));
            if (cpu_allocated_) memset(data_cpu_,0,data_count_ * sizeof(T));
        }
        
        
        /** @brief sync data host->device internally */
        void updateDeviceFromHost()
        {
            if (gpu_allocated_ && cpu_allocated_)
                loadData(data_cpu_,CPU_TO_GPU);
            else 
                printf("CUDA ERROR: Memory not allocated!\n");
        }
        
        /** @brief sync data device->host internally */
        void updateHostFromDevice()
        {
            if (gpu_allocated_ && cpu_allocated_)
                loadData(data_gpu_,GPU_TO_CPU);
            else 
                printf("CUDA ERROR: Memory not allocated!\n");
        }                
        
        /** @brief getd data pointer to cpu memory */
        inline const T * getPtrCPU() const {return (T*)data_cpu_;}
        inline T * getPtrCPU() {return (T*)data_cpu_;}
        
        /** @brief get data pointer to gpu memory */
        inline const T * getPtrGPU() const {return data_gpu_;}
        inline T * getPtrGPU() {return data_gpu_;}
        
        /** @brief get the number of elements */
        inline size_t count() {return data_count_;}
        
        /** @brief whether gpu memory is allocated */
        inline bool gpuAllocated() const {return gpu_allocated_;}
        /** @brief whether cpu memeory is allocated */
        inline bool cpuAllocated() const {return cpu_allocated_;}
    };
}
