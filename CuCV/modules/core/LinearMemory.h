//#pragma once
//#include <stdlib.h>
//#include <cstring>
//#include <boost/shared_ptr.hpp>


//namespace CuCv {

//    template <typename T>
//    class LinearMemory
//    {
//    private:
        
//        T * data_cpu_;
//        T * data_gpu_;

//        size_t data_count_;
        
//        bool gpu_allocated_;
//        bool cpu_allocated_;
                
//        /// free all memory
//        void free()
//        {
//            if (gpu_allocated_ && data_gpu_!=NULL) CuCvSafeCall(cudaFree(data_gpu_));
//            if (cpu_allocated_ && data_cpu_!=NULL) CuCvSafeCall(cudaFreeHost(data_cpu_));
            
//            gpu_allocated_ = false;
//            cpu_allocated_ = false;
//        }
        
        
//        /// allocate memory
//        void
//        allocate(  
//                const size_t data_count, 
//                const bool allocate_CPU = true, 
//                const bool allocate_GPU = true)
//        {
//            data_count_ = data_count;
            
//            if (allocate_GPU) CuCvSafeCall(cudaMalloc((void**)&data_gpu_, sizeof(T)*data_count_));
//            if (allocate_CPU) CuCvSafeCall(cudaMallocHost((void**)&data_cpu_, sizeof(T)*data_count_));
            
//            gpu_allocated_ = allocate_GPU;
//            cpu_allocated_ = allocate_CPU;
//        }
        
//    public:
        
//        typedef boost::shared_ptr<LinearMemory<T> > Ptr;
        
//        /// constructor 
//        explicit 
//        LinearMemory(  
//                const int data_count_,
//                const bool allocate_CPU = true, 
//                const bool allocate_GPU = true)
//        {
//            allocate(
//                        data_count_,
//                        allocate_CPU,
//                        allocate_GPU);
//        }
        
//        /// destructor
//        ~LinearMemory()
//        {
//            free();
//        }
        
//        /// copy data from a pointer
//        /// assuming the data in the pointer has the same size as the allocated size
//        void
//        loadData(T * data, MemoryCopyDirection mcd)
//        {
//            switch (mcd)
//            {
//            case CPU_TO_CPU:
//                CUDASafeCall(cudaMemcpy(data_cpu_, data, sizeof(T)*data_count_, cudaMemcpyHostToHost));
//                break;
//            case GPU_TO_CPU:
//                CUDASafeCall(cudaMemcpy(data_cpu_, data, sizeof(T)*data_count_, cudaMemcpyDeviceToHost));
//                break;
//            case CPU_TO_GPU:
//                CUDASafeCall(cudaMemcpy(data_gpu_, data, sizeof(T)*data_count_, cudaMemcpyHostToDevice));
//                break;
//            case GPU_TO_GPU:
//                CUDASafeCall(cudaMemcpy(data_gpu_, data, sizeof(T)*data_count_, cudaMemcpyDeviceToDevice));
//                break;
//            default:
//                break;
//            }
//        }
        
//        void
//        copyFrom(Ptr source)
//        {
//            if ((source->gpuAllocated() != gpu_allocated_) || 
//                (source->cpuAllocated() != cpu_allocated_) ||
//                (source->size() != data_count_))
//            {
//                free();
//                allocate(source->size(),
//                         source->cpuAllocated(),
//                         source->gpuAllocated());
                
//                if (source->cpuAllocated()) loadData(source->getDataPtr(MEM_CPU),CPU_TO_CPU);
//                if (source->gpuAllocated()) loadData(source->getDataPtr(MEM_GPU),GPU_TO_GPU);
//            }
            
//        }
        
//        void setZero()
//        {
//            if (gpu_allocated_) CUDASafeCall(cudaMemset(data_gpu_,0,data_count_ * sizeof(T)));
//            if (cpu_allocated_) memset(data_cpu_,0,data_count_ * sizeof(T));
//        }
        
//        /// sync data host->device internally
//        void updateDeviceFromHost()
//        {
//            if (gpu_allocated_ && cpu_allocated_)
//                loadData(data_cpu_,CPU_TO_GPU);
//            else 
//                printf("CUDA ERROR: Memory not allocated!\n");
//        }
        
//        /// sync data device->host internally
//        void updateHostFromDevice()
//        {
//            if (gpu_allocated_ && cpu_allocated_)
//                loadData(data_gpu_,GPU_TO_CPU);
//            else 
//                printf("CUDA ERROR: Memory not allocated!\n");
//        }                
        
//        /// Get the data pointer on CPU or GPU.
//		inline T* getDataPtr(MemoryType memoryType)
//		{
//			switch (memoryType)
//			{
//			case MEM_CPU: return data_cpu_;
//			case MEM_GPU: return data_gpu_;
//			}
            
//            return 0;
//		}

//		/// Get the data pointer on CPU or GPU.
//		inline const T* getDataPtr(MemoryType memoryType) const
//		{
//			switch (memoryType)
//			{
//			case MEM_CPU: return data_cpu_;
//			case MEM_GPU: return data_gpu_;
//			}
            
//            return 0;
//		}
        
//        /// get the number of elements 
//        inline size_t count() {return data_count_;}
        
//        inline bool gpuAllocated() const {return gpu_allocated_;}
//        inline bool cpuAllocated() const {return cpu_allocated_;}
        
//        int size() const {return data_count_;}
        
//    };
//}
