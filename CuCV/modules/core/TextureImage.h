#pragma once
#include <stdlib.h>
#include <cstring>

#include "base/BaseDefine.h"


namespace CuCv {

    template <typename T>
    class TextureImage
    {
    private:
        
        T * data_cpu_;
        T * data_gpu_;
        cudaArray_t data_tex_;

        cudaChannelFormatDesc channel_desc_;
        cudaResourceDesc res_desc_;
        cudaTextureDesc tex_desc_;
        cudaTextureObject_t tex_obj_;        
        
        size_t data_count_;
        
        int width_;
        int height_;
        
        bool gpu_allocated_;
        bool cpu_allocated_;
        bool tex_allocated_;
        bool texture_binded_;
        
        
        /** @brief free all memory */
        void free()
        {
            if (texture_binded_) CuCvSafeCall(cudaDestroyTextureObject(tex_obj_));
            if (gpu_allocated_) CuCvSafeCall(cudaFree(data_gpu_));
            if (cpu_allocated_) CuCvSafeCall(cudaFreeHost(data_cpu_));
            if (tex_allocated_) CuCvSafeCall(cudaFreeArray(data_tex_));
            
            gpu_allocated_ = false;
            cpu_allocated_ = false;
            tex_allocated_ = false;
            texture_binded_ = false;
        }
        
        
        /** @brief allocate memory */
        void
        allocate(  
                const int width, 
                const int height, 
                const bool allocate_CPU = true, 
                const bool allocate_GPU = true)
        {
            width_ = width;
            height_ = height;
            data_count_ = width * height;
            
            if (allocate_GPU) CuCvSafeCall(cudaMalloc((void **)&data_gpu_,data_count_*sizeof(T)));
            if (allocate_CPU) CuCvSafeCall(cudaMallocHost((void**)&data_cpu_, data_count_ * sizeof(T)));
            
            gpu_allocated_ = allocate_GPU;
            cpu_allocated_ = allocate_CPU;
            texture_binded_ = false;
        }
        
    public:
       
        /** @brief constructor */
        explicit 
        TextureImage(  
                const int width, 
                const int height, 
                const bool allocate_CPU = true, 
                const bool allocate_GPU = true,
                const bool use_normalized_corrd_in_texture = true,
                const cudaTextureFilterMode filter_mode = cudaFilterModeLinear)
            :tex_allocated_(false),
             cpu_allocated_(false),
             gpu_allocated_(false),
             texture_binded_(false),
             channel_desc_(cudaCreateChannelDesc<T>()) 
        {
            // setup texture descriptor
            // texuture is border mirror
            memset(&tex_desc_,0,sizeof(cudaTextureDesc));
            tex_desc_.normalizedCoords = (int) use_normalized_corrd_in_texture;
            tex_desc_.filterMode = filter_mode;
            tex_desc_.addressMode[0] = cudaAddressModeMirror;
            tex_desc_.addressMode[1] = cudaAddressModeMirror;
            tex_desc_.addressMode[2] = cudaAddressModeMirror;
            tex_desc_.readMode = cudaReadModeNormalizedFloat;
            
            // allocate memory
            allocate(
                        width,
                        height,
                        allocate_CPU,
                        allocate_GPU);
        }
        
        /** @brief desctructor */
        ~TextureImage()
        {
            free();
        }
        
        /** @brief change the size of the image, all data are wiped */
        void
        reallocate(int width, int height)
        {
            if (width!=width_ || height!=height_)
            {
                free();
                allocate(width,height,cpu_allocated_,gpu_allocated_);                
            }
        }
                
        /**
         * @brief copy data from a pointer
         * assuming the data in the pointer has the same size as the allocated size
         */
        void
        loadData(void * data, MemoryCopyDirection mcd)
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
        
        
        /** @brief copy data from another image */
        void
        copyFrom(const TextureImage<T> & source_image)
        {
            if ((source_image.gpuAllocated() != gpu_allocated_) || 
                (source_image.cpuAllocated() != cpu_allocated_) ||
                (source_image.width()!=width_) ||
                (source_image.height()!=height_))
            {
                free();
                allocate(source_image.width(),
                         source_image.height(),
                         source_image.cpuAllocated(),
                         source_image.gpuAllocated());
            }
            
            if (source_image.cpuAllocated()) loadData(source_image.getPtrCPU(),CPU_TO_CPU);
            if (source_image.gpuAllocated()) loadData(source_image.getPtrGPU(),GPU_TO_GPU);
        }
        
        /** @brief set image to zero */
        void setZero()
        {
            if (gpu_allocated_) CuCvSafeCall(cudaMemset(data_gpu_,0,data_count_ * sizeof(T)));
            if (cpu_allocated_) memset(data_cpu_,0,data_count_ * sizeof(T));
        }
        
        /** @brief sync data host->device internally */
        void updateDeviceFromHost()
        {
            if (gpu_allocated_ && cpu_allocated_) loadData(getPtrCPU(),CPU_TO_GPU);
            else printf("CUDA ERROR: Memory not allocated!\n");
        }
        
        /** @brief sync data device->host internally */
        void updateHostFromDevice()
        {               
            if (gpu_allocated_ && cpu_allocated_) loadData(getPtrGPU(),GPU_TO_CPU);
            else printf("CUDA ERROR: Memory not allocated!\n");
        }                
        
        /** @brief get data pointer to cpu memory */
        inline const T * getPtrCPU() const {return (T*)data_cpu_;}
        inline T * getPtrCPU() {return (T*)data_cpu_;}
        
        /** @brief get data pointer to gpu memory */
        inline const T * getPtrGPU() const {return data_gpu_;}
        inline T * getPtrGPU() {return data_gpu_;}
        
        /** @brief bind the data to texture memory and return the texture object */
        inline cudaTextureObject_t getTextureObject()
        {   
            if (!tex_allocated_)
            {
                CuCvSafeCall(cudaMallocArray(&data_tex_,&channel_desc_,width_,height_));
                tex_allocated_ = true;
            }
            
            if (!texture_binded_)
            {
                // setup resource descriptor
                memset(&res_desc_,0,sizeof(cudaResourceDesc));
                res_desc_.resType = cudaResourceTypeArray;
                res_desc_.res.array.array = data_tex_;
                
                CuCvSafeCall(cudaCreateTextureObject(&tex_obj_,&res_desc_,&tex_desc_,NULL));
                texture_binded_ = true;
            }
            
            // sync gpu global memory to texture
            CuCvSafeCall(cudaMemcpyToArray(data_tex_,0,0,data_gpu_,sizeof(T)*data_count_,cudaMemcpyDeviceToDevice));
            
            return tex_obj_;
        }
        
        
        /** @brief get the number of elements  */
        inline size_t count() {return data_count_;}
        
        /** @brief whether gpu memory is allocated */
        inline bool gpuAllocated() const {return gpu_allocated_;}
        /** @brief whether cpu memory is allocated */
        inline bool cpuAllocated() const {return cpu_allocated_;}
        /** @brief whether gpu memory binded to texture memory */
        inline bool textureBined() const {return texture_binded_;}
        
        /** @brief image width */
        inline int width() const {return width_;}
        /** @brief image height */
        inline int height() const {return height_;}
        /** @brief image size */
        inline uint2 size() const {return make_uint2(width_,height_);}
        
        /** @brief bare pointer */
        inline TextureImage<T> * rawPtr() {return this;}
        
        /** @brief const bare pointer */
        inline const TextureImage<T> * rawPtr() const {return this;}
        
    };
}
