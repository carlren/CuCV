#include "BasicProc.h"

/// ----------------------------------------------------
///	kernel function defines
/// ----------------------------------------------------

__global__ void
samplePixel_device(
        cudaTextureObject_t src_tex,
        uchar4 * dst_img,
        int width,
        int height
        );


/// ----------------------------------------------------
///	host function
/// ----------------------------------------------------
void
CuCv::
resize(
        TextureImage<uchar4> &src_image, 
        TextureImage<uchar4> & dst_image, 
        int width, 
        int height)
{
    if (dst_image.width()!=width || dst_image.height()!=height)
    {
        dst_image.reallocate(width,height);
    }
    
    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((int)ceil((float) dst_image.width() / (float) blockSize.x), (int)ceil((float) dst_image.height() / (float) blockSize.y));
    
    samplePixel_device<<<gridSize,blockSize>>>(
                                                 src_image.getTextureObject(),
                                                 dst_image.getPtrGPU(),
                                                 dst_image.width(),
                                                 dst_image.height());
}


/// ----------------------------------------------------
///	device function defines
/// ----------------------------------------------------
__global__ void
samplePixel_device(
        cudaTextureObject_t src_tex, 
        uchar4 * dst_img, 
        int width, 
        int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > width - 1 || y > height - 1) return;
    
    float fx = (float) x / (float) width;
    float fy = (float) y / (float) height;
    
    float4 p = tex2D<float4>(src_tex,fx,fy);
    
    dst_img[y * width + x] = make_uchar4(p.x*255,p.y*255,p.z*255,1);
}
