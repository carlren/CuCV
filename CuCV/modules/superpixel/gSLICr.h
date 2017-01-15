#pragma once

#include "CuCV/Core.h"

namespace CuCv {


class gSLICr{
    
public:
    
    struct SPixelInfo
    {
        float4 center;
        float4 rgba;
        long id;
        int num_pixels;
    };
    
    typedef LinearMemory<SPixelInfo> sPixelMap;
    
    enum ColorSpace
    {
        CIELAB = 0,
		XYZ,
		RGB
    };
    
    enum SizeControlMethod
    {
        GIVEN_NUM = 0,
        GIVEN_SIZE
    };
    
    struct Settings
    {
        int2 _image_size;
        int _num_segmentations;
        int _spixel_size;
        int _num_iterations;
        float _coherent_weight;
        bool _enforce_connectivity;
        ColorSpace _color_space;
        SizeControlMethod _size_control_method;
    };
    
    typedef TextureImage<uchar4> RGBAImage;
    typedef TextureImage<long> IndexImage;
    
private:
    
    Settings settings_;

    RGBAImage source_image_;
    IndexImage index_image_;
    sPixelMap accum_map_;
    int spixel_size_;
    
    
    void convertColor();
    void initClusterCenters();
    void findCenterAssociations();
    void updateClusterCenters();
    void enforceConnectivity();
    
public:
    
    explicit
    gSLICr(const Settings & settings);
    
    IndexImage & segment(void * image_data);
    void drawSegmentationOverlay(void * image_data);
    
};


}
