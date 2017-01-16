#pragma once
#include "CuCV/Core.h"

_DEVICE_AND_HOST_ 
inline float4 
rgb2xyz(const float4 & pixel_in)
{
    float4 pixel_out;
	pixel_out.x = pixel_in.z*0.412453f + pixel_in.y*0.357580f + pixel_in.x*0.180423f;
	pixel_out.y = pixel_in.z*0.212671f + pixel_in.y*0.715160f + pixel_in.x*0.072169f;
	pixel_out.z = pixel_in.z*0.019334f + pixel_in.y*0.119193f + pixel_in.x*0.950227f;
    pixel_in.w = 1.0f;
    return pixel_out;
}


_DEVICE_AND_HOST_
inline float
Labf(const float t)
{
    if (t > 0.008856f) return pow(t, 1.0f/3.0f);
    else return 7.787 * t + 0.13793;
}
