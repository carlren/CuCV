#pragma once
#include "BaseDefine.h"
#include "VectorHelper.h"

namespace CuCv {

struct Mat3f
{
    union {
        struct{
            float m00, m01, m02;
            float m10, m11, m12;
            float m20, m21, m22;
        };
        float m[9];
    };
    
    /** @brief Constructors */
     _DEVICE_AND_HOST_ Mat3f(){}
     _DEVICE_AND_HOST_ Mat3f(
             float _m00, float _m01, float _m02, 
             float _m10, float _m11, float _m12,
             float _m20, float _m21, float _m22)
     {
         m[0]=_m00; m[1]=_m01; m[2]=_m02;
         m[3]=_m10; m[4]=_m11; m[5]=_m12;
         m[6]=_m20; m[7]=_m21; m[8]=_m22;
     }
     _DEVICE_AND_HOST_ Mat3f (const Mat3f & _m)
     {for (int i=0;i<9;i++) m[i] = _m.m[i];}
    
     /** @brief Premade Matrix */
     _DEVICE_AND_HOST_ static inline Mat3f eye()
     {
         return Mat3f(1, 0, 0,
                      0, 1, 0,
                      0, 0, 1);
     }
     _DEVICE_AND_HOST_ static inline Mat3f zeros()
     {
         return Mat3f(0,0,0,
                      0,0,0,
                      0,0,0);
     }
    
     /** @brief Row & column access */
     _DEVICE_AND_HOST_ inline float3 col(int i) const {return make_float3(m[i],m[3+i],m[6+i]);}
     _DEVICE_AND_HOST_ inline float3 row(int i) const {return make_float3(m[i*3],m[i*3+1],m[i*3+2]);}
     
     /** @brief Element access */
     _DEVICE_AND_HOST_ inline float & operator () (int x, int y){return m[y*3+x]; }
     _DEVICE_AND_HOST_ inline const float & operator () (int x, int y) const {return m[y*3+x]; }
     
     /** @brief Scalar product */
     _DEVICE_AND_HOST_ inline Mat3f operator * (float rhs ) const
     {
         return Mat3f(m00*rhs, m01*rhs, m02*rhs, 
                      m10*rhs, m11*rhs, m12*rhs, 
                      m20*rhs, m21*rhs, m22*rhs);
     }
     
     /** @brief Matrix-Vector product */
     _DEVICE_AND_HOST_ inline float3 operator * (float3 rhs) const
     {
         float3 of;
         of.x = m00 * rhs.x + m01 * rhs.y + m02 * rhs.z;
         of.y = m10 * rhs.x + m11 * rhs.y + m12 * rhs.z;
         of.z = m20 * rhs.x + m21 * rhs.y + m22 * rhs.z;
         return of;
     }
     
     /** @brief Matrix-Matrix product */
     _DEVICE_AND_HOST_ inline Mat3f operator * (const Mat3f & rhs) const
     {
         float3 c1 = operator *(rhs.col(0));
         float3 c2 = operator *(rhs.col(1));
         float3 c3 = operator *(rhs.col(2));
         
         return Mat3f(c1.x, c2.x, c3.x,
                      c1.y, c2.y, c3.y,
                      c1.z, c2.z, c3.z);
     }
};

#ifndef __CUDACC__
#include <iostream>

/** @brief print out only enabled for cpp */
std::ostream &
operator << (std::ostream & os, const Mat3f & rhs)
{
    return os << "|" << rhs.m00 << ",\t" << rhs.m01 << ",\t" << rhs.m02 << "\t|" << std::endl
              << "|" << rhs.m10 << ",\t" << rhs.m11 << ",\t" << rhs.m12 << "\t|" << std::endl
              << "|" << rhs.m20 << ",\t" << rhs.m21 << ",\t" << rhs.m22 << "\t|" << std::endl;
}


#endif


}
