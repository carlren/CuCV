#pragma once
#include "BaseDefine.h"
#include "VectorHelper.h"

namespace CuCv {

struct Mat4f
{
    union {
        struct{
            float m00, m01, m02, m03;
            float m10, m11, m12, m13;
            float m20, m21, m22, m23;
            float m30, m31, m32, m33;
        };
        float m[16];
    };
    
    /** @brief Constructors */
     _DEVICE_AND_HOST_ Mat4f(){}
     _DEVICE_AND_HOST_ Mat4f(
             float _m00, float _m01, float _m02, float _m03,
             float _m10, float _m11, float _m12, float _m13,
             float _m20, float _m21, float _m22, float _m23,
             float _m30, float _m31, float _m32, float _m33,)
     {
         m[ 0]=_m00; m[ 1]=_m01; m[ 2]=_m02; m[ 3]=_m03;
         m[ 4]=_m10; m[ 5]=_m11; m[ 6]=_m12; m[ 7]=_m13;
         m[ 8]=_m20; m[ 9]=_m21; m[10]=_m22; m[11]=_m23;
         m[12]=_m30; m[13]=_m31; m[14]=_m32; m[15]=_m33;
     }
     _DEVICE_AND_HOST_ Mat4f (const Mat4f & _m)
     {for (int i=0;i<16;i++) m[i] = _m.m[i];}
    
     /** @brief Premade Matrix */
     _DEVICE_AND_HOST_ static inline Mat4f eye()
     {
         return Mat4f(1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1);
     }
     _DEVICE_AND_HOST_ static inline Mat4f zeros()
     {
         return Mat4f(0,0,0,
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
     _DEVICE_AND_HOST_ inline Mat4f operator * (float rhs ) const
     {
         return Mat4f(m00*rhs, m01*rhs, m02*rhs, 
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
     _DEVICE_AND_HOST_ inline Mat4f operator * (const Mat4f & rhs) const
     {
         float3 c1 = operator *(rhs.col(0));
         float3 c2 = operator *(rhs.col(1));
         float3 c3 = operator *(rhs.col(2));
         
         return Mat4f(c1.x, c2.x, c3.x,
                      c1.y, c2.y, c3.y,
                      c1.z, c2.z, c3.z);
     }
};

#ifndef __CUDACC__
#include <iostream>

/** @brief print out only enabled for cpp */
std::ostream &
operator << (std::ostream & os, const Mat4f & rhs)
{
    return os << "|" << rhs.m00 << ",\t" << rhs.m01 << ",\t" << rhs.m02 << "\t|" << std::endl
              << "|" << rhs.m10 << ",\t" << rhs.m11 << ",\t" << rhs.m12 << "\t|" << std::endl
              << "|" << rhs.m20 << ",\t" << rhs.m21 << ",\t" << rhs.m22 << "\t|" << std::endl;
}


#endif


}
