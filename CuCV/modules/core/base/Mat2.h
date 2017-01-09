#pragma once
#include "BaseDefine.h"
#include "VectorHelper.h"

namespace CuCv {

struct Mat2f
{
    union {
        struct{
            float m00, m01;
            float m10, m11;
        };
        float m[4];
    };
    
    /** @brief Constructors */
     _DEVICE_AND_HOST_ Mat2f(){}
     _DEVICE_AND_HOST_ Mat2f(float _m00, float _m01, float _m10, float _m11)
     {m[0] = _m00; m[1] = _m01; m[2] = _m10; m[3] = _m11;}
     _DEVICE_AND_HOST_ Mat2f (const Mat2f & _m)
     {m[0] = _m.m00; m[1] = _m.m01; m[2] = _m.m10; m[3] = _m.m11;}
    
     /** @brief Premade Matrix */
     _DEVICE_AND_HOST_ static inline Mat2f eye(){return Mat2f(1,0,1,0);}
     _DEVICE_AND_HOST_ static inline Mat2f zeros(){return Mat2f(0,0,0,0);}
    
     /** @brief Row & column access */
     _DEVICE_AND_HOST_ inline float2 col(int i) const {return make_float2(m[i],m[2+i]);}
     _DEVICE_AND_HOST_ inline float2 row(int i) const {return make_float2(m[i*2],m[i*2+1]);}
     
     /** @brief Element access */
     _DEVICE_AND_HOST_ inline float & operator () (int x, int y){return m[y*2+x]; }
     _DEVICE_AND_HOST_ inline const float & operator () (int x, int y) const {return m[y*2+x]; }
     
     /** @brief Scalar product */
     _DEVICE_AND_HOST_ inline Mat2f operator * (float rhs ) const
     {return Mat2f(m00*rhs, m01*rhs, m10*rhs, m11*rhs);}
     
     /** @brief Matrix-Vector product */
     _DEVICE_AND_HOST_ inline float2 operator * (float2 rhs) const
     {
         float2 of;
         of.x = m00 * rhs.x + m01 * rhs.y;
         of.y = m10 * rhs.x + m11 * rhs.y;
         return of;
     }
     
     /** @brief Matrix-Matrix product */
     _DEVICE_AND_HOST_ inline Mat2f operator * (const Mat2f & rhs) const
     {
        float om00, om01, om10, om11;
        
        om00 = m00 * rhs.m00 + m01 * rhs.m10; 
        om01 = m00 * rhs.m01 + m01 * rhs.m11;
        om10 = m10 * rhs.m00 + m11 * rhs.m10;
        om11 = m10 * rhs.m01 + m11 * rhs.m11;
        
        return Mat2f(om00, om01, om10, om11);
     }
};

#ifndef __CUDACC__
#include <iostream>

/** @brief print out only enabled for cpp */
std::ostream &
operator << (std::ostream & os, const Mat2f & rhs)
{
    return os << "|" << rhs.m00 << ",\t" << rhs.m01 << "\t|" << std::endl
              << "|" << rhs.m10 << ",\t" << rhs.m11 << "\t|" << std::endl;
}


#endif


}
