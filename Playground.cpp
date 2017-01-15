#include <iostream>

#include <base/Base.h>
#include <LinearMemory.h>
#include <TextureImage.h>

#include <memory>

using namespace std;


int main(int argc, char** argv)
{
    
    std::cout << " ------------ Mat3 -------------" << std::endl;
    
    CuCv::Mat3f mat3f(1,2,3,4,5,6,7,8,9);
    
    cout << mat3f << endl;
    cout << mat3f * 2 << endl;
    cout << mat3f * make_float3(1,2,3) << endl << endl;
    cout << mat3f * mat3f << endl;
    
    
    std::cout << std::endl << " ------------ Mat4 -------------" << std::endl;
    CuCv::Mat4f mat4f(1,2,3,4,
                      5,6,7,8,
                      9,1,2,3,
                      4,5,6,7);
    float4 b = make_float4(1,2,3,4);

    cout << mat4f << endl; 
    cout << mat4f * 3 << endl;
    cout << mat4f *  b << endl << endl;
    cout << mat4f * mat4f << endl;
        
    return 0;
}
