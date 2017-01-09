#include <iostream>

#include <base/Base.h>
#include <LinearMemory.h>

#include <memory>

using namespace std;


int main(int argc, char** argv)
{
    CuCv::Mat3f mat3f(1,2,3,4,5,6,7,8,9);
    
    cout << mat3f << endl;
    cout << mat3f * 2 << endl;
    cout << mat3f * make_float3(1,2,3) << endl << endl;
    cout << mat3f * mat3f << endl;
    
    
    std::cout << "hello world!" << std::endl;
    return 0;
}
