#include <iostream>
#include <memory>

#include "CuCV/Core.h"
#include "CuCV/ImgProc.h"

#include <opencv2/opencv.hpp>

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
    
//    cv::Mat origin_mat = cv::imread("/Users/carlren/_Code/CuCV/CuCV/data/SD_sea_world.JPG");
//    cv::cvtColor(origin_mat,origin_mat,cv::COLOR_BGR2BGRA);
    
//    CuCv::TextureImage<uchar4> origin_gpu_image(origin_mat.cols, origin_mat.rows);
//    CuCv::TextureImage<uchar4> small_gpu_image(origin_mat.cols/3, origin_mat.rows/3);
    
//    size_t start = cv::getCPUTickCount();
    
//    origin_gpu_image.loadData(origin_mat.data,CuCv::CPU_TO_GPU);
//    for (int i=0;i<200;i++) CuCv::resize(origin_gpu_image,small_gpu_image,origin_mat.cols/3, origin_mat.rows/3);
//    small_gpu_image.updateHostFromDevice();
//    cv::Mat small_mat(small_gpu_image.height(),small_gpu_image.width(),CV_8UC4,small_gpu_image.getPtrCPU());
    
//    std::cout << "gpu resize take = " << (cv::getCPUTickCount() - start) / cv::getTickFrequency() * 5 << "ms" << std::endl;
    
//    start = cv::getTickCount();
    
//    for (int i=0;i<200;i++) cv::resize(origin_mat,small_mat,small_mat.size());
//    std::cout << "OpenCV resize take = " << (cv::getCPUTickCount() - start) / cv::getTickFrequency() * 5 << "ms" << std::endl;
    
//    cv::imshow("origin",origin_mat);
//    cv::imshow("small",small_mat);
    
//    cv::waitKey();
    
    return 0;
}
