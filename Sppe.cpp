// One-stop header.
#include "Sppe.h"
#include <torch/script.h>


// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define kIMAGE_SIZE_W 256
#define kIMAGE_SIZE_H 320
#define kCHANNELS 3

using namespace std; 
using namespace std::chrono; 

cv::Mat SPPE::CroptheImage(cv::Mat &image , int &xmin , int &ymin , int &width , int &height){
    cv::Mat croppedimage = image.clone();
    cv::Rect myROI(xmin , ymin , width , height);
    croppedimage = croppedimage(myROI);
    return croppedimage ;
}

torch::Tensor SPPE::ImagetoTensor(cv::Mat &image) {
    cv::cvtColor(image, image, CV_BGR2RGB);
    cv::Size scale(kIMAGE_SIZE_W, kIMAGE_SIZE_H);
    cv::resize(image, image, scale);
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
    torch::Tensor input_tensor = torch::from_blob(image.data, {1, kIMAGE_SIZE_H, kIMAGE_SIZE_W, kCHANNELS});
    input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
    input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
    input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    input_tensor = input_tensor.to(at::kCUDA);// to GPU
    return input_tensor;
}

void SPPE::OutputtoOrigImg(int personid , cv::Mat &image , torch::Tensor maxid ,int &croppedimage_w ,int &croppedimage_h , int &xmin , int &ymin ){
      int coor[17][2];
      int count = 0;
      int max_x = 0;
      int max_y = 0;
      int prob = 0;

      for(int kpts=0;kpts<17;kpts++){

        int i = 0;
        i = (int)(maxid[personid][kpts].item().toFloat()) ;
        max_x = (i % 64)+1;
        max_y = (i / 64)+1;
        coor[kpts][0] = max_x ;
        coor[kpts][1] = max_y ;
      }
          
//-------------------Display Result -------------------------------------------// 
        cv::Point p(0,0);
        for(int kpts=0;kpts<17;kpts++){
        p.x = (xmin + (coor[kpts][0] * croppedimage_w) / 64 ) ; 
        p.y = (ymin + (coor[kpts][1] * croppedimage_h) / 80 ) ;
        circle(image, p, 4, cvScalar(0, 0, 255 , 1));
      }
      

}