#include <torch/script.h>


// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define kIMAGE_SIZE_W 256
#define kIMAGE_SIZE_H 320
#define kCHANNELS 3


class SPPE{

public:
    cv::Mat CroptheImage(cv::Mat &image , int &xmin , int &ymin , int &xmax , int &ymax);
    torch::Tensor ImagetoTensor(cv::Mat &image);
    void OutputtoOrigImg(int personid , cv::Mat &image , torch::Tensor maxid ,int &croppedimage_w ,int &croppedimage_h , int &xmin , int &ymin );
protected:
private:

};
