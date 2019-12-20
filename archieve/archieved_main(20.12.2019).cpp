#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Darknet.h"
#include "Sppe.h"

using namespace std; 
using namespace std::chrono; 

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: yolo-app <image path>\n";
        return -1;
    }

    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {        
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // input image size for YOLO v3
    int input_image_size = 608;

    SPPE sppe;

    Darknet net("../models/yolo/yolov3-swim.cfg", &device);

    map<string, string> *info = net.get_net_info();

    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight ..." << endl;
    net.load_weights("../models/yolo/yolov3-swim_50000.weights");
    std::cout << "weight loaded ..." << endl;
    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "start to inference ..." << endl;
    
    torch::jit::script::Module sppemodule = torch::jit::load("../models/sppe/posemodel.pt");
    std::cout << "== Switch to GPU mode" << std::endl;
    // to GPU
    sppemodule.to(at::kCUDA);

  //assert(module != nullptr);
    std::cout << "== PoseModel loaded!\n";

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat origin_image, resized_image;

    origin_image = cv::imread(argv[1]);
    
    cv::cvtColor(origin_image, resized_image,  cv::COLOR_RGB2BGR);
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0/255);

    auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
    img_tensor = img_tensor.permute({0,3,1,2});

    auto yolostart = std::chrono::high_resolution_clock::now();   
    auto output = net.forward(img_tensor);
    auto yoloend = std::chrono::high_resolution_clock::now();
    auto yoloduration = duration_cast<milliseconds>(yoloend - yolostart); 
    std::cout << "Time taken for yolo: " << yoloduration.count() << " ms" << endl;
    // filter result by NMS 
    // class_num = 80
    // confidence = 0.6
    float confidence = 0.3;
    float nms_conf = 0.5;
    int class_num = 1;

    auto result = net.write_results(output, class_num , confidence, nms_conf );

    if (result.dim() == 1)
    {
        std::cout << "no object found" << endl;
    }
    else
    {
        int obj_num = result.size(0);

        float w_scale = float(origin_image.cols) / input_image_size;
        float h_scale = float(origin_image.rows) / input_image_size;

        result.select(1,1).mul_(w_scale);
        result.select(1,2).mul_(h_scale);
        result.select(1,3).mul_(w_scale);
        result.select(1,4).mul_(h_scale);

        auto result_data = result.accessor<float, 2>();
        //setting up tensor for inputting a batch of data
        torch::Tensor input_vec = torch::rand({result.size(0), 3, 320, 256});
        std::vector<torch::jit::IValue> input_tensor;
        int xmin [result.size(0)];
        int ymin [result.size(0)];
        int xmax [result.size(0)];
        int ymax [result.size(0)];
        int width [result.size(0)];
        int height [result.size(0)];

        for (int i = 0; i < result.size(0) ; i++)
        {   xmin[i] = result_data[i][1];
            ymin[i] = result_data[i][2];
            xmax[i] = result_data[i][3];
            ymax[i] = result_data[i][4];
            //Relocating the bounding box that is out of the image
            if( xmin[i] <= 0 ){ xmin[i] = 0;}
            if( ymin[i] <= 0 ){ ymin[i] = 0;}
            if( xmax[i] >= origin_image.size().width ){ xmax[i] = origin_image.size().width;}
            if( ymax[i] >= origin_image.size().height ){ ymax[i] = origin_image.size().height;}
            width[i] =  xmax[i] - xmin[i];
            height[i] = ymax[i] - ymin[i];
            cv::rectangle(origin_image, cv::Point(xmin[i], ymin[i]), cv::Point(xmax[i], ymax[i]), cv::Scalar(0, 0, 255), 1, 1, 0);
            cv::Mat croppedImage ;
            croppedImage = sppe.CroptheImage(origin_image, xmin[i], ymin[i], width[i], height[i]);
            input_vec[i] = sppe.ImagetoTensor(croppedImage)[0];
        }

            // The difference btw .forward({x})<--x is torch::Tensor and .forward(x)<--x is std::vector<torch::jit::IValue>
            input_vec = input_vec.to(at::kCUDA);
            auto output = sppemodule.forward({input_vec}); 
            torch::Tensor output_tensor = output.toTensor();
            output_tensor = output_tensor.to(at::kCPU);
            
            auto ft = output_tensor.flatten(2,3); //flattening
            auto maxresult = at::max(ft,2); //find the coordinate with the highest confidence
            auto maxid = std::get<1>(maxresult); //get the tensor

        for (int j = 0; j < result.size(0) ; j++){
            sppe.OutputtoOrigImg(j,origin_image,maxid,width[j],height[j],xmin[j],ymin[j]);   
        }
        
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start); 

        // It should be known that it takes longer time at first time
        std::cout << "Time taken : " << duration.count() << " ms" << std::endl; 

        cv::imwrite("out-det.jpg", origin_image);
    }

    std::cout << "Done" << endl;

    
    return 0;
}
