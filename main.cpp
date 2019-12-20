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
    cv::VideoCapture cap;
    if(!cap.open(1))
    return 0;
    for(;;){

    cv::Mat frame;
    cap >> frame;
    
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat origin_image, resized_image;

    //origin_image = cv::imread(argv[1]);

    origin_image = frame;
    
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
    //std::cout << "Time taken for yolo: " << yoloduration.count() << " ms" << endl;
    // filter result by NMS 
    // class_num = 80
    // confidence = 0.6
    float confidence = 0.3;
    float nms_conf = 0.5;
    int class_num = 80;

    auto result = net.write_results(output, class_num , confidence, nms_conf );



    if (result.dim() == 1 )
    {
        //std::cout << "no object found" << endl;
        if( frame.empty() ) break; // end of video stream
        imshow("this is you, smile! :)", origin_image);
        if( cv::waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
    }
    else
    {   std::vector< int > human_arr;
        int human_num = 0;
        float human_classnum = 0;
        //std::cout << "result : " << result << std::endl ;
        int obj_num = result.size(0);
        
        for(int h = 0; h < obj_num ;h++){
            //std::cout << "result : " << result[h][7].item().toFloat() << std::endl ;
            if(result[h][7].item().toFloat() == human_classnum){
                human_num++;
                human_arr.push_back(h);
            }
        }

        //std::cout << "Number of Human found : " << human_num << std::endl ;
        //std::cout << "human_arr found : " << human_arr << std::endl ;

        if(human_num != 0){
        float w_scale = float(origin_image.cols) / input_image_size;
        float h_scale = float(origin_image.rows) / input_image_size;

        result.select(1,1).mul_(w_scale);
        result.select(1,2).mul_(h_scale);
        result.select(1,3).mul_(w_scale);
        result.select(1,4).mul_(h_scale);

        auto result_data = result.accessor<float, 2>();
        //setting up tensor for inputting a batch of data
        torch::Tensor input_vec = torch::rand({human_num, 3, 320, 256});
        std::vector<torch::jit::IValue> input_tensor;
        int xmin [human_num];
        int ymin [human_num];
        int xmax [human_num];
        int ymax [human_num];
        int width [human_num];
        int height [human_num];

        for (int j = 0; j < human_num ; j++)

        {   int i = human_arr[j];
            xmin[j] = result_data[i][1];
            ymin[j] = result_data[i][2];
            xmax[j] = result_data[i][3];
            ymax[j] = result_data[i][4];
            //Relocating the bounding box that is out of the image
            if( xmin[j] <= 0 ){ xmin[j] = 0;}
            if( ymin[j] <= 0 ){ ymin[j] = 0;}
            if( xmax[j] >= origin_image.size().width ){ xmax[j] = origin_image.size().width;}
            if( ymax[j] >= origin_image.size().height ){ ymax[j] = origin_image.size().height;}
            width[j] =  xmax[j] - xmin[j];
            height[j] = ymax[j] - ymin[j];
            cv::rectangle(origin_image, cv::Point(xmin[j], ymin[j]), cv::Point(xmax[j], ymax[j]), cv::Scalar(0, 0, 255), 1, 1, 0);
            cv::Mat croppedImage ;
            croppedImage = sppe.CroptheImage(origin_image, xmin[j], ymin[j], width[j], height[j]);
            input_vec[j] = sppe.ImagetoTensor(croppedImage)[0];
        }

            // The difference btw .forward({x})<--x is torch::Tensor and .forward(x)<--x is std::vector<torch::jit::IValue>
            input_vec = input_vec.to(at::kCUDA);
            auto output = sppemodule.forward({input_vec}); 
            torch::Tensor output_tensor = output.toTensor();
            output_tensor = output_tensor.to(at::kCPU);
            
            auto ft = output_tensor.flatten(2,3); //flattening
            auto maxresult = at::max(ft,2); //find the coordinate with the highest confidence
            auto maxid = std::get<1>(maxresult); //get the tensor

        for (int k = 0; k < human_num ; k++){
            sppe.OutputtoOrigImg(k,origin_image,maxid,width[k],height[k],xmin[k],ymin[k]);   
        }
        
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start); 

        // It should be known that it takes longer time at first time
        std::cout << "Time taken : " << duration.count() << " ms" << std::endl; 

        //cv::imwrite("out-det.jpg", origin_image);
        //std::cout << "Done" << endl;

        }

        if( frame.empty() ) break; // end of video stream
          imshow("this is you, smile! :)", origin_image);
        if( cv::waitKey(10) == 27 ) break; // stop capturing by pressing ESC 

        
    }

    }

    
    return 0;
}
