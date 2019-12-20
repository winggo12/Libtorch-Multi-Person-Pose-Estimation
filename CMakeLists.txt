cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(yolo-app)

set(OpenCV_DIR /home/hkuit104/opencv-3.4.6/include)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(TORCH_DIR /home/hkuit104/libtorch)
find_package(Torch REQUIRED)

include_directories("/home/hkuit104/Desktop/C++/libtorch-yolov3")
link_directories("/home/hkuit104/Desktop/C++/libtorch-yolov3")

aux_source_directory(. DIR_SRCS)

add_executable(yolo-app ${DIR_SRCS})
target_link_libraries(yolo-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET yolo-app PROPERTY CXX_STANDARD 11)
