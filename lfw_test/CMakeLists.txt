cmake_minimum_required (VERSION 2.8.11)
project (face_rec_sample_lfw_test)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
find_package(Flann REQUIRED)
include_directories(${FLANN_INCLUDE_DIRS})
#set(FaceRecSrzn_DIR "/home/mythxcq/work/face_recognition_srzn/face_rec_lib/install/lib/CMake/FaceRecSrzn")
#find_package(FaceRecSrzn REQUIRED)
include_directories("../include")
link_directories("../lib")
set(FaceRecSrzn_LIBRARIES "face_rec_srzn;caffe")

#find_package(boost REQUIRED)
#include_directories(${boost_INCLUDE_DIRS})

add_executable(lfw_test lfw_test.cpp)
target_link_libraries(lfw_test ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${FaceRecSrzn_LIBRARIES} ${FLANN_LIBRARY})

