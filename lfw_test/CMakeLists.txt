cmake_minimum_required (VERSION 2.8.11)
project (face_rec_sample_lfw_test)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Boost REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
find_package(Flann REQUIRED)
include_directories(${FLANN_INCLUDE_DIRS})
#set(FaceRecSrzn_DIR "/home/mythxcq/work/face_recognition_srzn/face_rec_lib/install/lib/CMake/FaceRecSrzn")
#find_package(FaceRecSrzn REQUIRED)

include_directories("../include")
link_directories("../lib")
set(FaceRecSrzn_LIBRARIES "face_rec_srzn;caffe")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(lfw_test lfw_test.cpp)
target_link_libraries(lfw_test ${FaceRecSrzn_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${FLANN_LIBRARY} ${HDF5_LIBRARIES})

