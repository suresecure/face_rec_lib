cmake_minimum_required (VERSION 2.8.11)
project (face_recognition_srzn)

set(FaceRecSrzn_MAJOR_VERSION 1)
set(FaceRecSrzn_MINOR_VERSION 0)
set(FaceRecSrzn_PATCH_VERSION 7)
set(FaceRecSrzn_VERSION
  ${FaceRecSrzn_MAJOR_VERSION}.${FaceRecSrzn_MINOR_VERSION}.${FaceRecSrzn_PATCH_VERSION})

# ---[ Using cmake scripts and modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install" CACHE PATH "Default install path" FORCE) 
endif() 

# Offer the user the choice of overriding the installation directories
#set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
#set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
#set(INSTALL_INCLUDE_DIR include CACHE PATH
  #"Installation directory for header files")
#if(WIN32 AND NOT CYGWIN)
  #set(DEF_INSTALL_CMAKE_DIR CMake)
#else()
  #set(DEF_INSTALL_CMAKE_DIR lib/CMake/FaceRecSrzn)
#endif()
#set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  #"Installation directory for CMake files")

# set up include-directories
include_directories(
  "${PROJECT_SOURCE_DIR}"   # to find foo/foo.h
  "${PROJECT_BINARY_DIR}")  # to find foo/config.h

# Make relative paths absolute (needed later on)
#foreach(p LIB BIN INCLUDE CMAKE)
  #set(var INSTALL_${p}_DIR)
  #if(NOT IS_ABSOLUTE "${${var}}")
    #set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  #endif()
#endforeach()

find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(include)
include_directories(face_alignment)
include_directories(src/bayesian_model)
add_definitions(${Caffe_DEFINITIONS})    

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#find_package(Boost REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
find_package(Flann REQUIRED)
include_directories(${FLANN_INCLUDE_DIRS})

configure_file(config.h.in "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

FIND_PACKAGE( OpenMP REQUIRED)

add_subdirectory(face_alignment/align)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


add_library (face_rec_srzn SHARED 
  src/face_repository.cpp
  src/face_recognition.cpp
  src/classifier.cpp
  src/bayesian_model/BayesianModel.cpp
  include/face_repository.hpp
  include/face_recognition.hpp 
  config.h.in)
#target_include_directories (face_rec_srzn PUBLIC ./include )
target_link_libraries(face_rec_srzn face_alignment_lib ${Caffe_LIBRARIES})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(lfw_val 
  lfw_val/lfw_val.cpp)
  
#target_include_directories (lfw_val PRIVATE ./include )
target_link_libraries(lfw_val face_rec_srzn face_alignment_lib ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(lfw_test/test 
  lfw_test/test.cpp)
target_link_libraries(lfw_test/test face_rec_srzn face_alignment_lib ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${FLANN_LIBRARY} ${HDF5_LIBRARIES})

set_target_properties(face_rec_srzn PROPERTIES
  PUBLIC_HEADER "include/face_recognition.hpp;include/face_repository.hpp;${CMAKE_CURRENT_BINARY_DIR}/config.h"
  VERSION ${FaceRecSrzn_VERSION})

add_subdirectory(lfw_test)

#FRAMEWORK ON
#SOVERSION 0
install(TARGETS face_rec_srzn 
  # IMPORTANT: Add the foo library to the "export-set"
  #EXPORT FaceRecSrznTargets
  RUNTIME DESTINATION "bin" COMPONENT bin
  LIBRARY DESTINATION "lib" COMPONENT shlib
  PUBLIC_HEADER DESTINATION "include"
  COMPONENT dev)

install(FILES lfw_test/lfw_test.cpp
  lfw_test/CMakeLists.txt
  DESTINATION lfw_test)
install(FILES  cmake/Modules/FindFlann.cmake
  DESTINATION lfw_test/cmake/Modules)
install(FILES caffe_lib/libcaffe.so
  caffe_lib/libcaffe.so.1.0.0-rc3
  DESTINATION lib)
