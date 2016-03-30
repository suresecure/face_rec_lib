cmake_minimum_required (VERSION 2.8.11)
project (face_recognition_srzn)

set(FaceRecSrzn_MAJOR_VERSION 1)
set(FaceRecSrzn_MINOR_VERSION 0)
set(FaceRecSrzn_PATCH_VERSION 3)
set(FaceRecSrzn_VERSION
  ${FaceRecSrzn_MAJOR_VERSION}.${FaceRecSrzn_MINOR_VERSION}.${FaceRecSrzn_PATCH_VERSION})

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

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(include)
include_directories(src/bayesian_model)
add_definitions(${Caffe_DEFINITIONS})    

#message(${Caffe_INCLUDE_DIRS})

configure_file(config.h.in "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

add_executable(lfw_val 
  src/face_recognition.cpp 
  src/bayesian_model/BayesianModel.cpp
  lfw_val/lfw_val.cpp 
  config.h.in)
target_include_directories (lfw_val PRIVATE ./include )
target_link_libraries(lfw_val ${Caffe_LIBRARIES})

add_library (face_rec_srzn SHARED 
  src/face_recognition.cpp
  src/bayesian_model/BayesianModel.cpp
  include/face_recognition.hpp 
  config.h.in)
#target_include_directories (face_rec_srzn PUBLIC ./include )
target_link_libraries(face_rec_srzn ${Caffe_LIBRARIES})

set_target_properties(face_rec_srzn PROPERTIES
  PUBLIC_HEADER "include/face_recognition.hpp;${CMAKE_CURRENT_BINARY_DIR}/config.h"
  VERSION ${FaceRecSrzn_VERSION})

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
install(FILES caffe_lib/libcaffe.so
  caffe_lib/libcaffe.so.1.0.0-rc3
  DESTINATION lib)

#INSTALL(TARGETS face_rec_srzn 
#RUNTIME DESTINATION bin
#LIBRARY DESTINATION lib
#PUBLIC_HEADER DESTINATION include
#)

## The interesting stuff goes here
## ===============================

## Add all targets to the build-tree export set
#export(TARGETS face_rec_srzn
#FILE "${PROJECT_BINARY_DIR}/FaceRecSrznTargets.cmake")

## Export the package for use from the build-tree
## (this registers the build-tree with a global CMake-registry)
#export(PACKAGE FaceRecSrzn)

## Create the FooBarConfig.cmake and FooBarConfigVersion files
#file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
#"${INSTALL_INCLUDE_DIR}")
## ... for the build tree
#set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
#configure_file(FaceRecSrznConfig.cmake.in
#"${PROJECT_BINARY_DIR}/FaceRecSrznConfig.cmake" @ONLY)
## ... for the install tree
#set(CONF_INCLUDE_DIRS "\${FaceRecSrzn_CMAKE_DIR}/${REL_INCLUDE_DIR}")
#configure_file(FaceRecSrznConfig.cmake.in
#"${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FaceRecSrznConfig.cmake" @ONLY)
## ... for both
#configure_file(FaceRecSrznConfigVersion.cmake.in
#"${PROJECT_BINARY_DIR}/FaceRecSrznConfigVersion.cmake" @ONLY)

## Install the FooBarConfig.cmake and FooBarConfigVersion.cmake
#install(FILES
#"${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FaceRecSrznConfig.cmake"
#"${PROJECT_BINARY_DIR}/FaceRecSrznConfigVersion.cmake"
#DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

## Install the export set for use with the install-tree
#install(EXPORT FaceRecSrznTargets DESTINATION
#"${INSTALL_CMAKE_DIR}" COMPONENT dev)

