cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
#find_package( Boost REQUIRED )
set(FaceAlignmentSrzn_MAJOR_VERSION 1)
set(FaceAlignmentSrzn_MINOR_VERSION 0)
set(FaceAlignmentSrzn_PATCH_VERSION 0)
set(FaceAlignmentSrzn_VERSION
  ${FaceAlignmentSrzn_MAJOR_VERSION}.${FaceAlignmentSrzn_MINOR_VERSION}.${FaceAlignmentSrzn_PATCH_VERSION})

include_directories(../)
#include_directories(${Boost_INCLUDE_DIRS})
set(alignment_source ./../liblinear/blas/blas.h ./../liblinear/blas/blasp.h ./../liblinear/blas/daxpy.c ./../liblinear/blas/ddot.c ./../liblinear/blas/dnrm2.c ./../liblinear/blas/dscal.c 
./../liblinear/tron.cpp ./../liblinear/tron.h ./../liblinear/linear.h ./../liblinear/linear.cpp  
./../LBF.h ./../Utils.cpp ./../Tree.h ./../Tree.cpp ./../RandomForest.cpp ./../RandomForest.h ./../LBFRegressor.h ./../LBFRegressor.cpp)

#if you don't want to use openMP, you should comment this line
FIND_PACKAGE( OpenMP REQUIRED)


if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#add_library (face_alignment_lib STATIC ${alignment_source})
add_library (face_alignment_lib SHARED ${alignment_source})
#add_executable(align ./align.cpp ${alignment_source})
#target_link_libraries(align ${OpenCV_LIBS}) 
set_target_properties(face_alignment_lib PROPERTIES
  #PUBLIC_HEADER "include/face_recognition.hpp;${CMAKE_CURRENT_BINARY_DIR}/config.h"
  VERSION ${FaceAlignmentSrzn_VERSION})
install(TARGETS face_alignment_lib 
  # IMPORTANT: Add the foo library to the "export-set"
  #EXPORT FaceRecSrznTargets
  RUNTIME DESTINATION "bin" COMPONENT bin
  LIBRARY DESTINATION "lib" COMPONENT shlib
  PUBLIC_HEADER DESTINATION "include"
  COMPONENT dev)
