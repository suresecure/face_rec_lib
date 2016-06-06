#ifndef FACE_ALIGN_H
#define FACE_ALIGN_H
// Face alignment library using dlib. 
// Author: Luo Lei 
// 2016.5.25
// Email: robert165 AT 163.com

#include<string>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

namespace face_rec_srzn {

class FaceAlign
{
public:
    FaceAlign(const std::string & facePredictor);
    ~FaceAlign();
    std::vector<dlib::rectangle> getAllFaceBoundingBoxes(dlib::cv_image<dlib::bgr_pixel> & rgbImg);
    dlib::rectangle getLargestFaceBoundingBox(dlib::cv_image<dlib::bgr_pixel> & rgbImg);
    std::vector<dlib::point> findLandmarks(dlib::cv_image<dlib::bgr_pixel> &rgbImg, dlib::rectangle bb);
    cv::Mat align(dlib::cv_image<dlib::bgr_pixel> &rgbImg,  
        dlib::rectangle bb=dlib::rectangle(),
        const int imgDim=224,
        const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_BOTTOM_LIP,
        const float scale_factor=0.0);

    // Landmark indices corresponding to the inner eyes and bottom lip.
    static const int INNER_EYES_AND_BOTTOM_LIP[];
    // Landmark indices corresponding to the inner eyes and bottom lip.
    static const int OUTER_EYES_AND_NOSE[];

private:
    // Face landmark template data
    static float TEMPLATE_DATA[][2];
    // Face landmark template
    cv::Mat TEMPLATE;
    // Column normalized face landmark template
    cv::Mat MINMAX_TEMPLATE;

    dlib::frontal_face_detector detector;
    dlib::shape_predictor predictor;
};
}
#endif // FACE_ALIGN_H
