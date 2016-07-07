#ifndef FACE_ALIGN_H
#define FACE_ALIGN_H
// Face alignment library using dlib. 
// Author: Luo Lei 
// Create: 2016.5.25
// Email: robert165 AT 163.com

#include<string>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

namespace face_rec_srzn {

static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
{
    if (r.is_empty())
        return cv::Rect();
    return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
    if (r.area() <=0 )
        return dlib::rectangle();
    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

class FaceAlign
{
public:
    FaceAlign(const std::string & facePredictor);
    ~FaceAlign();

    // Detect face using dlib.
    std::vector<dlib::rectangle> getAllFaceBoundingBoxes(dlib::cv_image<dlib::bgr_pixel> & rgbImg);
    dlib::rectangle getLargestFaceBoundingBox(dlib::cv_image<dlib::bgr_pixel> & rgbImg);
    // Find face landmarks.
    std::vector<dlib::point> findLandmarks(dlib::cv_image<dlib::bgr_pixel> &rgbImg, dlib::rectangle bb);
    // Do affine transform to align face.
    cv::Mat align(dlib::cv_image<dlib::bgr_pixel> &rgbImg,
                  dlib::rectangle bb=dlib::rectangle(),
                  const int imgDim=224,
                  const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_BOTTOM_LIP,
                  const float scale_factor=0.0);
    cv::Mat  align(cv::Mat & rgbImg,
                   cv::Rect rect=cv::Rect(),
                   const int imgDim=224,
                   const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_BOTTOM_LIP,
                   const float scale_factor=0.0);
    cv::Mat align(dlib::cv_image<dlib::bgr_pixel> &rgbImg,
                  cv::Mat & H,  // The affine matrix to the template
                  cv::Mat & inv_H, // Inverse affine matrix
                  dlib::rectangle bb=dlib::rectangle(),
                  const int imgDim=224,
                  const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_BOTTOM_LIP,
                  const float scale_factor=0.0);
    cv::Mat align(cv::Mat &rgbImg,
                  cv::Mat & H,  // The affine matrix to the template
                  cv::Mat & inv_H, // Inverse affine matrix
                  cv::Rect rect=cv::Rect(),
                  const int imgDim=224,
                  const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_BOTTOM_LIP,
                  const float scale_factor=0.0);

    // Detect the largest face, align and crop it.
    cv::Mat detectAlignCrop(const cv::Mat &img,
                            cv::Rect & rect,
                            const int imgDim=224,
                            const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_BOTTOM_LIP,
                            const float scale_factor=0.0);
    cv::Mat detectAlignCrop(const cv::Mat &img,
                            cv::Rect & rect,
                            cv::Mat & H,  // The affine matrix to the template
                            cv::Mat & inv_H, // Inverse affine matrix
                            const int imgDim=224,
                            const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_BOTTOM_LIP,
                            const float scale_factor=0.0);

    // Detect face(s);
    void detectFace(const cv::Mat & img, std::vector<cv::Rect> & rects);
    void detectFace(const cv::Mat & img, cv::Rect & rect);

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
