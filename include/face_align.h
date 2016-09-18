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

#define DEFAULT_RESIZE_DIM 192

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

typedef struct FACE_ALIGN_TRANS_VAR {
  float rotation;        // Rotation angle
  float pitching; // Pitching rate (good in raise face). Use the change of mid_eye_to_top_lip/whole_face_height as the approximate estimate of the face pitching rate.
  float ar_change; // Change of face aspect ratio (fair in down face, good in face fragment).
  float lr_change; // Change of width_of_left_face / width_of_right_face
} FaceAlignTransVar;

class FaceAlign
{
public:
    FaceAlign(const std::string & facePredictor);
    ~FaceAlign();

    // Detect face using dlib.
    std::vector<dlib::rectangle> getAllFaceBoundingBoxes(dlib::cv_image<dlib::bgr_pixel> & rgbImg);
    std::vector<cv::Rect> getAllFaceBoundingBoxes(cv::Mat & rgbImg);
    dlib::rectangle getLargestFaceBoundingBox(dlib::cv_image<dlib::bgr_pixel> & rgbImg);
    cv::Rect getLargestFaceBoundingBox(cv::Mat & rgbImg);
    dlib::full_object_detection getLargestFaceLandmarks(dlib::cv_image<dlib::bgr_pixel> & rgbImg);
    std::vector<cv::Point2f> getLargestFaceLandmarks(cv::Mat & rgbImg);
    cv::vector<dlib::full_object_detection> getAllFaceLandmarks(dlib::cv_image<dlib::bgr_pixel> & rgbImg);
    std::vector<std::vector<cv::Point2f> > getAllFaceLandmarks(cv::Mat & rgbImg);
    // Find face landmarks.
    std::vector<dlib::point> findLandmarks(dlib::cv_image<dlib::bgr_pixel> &rgbImg, dlib::rectangle bb);
    // Do affine transform to align face.
    cv::Mat align(dlib::cv_image<dlib::bgr_pixel> &rgbImg,
                  dlib::rectangle bb=dlib::rectangle(),
                  const int imgDim=DEFAULT_RESIZE_DIM,
                  const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                  const float scale_factor=0.0);
    cv::Mat  align(cv::Mat & rgbImg,
                   cv::Rect rect=cv::Rect(),
                   const int imgDim=DEFAULT_RESIZE_DIM,
                   const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                   const float scale_factor=0.0);
    cv::Mat align(dlib::cv_image<dlib::bgr_pixel> &rgbImg,
                  cv::Mat & H,  // The affine matrix to the template
                  cv::Mat & inv_H, // Inverse affine matrix
                  dlib::rectangle bb=dlib::rectangle(),
                  const int imgDim=DEFAULT_RESIZE_DIM,
                  const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                  const float scale_factor=0.0);
    cv::Mat align(cv::Mat &rgbImg,
                  cv::Mat & H,  // The affine matrix to the template
                  cv::Mat & inv_H, // Inverse affine matrix
                  cv::Rect rect=cv::Rect(),
                  const int imgDim=DEFAULT_RESIZE_DIM,
                  const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                  const float scale_factor=0.0);
    cv::Mat  align(dlib::cv_image<dlib::bgr_pixel> &rgbImg, 
        cv::Mat & H, // The affine matrix to the template
        FaceAlignTransVar & V,  // Transform variables in face alignment
        dlib::rectangle bb=dlib::rectangle(),
        const int imgDim=DEFAULT_RESIZE_DIM,
        const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
        const float scale_factor=0.0);
    cv::Mat  align(cv::Mat &rgbImg,
        cv::Mat & H, // The affine matrix to the template
        FaceAlignTransVar & V, // Transform variables in face alignment
        cv::Rect rect=cv::Rect(),
        const int imgDim=DEFAULT_RESIZE_DIM,
        const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
        const float scale_factor=0.0);

    // Detect the largest face, align and crop it.
    cv::Mat detectAlignCrop(const cv::Mat &img,
                            cv::Rect & rect,
                            const int imgDim=DEFAULT_RESIZE_DIM,
                            const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                            const float scale_factor=0.0);
    cv::Mat detectAlignCrop(const cv::Mat &img,
                            cv::Rect & rect,
                            cv::Mat & H,  // The affine matrix to the template
                            cv::Mat & inv_H, // Inverse affine matrix
                            const int imgDim=DEFAULT_RESIZE_DIM,
                            const int landmarkIndices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                            const float scale_factor=0.0);

    // Detect face(s);
    void detectFace(const cv::Mat & img, std::vector<cv::Rect> & rects);
    void detectFace(const cv::Mat & img, cv::Rect & rect);

    // Landmark indices corresponding to the inner eyes and top lip.
    static const int INNER_EYES_AND_TOP_LIP[];
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
