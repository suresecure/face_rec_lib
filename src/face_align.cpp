#include <dlib/assert.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "face_align.h"

namespace face_rec_srzn {
const int FaceAlign::INNER_EYES_AND_BOTTOM_LIP[] =  {3, 39, 42, 57};
const int FaceAlign::OUTER_EYES_AND_NOSE[] = {3, 36, 45, 33};
float FaceAlign::TEMPLATE_DATA[][2] =
{
    {0.0792396913815, 0.339223741112}, {0.0829219487236, 0.456955367943},
    {0.0967927109165, 0.575648016728}, {0.122141515615, 0.691921601066},
    {0.168687863544, 0.800341263616}, {0.239789390707, 0.895732504778},
    {0.325662452515, 0.977068762493}, {0.422318282013, 1.04329000149},
    {0.531777802068, 1.06080371126}, {0.641296298053, 1.03981924107},
    {0.738105872266, 0.972268833998}, {0.824444363295, 0.889624082279},
    {0.894792677532, 0.792494155836}, {0.939395486253, 0.681546643421},
    {0.96111933829, 0.562238253072}, {0.970579841181, 0.441758925744},
    {0.971193274221, 0.322118743967}, {0.163846223133, 0.249151738053},
    {0.21780354657, 0.204255863861}, {0.291299351124, 0.192367318323},
    {0.367460241458, 0.203582210627}, {0.4392945113, 0.233135599851},
    {0.586445962425, 0.228141644834}, {0.660152671635, 0.195923841854},
    {0.737466449096, 0.182360984545}, {0.813236546239, 0.192828009114},
    {0.8707571886, 0.235293377042}, {0.51534533827, 0.31863546193},
    {0.516221448289, 0.396200446263}, {0.517118861835, 0.473797687758},
    {0.51816430343, 0.553157797772}, {0.433701156035, 0.604054457668},
    {0.475501237769, 0.62076344024}, {0.520712933176, 0.634268222208},
    {0.565874114041, 0.618796581487}, {0.607054002672, 0.60157671656},
    {0.252418718401, 0.331052263829}, {0.298663015648, 0.302646354002},
    {0.355749724218, 0.303020650651}, {0.403718978315, 0.33867711083},
    {0.352507175597, 0.349987615384}, {0.296791759886, 0.350478978225},
    {0.631326076346, 0.334136672344}, {0.679073381078, 0.29645404267},
    {0.73597236153, 0.294721285802}, {0.782865376271, 0.321305281656},
    {0.740312274764, 0.341849376713}, {0.68499850091, 0.343734332172},
    {0.353167761422, 0.746189164237}, {0.414587777921, 0.719053835073},
    {0.477677654595, 0.706835892494}, {0.522732900812, 0.717092275768},
    {0.569832064287, 0.705414478982}, {0.635195811927, 0.71565572516},
    {0.69951672331, 0.739419187253}, {0.639447159575, 0.805236879972},
    {0.576410514055, 0.835436670169}, {0.525398405766, 0.841706377792},
    {0.47641545769, 0.837505914975}, {0.41379548902, 0.810045601727},
    {0.380084785646, 0.749979603086}, {0.477955996282, 0.74513234612},
    {0.523389793327, 0.748924302636}, {0.571057789237, 0.74332894691},
    {0.672409137852, 0.744177032192}, {0.572539621444, 0.776609286626},
    {0.5240106503, 0.783370783245}, {0.477561227414, 0.778476346951}
};

FaceAlign::FaceAlign(const std::string & facePredictor)
{
    detector = dlib::get_frontal_face_detector();
    dlib::deserialize(facePredictor) >> predictor;

    TEMPLATE = cv::Mat(68, 2, CV_32FC1, TEMPLATE_DATA);
    TEMPLATE.copyTo(MINMAX_TEMPLATE);
    // Column normalize template to (0,1). It will make the face landmarks tightly inside (0,1) box.
    cv::Mat temp, cp;
    for (int i=0; i<TEMPLATE.cols; i++)
    {
        cv::normalize(TEMPLATE.col(i), temp, 1, 0, cv::NORM_MINMAX);
        cp = MINMAX_TEMPLATE.colRange(i, i+1);
        temp.copyTo(cp);
    }
}

FaceAlign::~FaceAlign()
{
}

std::vector<dlib::rectangle> FaceAlign::getAllFaceBoundingBoxes(dlib::cv_image<dlib::bgr_pixel> & rgbImg)
{
    return detector(rgbImg);
}

dlib::rectangle FaceAlign::getLargestFaceBoundingBox(dlib::cv_image<dlib::bgr_pixel> & rgbImg)
{
    std::vector<dlib::rectangle> dets = this->getAllFaceBoundingBoxes(rgbImg);
    if (dets.size() > 0)
    {
        int i_max = 0;
        float max = 0;
        for (int i=0; i< dets.size(); i++)
        {
            float rect = dets[i].width() * dets[i].height();
            if( rect > max)
            {
                max = rect;
                i_max = i;
            }
        }
        return dets[i_max];
    }
    else
    {
        return dlib::rectangle();
    }
}

cv::Mat  FaceAlign::align(dlib::cv_image<dlib::bgr_pixel> &rgbImg,
                          dlib::rectangle bb,
                          const int imgDim,
                          const int landmarkIndices[],
                          const float scale_factor)
{
    cv::Mat H, inv_H;
    return align(rgbImg, H, inv_H, bb, imgDim, landmarkIndices, scale_factor);
}

cv::Mat  FaceAlign::align(cv::Mat & rgbImg,
                          cv::Rect rect,
                          const int imgDim,
                          const int landmarkIndices[],
                          const float scale_factor)
{
    cv::Mat H, inv_H;
    dlib::cv_image<dlib::bgr_pixel> cimg(rgbImg);
    return align(cimg, H, inv_H, openCVRectToDlib(rect), imgDim, landmarkIndices, scale_factor);
}

cv::Mat  FaceAlign::align(cv::Mat &rgbImg,
                          cv::Mat & H,
                          cv::Mat & inv_H,
                          cv::Rect rect,
                          const int imgDim,
                          const int landmarkIndices[],
                          const float scale_factor)
{
    dlib::cv_image<dlib::bgr_pixel> cimg(rgbImg);
    return align(cimg, H, inv_H, openCVRectToDlib(rect), imgDim, landmarkIndices, scale_factor);
}

cv::Mat  FaceAlign::align(dlib::cv_image<dlib::bgr_pixel> &rgbImg, 
                          cv::Mat & H,
                          cv::Mat & inv_H,
                          dlib::rectangle bb,
                          const int imgDim,
                          const int landmarkIndices[],
                          const float scale_factor)
{
    if (bb.is_empty())
        bb = this->getLargestFaceBoundingBox(rgbImg);

    dlib::full_object_detection landmarks = this->predictor(rgbImg, bb);

    int nPoints = landmarkIndices[0];
    cv::Point2f srcPoints[nPoints];
    cv::Point2f dstPoints[nPoints];

    cv::Mat template_face = TEMPLATE;
    if (scale_factor > 0 && scale_factor < 1) {
        template_face = MINMAX_TEMPLATE;
    }

    for (int i=1; i<=nPoints; i++)
    {
        dlib::point p = landmarks.part(landmarkIndices[i]);
        srcPoints[i-1] = cv::Point2f(p.x(), p.y());
        dstPoints[i-1] = cv::Point2f((float)imgDim * template_face.at<float>(landmarkIndices[i], 0),
                                     (float)imgDim * template_face.at<float>(landmarkIndices[i], 1));
        //std::cout<<dstPoints[i-1]<<std::endl;
    }
    float resize_factor = 1.0;
    if (scale_factor > 0 && scale_factor < 1) {
        // The first two landmarks (inner/outer eyes) and the third landmark (bottom lip/nose) form an isosceles triangle approximately.
        float d1, d2, d3, h1, h2, h;
        d1 = cv::norm(dstPoints[0] - dstPoints[1]);
        d2 = cv::norm(dstPoints[2] - dstPoints[0]);
        d3 = cv::norm(dstPoints[2] - dstPoints[1]);
        h1 = std::sqrt(d2*d2 - d1*d1/4); // Height computed by landmark 0, 2
        h2 = std::sqrt(d3*d3 - d1*d1/4); // Height computed by landmark 1, 3
        h = (h1 + h2)/2; // Use their average
        resize_factor = scale_factor/ h * imgDim;
        //std::cout<<" "<<d1<<" "<<d2<<" "<<d3<<" "<<h1<<" "<<h2<<" "<<h<<" "<<resize_factor<<std::endl;
    }
    for (int i=0; i<nPoints; i++)
    {
        dstPoints[i] -= cv::Point2f(0.5*imgDim, 0.5*imgDim);
        dstPoints[i] *= resize_factor;
        dstPoints[i] += cv::Point2f(0.5*imgDim, 0.5*imgDim);
        //std::cout<<dstPoints[i]<<std::endl;
    }
    H = cv::getAffineTransform(srcPoints, dstPoints);
    inv_H = cv::getAffineTransform(dstPoints, srcPoints);
    cv::Mat warpedImg = dlib::toMat(rgbImg);
    cv::warpAffine(warpedImg, warpedImg, H, cv::Size(imgDim, imgDim));
    return warpedImg;
}

// Find and crop face by dlib.
cv::Mat FaceAlign::detectAlignCrop(const cv::Mat &img,
                                   cv::Rect & rect,
                                   cv::Mat & H,
                                   cv::Mat & inv_H,
                                   const int imgDim,
                                   const int landmarkIndices[],
                                   const float scale_factor)
{
    // Detection
    /* Dlib detects a face larger than 80x80 pixels.
     * We can use pyramid_up to double the size of image,
     * then dlib can find faces in size of 40x40 pixels in the original image.*/
    // dlib::pyramid_up(img);
    dlib::cv_image<dlib::bgr_pixel> cimg(img);
    std::vector<dlib::rectangle> dets;
    dets.push_back(getLargestFaceBoundingBox(cimg)); // Use the largest detected face only
    if (0 == dets.size() || dets[0].is_empty())
    {
        rect = cv::Rect();
        return cv::Mat();
    }
    rect = dlibRectangleToOpenCV(dets[0]);

    // --Alignment
    return align(cimg, H, inv_H, dets[0], imgDim, landmarkIndices, scale_factor);
}

cv::Mat FaceAlign::detectAlignCrop(const cv::Mat &img,
                        cv::Rect & rect,
                        const int imgDim,
                        const int landmarkIndices[],
                        const float scale_factor)
{
    cv::Mat H, inv_H;
    return detectAlignCrop(img, rect, H, inv_H, imgDim, landmarkIndices, scale_factor);
}

void FaceAlign::detectFace(const cv::Mat & img, cv::Rect & rect)
{
    dlib::cv_image<dlib::bgr_pixel> cimg(img);
    dlib::rectangle det = getLargestFaceBoundingBox(cimg);
    rect = dlibRectangleToOpenCV(det);
}

void FaceAlign::detectFace(const cv::Mat & img, std::vector<cv::Rect> & rects)
{
    dlib::cv_image<dlib::bgr_pixel> cimg(img);
    std::vector<dlib::rectangle> dets = getAllFaceBoundingBoxes(cimg);
    rects.clear();
    for (int i = 0; i < dets.size(); i++)
        rects.push_back(dlibRectangleToOpenCV(dets[i]));
}

}
