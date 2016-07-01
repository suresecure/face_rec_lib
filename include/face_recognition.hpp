#ifndef HEADER_FACE_RECOGNITION
#define HEADER_FACE_RECOGNITION
// face recognition library
// author: xun changqing
// 2016.3.28
// email: xunchangqing AT qq.com

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

namespace face_rec_srzn {
using namespace std;
using namespace cv;

class LightFaceRecognizer {
public:
  //LightFaceRecognizer(const string &cnn_model_path,
                      //const string &face_alignment_model_path,
                      //const string &bayesian_model, const string &feature_name,
                      //bool use_gpu = true);
  LightFaceRecognizer(const string &cnn_model_path,
                      const string &bayesian_model, 
                      const string &feature_name,
                      bool use_gpu = true);
  void ExtractFaceFeature(const cv::Mat &, Mat &);
  //void FaceAlign(const cv::Mat &img, const Rect face_rect, cv::Mat &out_img);
  //void CropFace(const cv::Mat &img, const Rect face_rect,
                //Rect &cropped_face_rect);
  float CalculateCosDistance(const Mat &feature1, const Mat &feature2);
  float CalculateBayesianDistance(const Mat &feature1, const Mat &feature2);
  float CalculateEculideanDistance(const Mat &feature1, const Mat &feature2);
  float CalculateSimilarity(const Mat &feature1, const Mat &feature2);

private:
  //void ImageAlign(const Mat &orig, Point2d leftEye, Point2d rightEye,
                  //Mat &outputarray);
  string _feature_name;
  void *_conv_net;
  //void *_alignment_regressor;
  void *_bayesian_model;
};

}
#endif
