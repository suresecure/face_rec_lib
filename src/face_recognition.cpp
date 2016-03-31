#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include <glog/logging.h>
#include "classifier.hpp"

#include "face_recognition.hpp"
#include "BayesianModel.h"

namespace face_rec_srzn {
using namespace cv;
using namespace std;
using namespace BayesianModelNs;
struct Recognizer {
  Classifier *cnn_net;
  BayesianModel *bayesian_model;
};

void *InitRecognizer(const string &cfgname, const string &modelname,
                     const string &mean_file, const string &similarity_bin) {
  Recognizer *recognizer = new Recognizer();
  // caffe setting
  FLAGS_minloglevel = 3; // close INFO and WARNING level log
  ::google::InitGoogleLogging("SRZNFaceRecognitionLib");
  recognizer->cnn_net = new Classifier(cfgname, modelname, mean_file);
  recognizer->bayesian_model = new BayesianModel(similarity_bin.c_str());
  return (void *)recognizer;
}
vector<float> ExtractFaceFeatureFromImage(Mat face, Classifier *classifier) {
  return classifier->extract_layer_by_name(face, "prob");
  // return classifier->extract_layer_by_name(face, "fc7");
}
vector<float> ExtractFaceFeatureFromBuffer(void *rec, void *imgbuf, int w,
                                           int h) {
  Mat face_mat(h, w, CV_8UC3, imgbuf);
  Recognizer *recognizer = (Recognizer *)rec;
  return ExtractFaceFeatureFromImage(face_mat, recognizer->cnn_net);
}
vector<float> ExtractFaceFeatureFromMat(void *rec, cv::Mat img)
{
  Recognizer *recognizer = (Recognizer *)rec;
  return ExtractFaceFeatureFromImage(img, recognizer->cnn_net);
}
float FaceDistance(void *rec, const vector<float> &face1_feature,
                   const vector<float> &face2_feature) {
  float distance = 0.f;
  for (int i = 0; i < face1_feature.size(); i++) {
    distance += (face1_feature[i] - face2_feature[i]) *
           (face1_feature[i] - face2_feature[i]);
  }
  distance = sqrtf(distance);
  return distance;

  //Recognizer *recognizer = (Recognizer *)rec;
  //BayesianModel *bayesian_model = recognizer->bayesian_model;
  //float sum = 0;
  //vector<double> face1(face1_feature.begin(), face1_feature.end());
  //vector<double> face2(face2_feature.begin(), face2_feature.end());
  //float distance = bayesian_model->CalcSimilarity(&face1[0], &face2[0], 160);
  //return distance;
}
float FaceVerification(void *rec, const vector<float> &face1_feature,
                       const vector<float> &face2_feature) {
  Recognizer *recognizer = (Recognizer *)rec;
  BayesianModel *bayesian_model = recognizer->bayesian_model;
  float sum = 0;
  vector<double> face1(face1_feature.begin(), face1_feature.end());
  vector<double> face2(face2_feature.begin(), face2_feature.end());
  float distance = bayesian_model->CalcSimilarity(&face1[0], &face2[0], 160);
  // for (int i = 0; i < face1_feature.size(); i++) {
  // sum += (face1_feature[i] - face2_feature[i]) *
  //(face1_feature[i] - face2_feature[i]);
  //}
  // sum = sqrtf(sum);
  float best_distance = 0;
  float range = 0.f;
  // if (face1_feature.size() == 160) {
  best_distance = 54.61f;
  range = 10.f;
  //} else if (face1_feature.size() == 4096) {
  // best_distance = 0;
  // range = 100.f;
  //} else
  // return false;
  // if (sum <= min_distance)
  // return 1.0f;
  // else if (sum >= max_distance)
  // return 0.0f;
  // else
  float similarity = 0.9f - (sum - best_distance) / range;
  if (similarity < 0.f)
    similarity = 0.f;
  else if (similarity > 1.0f)
    similarity = 1.0f;
  return similarity;
}
void ReleaseRecognizer(void *rec) {
  Recognizer *recognizer = (Recognizer *)rec;
  delete recognizer->cnn_net;
  delete recognizer->bayesian_model;
  delete recognizer;
}
}
