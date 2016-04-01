#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include <glog/logging.h>

#include "face_recognition.hpp"
#include "classifier.hpp"
#include "BayesianModel.h"
#include "LBF.h"
#include "LBFRegressor.h"

Params global_params;

string modelPath;
string dataPath;

namespace face_rec_srzn {
using namespace cv;
using namespace std;
using namespace BayesianModelNs;

void ReadGlobalParamFromFile(string path) {
  cout << "Loading GlobalParam..." << endl;
  ifstream fin;
  fin.open(path.c_str());
  fin >> global_params.bagging_overlap;
  fin >> global_params.max_numtrees;
  fin >> global_params.max_depth;
  fin >> global_params.max_numthreshs;
  fin >> global_params.landmark_num;
  fin >> global_params.initial_num;
  fin >> global_params.max_numstage;

  for (int i = 0; i < global_params.max_numstage; i++) {
    fin >> global_params.max_radio_radius[i];
  }

  for (int i = 0; i < global_params.max_numstage; i++) {
    fin >> global_params.max_numfeats[i];
  }
  cout << "Loading GlobalParam end" << endl;
  fin.close();
}

LightFaceRecognizer::LightFaceRecognizer(
    const string &cnn_model_path, const string &face_alignment_model_path,
    const string &bayesian_model, bool use_gpu) {
  FLAGS_minloglevel = 3; // close INFO and WARNING level log
  ::google::InitGoogleLogging("SRZNFaceRecognitionLib");
  _conv_net = new Classifier(
      cnn_model_path + "/small.prototxt", cnn_model_path + "/small.caffemodel",
      cnn_model_path + "/small_mean_image.binaryproto", use_gpu);
  modelPath = face_alignment_model_path;
  ReadGlobalParamFromFile(modelPath + "/LBF.model");
  LBFRegressor *regressor = new LBFRegressor();
  regressor->Load(modelPath + "/LBF.model");
  _alignment_regressor = regressor;

  _bayesian_model = new BayesianModel(bayesian_model.c_str());
}

void LightFaceRecognizer::ExtractFaceFeature(const cv::Mat &img,
                                             vector<float> &feature) {
  Classifier *conv_net = (Classifier *)_conv_net;
  conv_net->extract_layer_by_name(img, "prob", feature);
  return;
}

// Face Alignment
void LightFaceRecognizer::ImageAlign(const Mat &orig, Point2d leftEye,
                                     Point2d rightEye, Mat &outputarray) {
  int desiredFaceWidth = orig.cols;
  int desiredFaceHeight = desiredFaceWidth;

  // Get the center between the 2 eyes center-points
  Point2f eyesCenter =
      Point2f((leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f);

  // Get the angle between the line eyes and horizontal line.
  double dy = (rightEye.y - leftEye.y);
  double dx = (rightEye.x - leftEye.x);
  double len = sqrt(dx * dx + dy * dy);
  double angle =
      atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.
  double scale = 1;
  // Get the transformation matrix for rotating and scaling the face to the
  // desired angle & size.
  Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
  outputarray.create(desiredFaceHeight, desiredFaceWidth, CV_8UC3);
  warpAffine(orig, outputarray, rot_mat, outputarray.size());
  return;
}

void LightFaceRecognizer::FaceAlign(const Mat &img, const Rect face_rect,
                                    Mat &aligned_img) {
  LBFRegressor *regressor = (LBFRegressor *)_alignment_regressor;

  // --Alignment
  BoundingBox boundingbox;

  boundingbox.start_x = face_rect.x;
  boundingbox.start_y = face_rect.y;
  boundingbox.width = face_rect.width;
  boundingbox.height = face_rect.height;
  boundingbox.centroid_x = boundingbox.start_x + boundingbox.width / 2.0;
  boundingbox.centroid_y = boundingbox.start_y + boundingbox.height / 2.0;
  Rect origin_max_face(boundingbox.start_x, boundingbox.start_y,
                       boundingbox.width, boundingbox.height);

  Mat_<double> current_shape = regressor->Predict(img, boundingbox, 1);
  Mat after_aligned;
  ImageAlign(img, Point2d(current_shape(36, 0), current_shape(36, 1)),
             Point2d(current_shape(45, 0), current_shape(45, 1)), aligned_img);
  return;
}

void LightFaceRecognizer::CropFace(const cv::Mat &img, const Rect face_rect,
                                   Rect &cropped_face_rect) {
  float scale = 0.55f;
  float scale_top_ratio = 0.3f;
  int w = face_rect.width;
  int h = face_rect.height;
  int s = std::max(w, h);
  int x = face_rect.x - (s - w) * 0.5f;
  int y = face_rect.y - (s - h) * 0.5f;
  int left_dist = x;
  int right_dist = img.cols - x - s;
  int hor_min_dist = std::min(left_dist, right_dist);
  int top_dist = y;
  int bot_dist = img.rows - y - s;
  int max_hor_padding = hor_min_dist * 2;
  int max_ver_padding = std::min(top_dist, (int)(bot_dist * scale_top_ratio /
                                                 (1.f - scale_top_ratio))) *
                        2;
  int max_padding = std::min(max_hor_padding, max_ver_padding);
  int padding = std::min(max_padding, (int)(s * scale));
  x = x - padding / 2;
  y = y - padding * scale_top_ratio;
  s = s + padding;

  cropped_face_rect = Rect(x, y, s, s);
}

float LightFaceRecognizer::CalculateDistance(const vector<float> &feature1,
                                             const vector<float> &feature2) {
  BayesianModel *bayesian_model = (BayesianModel *)_bayesian_model;
  double d1[BVLENGTH] = {0.f};
  double d2[BVLENGTH] = {0.f};
  for (int j = 0; j < BVLENGTH; j++) {
    d1[j] = feature1[j];
    d2[j] = feature2[j];
  }
  return bayesian_model->CalcSimilarity(d1, d2, BVLENGTH);
}
float LightFaceRecognizer::CalculateSimilarity(const vector<float> &feature1,
                                               const vector<float> &feature2) {
  float distance = CalculateDistance(feature1, feature2);
  float best_distance = 0;
  float range = 0.f;
  // best_distance = 54.61f;
  // range = 10.f;
  best_distance = -21.2243f;
  range = 20.f;
  float similarity = 0.9f + (distance - best_distance) / range;
  if (similarity < 0.f)
    similarity = 0.f;
  else if (similarity > 1.0f)
    similarity = 1.0f;
  return similarity;
}
}
