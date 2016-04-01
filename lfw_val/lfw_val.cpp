#include <time.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <iomanip>

#include "opencv2/highgui/highgui.hpp"
#include "face_recognition.hpp"
#include "BayesianModel.h"
#include "LBF.h"
#include "LBFRegressor.h"

using namespace cv;
using namespace std;
using namespace face_rec_srzn;
using namespace BayesianModelNs;

#define BVLENTH 160
// parameters
Params global_params;

string modelPath = "./../../model_69/";
string dataPath;
string cascadeName = "haarcascade_frontalface_alt.xml";

std::vector<std::string> &split(const std::string &s, char delim,
                                std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}

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

Rect FindMaxFace(const vector<Rect> &faces) {
  float max_area = 0.f;
  Rect max_face;
  for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++) {
    if (r->width * r->height > max_area) {
      max_area = r->width * r->height;
      max_face = *r;
    }
  }
  return max_face;
}

// Face Alignment
void FaceAlign(const Mat &orig, Point2d leftEye, Point2d rightEye,
               Mat &outputarray) {
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

Mat detectAlignCrop(Mat &img, CascadeClassifier &cascade,
                    LBFRegressor &regressor) {
  int i = 0;
  double t = 0;
  float scale = 0.55f;
  float scale_top_ratio = 0.3f;
  vector<Rect> faces;
  Mat gray;

  cvtColor(img, gray, CV_BGR2GRAY);
  equalizeHist(gray, gray);

  // --Detection
  cascade.detectMultiScale(gray, faces, 1.1, 2,
                           0
                               //|CV_HAAR_FIND_BIGGEST_OBJECT
                               //|CV_HAAR_DO_ROUGH_SEARCH
                               |
                               CV_HAAR_SCALE_IMAGE,
                           Size(30, 30));
  if (faces.size() <= 0) {
    cout << "Cannot detect face!\n";
    return Mat(img, Rect(68 - 113.f * scale * 0.5f,
                         68 - 113.f * scale * scale_top_ratio,
                         113 + 113.f * scale, 113 + 113.f * scale));
  }

  // --Alignment
  float max_area = 0.f;
  Rect max_face = FindMaxFace(faces);
  BoundingBox boundingbox;

  boundingbox.start_x = max_face.x;
  boundingbox.start_y = max_face.y;
  boundingbox.width = max_face.width;
  boundingbox.height = max_face.height;
  boundingbox.centroid_x = boundingbox.start_x + boundingbox.width / 2.0;
  boundingbox.centroid_y = boundingbox.start_y + boundingbox.height / 2.0;
  Rect origin_max_face(boundingbox.start_x, boundingbox.start_y,
                       boundingbox.width, boundingbox.height);

  Mat_<double> current_shape = regressor.Predict(gray, boundingbox, 1);

  // for (int i = 0; i < 46; i += 45 - 36) {
  // for(int i = 0;i < global_params.landmark_num;i++){
  // circle(img, Point2d(current_shape(i, 0), current_shape(i, 1)), 3,
  // Scalar(255, 255, 255), -1, 8, 0);
  ////stringstream id;
  ////id << i;
  ////putText(img, id.str(), Point2d(current_shape(i, 0), current_shape(i, 1)),
  ////CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
  //}
  // imshow("result", img);
  // waitKey(10);
  Mat after_aligned;
  FaceAlign(img, Point2d(current_shape(36, 0), current_shape(36, 1)),
            Point2d(current_shape(45, 0), current_shape(45, 1)), after_aligned);

  // rectangle(after_aligned, origin_max_face, Scalar(255,0,0));
  // rectangle(img, origin_max_face, Scalar(255,0,0));

  // detect faces on aligned image again
  cascade.detectMultiScale(after_aligned, faces, 1.1, 2,
                           0
                               //|CV_HAAR_FIND_BIGGEST_OBJECT
                               //|CV_HAAR_DO_ROUGH_SEARCH
                               |
                               CV_HAAR_SCALE_IMAGE,
                           Size(30, 30));

  if (faces.size() <= 0) {
    cout << "Cannot detect face after aligned!\n";
    return Mat(img, Rect(68 - 113.f * scale * 0.5f,
                         68 - 113.f * scale * scale_top_ratio,
                         113 + 113.f * scale, 113 + 113.f * scale));
  }
  max_face = FindMaxFace(faces);

  int w = max_face.width;
  int h = max_face.height;
  int s = std::max(w, h);
  int x = max_face.x - (s - w) * 0.5f;
  int y = max_face.y - (s - h) * 0.5f;
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

  Rect roi(x, y, s, s);
  // Mat aligned_show = after_aligned.clone();
  // rectangle(aligned_show, max_face, Scalar(0,255,0));
  // rectangle(aligned_show, roi, Scalar(255,0,0));
  // imshow("aligned", aligned_show);
  // waitKey(0);
  return Mat(after_aligned, roi);
}

void validate_on_lfw_data(void *recognizer) {
  ifstream pairs_file("../../lfw_data/pairs.txt");
  // ifstream pair_file("../../lfw_data/COLOR224issame.txt");
  ofstream distance_file("distance_file.txt");
  // initialize parameters
  ReadGlobalParamFromFile(modelPath + "LBF.model");
  vector<float> distance_vector;
  vector<bool> gt_vector;

  string line;
  getline(pairs_file, line);
  cout << line << endl;

  CascadeClassifier cascade;
  LBFRegressor regressor;
  regressor.Load(modelPath + "LBF.model");
  // -- 1. Load the cascades
  if (!cascade.load(cascadeName)) {
    cerr << "ERROR: Could not load classifier cascade" << endl;
    return;
  }

  bool gt;
  int fnum = 0;
  BayesianModel BM("../../models/small/bayesianModel.bin");
  double d1[BVLENGTH] = {0.f};
  double d2[BVLENGTH] = {0.f};
  while (getline(pairs_file, line)) {
    // split line
    // vector<string> sline{"1","2"};
    // vector<string> sline(istream_iterator<string>(line),
    // istream_iterator<string>());
    vector<string> sline = split(line, '\t');
    string name1, name2;
    int id1, id2;
    if (sline.size() == 3) {
      name1 = name2 = sline[0];
      stringstream idstr1(sline[1]);
      idstr1 >> id1;
      stringstream idstr2(sline[2]);
      idstr2 >> id2;
      gt = true;
    } else if (sline.size() == 4) {
      name1 = sline[0];
      stringstream idstr1(sline[1]);
      idstr1 >> id1;
      name2 = sline[2];
      stringstream idstr2(sline[3]);
      idstr2 >> id2;
      gt = false;
    } else {
      cout << "Read pair error!" << endl;
      exit(1);
    }
    // cout << name1 << "\t" << id1 << "\t" << name2 << "\t" << id2 << endl;

    // Load face images
    std::ostringstream face1_string_stream;
    face1_string_stream << "../../lfw_data/lfw/" << name1 << "/" << name1 << "_"
                        << setw(4) << setfill('0') << id1 << ".jpg";
    std::string face1_name = face1_string_stream.str();
    std::ostringstream face2_string_stream;
    face2_string_stream << "../../lfw_data/lfw/" << name2 << "/" << name2 << "_"
                        << setw(4) << setfill('0') << id2 << ".jpg";
    std::string face2_name = face2_string_stream.str();

    cout << face1_name << "\t" << face2_name << endl;

    Mat face1 = imread(face1_name);
    Mat face2 = imread(face2_name);

    // crop
    // Mat face1_cropped(face1, Rect(68, 68, 113, 113));
    // Mat face2_cropped(face2, Rect(68, 68, 113, 113));

    Mat face1_cropped = detectAlignCrop(face1, cascade, regressor);
    Mat face2_cropped = detectAlignCrop(face2, cascade, regressor);

    //// compute frame per second (fps)
    ////int64 start_tick = getTickCount();

    ////Extract feature from images
    vector<float> face1_feature =
        ExtractFaceFeatureFromMat(recognizer, face1_cropped);
    vector<float> face2_feature =
        ExtractFaceFeatureFromMat(recognizer, face2_cropped);

    for (int j = 0; j < BVLENGTH; j++) {
      d1[j] = face1_feature[j];
      d2[j] = face2_feature[j];
    }
    float distance = BM.CalcSimilarity(d1, d2, BVLENGTH);
    //float distance = FaceDistance(recognizer, face1_feature, face2_feature);
    distance_vector.push_back(distance);
    gt_vector.push_back(gt);
    distance_file << distance << "\t" << gt << endl;
    ++fnum;
    cout << "pair num: " << fnum << "distance: " << distance << endl;
  }
  distance_file.close();

  float max_precision = 0.0f;
  float max_thd = 0.0f;
  for (int i = 0; i < distance_vector.size(); ++i) {
    int correct = 0;
    float thd = distance_vector[i];
    for (int j = 0; j < distance_vector.size(); ++j) {
      float dist = distance_vector[j];
      if ((dist >= thd && gt_vector[j]) || (dist < thd && !gt_vector[j]))
        ++correct;
    }
    float precision = (float)correct / distance_vector.size();
    if (precision > max_precision) {
      max_precision = precision;
      max_thd = thd;
    }
  }
  cout << "Max precision: " << max_precision << "\tThd: " << max_thd << endl;
}

void validate_on_prepared_data(void *recognizer) {
  ifstream pair_file("../../lfw_data/BW100issame.txt");
  // ifstream pair_file("../../lfw_data/prepared_self.txt");
  ofstream distance_file("distance_file.txt");
  vector<float> distance_vector;
  vector<bool> gt_vector;

  bool gt;
  int fnum = 0;
  FILE *fid_face_features = fopen("face_pair_features.bin", "rb");
  // FILE* fid_face_features = fopen("bw100feat_from_mat.bin", "rb");
  float face1_feature[160];
  float face2_feature[160];

  BayesianModel BM("../../models/small/bayesianModel.bin");
  double d1[BVLENGTH] = {0.f};
  double d2[BVLENGTH] = {0.f};
  //float distance = BM.CalcSimilarity(d1, d2, BVLENGTH);
  //cout<<distance<<endl;
  //return;
  while (pair_file >> gt) {
    // Load face images
    std::ostringstream face1_string_stream;
    // face1_string_stream << "../../lfw_data/faces_cropped/" << (fnum * 2 + 0)
    //<< ".jpg";
    face1_string_stream << "../../lfw_data/100BW/" << (fnum * 2 + 1) << ".png";
    std::string face1_name = face1_string_stream.str();
    std::ostringstream face2_string_stream;
    // face2_string_stream << "../../lfw_data/faces_cropped/" << (fnum * 2 + 1)
    //<< ".jpg";
    face2_string_stream << "../../lfw_data/100BW/" << (fnum * 2 + 2) << ".png";
    std::string face2_name = face2_string_stream.str();
    cout << face1_name << "\t" << face2_name << endl;
    // string face1_name =
    //(boost::format("../../lfw_data/224COLOR/%1%.png") % (fnum * 2 + 1)).str();
    // string face2_name =
    //(boost::format("../../lfw_data/224COLOR/%1%.png") % (fnum * 2 + 2)).str();
    // Mat face1 = imread(face1_name);
    // Mat face2 = imread(face2_name);
    // imshow("face1", face1);
    // imshow("face2", face2);
    // waitKey(1);
    //// compute frame per second (fps)
    ////int64 start_tick = getTickCount();

    ////Extract feature from images
    // vector<float> face1_feature = ExtractFaceFeatureFromMat(recognizer,
    // face1);
    // vector<float> face2_feature = ExtractFaceFeatureFromMat(recognizer,
    // face2);

    fread(&face1_feature, sizeof(float), 160, fid_face_features);
    fread(&face2_feature, sizeof(float), 160, fid_face_features);

    // float sum = 0;
    // for (int i = 0; i < face1_feature.size(); i++) {
    // sum += (face1_feature[i] - face2_feature[i]) *
    //(face1_feature[i] - face2_feature[i]);
    //}
    // sum = sqrtf(sum);

    // float distance = FaceDistance(recognizer, face1_feature, face2_feature);

    for (int j = 0; j < BVLENGTH; j++) {
      d1[j] = face1_feature[j];
      d2[j] = face2_feature[j];
      // cout<<d1[j]<<"\t"<<d2[j]<<endl;
    }
    float distance = BM.CalcSimilarity(d1, d2, BVLENGTH);

    distance_vector.push_back(distance);
    gt_vector.push_back(gt);
    distance_file << distance << "\t" << gt << endl;
    ++fnum;
    cout << "pair num: " << fnum << "\tdistacne: " << distance << endl;
  }
  distance_file.close();

  float max_precision = 0.0f;
  float max_thd = 0.0f;
  for (int i = 0; i < distance_vector.size(); ++i) {
    int correct = 0;
    float thd = distance_vector[i];
    for (int j = 0; j < distance_vector.size(); ++j) {
      float dist = distance_vector[j];
      if ((dist >= thd && gt_vector[j]) || (dist < thd && !gt_vector[j]))
        ++correct;
    }
    float precision = (float)correct / distance_vector.size();
    if (precision > max_precision) {
      max_precision = precision;
      max_thd = thd;
    }
  }
  cout << "Max precision: " << max_precision << "\tThd: " << max_thd << endl;

  // double t = ((double)getTickCount() - start_tick) /
  // getTickFrequency();//elapsed time
  // cout << "Feature extraction cost: " << t << " seconds" << endl;

  // float similarity_12 = FaceVerification(face1_feature, face2_feature);
  // float similarity_23 = FaceVerification(face2_feature, face3_feature);
  // string result = is_the_same?string("PASS"):string("FAILED");
  // cout << face_1_name << " and " << face_2_name << " verification " << result
  // << endl;
  // imshow("face1", face1);
  // imshow("face2", face2);
  // imshow("face3", face3);
  // waitKey(0);
}

int main(int argc, char **argv) {
  // Init Recognizer
  // void *recognizer = InitRecognizer("../../models/big/big.prototxt",
  //"../../models/big/big.caffemodel", "");
  void *recognizer =
      InitRecognizer("../../models/small/small.prototxt",
                     "../../models/small/small.caffemodel",
                     "../../models/small/small_mean_image.binaryproto",
                     "../../models/small/similarity.bin");

  validate_on_lfw_data(recognizer);
  //validate_on_prepared_data(recognizer);
  return 0;
}
