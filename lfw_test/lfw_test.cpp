#include <time.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <iomanip>

//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "face_recognition.hpp"
#include "boost/filesystem.hpp"
namespace fs = ::boost::filesystem;

using namespace cv;
using namespace std;
using namespace face_rec_srzn;

string cascadeName = "../haarcascade_frontalface_alt.xml";

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

void get_all(const fs::path &root, const string &ext, vector<fs::path> &ret) {
  if (!fs::exists(root) || !fs::is_directory(root))
    return;

  fs::recursive_directory_iterator it(root);
  fs::recursive_directory_iterator endit;

  while (it != endit) {
    if (fs::is_regular_file(*it) && it->path().extension() == ext)
      ret.push_back(it->path());
    // cout<<it->path();
    ++it;
  }
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

Mat detectAlignCrop(const Mat &img, CascadeClassifier &cascade,
                    LightFaceRecognizer &recognizer) {
  vector<Rect> faces;
  Mat gray, gray_org;

  cvtColor(img, gray_org, CV_BGR2GRAY);
  equalizeHist(gray_org, gray);

  // --Detection
  cascade.detectMultiScale(gray, faces, 1.1, 2,
                           0
                               //|CV_HAAR_FIND_BIGGEST_OBJECT
                               //|CV_HAAR_DO_ROUGH_SEARCH
                               |
                               CV_HAAR_SCALE_IMAGE,
                           Size(30, 30));
  Rect max_face;
  if (faces.size() <= 0) {
    cout << "Cannot detect face!\n";
    max_face = Rect(68, 68, 113, 113);
    Rect out_rect;
    recognizer.CropFace(gray_org, max_face, out_rect);
    return Mat(gray_org, out_rect);
  }

  // --Alignment
  max_face = FindMaxFace(faces);

  Mat after_aligned, after_aligned_org;
  recognizer.FaceAlign(gray_org, max_face, after_aligned_org);
  equalizeHist(after_aligned_org, after_aligned);

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
    Rect face_rect(68, 68, 113, 113);
    Rect out_rect;
    recognizer.CropFace(img, face_rect, out_rect);
    return Mat(img, out_rect);
  }
  max_face = FindMaxFace(faces);
  Rect roi;
  recognizer.CropFace(gray, max_face, roi);
  return Mat(after_aligned_org, roi);
}

void validate_on_lfw_data(LightFaceRecognizer &recognizer) {
  ifstream pairs_file("../../../lfw_data/pairs.txt");

  string line;
  getline(pairs_file, line);
  cout << line << endl;

  CascadeClassifier cascade;
  // -- 1. Load the cascades
  if (!cascade.load(cascadeName)) {
    cerr << "ERROR: Could not load classifier cascade" << endl;
    return;
  }

  bool gt;
  int fnum = 0;
  int correct = 0;
  while (getline(pairs_file, line)) {
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
    face1_string_stream << "../../../lfw_data/lfw/" << name1 << "/" << name1
                        << "_" << setw(4) << setfill('0') << id1 << ".jpg";
    std::string face1_name = face1_string_stream.str();
    std::ostringstream face2_string_stream;
    face2_string_stream << "../../../lfw_data/lfw/" << name2 << "/" << name2
                        << "_" << setw(4) << setfill('0') << id2 << ".jpg";
    std::string face2_name = face2_string_stream.str();

    cout << face1_name << "\t" << face2_name << endl;

    Mat face1 = imread(face1_name);
    Mat face2 = imread(face2_name);

    Mat face1_cropped = detectAlignCrop(face1, cascade, recognizer);
    Mat face2_cropped = detectAlignCrop(face2, cascade, recognizer);

    ////Extract feature from images
    Mat face1_feature;
    Mat face2_feature;
    recognizer.ExtractFaceFeature(face1_cropped, face1_feature);
    recognizer.ExtractFaceFeature(face2_cropped, face2_feature);

    float similarity =
        recognizer.CalculateSimilarity(face1_feature, face2_feature);
    if ((gt && similarity >= 0.5f) || (!gt && similarity < 0.5f))
      ++correct;
    // float distance = FaceDistance(recognizer, face1_feature, face2_feature);
    ++fnum;
    cout << "pair num: " << fnum << "similarity: " << similarity << endl;
  }

  cout << "Precision: " << (float)correct / fnum << endl;
}

void validate_on_prepared_data(LightFaceRecognizer &recognizer) {
  ifstream pair_file("../../../lfw_data/BW100issame.txt");

  bool gt;
  int fnum = 0;
  int correct = 0;
  while (pair_file >> gt) {
    // Load face images
    std::ostringstream face1_string_stream;
    face1_string_stream << "../../../lfw_data/100BW/" << (fnum * 2 + 1)
                        << ".png";
    std::string face1_name = face1_string_stream.str();
    std::ostringstream face2_string_stream;
    face2_string_stream << "../../../lfw_data/100BW/" << (fnum * 2 + 2)
                        << ".png";
    std::string face2_name = face2_string_stream.str();
    cout << face1_name << "\t" << face2_name << endl;

    Mat face1_feature;
    Mat face2_feature;
    Mat face1 = imread(face1_name);
    Mat face2 = imread(face2_name);
    recognizer.ExtractFaceFeature(face1, face1_feature);
    recognizer.ExtractFaceFeature(face2, face2_feature);

    float similarity =
        recognizer.CalculateSimilarity(face1_feature, face2_feature);

    if ((gt && similarity >= 0.5f) || (!gt && similarity < 0.5f))
      ++correct;
    ++fnum;
    cout << "pair num: " << fnum << "similarity: " << similarity << endl;
  }
  cout << "Precision: " << (float)correct / fnum << endl;
}

float FaceVerification(LightFaceRecognizer &recognizer,
                       CascadeClassifier &cascade, const string &f1_name,
                       const string &f2_name) {
  Mat face1 = imread(f1_name);
  Mat face2 = imread(f2_name);
  Mat face1_cropped = detectAlignCrop(face1, cascade, recognizer);
  Mat face2_cropped = detectAlignCrop(face2, cascade, recognizer);
  imshow("face1_cropped", face1_cropped);
  imshow("face2_cropped", face2_cropped);
  waitKey(0);
  Mat face1_feature, face2_feature;
  recognizer.ExtractFaceFeature(face1_cropped, face1_feature);
  recognizer.ExtractFaceFeature(face2_cropped, face2_feature);
  float similarity =
      recognizer.CalculateSimilarity(face1_feature, face2_feature);
  float cos_distance =
      recognizer.CalculateCosDistance(face1_feature, face2_feature);
  float bayesian_distance =
      recognizer.CalculateDistance(face1_feature, face2_feature);
  cout << "Cos distance: " << cos_distance
       << "\tBayesian distance: " << bayesian_distance << endl;
  return similarity;
}

void FaceSearch(LightFaceRecognizer &recognizer, CascadeClassifier &cascade,
                const string &target_name, const string &dir_name) {
  Mat target_face = imread(target_name);
  Mat target_face_cropped = detectAlignCrop(target_face, cascade, recognizer);
  imshow("target_face_cropped", target_face_cropped);
  //waitKey(0);
  Mat target_face_feat;
  recognizer.ExtractFaceFeature(target_face_cropped, target_face_feat);
  vector<fs::path> files;
  get_all("../../../test_faces", ".jpg", files);
  for (int i = 0; i < files.size(); ++i) {
    cout<<files[i].string()<<"\t";
    Mat face2 = imread(files[i].string());
    Mat face2_cropped = detectAlignCrop(face2, cascade, recognizer);
    imshow("face2_cropped", face2_cropped);
    //waitKey(0);
    Mat face2_feature;
    recognizer.ExtractFaceFeature(face2_cropped, face2_feature);
    //float similarity =
        //recognizer.CalculateSimilarity(target_face_feat, face2_feature);
    float cos_distance =
        recognizer.CalculateCosDistance(target_face_feat, face2_feature);
    float bayesian_distance =
        recognizer.CalculateDistance(target_face_feat, face2_feature);
    cout << "Cos distance: " << cos_distance
         << "\tBayesian distance: " << bayesian_distance << endl;
  }

  // Mat face2 = imread(f2_name);
  // Mat face2_cropped = detectAlignCrop(face2, cascade, recognizer);
  // imshow("face1_cropped", face1_cropped);
  // imshow("face2_cropped", face2_cropped);
  // waitKey(0);
  // Mat face1_feature, face2_feature;
  // recognizer.ExtractFaceFeature(face2_cropped, face2_feature);
  // float similarity =
  // recognizer.CalculateSimilarity(face1_feature, face2_feature);
  // float cos_distance =
  // recognizer.CalculateCosDistance(face1_feature, face2_feature);
  // float bayesian_distance =
  // recognizer.CalculateDistance(face1_feature, face2_feature);
  // cout << "Cos distance: " << cos_distance
  //<< "\tBayesian distance: " << bayesian_distance << endl;
  // return similarity;
}

int main(int argc, char **argv) {
  // Init Recognizer
  // void *recognizer = InitRecognizer("../../models/big/big.prototxt",
  //"../../models/big/big.caffemodel", "");
  LightFaceRecognizer recognizer(
      "../../../face_rec_models/model_cnn/small",
      "../../../face_rec_models/model_face_alignment",
      "../../../face_rec_models/model_bayesian/bayesian_model_lfw.bin", "prob",
      false);

  // validate_on_lfw_data(recognizer);
  // validate_on_prepared_data(recognizer);

  CascadeClassifier cascade;
  // -- 1. Load the cascades
  if (!cascade.load(cascadeName)) {
    cerr << "ERROR: Could not load classifier cascade" << endl;
    return 0;
  }
  //float similarity = FaceVerification(recognizer, cascade, argv[1], argv[2]);
  //cout << "similartiy: " << similarity << endl;
  FaceSearch(recognizer, cascade, argv[1], argv[2]);
  return 0;
}
