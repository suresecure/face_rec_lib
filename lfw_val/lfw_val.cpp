#include <time.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <iomanip>

#include "opencv2/highgui/highgui.hpp"
#include "face_recognition.hpp"

using namespace cv;
using namespace std;
using namespace face_rec_srzn;

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

void validate_on_lfw_data(void *recognizer) {
  ifstream pairs_file("../../lfw_data/pairs.txt");
  // ifstream pair_file("../../lfw_data/COLOR224issame.txt");
  ofstream distance_file("distance_file.txt");
  vector<float> distance_vector;
  vector<bool> gt_vector;

  string line;
  getline(pairs_file, line);
  cout << line << endl;

  bool gt;
  int fnum = 0;
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
    //cout << name1 << "\t" << id1 << "\t" << name2 << "\t" << id2 << endl;

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
    Mat face1_cropped(face1, Rect(68, 68, 113, 113));
    Mat face2_cropped(face2, Rect(68, 68, 113, 113));

    //// compute frame per second (fps)
    ////int64 start_tick = getTickCount();

    ////Extract feature from images
    vector<float> face1_feature =
        ExtractFaceFeatureFromMat(recognizer, face1_cropped);
    vector<float> face2_feature =
        ExtractFaceFeatureFromMat(recognizer, face2_cropped);
    float distance = FaceDistance(recognizer, face1_feature, face2_feature);
    distance_vector.push_back(distance);
    gt_vector.push_back(gt);
    distance_file << distance << "\t" << gt << endl;
    ++fnum;
    //cout << "pair num: " << fnum << endl;
  }
  distance_file.close();

  float max_precision = 0.0f;
  float max_thd = 0.0f;
  for (int i = 0; i < distance_vector.size(); ++i) {
    int correct = 0;
    float thd = distance_vector[i];
    for (int j = 0; j < distance_vector.size(); ++j) {
      float dist = distance_vector[j];
      if ((dist <= thd && gt_vector[j]) || (dist > thd && !gt_vector[j]))
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
  return 0;
}

void validate_on_prepared_data(void *recognizer) {
  ifstream pair_file("../../lfw_data/BW100issame.txt");
  // ifstream pair_file("../../lfw_data/COLOR224issame.txt");
  ofstream distance_file("distance_file.txt");
  vector<float> distance_vector;
  vector<bool> gt_vector;

  bool gt;
  int fnum = 0;
  while (pair_file >> gt) {
    // Load face images
    std::ostringstream face1_string_stream;
    face1_string_stream << "../../lfw_data/100BW/" << (fnum * 2 + 1) << ".png";
    std::string face1_name = face1_string_stream.str();
    std::ostringstream face2_string_stream;
    face1_string_stream << "../../lfw_data/100BW/" << (fnum * 2 + 2) << ".png";
    std::string face2_name = face2_string_stream.str();
    // string face1_name =
    //(boost::format("../../lfw_data/224COLOR/%1%.png") % (fnum * 2 + 1)).str();
    // string face2_name =
    //(boost::format("../../lfw_data/224COLOR/%1%.png") % (fnum * 2 + 2)).str();
    Mat face1 = imread(face1_name);
    Mat face2 = imread(face2_name);
    //// compute frame per second (fps)
    ////int64 start_tick = getTickCount();

    ////Extract feature from images
    vector<float> face1_feature = ExtractFaceFeatureFromBuffer(
        recognizer, face1.data, face1.cols, face1.rows);
    vector<float> face2_feature = ExtractFaceFeatureFromBuffer(
        recognizer, face2.data, face2.cols, face2.rows);
    // float sum = 0;
    // for (int i = 0; i < face1_feature.size(); i++) {
    // sum += (face1_feature[i] - face2_feature[i]) *
    //(face1_feature[i] - face2_feature[i]);
    //}
    // sum = sqrtf(sum);
    float distance = FaceDistance(recognizer, face1_feature, face2_feature);
    distance_vector.push_back(distance);
    gt_vector.push_back(gt);
    distance_file << sum << "\t" << gt << endl;
    ++fnum;
    cout << "pair num: " << fnum << endl;
  }
  distance_file.close();

  float max_precision = 0.0f;
  float max_thd = 0.0f;
  for (int i = 0; i < distance_vector.size(); ++i) {
    int correct = 0;
    float thd = distance_vector[i];
    for (int j = 0; j < distance_vector.size(); ++j) {
      float dist = distance_vector[j];
      if ((dist <= thd && gt_vector[j]) || (dist > thd && !gt_vector[j]))
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
