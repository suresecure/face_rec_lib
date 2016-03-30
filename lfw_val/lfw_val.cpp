#include <time.h>
#include <stdio.h>
#include <fstream>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "face_recognition.hpp"
#include "boost/format.hpp"

using namespace cv;
using namespace std;
using namespace face_rec_srzn;
using boost::format;

int main(int argc, char **argv) {
  // Init Recognizer
  // void *recognizer = InitRecognizer("../../models/big/big.prototxt",
  //"../../models/big/big.caffemodel", "");
  void *recognizer =
      InitRecognizer("../../models/small/small.prototxt",
                     "../../models/small/small.caffemodel",
                     "../../models/small/small_mean_image.binaryproto");

  ifstream pair_file("../../lfw_data/BW100issame.txt");
  ofstream distance_file("distance_file.txt");
  vector<float> distance_vector;
  vector<bool> gt_vector;

  bool gt;
  int fnum = 0;
  while (pair_file >> gt) {
    // Load face images
    // string face1_name = string("../../lfw_data/100BW/1.png");
    // string face2_name = string("../../lfw_data/100BW/2.png");
    string face1_name =
        (boost::format("../../lfw_data/100BW/%1%.png") % (fnum * 2 + 1)).str();
    string face2_name =
        (boost::format("../../lfw_data/100BW/%1%.png") % (fnum * 2 + 2)).str();
    Mat face1 = imread(face1_name);
    Mat face2 = imread(face2_name);
    //// compute frame per second (fps)
    ////int64 start_tick = getTickCount();

    ////Extract feature from images
    vector<float> face1_feature = ExtractFaceFeatureFromBuffer(
        recognizer, face1.data, face1.cols, face1.rows);
    vector<float> face2_feature = ExtractFaceFeatureFromBuffer(
        recognizer, face2.data, face2.cols, face2.rows);
    float sum = 0;
    for (int i = 0; i < face1_feature.size(); i++) {
      sum += (face1_feature[i] - face2_feature[i]) *
             (face1_feature[i] - face2_feature[i]);
    }
    sum = sqrtf(sum);
    distance_vector.push_back(sum);
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
  return 0;
}
