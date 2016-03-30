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
                     "../../models/small/small_mean_image.binaryproto",
                     "../../models/small/similarity.bin");

  ifstream pair_file("../../lfw_data/BW100issame.txt");
  // ifstream pair_file("../../lfw_data/COLOR200isname.txt");
  ofstream distance_file("distance_file.txt");

  bool gt;
  int fnum = 0;
  int correct = 0;
  while (pair_file >> gt) {
    // Load face images
    string face1_name =
        (boost::format("../../lfw_data/100BW/%1%.png") % (fnum * 2 + 1)).str();
    string face2_name =
        (boost::format("../../lfw_data/100BW/%1%.png") % (fnum * 2 + 2)).str();
    // string face1_name =
    //(boost::format("../../lfw_data//%1%.png") % (fnum * 2 + 1)).str();
    // string face2_name =
    //(boost::format("../../lfw_data/100BW/%1%.png") % (fnum * 2 + 2)).str();
    Mat face1 = imread(face1_name);
    Mat face2 = imread(face2_name);
    //// compute frame per second (fps)
    ////int64 start_tick = getTickCount();

    ////Extract feature from images
    vector<float> face1_feature = ExtractFaceFeatureFromBuffer(
        recognizer, face1.data, face1.cols, face1.rows);
    vector<float> face2_feature = ExtractFaceFeatureFromBuffer(
        recognizer, face2.data, face2.cols, face2.rows);
    float similarity = FaceVerification(recognizer, face1_feature, face2_feature);
    string result("Wrong!");
    if((similarity >= 0.9f && gt) ||
      (similarity < 0.9f && !gt))
    {
      ++correct;
      result = "Correct!";
    }

    ++fnum;
    cout << "pair num: " << fnum <<"\t"<< result <<endl;
  }
  cout << "Precision: " << (float)correct / (fnum + 1) << endl;
  return 0;
}
