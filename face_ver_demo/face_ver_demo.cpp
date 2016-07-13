#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "face_recognition.hpp"
#include "boost/filesystem.hpp"
#include "face_repository.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "face_align.h"

namespace fs = ::boost::filesystem;

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

void getAllFiles(const fs::path &root, const string &ext,
                 vector<fs::path> &ret) {
  if (!fs::exists(root) || !fs::is_directory(root))
    return;

  fs::recursive_directory_iterator it(root);
  fs::recursive_directory_iterator endit;

  while (it != endit) {
    if (fs::is_regular_file(*it) && it->path().extension() == ext)
      ret.push_back(it->path());
    ++it;
  }
}

string getNewFileName(string ext) {
  srand((int)time(0));
  int num = rand() % 1000;
  struct tm *p;
  time_t second;
  time(&second);
  p = localtime(&second);
  char buf[100] = {'\0'};
  sprintf(buf, "%04d%02d%02d_%02d%02d%02d_%03d", 1900 + p->tm_year,
          1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec, num);
  return string(buf) + ext;
}

float dist2sim(float dist) {
  float maxDist = 2;
  if (dist < 0 || dist > maxDist)
    return 0;
  return (maxDist - dist) / maxDist;
}

// TODO
float affineDist(const Mat &H1, const Mat &H2) {
  assert(H1.size() == Size(2, 3));
  assert(H2.size() == Size(2, 3));
  return -1;
}

Mat affine2square(const Mat &H) {
  assert(H.size() == Size(2, 3));

  Mat M(3, 3, H.type());
  H.row(0).copyTo(M.row(0));
  H.row(1).copyTo(M.row(1));
  M.at<double>(2, 0) = 0.0;
  M.at<double>(2, 1) = 0.0;
  M.at<double>(2, 2) = 1.0;

  return M;
}

bool faceVerfication(LightFaceRecognizer &recognizer, FaceAlign &face_align,
                     fs::path &path, VideoCapture &cap) {
  // Parameter settings.
  int num_max_check_frame = 150;
  int num_select_sample = 5;
  int num_min_accept_frame = 5;
  double dist_threshold = 0.6;
  int size_face_least = 160;

  cout << "Verification: " << path.string() << endl;

  string winName = "Face Verification";
  namedWindow(winName);

  // Get register faces.
  vector<fs::path> all_image_path;
  getAllFiles(path, ".jpg", all_image_path);
  if (0 >= all_image_path.size()) {
    cerr << "The person with name " << path.filename() << " does not exist!"
         << endl;
    return false;
  }

  // Select face samples and compute their feature.
  vector<Mat> features;
  Mat sample_to_show;
  srand(time(0));
  for (int i = 0; i < num_select_sample && i < all_image_path.size(); i++) {
    int j = rand() % all_image_path.size();
    Mat face = imread(all_image_path[j].string());
    Mat feature;
    recognizer.ExtractFaceFeature(face, feature);
    features.push_back(feature);
    if (0 == i) {
      resize(face, sample_to_show, Size(100, 100));
    }
  }

  Mat frame;
  bool is_start = false;
  int num_accepted = 0;
  for (int i = 0; i <= num_max_check_frame; i = i + (is_start ? 1 : 0)) {
    // Get frame.
    cap >> frame;

    // Detect face and show.
    Rect face_rect;
    face_align.detectFace(frame, face_rect);
    if (face_rect.width < size_face_least || face_rect.height < size_face_least) {
      imshow(winName, frame);
      char key = (char)waitKey(30);
      continue;
    }

    // Extract face feature
    Mat face = face_align.align(frame, face_rect, 192,
                                FaceAlign::INNER_EYES_AND_BOTTOM_LIP, 0.3);
    Mat f;
    recognizer.ExtractFaceFeature(face, f);

    // Compare with register face features.
    vector<double> dists;
    for (int j = 0; j < features.size(); j++) {
      dists.push_back(norm(f, features[j]));
    }
    vector<double>::iterator min_iter = min_element(dists.begin(), dists.end());
    cout << i << ": " << *min_iter << endl;

    // Check verification criteria.
    if (is_start && *min_iter < dist_threshold) {
      num_accepted++;
      char buf[50] = {'\0'};
      sprintf(buf, "Checking %d/%d", num_accepted, num_min_accept_frame);
      putText(frame, buf, Point(frame.cols - 250, 50), CV_FONT_HERSHEY_COMPLEX,
              1, Scalar(255, 0, 0));
    }

    // Show image.
    rectangle(frame, face_rect, Scalar(255, 0, 255));
    Mat roi = Mat(frame, Rect(0, frame.rows - 100, 100, 100));
    sample_to_show.copyTo(roi);
    char buf[20] = {'\0'};
    sprintf(buf, "%4f", dist2sim(*min_iter));
    putText(frame, buf, Point(20, 50), CV_FONT_HERSHEY_COMPLEX, 1,
            Scalar(255, 0, 0));
    string result = num_accepted == num_min_accept_frame
                        ? "SUCCESS"
                        : (i == num_max_check_frame ? "FAIL" : "");
    putText(frame, result, Point(frame.cols / 2, 100), CV_FONT_HERSHEY_COMPLEX,
            1, Scalar(255, 0, 0));
    imshow(winName, frame);

    // Wait key.
    char key;
    if (is_start &&
        (num_accepted == num_min_accept_frame || i == num_max_check_frame)) {
      key = (char)waitKey(-1);
      return num_accepted == num_min_accept_frame;
    } else
      key = (char)waitKey(30);
    switch (key) {
    case 27:
      return false;
    case ' ':
      is_start = true;
      break;
    }
  }
}

bool faceRegister(LightFaceRecognizer &recognizer, FaceAlign &face_align,
                  fs::path &path, VideoCapture &cap) {
  int size_face_least = 160;

  cout << "Register: " << path.string() << endl;
  string winName = "Face Register";
  namedWindow(winName);
  Mat frame;
  for (;;) {
    cap >> frame;
    Mat origin = frame.clone();

    Rect face_rect;
    face_align.detectFace(frame, face_rect);
    bool face_detected = true;
    if (face_rect.width < size_face_least || face_rect.height < size_face_least )
      face_detected = false;
    if (face_detected)
      rectangle(frame, face_rect, Scalar(255, 0, 255));
    imshow(winName, frame);

    char key = (char)waitKey(30);
    switch (key) {
    case 27:
      return false;
    case ' ':
      if (face_detected) {
        Mat face = face_align.align(origin, face_rect, 192,
                                    FaceAlign::INNER_EYES_AND_BOTTOM_LIP, 0.3);
        string filename = getNewFileName(".jpg");
        fs::path save_path = path / filename;
        cout << "Save face image to: " << save_path.string() << endl;
        fs::create_directories(path);
        imwrite(save_path.string(), face);
        waitKey(-1);
        return true;
      }
    }
  }
}

int main(int argc, char **argv) {
  // Initial face recognizer and face aligner
  LightFaceRecognizer recognizer(
      "../../models/cnn", "../../models/bayesian_model_lfw.bin", "prob", false);
  FaceAlign face_align(
      "../../models/dlib_shape_predictor_68_face_landmarks.dat");
  string image_root("../../images");
  int camera_index = 0;

  string action, name;
  if (3 == argc) {
    action = argv[1];
    name = argv[2];
  }

  if (action.empty() || name.empty() || "ver" != action && "reg" != action) {
    cout << "Usage: ./face_ver [reg|ver] [person_name]" << endl;
    cout << "Press Space to start process, ESC to close window" << endl;
    return -1;
  }

  // Get a handle to the camera
  VideoCapture cap(camera_index);
  if (!cap.isOpened()) {
    cerr << "Camera " << camera_index << " cannot be opened." << endl;
    return -1;
  }

  // Get face image path
  fs::path folder(image_root);
  folder /= name;

  // Action
  if ("ver" == action) {
    faceVerfication(recognizer, face_align, folder, cap);
  } else {
    faceRegister(recognizer, face_align, folder, cap);
  }

  return 0;
}
