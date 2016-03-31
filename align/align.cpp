//
//  LBF.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include <sstream>
#include "LBF.h"
#include "LBFRegressor.h"
using namespace std;
using namespace cv;

// parameters
Params global_params;

string modelPath = "./../../model_69/";
string dataPath;
string cascadeName = "haarcascade_frontalface_alt.xml";

int main(int argc, const char **argv) {
  // initialize parameters
  ReadGlobalParamFromFile(modelPath + "LBF.model");
  return FaceDetectionAndAlignment("");
}

void ReadGlobalParamFromFile(string path) {
  cout << "Loading GlobalParam..." << endl;
  ifstream fin;
  fin.open(path);
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

int save_count = 0;
void detectAndDraw(Mat &img, CascadeClassifier &nestedCascade,
                   LBFRegressor &regressor, double scale, bool tryflip);

int FaceDetectionAndAlignment(const char *inputname) {
  string inputName;
  CvCapture *capture = 0;
  Mat frame, frameCopy, image;
  bool tryflip = false;
  double scale = 1.3;
  CascadeClassifier cascade;

  // name is empty or a number
  capture = cvCaptureFromCAM(0);
  // -- 0. Load LBF model
  LBFRegressor regressor;
  regressor.Load(modelPath + "LBF.model");

  // -- 1. Load the cascades
  if (!cascade.load(cascadeName)) {
    cerr << "ERROR: Could not load classifier cascade" << endl;
    return -1;
  }

  // cvNamedWindow( "result", 1 );
  // -- 2. Read the video stream
  if (capture) {
    cout << "In capture ..." << endl;
    for (;;) {
      IplImage *iplImg = cvQueryFrame(capture);
      frame = cvarrToMat(iplImg);
      if (frame.empty())
        break;
      if (iplImg->origin == IPL_ORIGIN_TL)
        frame.copyTo(frameCopy);
      else
        flip(frame, frameCopy, 0);

      detectAndDraw(frameCopy, cascade, regressor, scale, tryflip);

      if (waitKey(10) >= 0)
        goto _cleanup_;
    }

    waitKey(0);

  _cleanup_:
    cvReleaseCapture(&capture);
  }
  cvDestroyWindow("result");

  return 0;
}

Rect FindMaxFace(const vector<Rect>& faces)
{
  float max_area = 0.f;
  Rect max_face;
  for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end();
       r++) {
    if (r->width * r->height > max_area) {
      max_area = r->width * r->height;
      max_face = *r;
    }
  }
  return max_face;
}

void FaceAlign(const Mat &orig, Point2d left_eye, Point2d right_eye,
               Mat &outputarray);
void detectAndDraw(Mat &img, CascadeClassifier &cascade,
                   LBFRegressor &regressor, double scale, bool tryflip) {
  int i = 0;
  double t = 0;
  vector<Rect> faces, faces2;
  const static Scalar colors[] = {CV_RGB(0, 0, 255),   CV_RGB(0, 128, 255),
                                  CV_RGB(0, 255, 255), CV_RGB(0, 255, 0),
                                  CV_RGB(255, 128, 0), CV_RGB(255, 255, 0),
                                  CV_RGB(255, 0, 0),   CV_RGB(255, 0, 255)};
  Mat gray,
      smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

  cvtColor(img, gray, CV_BGR2GRAY);
  resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
  equalizeHist(smallImg, smallImg);

  // --Detection
  cascade.detectMultiScale(smallImg, faces, 1.1, 2,
                           0
                               //|CV_HAAR_FIND_BIGGEST_OBJECT
                               //|CV_HAAR_DO_ROUGH_SEARCH
                               |
                               CV_HAAR_SCALE_IMAGE,
                           Size(30, 30));

  // --Alignment
  float max_area = 0.f;
  Rect max_face = FindMaxFace(faces);
  BoundingBox boundingbox;

  boundingbox.start_x = max_face.x * scale;
  boundingbox.start_y = max_face.y * scale;
  boundingbox.width = (max_face.width - 1) * scale;
  boundingbox.height = (max_face.height - 1) * scale;
  boundingbox.centroid_x = boundingbox.start_x + boundingbox.width / 2.0;
  boundingbox.centroid_y = boundingbox.start_y + boundingbox.height / 2.0;
  Rect origin_max_face(boundingbox.start_x, boundingbox.start_y,
      boundingbox.width, boundingbox.height);

  Mat_<double> current_shape = regressor.Predict(gray, boundingbox, 1);

  for (int i = 36; i < 46; i += 45 - 36) {
    circle(img, Point2d(current_shape(i, 0), current_shape(i, 1)), 3,
           Scalar(255, 255, 255), -1, 8, 0);
    stringstream id;
    id << i;
    putText(img, id.str(), Point2d(current_shape(i, 0), current_shape(i, 1)),
            CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
  }
  Mat after_aligned;
  FaceAlign(img, Point2d(current_shape(36, 0), current_shape(36, 1)),
            Point2d(current_shape(45, 0), current_shape(45, 1)), after_aligned);


  rectangle(after_aligned, origin_max_face, Scalar(255,0,0));
  rectangle(img, origin_max_face, Scalar(255,0,0));

  //detect faces on aligned image again
  cascade.detectMultiScale(after_aligned, faces, 1.1, 2,
                           0
                               //|CV_HAAR_FIND_BIGGEST_OBJECT
                               //|CV_HAAR_DO_ROUGH_SEARCH
                               |
                               CV_HAAR_SCALE_IMAGE,
                           Size(30, 30));

  max_face = FindMaxFace(faces);
  rectangle(after_aligned, max_face, Scalar(0,255,0));


  imshow("aligned", after_aligned);
  cv::imshow("result", img);
  char a = waitKey(0);
  if (a == 's') {
    save_count++;
    imwrite(to_string(save_count) + ".jpg", img);
  }
}

// Face Alignment
void FaceAlign(const Mat &orig, Point2d leftEye, Point2d rightEye,
               Mat &outputarray) {
  int desiredFaceWidth = orig.cols;
  int desiredFaceHeight = desiredFaceWidth;

  // Get the eyes center-point with the landmarks
  // Point2d leftEye = Point2d((landmarks[2] + landmarks[10]) * 0.5f,
  //(landmarks[3] + landmarks[11]) * 0.5f);
  // Point2d rightEye = Point2d((landmarks[4] + landmarks[12]) * 0.5f,
  //(landmarks[5] + landmarks[13]) * 0.5f);
  //;

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
