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
#include <iostream>
//#include "boost/filesystem.hpp"
using namespace std;
using namespace cv;
//using namespace boost::filesystem;
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

// parameters
Params global_params;

string modelPath = "./../../model_69/";
string dataPath;
string cascadeName = "haarcascade_frontalface_alt.xml";

void prepare_lfw_data();
int main(int argc, const char **argv) {
  // initialize parameters
  ReadGlobalParamFromFile(modelPath + "LBF.model");
  //return FaceDetectionAndAlignment("");
  prepare_lfw_data();
  return 0;
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

//int save_count = 0;

//int FaceDetectionAndAlignment(const char *inputname) {
  //string inputName;
  //CvCapture *capture = 0;
  //Mat frame, frameCopy, image;
  //bool tryflip = false;
  //double scale = 1.3;
  //CascadeClassifier cascade;

  //// name is empty or a number
  //capture = cvCaptureFromCAM(0);
  //// -- 0. Load LBF model
  //LBFRegressor regressor;
  //regressor.Load(modelPath + "LBF.model");

  //// -- 1. Load the cascades
  //if (!cascade.load(cascadeName)) {
    //cerr << "ERROR: Could not load classifier cascade" << endl;
    //return -1;
  //}

  //// cvNamedWindow( "result", 1 );
  //// -- 2. Read the video stream
  //if (capture) {
    //cout << "In capture ..." << endl;
    //for (;;) {
      //IplImage *iplImg = cvQueryFrame(capture);
      //frame = cvarrToMat(iplImg);
      //if (frame.empty())
        //break;
      //if (iplImg->origin == IPL_ORIGIN_TL)
        //frame.copyTo(frameCopy);
      //else
        //flip(frame, frameCopy, 0);

      //detectAndDraw(frameCopy, cascade, regressor, scale, tryflip);

      //if (waitKey(10) >= 0)
        //goto _cleanup_;
    //}

    //waitKey(0);

  //_cleanup_:
    //cvReleaseCapture(&capture);
  //}
  //cvDestroyWindow("result");

  //return 0;
//}

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

Mat detectAndAlign(Mat &img, CascadeClassifier &cascade,
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
  if(faces.size()<=0)
  {
    cout<<"Cannot detect face!\n";
    return Mat(img, Rect(68-113.f*scale, 68-113.f*scale, 113+113.f*scale, 113+113.f*scale));
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

  //for (int i = 0; i < 46; i += 45 - 36) {
  //for(int i = 0;i < global_params.landmark_num;i++){
    //circle(img, Point2d(current_shape(i, 0), current_shape(i, 1)), 3,
           //Scalar(255, 255, 255), -1, 8, 0);
    ////stringstream id;
    ////id << i;
    ////putText(img, id.str(), Point2d(current_shape(i, 0), current_shape(i, 1)),
            ////CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
  //}
  //imshow("result", img);
  //waitKey(10);
  Mat after_aligned;
  FaceAlign(img, Point2d(current_shape(36, 0), current_shape(36, 1)),
            Point2d(current_shape(45, 0), current_shape(45, 1)), after_aligned);


  //rectangle(after_aligned, origin_max_face, Scalar(255,0,0));
  //rectangle(img, origin_max_face, Scalar(255,0,0));

  //detect faces on aligned image again
  cascade.detectMultiScale(after_aligned, faces, 1.1, 2,
                           0
                               //|CV_HAAR_FIND_BIGGEST_OBJECT
                               //|CV_HAAR_DO_ROUGH_SEARCH
                               |
                               CV_HAAR_SCALE_IMAGE,
                           Size(30, 30));

  if(faces.size()<=0)
  {
    cout<<"Cannot detect face after aligned!\n";
    int x;
    //cin>>x;
    return Mat(img, Rect(68-113.f*scale, 68-113.f*scale, 113+113.f*scale, 113+113.f*scale));
  }
  max_face = FindMaxFace(faces);
  int w = max_face.width;
  int h = max_face.height;
  int s = std::max(w,h);
  int x = max_face.x - (s-w)*0.5f;
  int y = max_face.y - (s-h)*0.5f;
  int left_dist = x;
  int right_dist = img.cols-x-s;
  int hor_min_dist = std::min(left_dist, right_dist);
  int top_dist = y;
  int bot_dist = img.rows-y-s;
  int max_hor_padding = hor_min_dist*2;
  int max_ver_padding = std::min(top_dist, (int)(bot_dist*scale_top_ratio/(1.f-scale_top_ratio)))*2;
  int max_padding = std::min(max_hor_padding, max_ver_padding);
  int padding = std::min(max_padding, (int)(s*scale));
  x = x-padding/2;
  y = y-padding*scale_top_ratio;
  s = s+padding;

  Rect roi(x,y,s,s);
  //Mat aligned_show = after_aligned.clone();
  //rectangle(aligned_show, max_face, Scalar(0,255,0));
  //rectangle(aligned_show, roi, Scalar(255,0,0));
  //imshow("aligned", aligned_show);
  //waitKey(0);
  return Mat(after_aligned, roi);


  //imshow("aligned", after_aligned);
  //cv::imshow("result", img);
  //char a = waitKey(10);
  //if (a == 's') {
    //save_count++;
    //imwrite(to_string(save_count) + ".jpg", img);
  //}
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


void prepare_lfw_data() {
  ifstream pairs_file("../../lfw_data/pairs.txt");
  ofstream gt_file("../../lfw_data/prepared_self.txt");

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
    gt_file<<gt<<endl;
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


    Mat face1_cropped = detectAndAlign(face1, cascade, regressor);
    Mat face2_cropped = detectAndAlign(face2, cascade, regressor);
    // crop
    //Mat face1_cropped(face1, Rect(68, 68, 113, 113));
    //Mat face2_cropped(face2, Rect(68, 68, 113, 113));
    //imshow("face1_cropped", face1_cropped);
    //imshow("face2_cropped", face2_cropped);

    //std::ostringstream face1_string_stream_store;
    //face1_string_stream_store << "../../lfw_data/lfw_cropped/" << name1 << "/" << name1 << "_"
                        //<< setw(4) << setfill('0') << id1 << ".jpg";
    //std::string face1_name_store = face1_string_stream_store.str();
    //std::ostringstream face2_string_stream_store;
    //face2_string_stream_store << "../../lfw_data/lfw_cropped/" << name2 << "/" << name2 << "_"
                        //<< setw(4) << setfill('0') << id2 << ".jpg";
    //std::string face2_name_store = face2_string_stream_store.str();

    //path face1_path(face1_name_store);
    //path face2_path(face2_name_store);
    //if(!exists(face1_path))
      //create_directory(face1_path);
    //if(!exists(face2_path))
      //create_directory(face2_path);

    if(face1_cropped.cols<=0 || face2_cropped.cols<=0){
      cout<<"error"<<endl;
      int x;
      cin>>x;
    }

    imwrite("../../lfw_data/faces_cropped/"+to_string(fnum*2)+".jpg", face1_cropped);
    imwrite("../../lfw_data/faces_cropped/"+to_string(fnum*2+1)+".jpg", face2_cropped);

    //waitKey(10);

    //// compute frame per second (fps)
    ////int64 start_tick = getTickCount();

    ////Extract feature from images
    ++fnum;
    cout << "pair num: " << fnum << endl;
  }
  gt_file.close();
}
