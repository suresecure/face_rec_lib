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
#include "flann/flann.hpp"
#include "flann/io/hdf5.h"
#include "face_repository.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "face_align.h"

namespace fs = ::boost::filesystem;

using namespace cv;
using namespace std;
using namespace face_rec_srzn;

#define  FEATURE_DIM (160) 
typedef float FEATURE_TYPE;
string cascadeName = "../../../face_rec_models/haarcascade_frontalface_alt.xml";

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
  //recognizer.CropFace(gray, max_face, roi);
  recognizer.CropFace(after_aligned, max_face, roi);
  return Mat(after_aligned_org, roi);
}

/*
 * The formal detectAlignCrop() function is mainly for lfw images, 
 * where left-top corner is returned if no face detected.
 * This function will return empty Mat in this case. 
 */
Mat detectAlignCropPractical(const Mat &img, CascadeClassifier &cascade,
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
    return Mat();
  }

  // --Alignment
  max_face = FindMaxFace(faces);

  Mat after_aligned, after_aligned_org;
  recognizer.FaceAlign(gray_org, max_face, after_aligned_org);
  equalizeHist(after_aligned_org, after_aligned);

  // detect faces on aligned image again
  faces.clear();
  cascade.detectMultiScale(after_aligned, faces, 1.1, 2,
      0
      //|CV_HAAR_FIND_BIGGEST_OBJECT
      //|CV_HAAR_DO_ROUGH_SEARCH
      |
      CV_HAAR_SCALE_IMAGE,
      Size(30, 30));

  if (faces.size() <= 0) {
    cout << "Cannot detect face after aligned!\n";
    return Mat();
  }
  max_face = FindMaxFace(faces);
  Rect roi;
  //recognizer.CropFace(gray, max_face, roi);
  recognizer.CropFace(after_aligned, max_face, roi);
  //cout<<faces.size()<<endl;
  //for (int i = 0; i < faces.size(); i++)
    //cout<<faces[i].size()<<endl;
  //cout<<endl;
  //cout<<"max_face: "<<max_face<<endl;
  //cout<<"roi: "<<roi<<endl;
  //cout<<"Aligned image size: "<<after_aligned.size()<<endl;
  return Mat(after_aligned_org, roi);
}

/*
 * Find and crop face by dlib.
 */
Mat detectAlignCropDlib(FaceAlign & face_align, const Mat &img) {
  // Detection
  /* Dlib detects a face larger than 80x80 pixels.
   * We can use pyramid_up to double the size of image,
   * then dlib can find faces in size of 40x40 pixels in the original image.*/
  // dlib::pyramid_up(img);
  dlib::cv_image<dlib::bgr_pixel> cimg(img);
  // std::vector<dlib::rectangle> dets = face_align.getAllFaceBoundingBoxes(cimg);
  std::vector<dlib::rectangle> dets;
  dets.push_back(face_align.getLargestFaceBoundingBox(cimg)); // Use the largest detected face only
  // cout<<dets[0]<<endl; // position of the largest detected face
  if (0 == dets.size() || dets[0].is_empty())
  {
    cout << "Cannot detect face!\n";
    return Mat();
  }

  // --Alignment
  return face_align.align(cimg, dets[0]);
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
    recognizer.CalculateBayesianDistance(face1_feature, face2_feature);
  cout << "Cos distance: " << cos_distance
    << "\tBayesian distance: " << bayesian_distance << endl;
  return similarity;
}

void FaceSearch(LightFaceRecognizer &recognizer, CascadeClassifier &cascade,
    const string &target_name, const string &dir_name) {
  Mat target_face = imread(target_name);
  Mat target_face_cropped = detectAlignCrop(target_face, cascade, recognizer);
  imshow("target_face_cropped", target_face_cropped);
  // waitKey(0);
  Mat target_face_feat;
  recognizer.ExtractFaceFeature(target_face_cropped, target_face_feat);
  vector<fs::path> files;
  // vector<float> distances;
  vector<pair<fs::path, float>> distances;
  get_all("../../../test_faces", ".jpg", files);
  for (int i = 0; i < files.size(); ++i) {
    //cout << files[i].string() << "\t";
    Mat face2 = imread(files[i].string());
    Mat face2_cropped = detectAlignCrop(face2, cascade, recognizer);
    //imshow("face2_cropped", face2_cropped);
    //waitKey(0);
    Mat face2_feature;
    recognizer.ExtractFaceFeature(face2_cropped, face2_feature);
    // float similarity =
    // recognizer.CalculateSimilarity(target_face_feat, face2_feature);
    float cos_distance =
        recognizer.CalculateCosDistance(target_face_feat, face2_feature);
    distances.push_back(pair<fs::path, float>(files[i], cos_distance));
    float bayesian_distance =
        recognizer.CalculateBayesianDistance(target_face_feat, face2_feature);
    // cout << "Cos distance: " << cos_distance
    //<< "\tBayesian distance: " << bayesian_distance << endl;
  }
  std::sort(std::begin(distances), std::end(distances),
            [](const std::pair<fs::path, float> &left,
               const std::pair<fs::path, float> &right) {
              return left.second > right.second;
            });
  for (vector<pair<fs::path, float>>::iterator s = distances.begin();
       s != distances.end(); ++s) {
    cout << s->first << ": " << s->second << endl;
  }
  //[&](int i1, int i2) { return distances[i1] < distances[i2]; });

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
/*
void retrieval_on_lfw_batch(LightFaceRecognizer & recognizer,
    CascadeClassifier &cascade) {

  string lfw_indices_file = "../../../lfw_data/dataset_FaceRepo/lfw_indices.hdf5";
  string lfw_dists_file = "../../../lfw_data/dataset_FaceRepo/lfw_dists.hdf5";
  string lfw_directory = "../../../lfw_data/dataset_FaceRepo";
  string lfw_file_name = "../../../lfw_data/lfw_file_name.txt";
  string lfw_same_face_count = "../../../lfw_data/lfw_same_face_count.txt";

  FaceRepo faceRepo(recognizer, cascade);

  // Try load existed index. Create index if load failed, which takes quite a long time.
  bool bFreshRun = !faceRepo.Load(lfw_directory);
  if (bFreshRun) {
    long time1 = clock();
    faceRepo.InitialIndex(lfw_file_name);
    long time2 = clock();
    // Save index and features.
    faceRepo.Save(lfw_directory);
  }

  //// Rebuild index. FLANN index appears some kind of randomization, so you will get different return list if uncomment next line. 
  //faceRepo.RebuildIndex();

  // load dataset features 
  ::flann::Matrix<FEATURE_TYPE> dataset;
  ::flann::load_from_file(dataset, lfw_directory+"/dataset.hdf5", "dataset");
  // load all path of faces in the dataset 
  vector <string> data_file_path;
  ifstream inFile(lfw_file_name);
  string line;
  while (getline(inFile, line)) {
    data_file_path.push_back(line);
  }
  
  // Read face count
  int N = faceRepo.GetFaceNum();
  vector<int> face_count(N, 0);
  ifstream ifs_face_count(lfw_same_face_count.c_str());
  for (int i = 0; getline(ifs_face_count, line); i++) {
    face_count[i] = atoi(line.c_str());  
  }
  ifs_face_count.close();

  // Do retrieval
  int nFaces_UpBound = 100;  // Up bound of retrieval faces
  ::flann::Matrix<FEATURE_TYPE> dists;
  ::flann::Matrix<int> indices;
  //if (bFreshRun) {
  if (true) {
    long time3 = clock();
    // Search result matrix
    dists = ::flann::Matrix<FEATURE_TYPE> (new FEATURE_TYPE[N*nFaces_UpBound], N, nFaces_UpBound);
    indices = ::flann::Matrix<int> (new int[N*nFaces_UpBound], N, nFaces_UpBound);
    // Search one by one
    for (int i = 0; i < N; i++) {
      //cout<<i<<": "<<query[0][FEATURE_DIM-1]<<"\t"<<dataset[i][FEATURE_DIM-1]<<endl;
      vector<vector<string> > return_list;
      vector<vector<int> > return_list_pos;
      vector<vector<float> > return_dists;
      // Query by previously computed face features. 
      faceRepo.Query(::flann::Matrix<FEATURE_TYPE>(dataset[i], 1, FEATURE_DIM), nFaces_UpBound, return_list, return_list_pos, return_dists);
      //// Query by image. SLOW because you need to extract face feature for each image. 
      //faceRepo.Query(data_file_path[i], nFaces_UpBound, return_list, return_list_pos, return_dists);
      for (int j = 0; j < nFaces_UpBound; j++) {
        indices[i][j] = return_list_pos[0][j];
        dists[i][j] = return_dists[0][j];
      }
      //waitKey(0);
    }
    // Remove existed hdf5 files, otherwise an error will happen when size, of the data to be written, exceed the existed ones.
    remove(lfw_dists_file.c_str());
    remove(lfw_indices_file.c_str());
    ::flann::save_to_file(dists, lfw_dists_file, "dists");
    ::flann::save_to_file(indices, lfw_indices_file, "indices");
    long time4 = clock();
    cout<<"Query time:\t"<<float(time4-time3)/1000000<<" sec., ";
    cout<<float(time4-time3)/N<<" us per query."<<endl;
  }
  else {
    ::flann::load_from_file(dists, lfw_dists_file, "dists");
    ::flann::load_from_file(indices, lfw_indices_file, "indices");
  }

  // Statistics
  vector<int> num_images(nFaces_UpBound, 0); // number of images who have more than XX faces
  vector< vector<float> > precision_per_rank;
  vector< vector<int> > correct;
  for (int i = 0; i < nFaces_UpBound; i++) {
    precision_per_rank.push_back(vector<float>(nFaces_UpBound, 0));
    correct.push_back(vector<int>(N, 0));
  }
  for (int i = 0; i < N; i++) {
    string class_name = fs::canonical(data_file_path[i]).parent_path().filename().string();
    for (int j = 0; j < nFaces_UpBound; j++) {
      if (face_count[i] < j + 1)
        break;
      num_images[j]++;
      string class_name_j = fs::canonical(data_file_path[indices[i][j]]).parent_path().filename().string();
      if (class_name_j == class_name) {
        correct[j][i] = 1;
      }
    }
  }
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < nFaces_UpBound; k++) {
      if (face_count[i] > k) {
        for (int j = 0; j <= k; j++) {
          for (int m = j; m <= nFaces_UpBound; m++) {
            precision_per_rank[k][m] += correct[j][i];
          }
        }
      }
    }
  }
  for (int k = 0; k < nFaces_UpBound; k++) {
    for (int j = 0; j <= k; j++) {
      precision_per_rank[k][j] /= (num_images[k] * (j+1) );
    }
  }

  // Print results.
  for (int i = 0; i < nFaces_UpBound; i++) {
    if (num_images[i] < 1)
      break;
    cout<<"--------------------------------------------------"<<endl;
    cout<<"Images who have "<<i+1<<" faces: "<<num_images[i]<<" images"<<endl;
    cout<<"Precision per rank:"<<endl;
    for (int j = 0; j <= i; j++) {
      cout<<"Rank #"<<j+1<<": "<<precision_per_rank[i][j]<<endl;
    }
    cout<<endl;
  }

  delete [] dists.ptr();
  delete [] indices.ptr();
}

void retrieval_on_lfw(LightFaceRecognizer & recognizer,
    CascadeClassifier &cascade, 
    const string & query,
    const size_t num_return = 10) {
  string lfw_directory = "../../../lfw_data/dataset_FaceRepo";
  string lfw_file_name = "../../../lfw_data/lfw_file_name.txt";
  // Try load existed index. Create index if load failed, which takes quite a long time.
  FaceRepo faceRepo(recognizer, cascade);
  bool bFreshRun = !faceRepo.Load(lfw_directory);
  if (bFreshRun) {
    long time1 = clock();
    faceRepo.InitialIndex(lfw_file_name);
    long time2 = clock();
    // Save index and features.
    faceRepo.Save(lfw_directory);
  }

  vector<vector<string> > return_list;  // Path of return images.
  vector<vector<int> > return_list_pos; // Indices in the face dataset of the return images.
  vector<vector<float> > return_dists; // Distance between return and query images.
  // Do query. 
  faceRepo.Query(query, num_return, return_list, return_list_pos, return_dists);

  cout<<"Return list of \""<<query<<"\" in LFW dataset:"<<endl;
  for (int i = 0; i < num_return; i++)
    cout<<"#"<<i<<": "<<return_list[0][i]<<endl;
}

void FindValidFace(LightFaceRecognizer &recognizer, 
    CascadeClassifier &cascade,
    const string &image_root,
    const string &save_path,
    const string &ext = string(".jpg")) {

  // Read image file list 
  vector<fs::path> file_path;

  fs::path root(image_root); 
  get_all(root, ext, file_path);
  int N = file_path.size();
  if (0 == N)
  {
    cerr<<"No image found in the given path with \""<<ext<<"\" extension."<<endl;
    exit(-1);
  }

  // Detect face, then save to the disk.
  cout<<"Image(s):"<<endl;
  for (int i = 0; i < N; i++) {
    cout<<file_path[i]<<endl; 
    Mat face = imread(file_path[i].string());
    Mat face_cropped = detectAlignCropPractical(face, cascade, recognizer);
    if (face_cropped.empty())
      continue;

    //imshow("face_cropped", face_cropped);
    //waitKey(0);

    // Save to the disk.
    string relative_path = fs::canonical(file_path[i]).string().substr(fs::canonical(root).string().length());
    //cout<<relative_path<<endl;
    fs::path dest_path(save_path); 
    dest_path += fs::path(relative_path);
    fs::create_directories(dest_path.parent_path());
    cout<<"Save detected face to: "<<dest_path<<endl;
    imwrite(dest_path.string(), face_cropped);
  }
}
*/
void FindValidFaceDlib(FaceAlign & face_align, 
    const string &image_root,
    const string &save_path,
    const string &ext = string(".jpg")) {

  // Read image file list 
  vector<fs::path> file_path;

  fs::path root(image_root); 
  get_all(root, ext, file_path);
  int N = file_path.size();
  if (0 == N)
  {
    cerr<<"No image found in the given path with \""<<ext<<"\" extension."<<endl;
    exit(-1);
  }

  // Detect face, then save to the disk.
  cout<<"Image(s):"<<endl;
  for (int i = 0; i < N; i++) {
    cout<<file_path[i]<<endl; 
    Mat face = imread(file_path[i].string());
    Mat face_cropped = detectAlignCropDlib(face_align, face);
    if (face_cropped.empty())
      continue;

    // Save to the disk.
    string relative_path = fs::canonical(file_path[i]).string().substr(fs::canonical(root).string().length());
    //cout<<relative_path<<endl;
    fs::path dest_path(save_path); 
    dest_path += fs::path(relative_path);
    fs::create_directories(dest_path.parent_path());
    cout<<"Save detected face to: "<<dest_path<<endl;
    imwrite(dest_path.string(), face_cropped);
  }
}
/*
void retrieval_test(LightFaceRecognizer & recognizer,
    CascadeClassifier &cascade, 
    const string &file_name,
    const string &same_face_count,
    const string &save_dir
    ) {
  
  // Create index. It may take a while depends on the number of images to be indexed.
  FaceRepo faceRepo(recognizer, cascade);
  bool bFreshRun = !faceRepo.Load(save_dir);
  if (bFreshRun) {
    long time1 = clock();
    string input_path_file(file_name);
    faceRepo.InitialIndex(input_path_file);
    long time2 = clock();
    // Save index and features.
    fs::path save_root(save_dir);
    if (!fs::exists(save_root) || !fs::is_directory(save_root))
      fs::create_directories(save_root);
    faceRepo.Save(save_dir);
  }

  // load dataset features 
  ::flann::Matrix<FEATURE_TYPE> dataset;
  ::flann::load_from_file(dataset, save_dir+"/dataset.hdf5", "dataset");
  // load all path of faces in the dataset 
  vector <string> data_file_path;
  ifstream inFile(save_dir + "/dataset_file_path.txt");
  string line;
  while (getline(inFile, line)) {
    data_file_path.push_back(line);
  }
  int N = faceRepo.GetValidFaceNum();

  // read face count
  int n = data_file_path.size();
  vector<int> face_count(n, 0);
  ifstream ifs_face_count(same_face_count.c_str());
  for (int i = 0; getline(ifs_face_count, line); i++) {
    face_count[i] = atoi(line.c_str());  
  }
  ifs_face_count.close();

  // Do retrieval
  int nFaces_UpBound = N > 100 ? 100 : N - 1;  // Up bound of retrieval faces
  // Search result matrix
  ::flann::Matrix<FEATURE_TYPE> dists;
  ::flann::Matrix<int> indices;
  dists = ::flann::Matrix<FEATURE_TYPE> (new FEATURE_TYPE[N*nFaces_UpBound], N, nFaces_UpBound);
  indices = ::flann::Matrix<int> (new int[N*nFaces_UpBound], N, nFaces_UpBound);
  // Search one by one
  long time3 = clock();
  for (int i = 0; i < N; i++) {
    //cout<<i<<": "<<query[0][FEATURE_DIM-1]<<"\t"<<dataset[i][FEATURE_DIM-1]<<endl;
    vector<vector<string> > return_list;
    vector<vector<int> > return_list_pos;
    vector<vector<float> > return_dists;
    // Query by previously computed face features. 
    faceRepo.Query(::flann::Matrix<FEATURE_TYPE>(dataset[i], 1, FEATURE_DIM), nFaces_UpBound, return_list, return_list_pos, return_dists);
    //// Query by image. SLOW because you need to extract face feature for each image. 
    //faceRepo.Query(data_file_path[i], nFaces_UpBound, return_list, return_list_pos, return_dists);
    for (int j = 0; j < nFaces_UpBound; j++) {
      indices[i][j] = return_list_pos[0][j];
      dists[i][j] = return_dists[0][j];
    }
  }
  long time4 = clock();
  cout<<"Query time:\t"<<float(time4-time3)/1000000<<" sec., ";
  cout<<float(time4-time3)/N<<" us per query."<<endl;

  // Retrieval statistics
  cout<<endl;
  cout<<"**************************************"<<endl;
  cout<<"RETRIEVAL STATISTICS"<<endl;
  vector<int> num_images(nFaces_UpBound, 0); // number of images who have more than XX faces
  vector< vector<float> > precision_per_rank;
  vector< vector<int> > correct;
  for (int i = 0; i < nFaces_UpBound; i++) {
    precision_per_rank.push_back(vector<float>(nFaces_UpBound, 0));
    correct.push_back(vector<int>(N, 0));
  }
  for (int i = 0; i < N; i++) {
    string class_name = fs::canonical(data_file_path[i]).parent_path().filename().string();
    for (int j = 0; j < nFaces_UpBound; j++) {
      if (face_count[i] < j + 1)
        break;
      num_images[j]++;
      string class_name_j = fs::canonical(data_file_path[indices[i][j]]).parent_path().filename().string();
      if (class_name_j == class_name) {
        correct[j][i] = 1;
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int k = 0; k < nFaces_UpBound; k++) {
      if (face_count[i] > k) {
        for (int j = 0; j <= k; j++) {
          for (int m = j; m <= nFaces_UpBound; m++) {
            precision_per_rank[k][m] += correct[j][i];
          }
        }
      }
    }
  }
  for (int k = 0; k < nFaces_UpBound; k++) {
    for (int j = 0; j <= k; j++) {
      precision_per_rank[k][j] /= (num_images[k] * (j+1) );
    }
  }

  for (int i = 0; i < nFaces_UpBound; i++) {
    if (num_images[i] < 1)
      break;
    cout<<"--------------------------------------------------"<<endl;
    cout<<"Images who have "<<i+1<<" faces: "<<num_images[i]<<" images"<<endl;
    cout<<"Precision per rank:"<<endl;
    for (int j = 0; j <= i; j++) {
      cout<<"Rank #"<<j+1<<": "<<precision_per_rank[i][j]<<endl;
    }
    cout<<endl;
  }

  // Validation statistics
  cout<<endl;
  cout<<"**************************************"<<endl;
  cout<<"VALIDATION STATISTICS"<<endl;
  int n_class = 6; // Least number of faces for a person.
  int knn = 10; // Size of return knn;
  float th_dist = 0.8; // Distance threshold for same person.
  int th_n = 2; // Least number of retrieved knn with same label.
  int num_valid_test = 0;
  int num_correct_accept = 0;
  int num_wrong_accept = 0;
  int num_reject = 0;
  for (int i = 0; i < N; i++) {
    if (face_count[i] < n_class)
      continue;
    num_valid_test++; 
    // True class name.
    string class_name = fs::canonical(data_file_path[i]).parent_path().filename().string();
    // Analyze the returned knn:
    vector <string> class_name_return;
    vector <float> dist_return;
    vector <int> count_same_class;
    for (int j = 1; j < knn; j++) { //omit the first return (should the query itself)
      if (dists[i][j] > th_dist)
        break;
      string class_name_j = fs::canonical(data_file_path[indices[i][j]]).parent_path().filename().string();
      vector<string>::iterator iter = find(class_name_return.begin(), class_name_return.end(), class_name_j);
      if ( iter == class_name_return.end() ) {
        class_name_return.push_back(class_name_j);
        dist_return.push_back(dists[i][j]);
        count_same_class.push_back(1);
      } else {
        int pos = iter - class_name_return.begin();
        dist_return[pos] += dists[i][j];
        //dist_return[pos] += sqrt(j) * dists[i][j];
        count_same_class[pos] ++;
      }
    }
    cout<<data_file_path[i]<<endl;
    cout<<"True class name: "<<class_name<<", number of groups in knn:"<<class_name_return.size()<<endl;
    float min_dist = 10000.0;
    string predict_class_name;
    for (int j = 0; j < class_name_return.size(); j++) {
      dist_return[j] /= count_same_class[j];
      if (dist_return[j] < min_dist && count_same_class[j] > th_n) {
        min_dist = dist_return[j];
        predict_class_name = class_name_return[j];
      }
      cout<<j<<": "<<class_name_return[j]<<", count: "<<count_same_class[j]<<", ave dist: "<<dist_return[j]<<endl;
    }
    if (10000.0 == min_dist) {
      num_reject ++;
      continue;
    }
    if (predict_class_name == class_name)
      num_correct_accept++;
    else
      num_wrong_accept++;
    cout<<endl;
  }
  cout<<"----------------------------------------------"<<endl;
  cout<<"Number of valid validation test: "<<num_valid_test<<endl;
  cout<<"Number of correct accept: "<<num_correct_accept<<", "<<(float)num_correct_accept/num_valid_test*100<<endl;
  cout<<"Number of wrong accept: "<<num_wrong_accept<<", "<<(float)num_wrong_accept/num_valid_test*100<<endl;
  cout<<"Number of reject: "<<num_reject<<", "<<(float)num_reject/num_valid_test*100<<endl;

  delete [] dataset.ptr();
  delete [] dists.ptr();
  delete [] indices.ptr();
}
*/

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
  //FaceSearch(recognizer, cascade, argv[1], argv[2]);

  //if (1 == argc)
    //// Do query for each image in lfw dataset, and do statistic for the result.
    //retrieval_on_lfw_batch(recognizer, cascade);  
  //else
    //// Do query in lfw dataset for the input image. 
    //retrieval_on_lfw(recognizer, cascade, argv[1]);  

  //// argv[1]: Root path of face images; 
  //// argv[2]: Root path to save cropped face images.
  //FindValidFace(recognizer, cascade, argv[1], argv[2]);

  //// Detect and align face by using dlib.
  //// argv[1]: Root path of face images; 
  //// argv[2]: Root path to save cropped face images.
  FaceAlign face_align("../../../face_rec_models/shape_predictor_68_face_landmarks.dat");
  FindValidFaceDlib(face_align, argv[1], argv[2]);

  //// argv[1]: txt file, stored all image file path
  //// argv[2]: txt file, stored the class size of the image
  //// argv[3]: Folder name to save the extracted features and other data.
  //retrieval_test(recognizer, cascade, argv[1], argv[2], argv[3]);
  return 0;
}
