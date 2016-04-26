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

#define  FEATURE_DIM (160) 
typedef float FEATURE_TYPE;

namespace fs = ::boost::filesystem;

using namespace cv;
using namespace std;
using namespace face_rec_srzn;

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

namespace flann 
{
  /* Cosine distance with one all zero entry (<0, 0 ,..., 0>) vector is not 
   * defined.
   * A divide by zero exception will be triged in this case.
   */
  template<class T>
    struct CosDistance
    {
      typedef bool is_vector_space_distance;

      typedef T ElementType;
      typedef typename Accumulator<T>::Type ResultType;

      /**
       *  Compute the cosine distance between two vectors.
       *
       *  This distance is not a valid kdtree distance, it's not dimensionwise additive.
       */
      template <typename Iterator1, typename Iterator2>
        ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
        {
          ResultType result = ResultType();
          ResultType sum0 = 0, sum1 = 0, sum2 = 0;
          Iterator1 last = a + size;
          Iterator1 lastgroup = last - 3;

          /* Process 4 items with each loop for efficiency. */
          while (a < lastgroup) {
            sum0 += (ResultType)(a[0] * a[0]);
            sum1 += (ResultType)(b[0] * b[0]);
            sum2 += (ResultType)(a[0] * b[0]);
            a += 4;
            b += 4;
          }
          /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
          while (a < last) {
            sum0 += (ResultType)(*a++ * *a++);
            sum1 += (ResultType)(*b++ * *b++);
            sum2 += (ResultType)(*a++ * *b++);
          }
          result =  sum2 / sqrt(sum0*sum1) ;
          result = (1 - result) / 2;
          return result;
        }

      /* This distance functor is not dimension-wise additive, which
       * makes it an invalid kd-tree distance, not implementing the accum_dist method */

    };
}

void Index(LightFaceRecognizer &recognizer, CascadeClassifier &cascade,
    const string &target_name, const string &dir_name, const int &num_return = 10) {
  Mat target_face = imread(target_name);
  Mat target_face_cropped = detectAlignCrop(target_face, cascade, recognizer);
  //imshow("target_face_cropped", target_face_cropped);
  //waitKey(0);
  Mat target_face_feat;
  recognizer.ExtractFaceFeature(target_face_cropped, target_face_feat);

  string rep_dir = dir_name.empty()?"../../../test_faces":dir_name;
  vector<fs::path> files;
  get_all(rep_dir, ".jpg", files);

  ::flann::Matrix<FEATURE_TYPE> dataset(new FEATURE_TYPE[files.size()*FEATURE_DIM], 
      files.size(), FEATURE_DIM); 
  ::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[FEATURE_DIM], 
      1, FEATURE_DIM); 
  memcpy(query[0], target_face_feat.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);

  for (int i = 0; i < files.size(); ++i) {
    cout<<files[i].string()<<"\n";
    Mat face2 = imread(files[i].string());
    Mat face2_cropped = detectAlignCrop(face2, cascade, recognizer);
    //imshow("face2_cropped", face2_cropped);
    //waitKey(0);
    Mat face2_feature;
    recognizer.ExtractFaceFeature(face2_cropped, face2_feature);
    memcpy(dataset[i], face2_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
    //cout<<face2_feature.at<FEATURE_TYPE>(82)<<" "<<dataset[i][82]<<endl;
  }

  //::flann::Index< ::flann::CosDistance<FEATURE_TYPE> > index(dataset, ::flann::LinearIndexParams(), ::flann::CosDistance<FEATURE_TYPE>());
  ::flann::Index< ::flann::L2<FEATURE_TYPE> > index(dataset, ::flann::AutotunedIndexParams());
  index.buildIndex();

  ::flann::Matrix<int> indices(new int[num_return], 1, num_return);
  ::flann::Matrix<float> dists(new float[num_return], 1, num_return);
  index.knnSearch(query, indices, dists, num_return, ::flann::SearchParams(128)); //::flann::CHECKS_AUTOTUNED

  cout<<"Return list:"<<endl;
  for (int i = 0; i < num_return; i++) {
    cout<<"No. "<<setw(2)<<i<<":\t"<<files[indices[0][i]]<<"\tdist is\t"<<setprecision(6)<<dists[0][i]<<endl;
  }
  //cout<<indices[0][0]<<endl;
  //cout<<dataset[files.size()-1][82]<<endl;
  delete [] dataset.ptr(); 
  delete [] indices.ptr();
  delete [] dists.ptr();
}

void Index(LightFaceRecognizer &recognizer, 
    CascadeClassifier &cascade,
    const string &filelist, 
    const string& dataset_file=string("dataset.hdf5"), 
    const string& index_file=string("index.hdf5"), 
    const string& dataset_path_file=string("dataset_file_path.txt") ) {
  long time1 = clock();
  int N = 0;
  ifstream file_list(filelist.c_str());
  string line;
  while (getline(file_list, line)) {
    N++;
  }

  if (0 == N)
  {
    cerr<<"FLANN index: wrong input file list!"<<endl;
    exit(-1);
  }

  ::flann::Matrix<FEATURE_TYPE> dataset(new FEATURE_TYPE[N*FEATURE_DIM], 
      N, FEATURE_DIM); 
  vector<string> file_path;

  file_list.clear();
  file_list.seekg(0, ios::beg);

  for (int i = 0; i < N; i++) {
    getline(file_list, line);
    line = fs::canonical(line).string();
    cout<<i<<":\t"<<line<<"\n";
    Mat face = imread(line);
    Mat face_cropped = detectAlignCrop(face, cascade, recognizer);
    //imshow("face_cropped", face_cropped);
    //waitKey(0);
    Mat face_feature;
    recognizer.ExtractFaceFeature(face_cropped, face_feature);
    memcpy(dataset[i], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
    file_path.push_back(line);
    //cout<<face_feature.at<FEATURE_TYPE>(82)<<"\t"<<dataset[i][82]<<endl;
    /*
    // Test L2-normalized feature
    double sum = 0, max = 0;
    for (int k = 0; k < FEATURE_DIM; k++) {
      sum += face_feature.at<FEATURE_TYPE>(k) * face_feature.at<FEATURE_TYPE>(k);
      max = max > face_feature.at<FEATURE_TYPE>(k) ? max : face_feature.at<FEATURE_TYPE>(k);
    }
    cout<<"Feature sum: "<<sum<<"; max: "<<max<<endl;
    */
  }
  file_list.close();
  long time2 = clock();

  // Create index
  //::flann::Index< ::flann::CosDistance<FEATURE_TYPE> > index(dataset, ::flann::LinearIndexParams(), ::flann::CosDistance<FEATURE_TYPE>());
  ::flann::Index< ::flann::L2<FEATURE_TYPE> > index(dataset, ::flann::AutotunedIndexParams());
  index.buildIndex();
  long time3 = clock();

  // Save index to the disk
  // Remove existed hdf5 files, otherwise an error will happen when size, of the data to be written, exceed the existed ones.
  remove(dataset_file.c_str());
  remove(index_file.c_str());
  ::flann::save_to_file(dataset, dataset_file, "dataset");
  index.save(index_file);
  ofstream ofile(dataset_path_file.c_str());
  ostream_iterator<string> output_iterator(ofile, "\n");
  copy(file_path.begin(), file_path.end(), output_iterator);
  long time4 = clock();

  // Test search. Use the first face as query
  int num_return = 10; 
  ::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[FEATURE_DIM], 
      1, FEATURE_DIM); 
  memcpy(query[0], dataset[0], sizeof(FEATURE_TYPE)*FEATURE_DIM);
  ::flann::Matrix<int> indices(new int[num_return], 1, num_return);
  ::flann::Matrix<float> dists(new float[num_return], 1, num_return);
  index.knnSearch(query, indices, dists, num_return, ::flann::SearchParams(128)); //::flann::CHECKS_AUTOTUNED
  long time5 = clock();
  cout<<"Return list:"<<endl;
  for (int i = 0; i < num_return; i++) {
    cout<<"No. "<<setw(2)<<i<<":\t"<<file_path[indices[0][i]]<<"\tdist is\t"<<setprecision(6)<<dists[0][i]<<endl;
  }

  cout<<"Number of faces is\t"<<N<<endl;
  cout<<"Time consume statistics : Total-time(sec.) | Average-time (us)"<<endl;
  cout<<"Extract feature:\t"<<float(time2-time1)/1000000<<"\t"<<(time2-time1)/N<<endl;
  cout<<"Create index:\t"<<float(time3-time2)/1000000<<"\t"<<(time3-time2)/N<<endl;
  cout<<"Save to disk:\t"<<float(time4-time3)/1000000<<"\t"<<(time4-time3)/N<<endl;
  cout<<"Search test:\t"<<float(time5-time4)<<endl;

  delete [] indices.ptr();
  delete [] dists.ptr();
  delete [] dataset.ptr(); 
  delete [] query.ptr(); 
}

void Query(LightFaceRecognizer &recognizer, 
    CascadeClassifier &cascade,
    const string &query_file, 
    const int& num_return=10, 
    const string& dataset_file=string("dataset.hdf5"), 
    const string& index_file=string("index.hdf5"), 
    const string& dataset_path_file=string("dataset_file_path.txt") ) {

  // Read query 
  vector<string> query_file_path;
  int N = 0;
  if (query_file.substr(query_file.length()-4) == ".txt") {
    ifstream file_list(query_file.c_str());
    string line;
    while (getline(file_list, line)) {
      query_file_path.push_back(line);
      N++;
    }
  }
  else {
    query_file_path.push_back(query_file);
    N++;
  }
  ::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[N*FEATURE_DIM], N, FEATURE_DIM);
  cout<<"Query image(s):"<<endl;
  for (int i = 0; i < N; i++) {
    cout<<query_file_path[i]<<endl; 
    Mat face = imread(query_file_path[i]);
    Mat face_cropped = detectAlignCrop(face, cascade, recognizer);
    //imshow("face_cropped", face_cropped);
    //waitKey(0);
    Mat face_feature;
    recognizer.ExtractFaceFeature(face_cropped, face_feature);
    memcpy(query[i], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
    //cout<<face_feature.at<FEATURE_TYPE>(82)<<"\t"<<query[i][82]<<endl;
  }

  if (0 == N)
  {
    cerr<<"The specified input is not an image or valid \".txt\" file with image paths!"<<endl;
    exit(-1);
  }

  // load dataset features 
  ::flann::Matrix<FEATURE_TYPE> dataset;
  ::flann::load_from_file(dataset, dataset_file, "dataset");
  // load all path of faces in the dataset 
  vector <string> data_file_path;
  ifstream inFile(dataset_path_file.c_str());
  while (inFile) {
    string line;
    getline(inFile, line);
    data_file_path.push_back(line);
  }
  // load the index 
  //::flann::Index< ::flann::CosDistance<FEATURE_TYPE> > index(dataset, ::flann::SavedIndexParams(index_file), ::flann::CosDistance<FEATURE_TYPE>());
  ::flann::Index< ::flann::L2<FEATURE_TYPE> > index(dataset, ::flann::SavedIndexParams(index_file));

  /*
  //-----------------------------------------
  // "addPoints" and removePoint" testing 
  ::flann::Matrix<FEATURE_TYPE> added(new FEATURE_TYPE[N*FEATURE_DIM], N, FEATURE_DIM);
  for (int i = 0; i < N; i++) {
    Mat face = imread(query_file_path[i]);
    Mat face_cropped = detectAlignCrop(face, cascade, recognizer);
    //imshow("face_cropped", face_cropped);
    //waitKey(0);
    Mat face_feature;
    recognizer.ExtractFaceFeature(face_cropped, face_feature);
    memcpy(added[i], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
    data_file_path.push_back(query_file_path[i]);
    //cout<<face_feature.at<FEATURE_TYPE>(82)<<"\t"<<query[i][82]<<endl;
  }
  index.addPoints(added);  
  //delete [] added.ptr();
  //delete [] dataset.ptr(); // It seems that Index holds its own data. The orginal feature matrix does not matter after the construction of Index.
  cout<<"Dataset size after add: "<<dataset.rows<<endl;
  index.removePoint(22);
  index.removePoint(24);
  index.removePoint(26);
  index.removePoint(28);
  //cout<<index.getPoint(20)[0]<<"\t"<<query[0][0]<<endl;
  //-----------------------------------------
  */

  // prepare the search result matrix
  ::flann::Matrix<FEATURE_TYPE> dists(new FEATURE_TYPE[query.rows*num_return], query.rows, num_return);
  ::flann::Matrix<int> indices(new int[query.rows*num_return], query.rows, num_return);

  index.knnSearch(query, indices, dists, num_return, ::flann::SearchParams(128)); //::flann::CHECKS_AUTOTUNED

  for (int ind = 0; ind < query.rows; ind++) {
    cout<<"Query image is:\t"<<query_file_path[ind]<<endl;
    cout<<"Return list:"<<endl;
    for (int i = 0; i < num_return; i++) {
      cout<<"No. "<<setw(2)<<i<<":\t"<<data_file_path[indices[ind][i]]<<"\tdist is\t"<<setprecision(6)<<dists[ind][i]<<endl;
    }
    cout<<"-----------------------------------------"<<endl<<endl;
  }
  //cout<<indices[0][0]<<endl;
  //cout<<dataset[files.size()-1][82]<<endl;

  delete [] dataset.ptr();
  delete [] indices.ptr();
  delete [] dists.ptr();
  delete [] query.ptr();
}

void retrieval_on_lfw(LightFaceRecognizer & recognizer,
    CascadeClassifier &cascade) {

  bool bFreshRun = false;  // If true, recreate index, and redo all search
  string dataset_file("dataset.hdf5");
  string index_file("index.hdf5");
  string dataset_path_file("dataset_file_path.txt");
  string lfw_indices_file = "lfw_indices.hdf5";
  string lfw_dists_file = "lfw_dists.hdf5";
  string lfw_file_name = "../../../lfw_data/lfw_file_name.txt";
  string lfw_same_face_count = "../../../lfw_data/lfw_same_face_count.txt";

  // Create index. It takes quite a long time. 
  if (bFreshRun) {
    long time1 = clock();
    Index(recognizer, cascade, lfw_file_name);
    long time2 = clock();
  }

  // load dataset features 
  ::flann::Matrix<FEATURE_TYPE> dataset;
  ::flann::load_from_file(dataset, dataset_file, "dataset");
  // load all path of faces in the dataset 
  vector <string> data_file_path;
  ifstream inFile(dataset_path_file.c_str());
  string line;
  while (getline(inFile, line)) {
    data_file_path.push_back(line);
  }
  // load the index 
  //::flann::Index< ::flann::CosDistance<FEATURE_TYPE> > index(dataset, ::flann::SavedIndexParams(index_file), ::flann::CosDistance<FEATURE_TYPE>());
  ::flann::Index< ::flann::L2<FEATURE_TYPE> > index(dataset, ::flann::SavedIndexParams(index_file));

  // Read face count
  int N = data_file_path.size();
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
    // Do search. Here query = dataset.
    // The following method may run out of memory.
    //::flann::Matrix<FEATURE_TYPE> query(dataset.ptr(), N, FEATURE_DIM);
    //index.knnSearch(query, indices, dists, nFaces_UpBound, ::flann::SearchParams(128)); //::flann::CHECKS_AUTOTUNED
    // Search one by one
    ::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[FEATURE_DIM], 1, FEATURE_DIM);
    ::flann::Matrix<FEATURE_TYPE> dists_one_query(new FEATURE_TYPE[nFaces_UpBound], 1, nFaces_UpBound);
    ::flann::Matrix<int> indices_one_query(new int[nFaces_UpBound], 1, nFaces_UpBound);
    for (int i = 0; i < N; i++) {
      memcpy(query[0], dataset[i], sizeof(FEATURE_TYPE)*FEATURE_DIM);
      //cout<<i<<": "<<query[0][FEATURE_DIM-1]<<"\t"<<dataset[i][FEATURE_DIM-1]<<endl;
      index.knnSearch(query, indices_one_query, dists_one_query, nFaces_UpBound, ::flann::SearchParams(::flann::FLANN_CHECKS_AUTOTUNED)); //::flann::SearchParams(128)
      //index.knnSearch(query, indices_one_query, dists_one_query, nFaces_UpBound, ::flann::SearchParams(16)); //::flann::SearchParams(128)
      memcpy(indices[i], indices_one_query[0], sizeof(FEATURE_TYPE)*nFaces_UpBound);
      memcpy(dists[i], dists_one_query[0], sizeof(int)*nFaces_UpBound);
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
  //// A test for queries who have more than 70 faces in the dataset
  //int a = 0, b = 0;
  //for (int i = 0; i < N; i++) {
    //if (face_count[i] > 70)  {
      //a = 0;
      //for (int j = 0; j < 70; j++)
      //{
        //cout<<correct[j][i]<<"\t";
        //a+=correct[j][i];
      //}
      //cout<<a<<endl;
      //b += a;
    //}
  //}
  //cout<<float(b)/70/num_images[70]<<endl;

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

  delete [] dataset.ptr();
  delete [] dists.ptr();
  delete [] indices.ptr();
}

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

  //// Rebuild index
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
      faceRepo.Query(::flann::Matrix<FEATURE_TYPE>(dataset[i], 1, FEATURE_DIM), nFaces_UpBound, return_list, return_list_pos, return_dists);
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

void testFaceRepo(LightFaceRecognizer &recognizer, CascadeClassifier &cascade){
  FaceRepo faceRepo(recognizer, cascade);
  string txtFile = string("../../../test_faces.txt");
  //string queryFile = string("../../../test_faces.txt");
  string queryFile = string("../../../all.txt");
  string addFile = string("../../../more.txt");
  string removeFile = string("../../../test_faces/Abdullah_Gul_0003.jpg");
  string removeFile2 = string("../../../test_faces/Dennis_Hastert_0001.jpg");
  string directory = string("./test_old"); 
  string directory_save = string("./test"); 
  
  //faceRepo.InitialIndex(txtFile);
  //faceRepo.Save(directory);

  faceRepo.Load(directory);
  faceRepo.AddFace(addFile);

  faceRepo.RemoveFace(removeFile);
  faceRepo.RemoveFace(removeFile2);
  faceRepo.RebuildIndex();

  faceRepo.Save(directory_save);

  vector<vector<string> > return_list;
  vector<vector<int> > return_list_pos;
  vector<vector<float> > dists;
  int num_return = 10;
  faceRepo.Query(queryFile, num_return, return_list, return_list_pos, dists);
  for (int i = 0; i < return_list.size(); i++) {
    cout<<"Query "<<i<<":"<<endl;
    for (int j = 0; j < num_return; j++) {
      cout<<"#"<<j<<": "<<return_list_pos[i][j]<<" ("<<dists[i][j]<<"), "<<return_list[i][j]<<endl;
    }
    cout<<endl;
  }
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
  //FaceSearch(recognizer, cascade, argv[1], argv[2]);

  //Index(recognizer, cascade, argv[1], argv[2], 20);
  //Index(recognizer, cascade, argv[1]);
  //Query(recognizer, cascade, argv[1] ); // Need index first
  //retrieval_on_lfw(recognizer, cascade);  
  
  //testFaceRepo(recognizer, cascade);

  if (1 == argc)
    // Do query for each image in lfw dataset, and do statistic for the result.
    retrieval_on_lfw_batch(recognizer, cascade);  
  else
    // Do query in lfw dataset for the input image. 
    retrieval_on_lfw(recognizer, cascade, argv[1]);  
  return 0;
}
