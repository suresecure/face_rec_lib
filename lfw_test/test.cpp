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

//#define  FEATURE_DIM (160) 
//typedef float FEATURE_TYPE;
#define FACE_ALIGN_SIZE 192
#define FACE_ALIGN_LANDMARK FaceAlign::INNER_EYES_AND_TOP_LIP
//#define FACE_ALIGN_SCALE_FACTOR 0.2734
#define FACE_ALIGN_SCALE_FACTOR 0.26 // 25:100 as in CASIA dataset paper. Use 96*96 as input in small CNN, that's 25/96.
//#define FACE_ALIGN_LANDMARK FaceAlign::INNER_EYES_AND_BOTTOM_LIP
//#define FACE_ALIGN_SCALE_FACTOR 0.3385
string cascadeName = "../../../face_rec_models/haarcascade_frontalface_alt.xml";

// Face recognition parameters.
int N_CLASS = 10; // Least number of faces for a person.
int KNN = 10; // Size of return knn;
float TH_DIST = 0.1; // Distance threshold for same person.
int TH_N = 1; // Least number of retrieved knn with same label.


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

// Get the class name of an image
// Now we assume that all face images of the same person store
// in the same folder. So an image's class name is the parent fold name.
string get_class_name(const string path) {
  return fs::canonical(path).parent_path().filename().string();
}

//// Find the max face of OpenCV VJ face detector
//Rect FindMaxFace(const vector<Rect> &faces) {
  //float max_area = 0.f;
  //Rect max_face;
  //for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++) {
    //if (r->width * r->height > max_area) {
      //max_area = r->width * r->height;
      //max_face = *r;
    //}
  //}
  //return max_face;
//}

//// OpenCV VJ face detector. Align and crop the largest face, then detect again.
//Mat detectAlignCrop(const Mat &img, CascadeClassifier &cascade,
    //LightFaceRecognizer &recognizer) {
  //vector<Rect> faces;
  //Mat gray, gray_org;

  //cvtColor(img, gray_org, CV_BGR2GRAY);
  //equalizeHist(gray_org, gray);

  //// --Detection
  //cascade.detectMultiScale(gray, faces, 1.1, 2,
      //0
      ////|CV_HAAR_FIND_BIGGEST_OBJECT
      ////|CV_HAAR_DO_ROUGH_SEARCH
      //|
      //CV_HAAR_SCALE_IMAGE,
      //Size(30, 30));
  //Rect max_face;
  //if (faces.size() <= 0) {
    //cout << "Cannot detect face!\n";
    //max_face = Rect(68, 68, 113, 113);
    //Rect out_rect;
    //recognizer.CropFace(gray_org, max_face, out_rect);
    //return Mat(gray_org, out_rect);
  //}

  //// --Alignment
  //max_face = FindMaxFace(faces);

  //Mat after_aligned, after_aligned_org;
  //recognizer.FaceAlign(gray_org, max_face, after_aligned_org);
  //equalizeHist(after_aligned_org, after_aligned);

  //// detect faces on aligned image again
  //cascade.detectMultiScale(after_aligned, faces, 1.1, 2,
      //0
      ////|CV_HAAR_FIND_BIGGEST_OBJECT
      ////|CV_HAAR_DO_ROUGH_SEARCH
      //|
      //CV_HAAR_SCALE_IMAGE,
      //Size(30, 30));

  //if (faces.size() <= 0) {
    //cout << "Cannot detect face after aligned!\n";
    //Rect face_rect(68, 68, 113, 113);
    //Rect out_rect;
    //recognizer.CropFace(img, face_rect, out_rect);
    //return Mat(img, out_rect);
  //}
  //max_face = FindMaxFace(faces);
  //Rect roi;
  //recognizer.CropFace(gray, max_face, roi);
  //return Mat(after_aligned_org, roi);
//}

//// The formal detectAlignCrop() function is mainly for lfw images, 
//// where left-top corner is returned if no face detected.
//// This function will return empty Mat in this case. 
//Mat detectAlignCropPractical(const Mat &img, CascadeClassifier &cascade,
    //LightFaceRecognizer &recognizer) {
  //vector<Rect> faces;
  //Mat gray, gray_org;

  //cvtColor(img, gray_org, CV_BGR2GRAY);
  //equalizeHist(gray_org, gray);

  //// --Detection
  //cascade.detectMultiScale(gray, faces, 1.1, 2,
      //0
      ////|CV_HAAR_FIND_BIGGEST_OBJECT
      ////|CV_HAAR_DO_ROUGH_SEARCH
      //|
      //CV_HAAR_SCALE_IMAGE,
      //Size(30, 30));
  //Rect max_face;
  //if (faces.size() <= 0) {
    //cout << "Cannot detect face!\n";
    //return Mat();
  //}

  //// --Alignment
  //max_face = FindMaxFace(faces);

  //Mat after_aligned, after_aligned_org;
  //recognizer.FaceAlign(gray_org, max_face, after_aligned_org);
  //equalizeHist(after_aligned_org, after_aligned);

  //// detect faces on aligned image again
  //faces.clear();
  //cascade.detectMultiScale(after_aligned, faces, 1.1, 2,
      //0
      ////|CV_HAAR_FIND_BIGGEST_OBJECT
      ////|CV_HAAR_DO_ROUGH_SEARCH
      //|
      //CV_HAAR_SCALE_IMAGE,
      //Size(30, 30));

  //if (faces.size() <= 0) {
    //cout << "Cannot detect face after aligned!\n";
    //return Mat();
  //}
  //max_face = FindMaxFace(faces);
  //Rect roi;
  ////recognizer.CropFace(gray, max_face, roi);
  //recognizer.CropFace(after_aligned, max_face, roi);
  ////cout<<faces.size()<<endl;
  ////for (int i = 0; i < faces.size(); i++)
    ////cout<<faces[i].size()<<endl;
  ////cout<<endl;
  ////cout<<"max_face: "<<max_face<<endl;
  ////cout<<"roi: "<<roi<<endl;
  ////cout<<"Aligned image size: "<<after_aligned.size()<<endl;
  //return Mat(after_aligned_org, roi);
//}

// Find and crop face by dlib.
Mat detectAlignCropDlib(FaceAlign & face_align, const Mat &img) {
  // Detection

  if (img.empty()) { 
    return Mat();
  }

  Mat img_cache;
  long time1 = clock();
  if (img.channels() == 1)
    cvtColor(img, img_cache, CV_GRAY2BGR);
  else
    img_cache = img;
  dlib::cv_image<dlib::bgr_pixel> cimg(img);
  long time2 = clock();
  // std::vector<dlib::rectangle> dets = face_align.getAllFaceBoundingBoxes(cimg);
  std::vector<dlib::rectangle> dets;
  dets.push_back(face_align.getLargestFaceBoundingBox(cimg)); // Use the largest detected face only
  // cout<<dets[0]<<endl; // position of the largest detected face
  long time3 = clock();
  if (0 == dets.size() || dets[0].is_empty())
  {
    cout << "Cannot detect face!\n";
    return Mat();
  }

  // --Alignment
  long time4 = clock();
  //Mat ret = face_align.align(cimg, dets[0]);
  // The last argument control the size of face after align.
  // It is the proportion of the distance, which from the center point of inner eyes to the bottom/top lip, in the whole image.
  // It is best tuned to the SMALL CNN model's mean image.
  //Mat ret = face_align.align(cimg, dets[0], 224, FaceAlign::INNER_EYES_AND_BOTTOM_LIP, 0.3);
  Mat ret = face_align.align(cimg, dets[0], FACE_ALIGN_SIZE,
                              FACE_ALIGN_LANDMARK, FACE_ALIGN_SCALE_FACTOR);
  long time5 = clock();
  //cout<<time2 - time1<<endl;
  //cout<<time3 - time2<<endl;
  //cout<<time4 - time3<<endl;
  cout<<"Detect and align face in "<<(float)(time5 - time1)/1000.0<<"ms"<<endl;
  return ret;
  //return face_align.align(cimg, dets[0]);
}

//// Do face validation on lfw data
//void validate_on_lfw_data(LightFaceRecognizer &recognizer) {
  //ifstream pairs_file("../../../lfw_data/pairs.txt");

  //string line;
  //getline(pairs_file, line);
  //cout << line << endl;

  //CascadeClassifier cascade;
  //// -- 1. Load the cascades
  //if (!cascade.load(cascadeName)) {
    //cerr << "ERROR: Could not load classifier cascade" << endl;
    //return;
  //}

  //bool gt;
  //int fnum = 0;
  //int correct = 0;
  //while (getline(pairs_file, line)) {
    //vector<string> sline = split(line, '\t');
    //string name1, name2;
    //int id1, id2;
    //if (sline.size() == 3) {
      //name1 = name2 = sline[0];
      //stringstream idstr1(sline[1]);
      //idstr1 >> id1;
      //stringstream idstr2(sline[2]);
      //idstr2 >> id2;
      //gt = true;
    //} else if (sline.size() == 4) {
      //name1 = sline[0];
      //stringstream idstr1(sline[1]);
      //idstr1 >> id1;
      //name2 = sline[2];
      //stringstream idstr2(sline[3]);
      //idstr2 >> id2;
      //gt = false;
    //} else {
      //cout << "Read pair error!" << endl;
      //exit(1);
    //}
    //// cout << name1 << "\t" << id1 << "\t" << name2 << "\t" << id2 << endl;

    //// Load face images
    //std::ostringstream face1_string_stream;
    //face1_string_stream << "../../../lfw_data/lfw/" << name1 << "/" << name1
      //<< "_" << setw(4) << setfill('0') << id1 << ".jpg";
    //std::string face1_name = face1_string_stream.str();
    //std::ostringstream face2_string_stream;
    //face2_string_stream << "../../../lfw_data/lfw/" << name2 << "/" << name2
      //<< "_" << setw(4) << setfill('0') << id2 << ".jpg";
    //std::string face2_name = face2_string_stream.str();

    //cout << face1_name << "\t" << face2_name << endl;

    //Mat face1 = imread(face1_name);
    //Mat face2 = imread(face2_name);

    //Mat face1_cropped = detectAlignCrop(face1, cascade, recognizer);
    //Mat face2_cropped = detectAlignCrop(face2, cascade, recognizer);

    //////Extract feature from images
    //Mat face1_feature;
    //Mat face2_feature;
    //recognizer.ExtractFaceFeature(face1_cropped, face1_feature);
    //recognizer.ExtractFaceFeature(face2_cropped, face2_feature);

    //float similarity =
      //recognizer.CalculateSimilarity(face1_feature, face2_feature);
    //if ((gt && similarity >= 0.5f) || (!gt && similarity < 0.5f))
      //++correct;
    //// float distance = FaceDistance(recognizer, face1_feature, face2_feature);
    //++fnum;
    //cout << "pair num: " << fnum << "similarity: " << similarity << endl;
  //}

  //cout << "Precision: " << (float)correct / fnum << endl;
//}

//// Do face validation on prepared data (BW100issame.txt)
//void validate_on_prepared_data(LightFaceRecognizer &recognizer) {
  //ifstream pair_file("../../../lfw_data/BW100issame.txt");

  //bool gt;
  //int fnum = 0;
  //int correct = 0;
  //while (pair_file >> gt) {
    //// Load face images
    //std::ostringstream face1_string_stream;
    //face1_string_stream << "../../../lfw_data/100BW/" << (fnum * 2 + 1)
      //<< ".png";
    //std::string face1_name = face1_string_stream.str();
    //std::ostringstream face2_string_stream;
    //face2_string_stream << "../../../lfw_data/100BW/" << (fnum * 2 + 2)
      //<< ".png";
    //std::string face2_name = face2_string_stream.str();
    //cout << face1_name << "\t" << face2_name << endl;

    //Mat face1_feature;
    //Mat face2_feature;
    //Mat face1 = imread(face1_name);
    //Mat face2 = imread(face2_name);
    //recognizer.ExtractFaceFeature(face1, face1_feature);
    //recognizer.ExtractFaceFeature(face2, face2_feature);

    //float similarity =
      //recognizer.CalculateSimilarity(face1_feature, face2_feature);

    //if ((gt && similarity >= 0.5f) || (!gt && similarity < 0.5f))
      //++correct;
    //++fnum;
    //cout << "pair num: " << fnum << "similarity: " << similarity << endl;
  //}
  //cout << "Precision: " << (float)correct / fnum << endl;
//}

//// Face verification for the two input image
//float FaceVerification(LightFaceRecognizer &recognizer,
    //CascadeClassifier &cascade, const string &f1_name,
    //const string &f2_name) {
  //Mat face1 = imread(f1_name);
  //Mat face2 = imread(f2_name);
  //Mat face1_cropped = detectAlignCrop(face1, cascade, recognizer);
  //Mat face2_cropped = detectAlignCrop(face2, cascade, recognizer);
  //imshow("face1_cropped", face1_cropped);
  //imshow("face2_cropped", face2_cropped);
  //waitKey(0);
  //Mat face1_feature, face2_feature;
  //recognizer.ExtractFaceFeature(face1_cropped, face1_feature);
  //recognizer.ExtractFaceFeature(face2_cropped, face2_feature);
  //float similarity =
    //recognizer.CalculateSimilarity(face1_feature, face2_feature);
  //float cos_distance =
    //recognizer.CalculateCosDistance(face1_feature, face2_feature);
  //float bayesian_distance =
    //recognizer.CalculateBayesianDistance(face1_feature, face2_feature);
  //cout << "Cos distance: " << cos_distance
    //<< "\tBayesian distance: " << bayesian_distance << endl;
  //return similarity;
//}

/*
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
  vector<pair<fs::path, float> > distances;
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
*/

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

//// Flann index. Use VJ detector
//void Index(LightFaceRecognizer &recognizer, CascadeClassifier &cascade,
    //const string &target_name, const string &dir_name, const int &num_return = 10) {
  //Mat target_face = imread(target_name);
  //Mat target_face_cropped = detectAlignCrop(target_face, cascade, recognizer);
  ////imshow("target_face_cropped", target_face_cropped);
  ////waitKey(0);
  //Mat target_face_feat;
  //recognizer.ExtractFaceFeature(target_face_cropped, target_face_feat);

  //string rep_dir = dir_name.empty()?"../../../test_faces":dir_name;
  //vector<fs::path> files;
  //get_all(rep_dir, ".jpg", files);

  //::flann::Matrix<FEATURE_TYPE> dataset(new FEATURE_TYPE[files.size()*FEATURE_DIM], 
      //files.size(), FEATURE_DIM); 
  //::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[FEATURE_DIM], 
      //1, FEATURE_DIM); 
  //memcpy(query[0], target_face_feat.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);

  //for (int i = 0; i < files.size(); ++i) {
    //cout<<files[i].string()<<"\n";
    //Mat face2 = imread(files[i].string());
    //Mat face2_cropped = detectAlignCrop(face2, cascade, recognizer);
    ////imshow("face2_cropped", face2_cropped);
    ////waitKey(0);
    //Mat face2_feature;
    //recognizer.ExtractFaceFeature(face2_cropped, face2_feature);
    //memcpy(dataset[i], face2_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
    ////cout<<face2_feature.at<FEATURE_TYPE>(82)<<" "<<dataset[i][82]<<endl;
  //}

  ////::flann::Index< ::flann::CosDistance<FEATURE_TYPE> > index(dataset, ::flann::LinearIndexParams(), ::flann::CosDistance<FEATURE_TYPE>());
  //::flann::Index< ::flann::L2<FEATURE_TYPE> > index(dataset, ::flann::AutotunedIndexParams());
  //index.buildIndex();

  //::flann::Matrix<int> indices(new int[num_return], 1, num_return);
  //::flann::Matrix<float> dists(new float[num_return], 1, num_return);
  //index.knnSearch(query, indices, dists, num_return, ::flann::SearchParams(128)); //::flann::CHECKS_AUTOTUNED

  //cout<<"Return list:"<<endl;
  //for (int i = 0; i < num_return; i++) {
    //cout<<"No. "<<setw(2)<<i<<":\t"<<files[indices[0][i]]<<"\tdist is\t"<<setprecision(6)<<dists[0][i]<<endl;
  //}
  ////cout<<indices[0][0]<<endl;
  ////cout<<dataset[files.size()-1][82]<<endl;
  //delete [] dataset.ptr(); 
  //delete [] indices.ptr();
  //delete [] dists.ptr();
//}

//// Flann index. Use VJ detector
//void Index(LightFaceRecognizer &recognizer, 
    //CascadeClassifier &cascade,
    //const string &filelist, 
    //const string& dataset_file=string("dataset.hdf5"), 
    //const string& index_file=string("index.hdf5"), 
    //const string& dataset_path_file=string("dataset_file_path.txt") ) {
  //long time1 = clock();
  //int N = 0;
  //ifstream file_list(filelist.c_str());
  //string line;
  //while (getline(file_list, line)) {
    //N++;
  //}

  //if (0 == N)
  //{
    //cerr<<"FLANN index: wrong input file list!"<<endl;
    //exit(-1);
  //}

  //::flann::Matrix<FEATURE_TYPE> dataset(new FEATURE_TYPE[N*FEATURE_DIM], 
      //N, FEATURE_DIM); 
  //vector<string> file_path;

  //file_list.clear();
  //file_list.seekg(0, ios::beg);

  //for (int i = 0; i < N; i++) {
    //getline(file_list, line);
    //line = fs::canonical(line).string();
    //cout<<i<<":\t"<<line<<"\n";
    //Mat face = imread(line);
    //Mat face_cropped = detectAlignCrop(face, cascade, recognizer);
    ////imshow("face_cropped", face_cropped);
    ////waitKey(0);
    //Mat face_feature;
    //recognizer.ExtractFaceFeature(face_cropped, face_feature);
    //memcpy(dataset[i], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
    //file_path.push_back(line);
    ////cout<<face_feature.at<FEATURE_TYPE>(82)<<"\t"<<dataset[i][82]<<endl;
    //[>
    //// Test L2-normalized feature
    //double sum = 0, max = 0;
    //for (int k = 0; k < FEATURE_DIM; k++) {
      //sum += face_feature.at<FEATURE_TYPE>(k) * face_feature.at<FEATURE_TYPE>(k);
      //max = max > face_feature.at<FEATURE_TYPE>(k) ? max : face_feature.at<FEATURE_TYPE>(k);
    //}
    //cout<<"Feature sum: "<<sum<<"; max: "<<max<<endl;
    //*/
  //}
  //file_list.close();
  //long time2 = clock();

  //// Create index
  ////::flann::Index< ::flann::CosDistance<FEATURE_TYPE> > index(dataset, ::flann::LinearIndexParams(), ::flann::CosDistance<FEATURE_TYPE>());
  //::flann::Index< ::flann::L2<FEATURE_TYPE> > index(dataset, ::flann::AutotunedIndexParams());
  //index.buildIndex();
  //long time3 = clock();

  //// Save index to the disk
  //// Remove existed hdf5 files, otherwise an error will happen when size, of the data to be written, exceed the existed ones.
  //remove(dataset_file.c_str());
  //remove(index_file.c_str());
  //::flann::save_to_file(dataset, dataset_file, "dataset");
  //index.save(index_file);
  //ofstream ofile(dataset_path_file.c_str());
  //ostream_iterator<string> output_iterator(ofile, "\n");
  //copy(file_path.begin(), file_path.end(), output_iterator);
  //long time4 = clock();

  //// Test search. Use the first face as query
  //int num_return = 10; 
  //::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[FEATURE_DIM], 
      //1, FEATURE_DIM); 
  //memcpy(query[0], dataset[0], sizeof(FEATURE_TYPE)*FEATURE_DIM);
  //::flann::Matrix<int> indices(new int[num_return], 1, num_return);
  //::flann::Matrix<float> dists(new float[num_return], 1, num_return);
  //index.knnSearch(query, indices, dists, num_return, ::flann::SearchParams(128)); //::flann::CHECKS_AUTOTUNED
  //long time5 = clock();
  //cout<<"Return list:"<<endl;
  //for (int i = 0; i < num_return; i++) {
    //cout<<"No. "<<setw(2)<<i<<":\t"<<file_path[indices[0][i]]<<"\tdist is\t"<<setprecision(6)<<dists[0][i]<<endl;
  //}

  //cout<<"Number of faces is\t"<<N<<endl;
  //cout<<"Time consume statistics : Total-time(sec.) | Average-time (us)"<<endl;
  //cout<<"Extract feature:\t"<<float(time2-time1)/1000000<<"\t"<<(time2-time1)/N<<endl;
  //cout<<"Create index:\t"<<float(time3-time2)/1000000<<"\t"<<(time3-time2)/N<<endl;
  //cout<<"Save to disk:\t"<<float(time4-time3)/1000000<<"\t"<<(time4-time3)/N<<endl;
  //cout<<"Search test:\t"<<float(time5-time4)<<endl;

  //delete [] indices.ptr();
  //delete [] dists.ptr();
  //delete [] dataset.ptr(); 
  //delete [] query.ptr(); 
//}

//// Index without face detection. Just use the origin input image as the input of the face recogniton network.
//void IndexWithoutCrop(LightFaceRecognizer &recognizer, 
    //CascadeClassifier &cascade,
    //const string &filelist, 
    //const string& dataset_file=string("dataset.hdf5"), 
    //const string& index_file=string("index.hdf5"), 
    //const string& dataset_path_file=string("dataset_file_path.txt") ) {
  //long time1 = clock();
  //int N = 0;
  //ifstream file_list(filelist.c_str());
  //string line;
  //while (getline(file_list, line)) {
    //N++;
  //}

  //if (0 == N)
  //{
    //cerr<<"FLANN index: wrong input file list!"<<endl;
    //exit(-1);
  //}

  //::flann::Matrix<FEATURE_TYPE> dataset(new FEATURE_TYPE[N*FEATURE_DIM], 
      //N, FEATURE_DIM); 
  //vector<string> file_path;

  //file_list.clear();
  //file_list.seekg(0, ios::beg);

  //for (int i = 0; i < N; i++) {
    //getline(file_list, line);
    //line = fs::canonical(line).string();
    //cout<<i<<":\t"<<line<<"\n";
    //Mat face = imread(line);
    ////Mat face_cropped = detectAlignCrop(face, cascade, recognizer);
    //Mat & face_cropped = face;
    ////imshow("face_cropped", face_cropped);
    ////waitKey(0);
    //Mat face_feature;
    //recognizer.ExtractFaceFeature(face_cropped, face_feature);
    //memcpy(dataset[i], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
    //file_path.push_back(line);
    ////cout<<face_feature.at<FEATURE_TYPE>(82)<<"\t"<<dataset[i][82]<<endl;
    //[>
    //// Test L2-normalized feature
    //double sum = 0, max = 0;
    //for (int k = 0; k < FEATURE_DIM; k++) {
      //sum += face_feature.at<FEATURE_TYPE>(k) * face_feature.at<FEATURE_TYPE>(k);
      //max = max > face_feature.at<FEATURE_TYPE>(k) ? max : face_feature.at<FEATURE_TYPE>(k);
    //}
    //cout<<"Feature sum: "<<sum<<"; max: "<<max<<endl;
    //*/
  //}
  //file_list.close();
  //long time2 = clock();

  //// Create index
  ////::flann::Index< ::flann::CosDistance<FEATURE_TYPE> > index(dataset, ::flann::LinearIndexParams(), ::flann::CosDistance<FEATURE_TYPE>());
  //::flann::Index< ::flann::L2<FEATURE_TYPE> > index(dataset, ::flann::AutotunedIndexParams());
  //index.buildIndex();
  //long time3 = clock();

  //// Save index to the disk
  //// Remove existed hdf5 files, otherwise an error will happen when size, of the data to be written, exceed the existed ones.
  //remove(dataset_file.c_str());
  //remove(index_file.c_str());
  //::flann::save_to_file(dataset, dataset_file, "dataset");
  //index.save(index_file);
  //ofstream ofile(dataset_path_file.c_str());
  //ostream_iterator<string> output_iterator(ofile, "\n");
  //copy(file_path.begin(), file_path.end(), output_iterator);
  //long time4 = clock();

  //// Test search. Use the first face as query
  //int num_return = 10; 
  //::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[FEATURE_DIM], 
      //1, FEATURE_DIM); 
  //memcpy(query[0], dataset[0], sizeof(FEATURE_TYPE)*FEATURE_DIM);
  //::flann::Matrix<int> indices(new int[num_return], 1, num_return);
  //::flann::Matrix<float> dists(new float[num_return], 1, num_return);
  //index.knnSearch(query, indices, dists, num_return, ::flann::SearchParams(128)); //::flann::CHECKS_AUTOTUNED
  //long time5 = clock();
  //cout<<"Return list:"<<endl;
  //for (int i = 0; i < num_return; i++) {
    //cout<<"No. "<<setw(2)<<i<<":\t"<<file_path[indices[0][i]]<<"\tdist is\t"<<setprecision(6)<<dists[0][i]<<endl;
  //}

  //cout<<"Number of faces is\t"<<N<<endl;
  //cout<<"Time consume statistics : Total-time(sec.) | Average-time (us)"<<endl;
  //cout<<"Extract feature:\t"<<float(time2-time1)/1000000<<"\t"<<(time2-time1)/N<<endl;
  //cout<<"Create index:\t"<<float(time3-time2)/1000000<<"\t"<<(time3-time2)/N<<endl;
  //cout<<"Save to disk:\t"<<float(time4-time3)/1000000<<"\t"<<(time4-time3)/N<<endl;
  //cout<<"Search test:\t"<<float(time5-time4)<<endl;

  //delete [] indices.ptr();
  //delete [] dists.ptr();
  //delete [] dataset.ptr(); 
  //delete [] query.ptr(); 
//}

void IndexDlib(LightFaceRecognizer &recognizer, 
    //CascadeClassifier &cascade,
    FaceAlign & face_align,
    const string &filelist, 
    const string& dataset_file=string("dataset.hdf5"), 
    const string& index_file=string("index.hdf5"), 
    const string& dataset_path_file=string("dataset_file_path.txt"),
    const string& invalid_face_index_file=string("invalid_face_index.txt") ) {
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

  int pos = 0;
  vector<int> invalid_image_ind;
  vector<string> invalid_image_path;
  for (int i = 0; i < N; i++) {
    getline(file_list, line);
    line = fs::canonical(line).string();
    cout<<i<<":\t"<<line<<"\n";
    Mat face = imread(line);
    //Mat face_cropped = detectAlignCrop(face, cascade, recognizer);
    Mat face_cropped = detectAlignCropDlib(face_align, face);
    //imshow("face_cropped", face_cropped);
    //waitKey(0);
    if (face_cropped.empty()) { // If no face detected.
      invalid_image_ind.push_back(i);
      invalid_image_path.push_back(line);
      continue;
    }
    Mat face_feature;
    long time1 = clock();
    recognizer.ExtractFaceFeature(face_cropped, face_feature);
    long time2 = clock();
    cout<<"Extract feature in "<<float(time2-time1)/1000.0<<"ms"<<endl;
    memcpy(dataset[pos++], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
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
  dataset.rows = pos;
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
  for (int i = 0; i < file_path.size(); i++) {
    ofile<<file_path[i]<<endl;
    //cout<<file_path[i]<<endl;
  }
  // Donot use "copy" of ostream, may cause data lose.
  //ostream_iterator<string> output_iterator(ofile, "\n");
  //copy(file_path.begin(), file_path.end(), output_iterator);
  ofstream ofile2(invalid_face_index_file.c_str());
  for (int i = 0; i < invalid_image_path.size(); i++) {
    ofile2<<invalid_image_path[i]<<" "<<invalid_image_ind[i]<<endl;
    //cout<<invalid_image_path[i]<<" "<<invalid_image_ind[i]<<endl;
  }
  //ostream_iterator<int> output_iterator2(ofile2, "\n");
  //copy(invalid_image.begin(), invalid_image.end(), output_iterator2);
  ofile.close();
  ofile2.close();
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

  cout<<"Number of images is\t"<<N<<", where "<<N-pos-1<<" images do not be detected face."<<endl;
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

//// Flann query. Use VJ detector
//void Query(LightFaceRecognizer &recognizer, 
    //CascadeClassifier &cascade,
    //const string &query_file, 
    //const int& num_return=10, 
    //const string& dataset_file=string("dataset.hdf5"), 
    //const string& index_file=string("index.hdf5"), 
    //const string& dataset_path_file=string("dataset_file_path.txt") ) {

  //// Read query 
  //vector<string> query_file_path;
  //int N = 0;
  //if (query_file.substr(query_file.length()-4) == ".txt") {
    //ifstream file_list(query_file.c_str());
    //string line;
    //while (getline(file_list, line)) {
      //query_file_path.push_back(line);
      //N++;
    //}
  //}
  //else {
    //query_file_path.push_back(query_file);
    //N++;
  //}
  //::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[N*FEATURE_DIM], N, FEATURE_DIM);
  //cout<<"Query image(s):"<<endl;
  //for (int i = 0; i < N; i++) {
    //cout<<query_file_path[i]<<endl; 
    //Mat face = imread(query_file_path[i]);
    //Mat face_cropped = detectAlignCrop(face, cascade, recognizer);
    ////imshow("face_cropped", face_cropped);
    ////waitKey(0);
    //Mat face_feature;
    //recognizer.ExtractFaceFeature(face_cropped, face_feature);
    //memcpy(query[i], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
    ////cout<<face_feature.at<FEATURE_TYPE>(82)<<"\t"<<query[i][82]<<endl;
  //}

  //if (0 == N)
  //{
    //cerr<<"The specified input is not an image or valid \".txt\" file with image paths!"<<endl;
    //exit(-1);
  //}

  //// load dataset features 
  //::flann::Matrix<FEATURE_TYPE> dataset;
  //::flann::load_from_file(dataset, dataset_file, "dataset");
  //// load all path of faces in the dataset 
  //vector <string> data_file_path;
  //ifstream inFile(dataset_path_file.c_str());
  //while (inFile) {
    //string line;
    //getline(inFile, line);
    //data_file_path.push_back(line);
  //}
  //// load the index 
  ////::flann::Index< ::flann::CosDistance<FEATURE_TYPE> > index(dataset, ::flann::SavedIndexParams(index_file), ::flann::CosDistance<FEATURE_TYPE>());
  //::flann::Index< ::flann::L2<FEATURE_TYPE> > index(dataset, ::flann::SavedIndexParams(index_file));

  //[>
  ////-----------------------------------------
  //// "addPoints" and removePoint" testing 
  //::flann::Matrix<FEATURE_TYPE> added(new FEATURE_TYPE[N*FEATURE_DIM], N, FEATURE_DIM);
  //for (int i = 0; i < N; i++) {
    //Mat face = imread(query_file_path[i]);
    //Mat face_cropped = detectAlignCrop(face, cascade, recognizer);
    ////imshow("face_cropped", face_cropped);
    ////waitKey(0);
    //Mat face_feature;
    //recognizer.ExtractFaceFeature(face_cropped, face_feature);
    //memcpy(added[i], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
    //data_file_path.push_back(query_file_path[i]);
    ////cout<<face_feature.at<FEATURE_TYPE>(82)<<"\t"<<query[i][82]<<endl;
  //}
  //index.addPoints(added);  
  ////delete [] added.ptr();
  ////delete [] dataset.ptr(); // It seems that Index holds its own data. The orginal feature matrix does not matter after the construction of Index.
  //cout<<"Dataset size after add: "<<dataset.rows<<endl;
  //index.removePoint(22);
  //index.removePoint(24);
  //index.removePoint(26);
  //index.removePoint(28);
  ////cout<<index.getPoint(20)[0]<<"\t"<<query[0][0]<<endl;
  ////-----------------------------------------
  //*/

  //// prepare the search result matrix
  //::flann::Matrix<float> dists(new FEATURE_TYPE[query.rows*num_return], query.rows, num_return);
  //::flann::Matrix<int> indices(new int[query.rows*num_return], query.rows, num_return);

  //index.knnSearch(query, indices, dists, num_return, ::flann::SearchParams(128)); //::flann::CHECKS_AUTOTUNED

  //for (int ind = 0; ind < query.rows; ind++) {
    //cout<<"Query image is:\t"<<query_file_path[ind]<<endl;
    //cout<<"Return list:"<<endl;
    //for (int i = 0; i < num_return; i++) {
      //cout<<"No. "<<setw(2)<<i<<":\t"<<data_file_path[indices[ind][i]]<<"\tdist is\t"<<setprecision(6)<<dists[ind][i]<<endl;
    //}
    //cout<<"-----------------------------------------"<<endl<<endl;
  //}
  ////cout<<indices[0][0]<<endl;
  ////cout<<dataset[files.size()-1][82]<<endl;

  //delete [] dataset.ptr();
  //delete [] indices.ptr();
  //delete [] dists.ptr();
  //delete [] query.ptr();
//}

// TODO!! still use origin image as input while no face detected. change to dlib index
// Recoginition test with outlier faces (person whos face does not occupy in the FaceRepo)
void rec_test_outlier(LightFaceRecognizer & recognizer,
    //CascadeClassifier &cascade, 
    FaceAlign & face_align, 
    const string &query_file, // Txt file stores query faces' path
    const string &save_dir // FaceRepo directory
    ) {

  fs::path save_root(save_dir);
  if (!fs::exists(save_root) || !fs::is_directory(save_root)) {
    cerr<<"Dataset directory does not exist!"<<endl; 
    exit(-1);
  }

  fs::path dataset_file = save_root;
  dataset_file /= fs::path("dataset.hdf5");
  fs::path index_file = save_root;
  index_file /= fs::path("index.hdf5");
  fs::path dataset_path_file = save_root;
  dataset_path_file /= fs::path("dataset_file_path.txt");

  // Load index. 
  FaceRepo faceRepo(recognizer);
  if (!faceRepo.Load(save_dir)) {
    cerr<<"FaceRepo load fail from "<<save_dir<<endl;
    return;
  }
  int N = faceRepo.GetValidFaceNum();

  // Read query 
  vector<string> query_file_path;
  int NQ = 0;
  if (query_file.substr(query_file.length()-4) == ".txt") {
    ifstream file_list(query_file.c_str());
    string line;
    while (getline(file_list, line)) {
      query_file_path.push_back(line);
      NQ++;
    }
  }
  else {
    cerr<<"Query must a txt file with query image paths."<<endl;
    exit(-1);
  }

  vector<Mat> query;
  vector<string> invalid_query_file_path;
  cout<<"Query image(s):"<<endl;
  int pos = 0;
  for (vector<string>::iterator iter = query_file_path.begin(); iter != query_file_path.end();) {
    cout<<*iter<<endl; 
    Mat face = imread(*iter);
    Mat face_cropped = detectAlignCropDlib(face_align, face);
    if (face_cropped.empty()) {
      invalid_query_file_path.push_back(*iter);
      continue;
    }
    Mat face_feature;
    recognizer.ExtractFaceFeature(face_cropped, face_feature);
    query.push_back(face_feature);
  }
  NQ = pos;

  // Do search 
  int nFaces_UpBound = N > 100 ? 100 : N - 1;  // Up bound of retrieval faces
  vector<vector<string> > return_list;
  vector< vector<float> > dists;
  vector< vector<int> > indices;
  long time3 = clock();
  // Search
  faceRepo.Query(query, nFaces_UpBound, return_list, indices, dists);
  long time4 = clock();
  cout<<"Query time:\t"<<float(time4-time3)/1000000<<" sec., ";
  cout<<float(time4-time3)/NQ<<" us per query."<<endl;

  // Recognition statistics
  cout<<endl;
  cout<<"**************************************"<<endl;
  cout<<"RECOGNITION STATISTICS"<<endl;
  bool omit_first_return = false; //omit the first return (if query is also in the database, the first return should be the query itself)
  int & knn = KNN; // Size of return knn;
  float & th_dist = TH_DIST; // Distance threshold for same person.
  int & th_n = TH_N; // Least number of retrieved knn with same label.
  int num_valid_test = 0;
  int num_wrong_accept = 0;
  int num_reject = 0;
  for (int i = 0; i < NQ; i++) {
    num_valid_test++; 
    // Analyze the returned knn:
    vector <string> class_name_return;
    vector <float> dist_return;
    vector <int> count_same_class;
    for (int j = omit_first_return ? 1 : 0 ; j < knn; j++) { 
      if (dists[i][j] > th_dist)
        continue;
      string class_name_j = get_class_name(return_list[i][j]); 
      vector<string>::iterator iter = find(class_name_return.begin(), class_name_return.end(), class_name_j);
      if ( iter == class_name_return.end() ) {
        class_name_return.push_back(class_name_j);
        dist_return.push_back(dists[i][j]);
        count_same_class.push_back(1);
      } else {
        int pos = iter - class_name_return.begin();
        dist_return[pos] += dists[i][j];
        count_same_class[pos] ++;
      }
    }
    cout<<"#"<<i<<": "<<query_file_path[i]<<endl;
    cout<<"Number of groups in knn:"<<class_name_return.size()<<endl;
    float min_dist = 10000.0;
    string predict_class_name;
    for (int j = 0; j < class_name_return.size(); j++) {
      dist_return[j] /= count_same_class[j];
      if (dist_return[j] <= min_dist && count_same_class[j] >= th_n) {
        min_dist = dist_return[j];
        predict_class_name = class_name_return[j];
      }
      cout<<"Group #"<<j<<": "<<class_name_return[j]<<", count: "<<count_same_class[j]<<", ave dist: "<<dist_return[j]<<endl;
    }
    if (10000.0 == min_dist) {
      num_reject ++;
      cout<<"CORRECT REJECT\n"<<endl;
      continue;
    }
    else {
      num_wrong_accept++;
      cout<<"WRONG_ACCECPT"<<endl;
    }
    cout<<endl;
  }
  cout<<"----------------------------------------------"<<endl;
  cout<<"Number of valid recognition test: "<<num_valid_test<<endl;
  cout<<"Number of wrong accept: "<<num_wrong_accept<<", "<<(float)num_wrong_accept/num_valid_test*100<<endl;
  cout<<"Number of reject: "<<num_reject<<", "<<(float)num_reject/num_valid_test*100<<endl;
}

void retrieval_test_statistic(const int nFaces_UpBound, 
    const vector<string> & data_file_path,
    const vector<int> & face_count,
    const vector< vector<float> > & dists,
    const vector< vector<int> > &indices){
  // Retrieval statistics
  int N = data_file_path.size();
  assert(N == face_count.size());
  assert(N == dists.size());
  assert(N == indices.size());

  // RECOGNITION TEST parameters
  bool omit_first_return = true; //omit the first return (if query is also in the database, the first return should be the query itself)
  int & n_class = N_CLASS; // Least number of faces for a person.
  int & knn = KNN; // Size of return knn;
  float & th_dist = TH_DIST; // Distance threshold for same person.
  int & th_n = TH_N; // Least number of retrieved knn with same label.

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
    //string class_name = fs::canonical(data_file_path[i]).parent_path().filename().string();
    string class_name = get_class_name(data_file_path[i]); 
    for (int j = 0; j < nFaces_UpBound; j++) {
      if (face_count[i] < j + 1)
        break;
      num_images[j]++;
      //if (indices[i][j] >= N) {
        //cout<<"WRONG HERE! i:"<<i<<" j:"<<j<<"indices[i][j]: "<<indices[i][j]<<endl;
      //string class_name_j = fs::canonical(data_file_path[indices[i][j]]).parent_path().filename().string();
      string class_name_j = get_class_name(data_file_path[indices[i][j]]); 
      if (class_name_j == class_name) {
        correct[j][i] = 1;
      }
    }
  }
  float minDist = 100, maxDist = 0; // To test the max and min distance in flann repository
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < nFaces_UpBound; k++) {
      // To test the max and min distance of flann repository
      if (dists[i][k] < minDist)
        minDist = dists[i][k];
      if (dists[i][k] > maxDist)
        maxDist = dists[i][k];

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
  cout<<"Max distance is: "<<maxDist<<endl;
  cout<<"Min distance is: "<<minDist<<endl;

  // Recognition statistics
  cout<<endl;
  cout<<"**************************************"<<endl;
  cout<<"RECOGNITION STATISTICS"<<endl;
  int num_valid_test = 0;
  int num_correct_accept = 0;
  int num_wrong_accept = 0;
  int num_reject = 0;
  for (int i = 0; i < N; i++) {
    if (face_count[i] < n_class)
      continue;
    num_valid_test++; 
    // True class name.
    //string class_name = fs::canonical(data_file_path[i]).parent_path().filename().string();
      string class_name = get_class_name(data_file_path[i]); 
    // Analyze the returned knn:
    vector <string> class_name_return;
    vector <float> dist_return;
    vector <int> count_same_class;
    for (int j = omit_first_return ? 1 : 0 ; j < knn; j++) { 
      if (dists[i][j] > th_dist)
        continue;
      //string class_name_j = fs::canonical(data_file_path[indices[i][j]]).parent_path().filename().string();
      string class_name_j = get_class_name(data_file_path[indices[i][j]]); 
      vector<string>::iterator iter = find(class_name_return.begin(), class_name_return.end(), class_name_j);
      if ( iter == class_name_return.end() ) {
        class_name_return.push_back(class_name_j);
        dist_return.push_back(dists[i][j]);
        count_same_class.push_back(1);
      } else {
        int pos = iter - class_name_return.begin();
        dist_return[pos] += dists[i][j];
        count_same_class[pos] ++;
      }
    }
    cout<<"#"<<i<<": "<<data_file_path[i]<<endl;
    cout<<"True class name: "<<class_name<<", number of groups in knn:"<<class_name_return.size()<<endl;
    float min_dist = 10000.0;
    string predict_class_name;
    for (int j = 0; j < class_name_return.size(); j++) {
      dist_return[j] /= count_same_class[j];
      if (dist_return[j] <= min_dist && count_same_class[j] >= th_n) {
        min_dist = dist_return[j];
        predict_class_name = class_name_return[j];
      }
      cout<<"Group #"<<j<<": "<<class_name_return[j]<<", count: "<<count_same_class[j]<<", ave dist: "<<dist_return[j]<<endl;
    }
    if (10000.0 == min_dist) {
      num_reject ++;
      cout<<"REJECT\n"<<endl;
      continue;
    }
    if (predict_class_name == class_name) {
      num_correct_accept++;
      cout<<"CORRECT_ACCECPT"<<endl;
    }
    else {
      num_wrong_accept++;
      cout<<"WRONG_ACCECPT"<<endl;
    }
    cout<<endl;
  }
  cout<<"----------------------------------------------"<<endl;
  cout<<"knn: "<<knn<<", th_dist: "<<th_dist<<", th_n: "<<th_n<<endl;
  cout<<"Number of valid recognition test: "<<num_valid_test<<endl;
  cout<<"Number of correct accept: "<<num_correct_accept<<", "<<(float)num_correct_accept/num_valid_test*100<<endl;
  cout<<"Number of wrong accept: "<<num_wrong_accept<<", "<<(float)num_wrong_accept/num_valid_test*100<<endl;
  cout<<"Number of reject: "<<num_reject<<", "<<(float)num_reject/num_valid_test*100<<endl;

}

void retrieval_test_statistic(const int nFaces_UpBound, 
    const vector<string> & data_file_path,
    const vector<int> & face_count,
    const ::flann::Matrix<float> & dists,
    const ::flann::Matrix<int> &indices){
  int N = data_file_path.size();
  assert(N == face_count.size());
  assert(N == dists.rows);
  assert(N == indices.rows);

  vector< vector<float> > dists_vector;
  vector <vector<int> > indices_vector;
  
  for (int i = 0; i < N; i++) {
    vector<float> dists_i;
    vector<int> indices_i;
    for (int j = 0; j < nFaces_UpBound; j++) {
      indices_i.push_back(indices[i][j]);
      dists_i.push_back(dists[i][j]);
    }
    dists_vector.push_back(dists_i);
    indices_vector.push_back(indices_i);
  }
  retrieval_test_statistic(nFaces_UpBound, data_file_path, face_count, dists_vector, indices_vector);
}

// Retrieval and recognition test on a dataset.
// Each face perform as a query once.
// Statistic is done after every face has been used as a query.
void retrieval_test(LightFaceRecognizer & recognizer,
    //CascadeClassifier &cascade, 
    FaceAlign & face_align, 
    const string &file_name, // 
    const string &same_face_count, // Count of faces belong to the same person.
    const string &save_dir // Directory to save FaceRepo files.
    //const bool bCrop = false  // Whether to do detectAlignCrop()
    ) {

  bool bFreshRun = false;  // If true, recreate index

  fs::path save_root(save_dir);
  if (!fs::exists(save_root) || !fs::is_directory(save_root))
    fs::create_directories(save_root);

  fs::path dataset_file = save_root;
  dataset_file /= fs::path("dataset.hdf5");
  fs::path index_file = save_root;
  index_file /= fs::path("index.hdf5");
  fs::path dataset_path_file = save_root;
  dataset_path_file /= fs::path("dataset_file_path.txt");
  fs::path invalid_face_index_file = save_root;
  invalid_face_index_file /= fs::path("invalid_face_index.txt");

  // Create index. It may take a while depends on the number of images to be indexed.
  if (bFreshRun || !fs::exists(index_file) ) {
    long time1 = clock();
    //if (bCrop)
      //Index(recognizer, cascade, file_name, dataset_file.string(), index_file.string(), dataset_path_file.string());
      IndexDlib(recognizer, face_align, file_name, dataset_file.string(), index_file.string(), dataset_path_file.string(), invalid_face_index_file.string());
    //else
      //IndexWithoutCrop(recognizer, cascade, file_name, dataset_file.string(), index_file.string(), dataset_path_file.string());
    long time2 = clock();
    cout<<"Index DONE"<<time2-time1<<endl;
  }

  // load dataset features 
  ::flann::Matrix<FEATURE_TYPE> dataset;
  ::flann::load_from_file(dataset, dataset_file.string(), "dataset");
  // load all path of faces in the dataset 
  vector <string> data_file_path;
  ifstream inFile(dataset_path_file.string().c_str());
  string line;
  while (getline(inFile, line)) {
    data_file_path.push_back(line);
  }
  vector <int> invalid_face_index;
  vector <string> invalid_face_path;
  ifstream inFile2(invalid_face_index_file.string().c_str());
  while (getline(inFile2, line)) {
    vector <string> line_strings = split(line, ' '); 
    invalid_face_index.push_back(atoi(line_strings[1].c_str()));
    invalid_face_path.push_back(line_strings[0]);
  }
  // load the index 
  ::flann::Index< ::flann::L2<FEATURE_TYPE> > index(dataset, ::flann::SavedIndexParams(index_file.string()));
  int N = data_file_path.size();
  //cout<<"Dataset matrix rows: "<<dataset.rows<<endl;
  //cout<<dataset[dataset.rows-2][0]<<endl;
  //cout<<dataset[dataset.rows-2][1]<<endl;
  //cout<<dataset[dataset.rows-1][0]<<endl;
  //cout<<dataset[dataset.rows-1][1]<<endl;

  //// Read face count. Does not consider image cannot detect face.
  //int n = data_file_path.size();
  //cout<<"int n = data_file_path.size();"<<n<<endl;
  //vector<int> face_count(n, 0);
  //ifstream ifs_face_count(same_face_count.c_str());
  //for (int i = 0; getline(ifs_face_count, line); i++) {
    //face_count[i] = atoi(line.c_str());  
  //}
  //ifs_face_count.close();

  // Read face count. Exclude images that cannot detect face.
  // TODO!! Minus the no face image number!
  vector<int> face_count(N, 0);
  ifstream ifs_face_count(same_face_count.c_str());
  int pos1 = 0, pos2 = 0;
  for (int i = 0; getline(ifs_face_count, line); i++) {
    if (i != invalid_face_index[pos2])
      face_count[pos1++] = atoi(line.c_str());  
    else
      pos2++;
  }
  ifs_face_count.close();
  assert(face_count.size() == N);
  //cout<<face_count.size()<<" N: "<<N<<endl;
  // Subtract the number of images that cannot detect face.
  for (int i = 0; i < invalid_face_index.size(); i++) {
    string class_name_invalid_face = get_class_name(invalid_face_path[i]);
    for (int j = 0; j < N; j++) {
      if (get_class_name(data_file_path[j]) == class_name_invalid_face)
        face_count[j]--;
    }
  }

  // Do retrieval
  int nFaces_UpBound = N > 100 ? 100 : N - 1;  // Up bound of retrieval faces
  ::flann::Matrix<float> dists;
  ::flann::Matrix<int> indices;
  long time3 = clock();
  // Search result matrix
  dists = ::flann::Matrix<float> (new FEATURE_TYPE[N*nFaces_UpBound], N, nFaces_UpBound);
  indices = ::flann::Matrix<int> (new int[N*nFaces_UpBound], N, nFaces_UpBound);
  // Do search. Here query = dataset.
  // The following method may run out of memory.
  //::flann::Matrix<FEATURE_TYPE> query(dataset.ptr(), N, FEATURE_DIM);
  //index.knnSearch(query, indices, dists, nFaces_UpBound, ::flann::SearchParams(128)); //::flann::CHECKS_AUTOTUNED
  // Search one by one
  ::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[FEATURE_DIM], 1, FEATURE_DIM);
  ::flann::Matrix<float> dists_one_query(new FEATURE_TYPE[nFaces_UpBound], 1, nFaces_UpBound);
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
  long time4 = clock();
  cout<<"Query time:\t"<<float(time4-time3)/1000000<<" sec., ";
  cout<<float(time4-time3)/N<<" us per query."<<endl;

  retrieval_test_statistic(nFaces_UpBound, data_file_path, face_count, dists, indices);

  delete [] dataset.ptr();
  delete [] dists.ptr();
  delete [] indices.ptr();
  delete [] query.ptr();
  delete [] dists_one_query.ptr();
  delete [] indices_one_query.ptr();
}

// Retrieval and recognition test on a dataset.
// Each face perform as a query once.
// Statistic is done after every face has been used as a query.
void retrieval_test(FaceRepo & faceRepo, 
    const string &save_dir,
    const vector<int> &face_count,
    const vector<string> &data_file_path) {

  if (!faceRepo.Load(save_dir)) {
    cerr<<"FaceRepo load fail from "<<save_dir<<endl;
    return;
  }
  int N = faceRepo.GetValidFaceNum();

  // Do retrieval
  int nFaces_UpBound = N > 100 ? 100 : N - 1;  // Up bound of retrieval faces
  //vector<vector<string> > return_list;
  vector< vector<float> > dists;
  vector< vector<int> > indices;
  long time3 = clock();
  // Search one by one
  ::flann::Matrix<FEATURE_TYPE> query;
  for (int i = 0; i < N; i++) {
    vector<vector<string> > return_list;
    vector<vector<int> > return_list_pos;
    vector<vector<float> > return_dists;
    // Query by previously computed face features. 
    query = faceRepo.GetFeature(i);
    faceRepo.Query(query, nFaces_UpBound, return_list, return_list_pos, return_dists);
    delete [] query.ptr();
    dists.push_back(return_dists[0]);
    indices.push_back(return_list_pos[0]);
  }
  long time4 = clock();
  cout<<"Query time:\t"<<float(time4-time3)/1000000<<" sec., ";
  cout<<float(time4-time3)/N<<" us per query."<<endl;

  retrieval_test_statistic(nFaces_UpBound, data_file_path, face_count, dists, indices);
}

//// Specific retrieval test on lfw. No use now
//void retrieval_on_lfw(LightFaceRecognizer & recognizer,
    //CascadeClassifier &cascade) {

  //bool bFreshRun = false;  // If true, recreate index, and redo all search
  //string dataset_file("dataset.hdf5");
  //string index_file("index.hdf5");
  //string dataset_path_file("dataset_file_path.txt");
  //string lfw_indices_file = "lfw_indices.hdf5";
  //string lfw_dists_file = "lfw_dists.hdf5";
  //string lfw_file_name = "../../../lfw_data/lfw_file_name.txt";
  //string lfw_same_face_count = "../../../lfw_data/lfw_same_face_count.txt";

  //// Create index. It takes quite a long time. 
  //if (bFreshRun) {
    //long time1 = clock();
    //Index(recognizer, cascade, lfw_file_name);
    //long time2 = clock();
  //}

  //// load dataset features 
  //::flann::Matrix<FEATURE_TYPE> dataset;
  //::flann::load_from_file(dataset, dataset_file, "dataset");
  //// load all path of faces in the dataset 
  //vector <string> data_file_path;
  //ifstream inFile(dataset_path_file.c_str());
  //string line;
  //while (getline(inFile, line)) {
    //data_file_path.push_back(line);
  //}
  //// load the index 
  ////::flann::Index< ::flann::CosDistance<FEATURE_TYPE> > index(dataset, ::flann::SavedIndexParams(index_file), ::flann::CosDistance<FEATURE_TYPE>());
  //::flann::Index< ::flann::L2<FEATURE_TYPE> > index(dataset, ::flann::SavedIndexParams(index_file));

  //// Read face count
  //int N = data_file_path.size();
  //vector<int> face_count(N, 0);
  //ifstream ifs_face_count(lfw_same_face_count.c_str());
  //for (int i = 0; getline(ifs_face_count, line); i++) {
    //face_count[i] = atoi(line.c_str());  
  //}
  //ifs_face_count.close();

  //// Do retrieval
  //int nFaces_UpBound = 100;  // Up bound of retrieval faces
  //::flann::Matrix<float> dists;
  //::flann::Matrix<int> indices;
  ////if (bFreshRun) {
  //if (true) {
    //long time3 = clock();
    //// Search result matrix
    //dists = ::flann::Matrix<float> (new FEATURE_TYPE[N*nFaces_UpBound], N, nFaces_UpBound);
    //indices = ::flann::Matrix<int> (new int[N*nFaces_UpBound], N, nFaces_UpBound);
    //// Do search. Here query = dataset.
    //// The following method may run out of memory.
    ////::flann::Matrix<FEATURE_TYPE> query(dataset.ptr(), N, FEATURE_DIM);
    ////index.knnSearch(query, indices, dists, nFaces_UpBound, ::flann::SearchParams(128)); //::flann::CHECKS_AUTOTUNED
    //// Search one by one
    //::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[FEATURE_DIM], 1, FEATURE_DIM);
    //::flann::Matrix<float> dists_one_query(new FEATURE_TYPE[nFaces_UpBound], 1, nFaces_UpBound);
    //::flann::Matrix<int> indices_one_query(new int[nFaces_UpBound], 1, nFaces_UpBound);
    //for (int i = 0; i < N; i++) {
      //memcpy(query[0], dataset[i], sizeof(FEATURE_TYPE)*FEATURE_DIM);
      ////cout<<i<<": "<<query[0][FEATURE_DIM-1]<<"\t"<<dataset[i][FEATURE_DIM-1]<<endl;
      //index.knnSearch(query, indices_one_query, dists_one_query, nFaces_UpBound, ::flann::SearchParams(::flann::FLANN_CHECKS_AUTOTUNED)); //::flann::SearchParams(128)
      ////index.knnSearch(query, indices_one_query, dists_one_query, nFaces_UpBound, ::flann::SearchParams(16)); //::flann::SearchParams(128)
      //memcpy(indices[i], indices_one_query[0], sizeof(FEATURE_TYPE)*nFaces_UpBound);
      //memcpy(dists[i], dists_one_query[0], sizeof(int)*nFaces_UpBound);
      ////waitKey(0);
    //}
    //// Remove existed hdf5 files, otherwise an error will happen when size, of the data to be written, exceed the existed ones.
    //remove(lfw_dists_file.c_str());
    //remove(lfw_indices_file.c_str());
    //::flann::save_to_file(dists, lfw_dists_file, "dists");
    //::flann::save_to_file(indices, lfw_indices_file, "indices");
    //long time4 = clock();
    //cout<<"Query time:\t"<<float(time4-time3)/1000000<<" sec., ";
    //cout<<float(time4-time3)/N<<" us per query."<<endl;
  //}
  //else {
    //::flann::load_from_file(dists, lfw_dists_file, "dists");
    //::flann::load_from_file(indices, lfw_indices_file, "indices");
  //}

  //// Statistics
  //vector<int> num_images(nFaces_UpBound, 0); // number of images who have more than XX faces
  //vector< vector<float> > precision_per_rank;
  //vector< vector<int> > correct;
  //for (int i = 0; i < nFaces_UpBound; i++) {
    //precision_per_rank.push_back(vector<float>(nFaces_UpBound, 0));
    //correct.push_back(vector<int>(N, 0));
  //}
  //for (int i = 0; i < N; i++) {
      //string class_name = get_class_name(data_file_path[i]); 
    //for (int j = 0; j < nFaces_UpBound; j++) {
      //if (face_count[i] < j + 1)
        //break;
      //num_images[j]++;
      //string class_name_j = get_class_name(data_file_path[indices[i][j]]); 
      //if (class_name_j == class_name) {
        //correct[j][i] = 1;
      //}
    //}
  //}
  ////// A test for queries who have more than 70 faces in the dataset
  ////int a = 0, b = 0;
  ////for (int i = 0; i < N; i++) {
    ////if (face_count[i] > 70)  {
      ////a = 0;
      ////for (int j = 0; j < 70; j++)
      ////{
        ////cout<<correct[j][i]<<"\t";
        ////a+=correct[j][i];
      ////}
      ////cout<<a<<endl;
      ////b += a;
    ////}
  ////}
  ////cout<<float(b)/70/num_images[70]<<endl;

  //for (int i = 0; i < N; i++) {
    //for (int k = 0; k < nFaces_UpBound; k++) {
      //if (face_count[i] > k) {
        //for (int j = 0; j <= k; j++) {
          //for (int m = j; m <= nFaces_UpBound; m++) {
            //precision_per_rank[k][m] += correct[j][i];
          //}
        //}
      //}
    //}
  //}
  //for (int k = 0; k < nFaces_UpBound; k++) {
    //for (int j = 0; j <= k; j++) {
      //precision_per_rank[k][j] /= (num_images[k] * (j+1) );
    //}
  //}

  //for (int i = 0; i < nFaces_UpBound; i++) {
    //if (num_images[i] < 1)
      //break;
    //cout<<"--------------------------------------------------"<<endl;
    //cout<<"Images who have "<<i+1<<" faces: "<<num_images[i]<<" images"<<endl;
    //cout<<"Precision per rank:"<<endl;
    //for (int j = 0; j <= i; j++) {
      //cout<<"Rank #"<<j+1<<": "<<precision_per_rank[i][j]<<endl;
    //}
    //cout<<endl;
  //}

  //delete [] dataset.ptr();
  //delete [] dists.ptr();
  //delete [] indices.ptr();
//}

//void retrieval_on_lfw_batch(LightFaceRecognizer & recognizer,
    //CascadeClassifier &cascade) {

  //string lfw_indices_file = "../../../lfw_data/dataset_FaceRepo/lfw_indices.hdf5";
  //string lfw_dists_file = "../../../lfw_data/dataset_FaceRepo/lfw_dists.hdf5";
  //string lfw_directory = "../../../lfw_data/dataset_FaceRepo";
  //string lfw_file_name = "../../../lfw_data/lfw_file_name.txt";
  //string lfw_same_face_count = "../../../lfw_data/lfw_same_face_count.txt";

  //FaceRepo faceRepo(recognizer, cascade);

  //// Try load existed index. Create index if load failed, which takes quite a long time.
  //bool bFreshRun = !faceRepo.Load(lfw_directory);
  //if (bFreshRun) {
    //long time1 = clock();
    //faceRepo.InitialIndex(lfw_file_name);
    //long time2 = clock();
    //// Save index and features.
    //faceRepo.Save(lfw_directory);
  //}

  ////// Rebuild index
  ////faceRepo.RebuildIndex();

  //// load dataset features 
  //::flann::Matrix<FEATURE_TYPE> dataset;
  //::flann::load_from_file(dataset, lfw_directory+"/dataset.hdf5", "dataset");
  //// load all path of faces in the dataset 
  //vector <string> data_file_path;
  //ifstream inFile(lfw_file_name.c_str());
  //string line;
  //while (getline(inFile, line)) {
    //data_file_path.push_back(line);
  //}
  
  //// Read face count
  //int N = faceRepo.GetFaceNum();
  //vector<int> face_count(N, 0);
  //ifstream ifs_face_count(lfw_same_face_count.c_str());
  //for (int i = 0; getline(ifs_face_count, line); i++) {
    //face_count[i] = atoi(line.c_str());  
  //}
  //ifs_face_count.close();

  //// Do retrieval
  //int nFaces_UpBound = 100;  // Up bound of retrieval faces
  //::flann::Matrix<float> dists;
  //::flann::Matrix<int> indices;
  ////if (bFreshRun) {
  //if (true) {
    //long time3 = clock();
    //// Search result matrix
    //dists = ::flann::Matrix<float> (new FEATURE_TYPE[N*nFaces_UpBound], N, nFaces_UpBound);
    //indices = ::flann::Matrix<int> (new int[N*nFaces_UpBound], N, nFaces_UpBound);
    //// Search one by one
    //for (int i = 0; i < N; i++) {
      ////cout<<i<<": "<<query[0][FEATURE_DIM-1]<<"\t"<<dataset[i][FEATURE_DIM-1]<<endl;
      //vector<vector<string> > return_list;
      //vector<vector<int> > return_list_pos;
      //vector<vector<float> > return_dists;
      //faceRepo.Query(::flann::Matrix<FEATURE_TYPE>(dataset[i], 1, FEATURE_DIM), nFaces_UpBound, return_list, return_list_pos, return_dists);
      //for (int j = 0; j < nFaces_UpBound; j++) {
        //indices[i][j] = return_list_pos[0][j];
        //dists[i][j] = return_dists[0][j];
      //}
      ////waitKey(0);
    //}
    //// Remove existed hdf5 files, otherwise an error will happen when size, of the data to be written, exceed the existed ones.
    //remove(lfw_dists_file.c_str());
    //remove(lfw_indices_file.c_str());
    //::flann::save_to_file(dists, lfw_dists_file, "dists");
    //::flann::save_to_file(indices, lfw_indices_file, "indices");
    //long time4 = clock();
    //cout<<"Query time:\t"<<float(time4-time3)/1000000<<" sec., ";
    //cout<<float(time4-time3)/N<<" us per query."<<endl;
  //}
  //else {
    //::flann::load_from_file(dists, lfw_dists_file, "dists");
    //::flann::load_from_file(indices, lfw_indices_file, "indices");
  //}

  //// Statistics
  //vector<int> num_images(nFaces_UpBound, 0); // number of images who have more than XX faces
  //vector< vector<float> > precision_per_rank;
  //vector< vector<int> > correct;
  //for (int i = 0; i < nFaces_UpBound; i++) {
    //precision_per_rank.push_back(vector<float>(nFaces_UpBound, 0));
    //correct.push_back(vector<int>(N, 0));
  //}
  //for (int i = 0; i < N; i++) {
      //string class_name = get_class_name(data_file_path[i]); 
    //for (int j = 0; j < nFaces_UpBound; j++) {
      //if (face_count[i] < j + 1)
        //break;
      //num_images[j]++;
      //string class_name_j = get_class_name(data_file_path[indices[i][j]]); 
      //if (class_name_j == class_name) {
        //correct[j][i] = 1;
      //}
    //}
  //}
  //for (int i = 0; i < N; i++) {
    //for (int k = 0; k < nFaces_UpBound; k++) {
      //if (face_count[i] > k) {
        //for (int j = 0; j <= k; j++) {
          //for (int m = j; m <= nFaces_UpBound; m++) {
            //precision_per_rank[k][m] += correct[j][i];
          //}
        //}
      //}
    //}
  //}
  //for (int k = 0; k < nFaces_UpBound; k++) {
    //for (int j = 0; j <= k; j++) {
      //precision_per_rank[k][j] /= (num_images[k] * (j+1) );
    //}
  //}

  //// Print results.
  //for (int i = 0; i < nFaces_UpBound; i++) {
    //if (num_images[i] < 1)
      //break;
    //cout<<"--------------------------------------------------"<<endl;
    //cout<<"Images who have "<<i+1<<" faces: "<<num_images[i]<<" images"<<endl;
    //cout<<"Precision per rank:"<<endl;
    //for (int j = 0; j <= i; j++) {
      //cout<<"Rank #"<<j+1<<": "<<precision_per_rank[i][j]<<endl;
    //}
    //cout<<endl;
  //}

  //delete [] dists.ptr();
  //delete [] indices.ptr();
//}

//void retrieval_on_lfw(LightFaceRecognizer & recognizer,
    //CascadeClassifier &cascade, 
    //const string & query,
    //const size_t num_return = 10) {
    ////size_t num_return = 10) {
  ////num_return = 13233;
  //string lfw_directory = "../../../lfw_data/dataset_FaceRepo";
  //string lfw_file_name = "../../../lfw_data/lfw_file_name.txt";
  //// Try load existed index. Create index if load failed, which takes quite a long time.
  //FaceRepo faceRepo(recognizer, cascade);
  //bool bFreshRun = !faceRepo.Load(lfw_directory);
  //if (bFreshRun) {
    //long time1 = clock();
    //faceRepo.InitialIndex(lfw_file_name);
    //long time2 = clock();
    //// Save index and features.
    //faceRepo.Save(lfw_directory);
  //}

  //vector<vector<string> > return_list;  // Path of return images.
  //vector<vector<int> > return_list_pos; // Indices in the face dataset of the return images.
  //vector<vector<float> > return_dists; // Distance between return and query images.
  //// Do query. 
  //faceRepo.Query(query, num_return, return_list, return_list_pos, return_dists);

  //cout<<"Return list of \""<<query<<"\" in LFW dataset:"<<endl;
  //for (int i = 0; i < num_return; i++)
    //cout<<"#"<<i<<": "<<return_list[0][i]<<", dist is: "<<return_dists[0][i]<<endl;
//}

//void testFaceRepo(LightFaceRecognizer &recognizer, CascadeClassifier &cascade){
  //FaceRepo faceRepo(recognizer, cascade);
  //string txtFile = string("../../../test_faces.txt");
  ////string queryFile = string("../../../test_faces.txt");
  //string queryFile = string("../../../all.txt");
  //string addFile = string("../../../more.txt");
  //string removeFile = string("../../../test_faces/Abdullah_Gul_0003.jpg");
  //string removeFile2 = string("../../../test_faces/Dennis_Hastert_0001.jpg");
  //string directory = string("./test_old"); 
  //string directory_save = string("./test"); 
  
  ////faceRepo.InitialIndex(txtFile);
  ////faceRepo.Save(directory);

  //faceRepo.Load(directory);
  //faceRepo.AddFace(addFile);

  //faceRepo.RemoveFace(removeFile);
  //faceRepo.RemoveFace(removeFile2);
  //faceRepo.RebuildIndex();

  //faceRepo.Save(directory_save);

  //vector<vector<string> > return_list;
  //vector<vector<int> > return_list_pos;
  //vector<vector<float> > dists;
  //int num_return = 10;
  //faceRepo.Query(queryFile, num_return, return_list, return_list_pos, dists);
  //for (int i = 0; i < return_list.size(); i++) {
    //cout<<"Query "<<i<<":"<<endl;
    //for (int j = 0; j < num_return; j++) {
      //cout<<"#"<<j<<": "<<return_list_pos[i][j]<<" ("<<dists[i][j]<<"), "<<return_list[i][j]<<endl;
    //}
    //cout<<endl;
  //}
//}

// Detect face, then align and crop it to canonical front face.
// Save aligned face image to "save_path", one subfolder for each person.
void FindValidFaceDlib(FaceAlign & face_align, 
    const vector <fs::path> & file_path,
    //const string &save_path)
    const string &save_path,
    vector <string> &saved_image_path, // #size = n
    vector <int> &person_face_count, // #size = K
    // Indicator to person_face_count
    vector <int> &valid_image_person_ind, // #value = 0,...,K-1; #size = n 
    vector <string> &person_name, // #size = K
    vector <string> &valid_image_path, // #size = n
    vector <string> &invalid_image_path) // #size = N - n 
{
  assert(file_path.size() > 0);
    
  fs::path dest_root_path(save_path); 
  int N = file_path.size();
  cout<<"Image(s):"<<endl;
  for (int i = 0; i < N; i++) {
    cout<<i<<": "<<file_path[i]<<endl; 
    Mat face = imread(file_path[i].string());
    Mat face_cropped = detectAlignCropDlib(face_align, face);
    if (face_cropped.empty()) {
      invalid_image_path.push_back(file_path[i].string());
      continue;
    }
    // Valid face
    valid_image_path.push_back(file_path[i].string());
    string class_name = get_class_name(file_path[i].string()); 
    vector<string>::iterator iter = find(person_name.begin(), person_name.end(), class_name);
    //cout<<"CLASS_NAME: "<<class_name<<endl;
    if ( iter == person_name.end() ) { // New person.
      person_name.push_back(class_name);
      person_face_count.push_back(1);
      valid_image_person_ind.push_back(person_name.size()-1);
      //cout<<"NEW PERSON: "<<person_name.size()-1<<" "<<person_name[person_name.size()-1]<<endl;
    } else {
      int pos = iter - person_name.begin();
      person_face_count[pos] ++;
      valid_image_person_ind.push_back(pos);
      //cout<<"OLD PERSON: "<<pos<<" "<<person_name[pos]<<" "<<person_face_count[pos]<<endl;
    }
    // Save to the disk.
    //cout<<relative_path<<endl;
    fs::path dest_path = dest_root_path;
    dest_path /= fs::path(class_name);
    fs::create_directories(dest_path);
    cout<<"Save detected face to: "<<dest_path<<endl;
    dest_path /= file_path[i].filename(); 
    imwrite(dest_path.string(), face_cropped);
    saved_image_path.push_back(dest_path.string());
  }
  
  //cout<<"Valid images:"<<endl;
  //for (int i = 0; i < valid_image_path.size(); i++) {
    //cout<<i<<": "<<valid_image_path[i]<<"\t"; // #size = n
    //cout<<valid_image_person_ind[i]<<endl; // #value = 0,...,K-1; #size = n 
    //cout<<"Saved to "<<saved_image_path[i]<<endl; // #value = 0,...,K-1; #size = n 
  //}
  //cout<<"Person names:"<<endl;
  //for (int i = 0; i < person_name.size(); i++) {
    //cout<<i<<": "<<person_name[i]<<"\t"; // #size = K
    //cout<<person_face_count[i]<<endl; // #size = K
  //}
  //cout<<"Invalid images:"<<endl;
  //for (int i = 0; i < invalid_image_path.size(); i++) {
    //cout<<i<<": "<<invalid_image_path[i]<<endl; 
  //}
}

void FindValidFaceDlib(FaceAlign & face_align, 
    const string &image_path_file,
    //const string &save_path)
    const string &save_path,
    vector <string> &saved_image_path, // #size = n
    vector <int> &person_face_count, // #size = K
    // Indicator to person_face_count
    vector <int> &valid_image_person_ind, // #value = 0,...,K-1; #size = n 
    vector <string> &person_name, // #size = K
    vector <string> &valid_image_path, // #size = n
    vector <string> &invalid_image_path) // #size = N - n 
{
  vector <fs::path> file_path;
  ifstream inFile(image_path_file.c_str());
  while (inFile) {
    string line;
    getline(inFile, line);
    file_path.push_back(fs::path(line));
  }
  //FindValidFaceDlib(face_align, file_path, save_path);
  FindValidFaceDlib(face_align, file_path, save_path, 
      saved_image_path, person_face_count, valid_image_person_ind,
      person_name, valid_image_path, invalid_image_path);
}

void FindValidFaceDlib(FaceAlign & face_align, 
    const string &image_root,
    const string &save_path,
    //const string &ext) 
    const string &ext,
    vector <string> &saved_image_path, // #size = n
    vector <int> &person_face_count, // #size = K
    // Indicator to person_face_count
    vector <int> &valid_image_person_ind, // #value = 0,...,K-1; #size = n 
    vector <string> &person_name, // #size = K
    vector <string> &valid_image_path, // #size = n
    vector <string> &invalid_image_path) // #size = N - n 
{

  // Read image file list 
  vector<fs::path> file_path;

  fs::path root(image_root); 
  get_all(root, ext, file_path);

  FindValidFaceDlib(face_align, file_path, save_path, 
      saved_image_path, person_face_count, valid_image_person_ind,
      person_name, valid_image_path, invalid_image_path);
}

//void FindValidFace(LightFaceRecognizer &recognizer, 
    //CascadeClassifier &cascade,
    //const string &image_root,
    //const string &save_path,
    //const string &ext = string(".jpg")) {

  //// Read image file list 
  //vector<fs::path> file_path;

  //fs::path root(image_root); 
  //get_all(root, ext, file_path);
  //int N = file_path.size();
  //if (0 == N)
  //{
    //cerr<<"No image found in the given path with \""<<ext<<"\" extension."<<endl;
    //exit(-1);
  //}

  //// Detect face, then save to the disk.
  //cout<<"Image(s):"<<endl;
  //for (int i = 0; i < N; i++) {
    //cout<<file_path[i]<<endl; 
    //Mat face = imread(file_path[i].string());
    //Mat face_cropped = detectAlignCropPractical(face, cascade, recognizer);
    //if (face_cropped.empty())
      //continue;

    ////imshow("face_cropped", face_cropped);
    ////waitKey(0);

    //// Save to the disk.
    //string relative_path = fs::canonical(file_path[i]).string().substr(fs::canonical(root).string().length());
    ////cout<<relative_path<<endl;
    //fs::path dest_path(save_path); 
    //dest_path += fs::path(relative_path);
    //fs::create_directories(dest_path.parent_path());
    //cout<<"Save detected face to: "<<dest_path<<endl;
    //imwrite(dest_path.string(), face_cropped);
  //}
//}

void detect_and_recognition_test(LightFaceRecognizer & recognizer, FaceAlign &face_align, string input_image, string save_image_dir, string repo_dir)
{
  // Detect face and align it. Then save the aligned face to the disk.
  vector <string> saved_image_path;
  vector <int> person_face_count;
  vector <int> valid_image_person_ind;
  vector <int> image_face_count;
  vector <string> person_name;
  vector <string> valid_image_path;
  vector <string> invalid_image_path;
  string ext = ".jpg";
  cout<<"FIND VALID FACE"<<endl;
  if (fs::is_directory(input_image))
    FindValidFaceDlib(face_align, input_image, save_image_dir, ext,
        saved_image_path, person_face_count, valid_image_person_ind,
        person_name, valid_image_path, invalid_image_path);
  else
    FindValidFaceDlib(face_align, input_image, save_image_dir, 
        saved_image_path, person_face_count, valid_image_person_ind,
        person_name, valid_image_path, invalid_image_path);

  for (int i = 0; i < saved_image_path.size(); i++){
    image_face_count.push_back(person_face_count[valid_image_person_ind[i]]);
  }
  
  // Create face repository
  cout<<"CREATE FACE REPOSITORY"<<endl;
  FaceRepo faceRepo(recognizer);
  faceRepo.InitialIndex(saved_image_path);
  faceRepo.Save(repo_dir);
  // Save face count to disk
  fs::path face_count_save_file(repo_dir);
  face_count_save_file /= fs::path("face_count.txt");
  ofstream ofile(face_count_save_file.string().c_str());
  ostream_iterator<int> output_iterator(ofile, "\n");
  copy(image_face_count.begin(), image_face_count.end(), output_iterator);

  cout<<"RETRIEVAL TEST"<<endl;
  retrieval_test(faceRepo, repo_dir, image_face_count, saved_image_path);
}

void load_and_recognition_test(LightFaceRecognizer & recognizer, FaceAlign &face_align, string save_dir)
{
  // Load face repository.
  vector<int> image_face_count;
  vector<string> saved_image_path;
  FaceRepo faceRepo(recognizer);
  faceRepo.Load(save_dir);
  if (!faceRepo.Load(save_dir)) {
    cerr<<"FaceRepo load fail from "<<save_dir<<endl;
    exit(-1);
  }
  int N = faceRepo.GetValidFaceNum();
  fs::path face_count_save_file(save_dir);
  face_count_save_file /= fs::path("face_count.txt");
  fs::path save_image_path_file(save_dir);
  save_image_path_file /= fs::path("dataset_file_path.txt");
  ifstream ifile1(face_count_save_file.string().c_str());
  ifstream ifile2(save_image_path_file.string().c_str());
  string line;
  for (int i = 0; i < N; i++) {
    getline(ifile1, line);
    image_face_count.push_back(atoi(line.c_str()));
    getline(ifile2, line);
    saved_image_path.push_back(line);
  }
  ifile1.close();
  ifile2.close();

  retrieval_test(faceRepo, save_dir, image_face_count, saved_image_path);
}


void batch_do_recognition_test(LightFaceRecognizer & recognizer, FaceAlign &face_align, string repo_dir) {
//int N_CLASS = 10; // Least number of faces for a person.
//int KNN = 10; // Size of return knn;
//float TH_DIST = 0.1; // Distance threshold for same person.
//int TH_N = 1; // Least number of retrieved knn with same label.

    int alter_th_n[] = {1, 2, 3, 4, 5};
    float alter_th_dist[] = {0.1, 0.2, 0.4, 0.6};
    for (int i = 0; i < 5; i++)
      for (int j = 0; j < 4; j++) {
        TH_N = alter_th_n[i];
        TH_DIST = alter_th_dist[j];
        load_and_recognition_test(recognizer, face_align, repo_dir);
      }
}

int main(int argc, char **argv) {
  // Init Recognizer
  // void *recognizer = InitRecognizer("../../models/big/big.prototxt",
  //"../../models/big/big.caffemodel", "");
  LightFaceRecognizer recognizer(
      "../../../face_rec_models/model_cnn/small",
      "../../../face_rec_models/model_bayesian/bayesian_model_lfw.bin", 
      "prob", false);
  //LightFaceRecognizer recognizer(
      //"../../../face_rec_models/model_cnn/small",
      //"../../../face_rec_models/model_face_alignment",
      //"../../../face_rec_models/model_bayesian/bayesian_model_lfw.bin", "prob",
      //false);

  FaceAlign face_align("../../../face_rec_models/shape_predictor_68_face_landmarks.dat");

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

  //if (1 == argc)
    //// Do query for each image in lfw dataset, and do statistic for the result.
    //retrieval_on_lfw_batch(recognizer, cascade);  
  //else
    //// Do query in lfw dataset for the input image. 
    //retrieval_on_lfw(recognizer, cascade, argv[1]);  

  /*
  // Detect face and align it. Then save the aligned face to the disk.
  //FindValidFace(recognizer, cascade, argv[1], argv[2]);
  //FindValidFaceDlib(face_align, argv[1], argv[2]);
  vector <string> saved_image_path;
  vector <int> person_face_count;
  vector <int> valid_image_person_ind;
  vector <int> image_face_count;
  vector <string> person_name;
  vector <string> valid_image_path;
  vector <string> invalid_image_path;
  string ext = ".jpg";
  cout<<"FIND VALID FACE"<<endl;
  FindValidFaceDlib(face_align, argv[1], argv[2], ext,
      saved_image_path, person_face_count, valid_image_person_ind,
      person_name, valid_image_path, invalid_image_path);
  for (int i = 0; i < saved_image_path.size(); i++){
    image_face_count.push_back(person_face_count[valid_image_person_ind[i]]);
  }
  
  // Create face repository
  cout<<"CREATE FACE REPOSITORY"<<endl;
  FaceRepo faceRepo(recognizer);
  faceRepo.InitialIndex(saved_image_path);
  faceRepo.Save(argv[3]);
  // Save face count to disk
  fs::path face_count_save_file(argv[3]);
  face_count_save_file /= fs::path("face_count.txt");
  ofstream ofile(face_count_save_file.string().c_str());
  ostream_iterator<int> output_iterator(ofile, "\n");
  copy(image_face_count.begin(), image_face_count.end(), output_iterator);

  ////retrieval_test(recognizer, cascade, argv[1], argv[2], argv[3]);
  //retrieval_test(recognizer, face_align, argv[1], argv[2], argv[3]);
  cout<<"RETRIEVAL TEST"<<endl;
  retrieval_test(faceRepo, argv[3], image_face_count, saved_image_path);
*/
  
  //// Load face repository.
  //vector<int> image_face_count;
  //vector<string> saved_image_path;
  //FaceRepo faceRepo(recognizer);
  //faceRepo.Load(argv[1]);
  //if (!faceRepo.Load(argv[1])) {
    //cerr<<"FaceRepo load fail from "<<argv[1]<<endl;
    //return -1;
  //}
  //int N = faceRepo.GetValidFaceNum();
  //fs::path face_count_save_file(argv[1]);
  //face_count_save_file /= fs::path("face_count.txt");
  //fs::path save_image_path_file(argv[1]);
  //save_image_path_file /= fs::path("dataset_file_path.txt");
  //ifstream ifile1(face_count_save_file.string().c_str());
  //ifstream ifile2(save_image_path_file.string().c_str());
  //string line;
  //for (int i = 0; i < N; i++) {
    //getline(ifile1, line);
    //image_face_count.push_back(atoi(line.c_str()));
    //getline(ifile2, line);
    //saved_image_path.push_back(line);
  //}
  //ifile1.close();
  //ifile2.close();

  //retrieval_test(faceRepo, argv[1], image_face_count, saved_image_path);

  if (2 == argc) {
    // Load FaceRepo from argv[1], then do face recognition test. Each image as a query.
    load_and_recognition_test(recognizer, face_align, string(argv[1]));
    // Batch do load_and_recognition_test with different parameter setting.
    //batch_do_recognition_test(recognizer, face_align, string(argv[1]));
  }
  if (4 == argc) {
    // Detect face in argv[1], save aligned face to argv[2], save FaceRepo to argv[3]
    detect_and_recognition_test(recognizer, face_align, string(argv[1]), string(argv[2]), string(argv[3]));
    batch_do_recognition_test(recognizer, face_align, string(argv[3]));
  }

  return 0;
}
