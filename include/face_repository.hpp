#ifndef HEADER_FACE_REPOSITORY
#define HEADER_FACE_REPOSITORY
// Face retrieval library. Not thread safe!
// Author: Luo Lei 
// 2016.4.23
// Email: robert165 AT 163.com

#include <string>
#include <vector>

#include <flann/flann.hpp>
#include "face_recognition.hpp"
#include "opencv2/opencv.hpp"

#define FEATURE_DIM 160
typedef float FEATURE_TYPE;
#define CV_FEATURE_DATA_TYPE CV_32FC1

namespace face_rec_srzn {
  using namespace std;
  using namespace cv;

  // Face repository class. Create face repository based on their feature. 
  // Support create, add, delete, query in the repo.
  // Beware that FaceRepo supposes that the input image is an aligned face and it has no effect on face detection and alignment, so the input face images should be aligned before use.
  class FaceRepo {
    public:
      //FaceRepo(LightFaceRecognizer & recognizer,
          //CascadeClassifier & cascade);
      FaceRepo(LightFaceRecognizer & recognizer);
      ~FaceRepo();

      // Initial index from images whos paths specified in a ".txt" file. One face image per line.
      bool InitialIndex(string &path_list_file);
      // Initial index from images whos paths specified in a string vector. One face image per string.
      bool InitialIndex(const vector<string> & filelist, const vector<cv::Mat> & feature_list = vector<cv::Mat>());

      // Save face repository to the specified directory. Can customize file names. If any face add/remove after the initial index, FaceRepo::RebuildIndex() will be called before save.
      bool Save(const string & directory, 
          const string & dataset_file = string("dataset.hdf5"), // All face features.
          const string & index_file = string("index.hdf5"), // flann index file
          const string & dataset_path_file = string("dataset_file_path.txt") // Face image paths.
          );
      // Load face repository from the specified directory. Correct file names should be given. 
      bool Load(const string & directory, 
          const string & dataset_file = string("dataset.hdf5"),
          const string & index_file = string("index.hdf5"),
          const string & dataset_path_file = string("dataset_file_path.txt") );

      // Do query. The query images paths are specified by a string vector. Return result images' paths, their index in face repository, and their (feature) distance to the query face.
      void Query(const vector<string>  &query_list, // Query faces' paths.  
          const size_t & num_return, // The number of faces returned for each query face.
          vector<vector<string> >& return_list, // Paths of the returned face images. Each query correponds to a string vector with length of "num_return". 
          vector<vector<int> >& return_list_pos, // Indices of the returned faces in the repository. 
          vector<vector<float> > & dists // Distance of the returned face and the query.
          );
      // Do query. The query image is specified by "query_file", or images are listed in "query_file" (should be a ".txt" file, one query's path per line).
      void Query(const string &query_file, const size_t & num_return, vector<vector<string> >& return_list, vector<vector<int> >& return_list_pos, vector<vector<float> > & dists);
      // Do query by feature. 
      void Query(const ::flann::Matrix<FEATURE_TYPE> & query, const size_t & num_return, vector<vector<string> >& return_list, vector<vector<int> >& return_list_pos, vector<vector<float> > & dists);
      void Query(const vector<cv::Mat> & query, const size_t & num_return, vector<vector<string> >& return_list, vector<vector<int> >& return_list_pos, vector<vector<float> > & dists);
      void Query(const cv::Mat & query, const size_t & num_return, vector<string>& return_list, vector<int>& return_list_pos, vector<float> & dists);

      // Rebuild flann index. If any face is added/removed after the initial index, the features matrices will be re-orgnized to a uniform one, where the removed faces will truely deleted. Return false if FLANN index is not actually rebuilt.
      bool RebuildIndex();

      // Add face(s) to the repository. FLANN will not rebuild index immediately. An automatic rebuild will be triged when the repository size doubled.
      bool AddFace(const string &file);
      bool AddFace(const vector<string> & filelist, const vector<cv::Mat> & feature_list = vector<cv::Mat>());
      // Remove one face from the repository. Just mark as removed, not really remove it. 
      bool RemoveFace(const string & face_path);
      bool RemoveFace(const size_t point_id);

      // Get face feature from the repository.
      ::flann::Matrix<FEATURE_TYPE> GetFeature(const string & face_path);
      ::flann::Matrix<FEATURE_TYPE> GetFeature(const size_t point_id);
      cv::Mat GetFeatureCV(const string & face_path);
      cv::Mat GetFeatureCV(const size_t point_id);
      
      // Get path or id from its counterpart.
      int GetID(const string & face_path); // Return -1 if not found.
      string GetPath(const size_t point_id); // Return "" if not found.

      // Get the total number of faces in the repository. 
      size_t GetFaceNum();
      // Get the number of valid faces in the repository (exclude faces marked as "removed"). They will truely removed in FaceRepo::RebuildIndex(). 
      size_t GetValidFaceNum();

    private:
      vector<string> _file_path;   // Hold face image file absolute path.
      ::flann::Index< ::flann::L2<FEATURE_TYPE> > * _index; // The flann index.

      vector< ::flann::Matrix<FEATURE_TYPE> > _feature_list; // Face features.
      //vector< vector<bool> > _invalid_face_mask;    // Mask for invalid face feature. Because RemoveFace() will not actually change the data struct.
      vector<size_t> _removed_face_ind; // Indices of removed faces in "_file_path". Because RemoveFace() will not actually change the data struct.
      //CascadeClassifier & _cascade; // Face detector.
      LightFaceRecognizer & _recognizer; // Face recognizer, for face feature extraction.
  };

}
#endif
