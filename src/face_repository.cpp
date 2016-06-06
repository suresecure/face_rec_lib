#include <fstream>
#include <stdio.h>
#include <time.h>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "face_recognition.hpp"
#include "boost/filesystem.hpp"
#include "flann/flann.hpp"
#include "flann/io/hdf5.h"
#include "face_repository.hpp"


namespace face_rec_srzn {
  using namespace cv;
  using namespace std;
  namespace fs = ::boost::filesystem;

  // Helper funtion. Load paths specified in "file", and store their absolute path to the end of "path_vector". If "file" is not a ".txt" file, push its own absolute path to "path_vector".
  size_t ReadPath(const string & file, vector<string> & path_vector) {
    size_t N = 0;
    if (file.substr(file.length()-4) == ".txt") {
      ifstream file_list(file.c_str());
      string line;
      while (getline(file_list, line)) {
        line = fs::canonical(line).string();
        if (!(fs::exists(fs::path(line))&&fs::is_regular_file(fs::path(line))))
          continue;
        path_vector.push_back(line);
        N++;
      }
      file_list.close();
    }
    else {
      path_vector.push_back(fs::canonical(file).string());
      N++;
    }
    return N;
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

  /*
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
*/
  //FaceRepo::FaceRepo(LightFaceRecognizer & recognizer,
      //CascadeClassifier & cascade) : _cascade(cascade), _recognizer(recognizer)
  FaceRepo::FaceRepo(LightFaceRecognizer & recognizer) : _recognizer(recognizer)
  {
    _index = NULL;
  }

  FaceRepo::~FaceRepo() {

    if (_index) {
      delete _index;
      _index = NULL;
    }
    for ( int i = 0; i < _feature_list.size(); i++ ) {
      delete [] _feature_list[i].ptr();
    }
  }

  bool FaceRepo::InitialIndex(string &path_list_file) {
    //int N = 0;
    //ifstream file_list(path_list_file.c_str());
    //string line;
    //while (getline(file_list, line)) {
    //line = fs::canonical(line).string();
    //_file_path.push_back(line);
    //N++;
    //}
    //file_list.close();
    size_t N = ReadPath(path_list_file, _file_path);

    if (0 == N)
    {
      cerr<<"FLANN index: wrong input file list!"<<endl;
      return false; 
    }
    return InitialIndex(_file_path);

  }

  bool FaceRepo::InitialIndex(const vector<string> & filelist) {
    if (&filelist != &_file_path) {
      string line;
      for ( int i = 0; i < filelist.size(); i++ ) {
        line = fs::canonical(filelist[i]).string();
        _file_path.push_back(line);
      }
    }

    try {
      long time1 = clock();
      int N = _file_path.size();
      ::flann::Matrix<FEATURE_TYPE> dataset(new FEATURE_TYPE[N*FEATURE_DIM], N, FEATURE_DIM);
      _feature_list.push_back(dataset);
      for (int i = 0; i < N; i++) {
        cout<<i<<"/"<<N<<": "<<_file_path[i]<<endl;
        Mat face = imread(_file_path[i]);
        Mat & face_cropped = face;
        //Mat face_cropped = detectAlignCrop(face, _cascade, _recognizer);
        Mat face_feature;
        _recognizer.ExtractFaceFeature(face_cropped, face_feature);
        memcpy(dataset[i], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
      }
      long time2 = clock();
      cout<<N<<" face feature extracted in "<<float(time2-time1)/1000000<<" seconds, "<<float(time2-time1)/N/1000<<" ms per face"<<endl;
      cout<<"Building index...";
      _index = new ::flann::Index< ::flann::L2<FEATURE_TYPE> > (dataset, ::flann::AutotunedIndexParams());
      _index->buildIndex();
      long time3 = clock();
      cout<<"\t"<<"Done!\n"<<"Index built in "<<float(time3-time2)/1000000<<" seconds."<<endl;
    }
    //catch(Exception e)
    catch(...)
    {
      //cerr<<"Error happened in FaceRepoInitialIndex():"<<e<<endl;
      cerr<<"Error happened in FaceRepoInitialIndex()."<<endl;
      if (NULL != _index)
        delete _index;
      if (NULL != _feature_list[0].ptr())
        delete [] _feature_list[0].ptr();
      return false;
    }

    return true;
  }

  bool FaceRepo::RebuildIndex() {
    if (NULL == _index || 0 == _feature_list.size())
      return false;

    int N = GetValidFaceNum();

    long time1 = clock();
    if ( !(GetFaceNum() == N && 1 == _feature_list.size())) {
      // We have some faces added/removed. 
      // Re-orgnize all valid face feature into one FLANN matrix, then re-index.
      ::flann::Matrix<FEATURE_TYPE> dataset(new FEATURE_TYPE[N*FEATURE_DIM], N, FEATURE_DIM);
      sort(_removed_face_ind.begin(), _removed_face_ind.end());
      int k_invalid = _removed_face_ind.size() - 1;
      int k_all = GetFaceNum() - 1;
      int k_valid = N - 1;
      for (int i = _feature_list.size() - 1; i >= 0; i--) {
        for( int j = _feature_list[i].rows - 1; j >=0; j--) {
          if (k_invalid >= 0 && _removed_face_ind[k_invalid] == k_all) {
            k_invalid--;
            _file_path.erase((vector<string>::iterator)&_file_path[k_all]);
            k_all--;
            continue;
          }
          k_all--;
          memcpy(dataset[k_valid--], _feature_list[i][j], sizeof(FEATURE_TYPE)*FEATURE_DIM);
        }
        delete [] _feature_list[i].ptr();
      }
      _removed_face_ind.clear();
      _feature_list.clear();
      _feature_list.push_back(dataset);
    }
    delete _index;
    _index = new ::flann::Index< ::flann::L2<FEATURE_TYPE> > (_feature_list[0], ::flann::AutotunedIndexParams());
    _index->buildIndex();
    long time2 = clock();
    cout<<"Rebuild index in "<<float(time2-time1)/1000000<<" seconds."<<endl;
    return true;
  }

  bool FaceRepo::Save(const string & directory, 
      const string & dataset_file ,
      const string & index_file ,
      const string & dataset_path_file ) {

    if (NULL == _index) {
      cerr<<"Call FaceRepo::Save() before initial index."<<endl;
      return false;
    }

    if (!fs::is_directory(directory)) {
      cout<<"Save directory \""<<directory<<"\" does not exist. Create it."<<endl;
      fs::create_directories(directory);
    }

    fs::path path_dataset_file(directory), path_index_file(directory), path_dataset_path_file(directory);
    path_dataset_file /= dataset_file;
    path_index_file /= index_file;
    path_dataset_path_file /= dataset_path_file;

    // Always save clean and uniform repository.
    RebuildIndex();

    // Remove existed hdf5 files, otherwise an error will happen when size, of the data to be written, exceed the existed ones.
    //remove(dataset_file.c_str());
    //remove(index_file.c_str());
    fs::remove(path_dataset_file);
    fs::remove(path_index_file);
    try { 
      ::flann::save_to_file(_feature_list[0], path_dataset_file.string(), "dataset");
      _index->save(path_index_file.string());
      ofstream ofile(path_dataset_path_file.string().c_str());
      ostream_iterator<string> output_iterator(ofile, "\n");
      copy(_file_path.begin(), _file_path.end(), output_iterator);
      ofile.close();
    }
    catch(...)
    {
      cerr<<"Error happened in FaceRepo::Save()."<<endl;
      return false;
    }

    return true;
  }

  bool FaceRepo::Load(const string & directory, 
      const string & dataset_file,
      const string & index_file,
      const string & dataset_path_file ) {

    if (!fs::is_directory(directory)) {
      cerr<<"Load directory \""<<directory<<"\" does not exist."<<endl;
      return false;
    }

    fs::path path_dataset_file(directory), path_index_file(directory), path_dataset_path_file(directory);
    path_dataset_file /= dataset_file;
    path_index_file /= index_file;
    path_dataset_path_file /= dataset_path_file;

    ::flann::Index< ::flann::L2<FEATURE_TYPE> > * index = NULL;
    ::flann::Matrix<FEATURE_TYPE> dataset;
    vector<string> file_path;
    try {
      // Load dataset features.
      ::flann::load_from_file(dataset, path_dataset_file.string(), "dataset");
      string line;
      ifstream inFile(path_dataset_path_file.string().c_str());
      while (getline(inFile, line)) {
        file_path.push_back(line);
      }
      inFile.close();
      // Load index.
      index =  new ::flann::Index< ::flann::L2<FEATURE_TYPE> >(dataset, ::flann::SavedIndexParams(path_index_file.string()));
    }
    catch(...)
    {
      cerr<<"Error happened in FaceRepo::Load()."<<endl;
      if (NULL != index)
        delete index;
      if (NULL != dataset.ptr())
        delete [] dataset.ptr();
      return false;
    }

    // Clean old data.
    if (NULL != _index) {
      delete _index;
      for ( int i = 0; i < _feature_list.size() - 1; i++ ) {
        delete [] _feature_list[i].ptr();
      }
      _feature_list.clear();
      _file_path.clear();
      _removed_face_ind.clear();
    }

    // Assign new data.
    _index = index;
    _feature_list.push_back(dataset);
    _file_path.assign(file_path.begin(), file_path.end());
    return true;
  }

  void FaceRepo::Query(const ::flann::Matrix<FEATURE_TYPE> & query, 
      const size_t &num_return, 
      vector<vector<string> >& return_list, 
      vector<vector<int> >& return_list_pos, 
      vector<vector<float> > & dists) {

    if (NULL == _index) {
      cerr<<"Call FaceRepo::Query() before initial index."<<endl;
      return;
    }

    if (0 == query.rows || FEATURE_DIM != query.cols)
      return;

    // prepare the search result matrix
    ::flann::Matrix<FEATURE_TYPE> flann_dists(new FEATURE_TYPE[query.rows*num_return], query.rows, num_return);
    ::flann::Matrix<int> indices(new int[query.rows*num_return], query.rows, num_return);

    _index->knnSearch(query, indices, flann_dists, num_return, ::flann::SearchParams(::flann::FLANN_CHECKS_AUTOTUNED)); 

    //for (int ind = 0; ind < query.rows; ind++) {
    //cout<<"Query image is:\t"<<query_list[ind]<<endl;
    //cout<<"Return list:"<<endl;
    //for (int i = 0; i < num_return; i++) {
    //cout<<"No. "<<setw(2)<<i<<":\t"<<_file_path[indices[ind][i]]<<"\tdist is\t"<<setprecision(6)<<flann_dists[ind][i]<<endl;
    //}
    //cout<<"-----------------------------------------"<<endl<<endl;
    //}

    for (int ind = 0; ind < query.rows; ind++) {
      vector<string> return_ind;
      vector<int> return_ind_pos;
      vector<float> dists_ind;
      for (int i = 0; i < num_return; i++) {
        return_ind.push_back(_file_path[indices[ind][i]]);
        return_ind_pos.push_back(indices[ind][i]);
        dists_ind.push_back(flann_dists[ind][i]);
      }
      return_list.push_back(return_ind);
      return_list_pos.push_back(return_ind_pos);
      dists.push_back(dists_ind);
    }
  }

  void FaceRepo::Query(const vector<cv::Mat> & query, 
      const size_t &num_return, 
      vector<vector<string> >& return_list, 
      vector<vector<int> >& return_list_pos, 
      vector<vector<float> > & dists) {
    //::flann::Matrix<FEATURE_TYPE> feature((FEATURE_TYPE*)query.data, 1, FEATURE_DIM);
    ::flann::Matrix<FEATURE_TYPE> feature(new FEATURE_TYPE(query.size()*FEATURE_DIM), query.size(), FEATURE_DIM);
    for (int i = 0; i < query.size(); i++) {
      memcpy(feature[i], query[i].data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
    }
    Query(feature, num_return, return_list, return_list_pos, dists);
    delete feature.ptr();
  }

  void FaceRepo::Query(const vector<string>  &query_list, 
      const size_t &num_return, 
      vector<vector<string> >& return_list, 
      vector<vector<int> >& return_list_pos, 
      vector<vector<float> > & dists) {

    if (NULL == _index) {
      cerr<<"Call FaceRepo::Query() before initial index."<<endl;
      return;
    }

    int N = query_list.size();
    ::flann::Matrix<FEATURE_TYPE> query(new FEATURE_TYPE[N*FEATURE_DIM], N, FEATURE_DIM);
    //cout<<"Query image(s):"<<endl;
    for (int i = 0; i < N; i++) {
      //cout<<query_list[i]<<endl; 
      Mat face = imread(query_list[i]);
      Mat & face_cropped = face;
      //Mat face_cropped = detectAlignCrop(face, _cascade, _recognizer);
      //imshow("face_cropped", face_cropped);
      //waitKey(0);
      Mat face_feature;
      _recognizer.ExtractFaceFeature(face_cropped, face_feature);
      memcpy(query[i], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
      //cout<<face_feature.at<FEATURE_TYPE>(82)<<"\t"<<query[i][82]<<endl;
    }

    Query(query, num_return, return_list, return_list_pos, dists);
    delete [] query.ptr();

    //// prepare the search result matrix
    //::flann::Matrix<FEATURE_TYPE> flann_dists(new FEATURE_TYPE[query.rows*num_return], query.rows, num_return);
    //::flann::Matrix<int> indices(new int[query.rows*num_return], query.rows, num_return);

    //_index->knnSearch(query, indices, flann_dists, num_return, ::flann::SearchParams(::flann::FLANN_CHECKS_AUTOTUNED)); 

    ////for (int ind = 0; ind < query.rows; ind++) {
    ////cout<<"Query image is:\t"<<query_list[ind]<<endl;
    ////cout<<"Return list:"<<endl;
    ////for (int i = 0; i < num_return; i++) {
    ////cout<<"No. "<<setw(2)<<i<<":\t"<<_file_path[indices[ind][i]]<<"\tdist is\t"<<setprecision(6)<<flann_dists[ind][i]<<endl;
    ////}
    ////cout<<"-----------------------------------------"<<endl<<endl;
    ////}

    //for (int ind = 0; ind < query.rows; ind++) {
      //vector<string> return_ind;
      //vector<int> return_ind_pos;
      //vector<float> dists_ind;
      //for (int i = 0; i < num_return; i++) {
        //return_ind.push_back(_file_path[indices[ind][i]]);
        //return_ind_pos.push_back(indices[ind][i]);
        //dists_ind.push_back(flann_dists[ind][i]);
      //}
      //return_list.push_back(return_ind);
      //return_list_pos.push_back(return_ind_pos);
      //dists.push_back(dists_ind);
    //}
  }

  void FaceRepo::Query(const string &query_file, 
      const size_t &num_return, 
      vector<vector<string> >& return_list, 
      vector<vector<int> >& return_list_pos, 
      vector<vector<float> > & dists) {

    vector<string> query_list;
    size_t N = ReadPath(query_file, query_list);

    if (0 == N)
    {
      cerr<<"FLANN query: wrong input query file list!"<<endl;
      return; 
    }
    Query(query_list, num_return, return_list, return_list_pos, dists);
  }

  bool FaceRepo::AddFace(const string &path_list_file) {
    vector<string> filelist;  
    size_t N = ReadPath(path_list_file, filelist);
    return AddFace(filelist);
  }

  bool FaceRepo::AddFace(const vector<string> & filelist) {
    if (0 == GetFaceNum())
      return InitialIndex(filelist);

    long time1 = clock();
    int N = filelist.size();
    ::flann::Matrix<FEATURE_TYPE> dataset(new FEATURE_TYPE[N*FEATURE_DIM], N, FEATURE_DIM);
    try {
      for (int i = 0; i < N; i++) {
        cout<<i<<"/"<<N<<": "<<filelist[i]<<endl;
        Mat face = imread(filelist[i]);
        Mat & face_cropped = face;
        //Mat face_cropped = detectAlignCrop(face, _cascade, _recognizer);
        Mat face_feature;
        _recognizer.ExtractFaceFeature(face_cropped, face_feature);
        memcpy(dataset[i], face_feature.data, sizeof(FEATURE_TYPE)*FEATURE_DIM);
      }
    }
    catch(...) {
      cerr<<"Read image of extract feature failed in FaceRepo::AddFace()"<<endl;
      return false;
    }
    long time2 = clock();
    cout<<"Add "<<N<<" faces into the face repository in "<<float(time2-time1)/1000000<<" seconds."<<endl;
    _index->addPoints(dataset);
    _feature_list.push_back(dataset);

    cout<<"BEFORE INSERT "<<_file_path.size()<<endl;
    for(vector<string>::iterator t = _file_path.begin(); t != _file_path.end(); t++)
      cout<<*t<<endl;
    _file_path.insert(_file_path.end(), filelist.begin(), filelist.end());
    cout<<"AFTER INSERT"<<_file_path.size()<<endl;
    for(vector<string>::iterator t = _file_path.begin(); t != _file_path.end(); t++)
      cout<<*t<<endl;
    return true;
  }

  bool FaceRepo::RemoveFace(const string & face_path) {
    vector<string>::iterator t = find(_file_path.begin(), _file_path.end(), fs::canonical(face_path).string());
    if (t == _file_path.end()) {// Not found.
      cout<<"The specified face to be removed is not found."<<endl;
      return false;
    }
    size_t ind = t - _file_path.begin();
    return RemoveFace(ind);
  }

  bool FaceRepo::RemoveFace(const size_t point_id) {
    if (find(_removed_face_ind.begin(), _removed_face_ind.end(), point_id) != _removed_face_ind.end()) {
      cout<<"The specified face already removed."<<endl;
      return false;
    }
    _index->removePoint(point_id);
    _removed_face_ind.push_back(point_id);
    return true;
  }

  ::flann::Matrix<FEATURE_TYPE> FaceRepo::GetFeature(const string & face_path){
    vector<string>::iterator t = find(_file_path.begin(), _file_path.end(), fs::canonical(face_path).string());
    if (t == _file_path.end()) {// Not found.
      cout<<"The specified face to be get is not found."<<endl;
      return ::flann::Matrix<FEATURE_TYPE>();
    }
    size_t ind = t - _file_path.begin();
    return GetFeature(ind);
  }

  ::flann::Matrix<FEATURE_TYPE> FaceRepo::GetFeature(const size_t point_id) {
    if (point_id < 0 || point_id > _file_path.size() )
      return ::flann::Matrix<FEATURE_TYPE>();

    ::flann::Matrix<FEATURE_TYPE> feature(new FEATURE_TYPE[FEATURE_DIM], 1, FEATURE_DIM);
    int sum = 0;
      for (int i = 0; i < _feature_list.size(); i++) {
        if ( sum + _feature_list[i].rows >= point_id ) 
          memcpy(feature[0], _feature_list[i][point_id-sum], sizeof(FEATURE_TYPE)*FEATURE_DIM);
        else
          sum += _feature_list[i].rows;
        }
      return feature;
  }

  size_t FaceRepo::GetFaceNum() {
    return _file_path.size();
  }

  size_t FaceRepo::GetValidFaceNum(){
    return _file_path.size() - _removed_face_ind.size(); 
  }
}
