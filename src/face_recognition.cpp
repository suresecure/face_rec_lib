#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include <glog/logging.h>
#include "classifier.hpp"

#include "face_recognition.hpp"

namespace face_rec_srzn{
	using namespace cv;
	using namespace std;
	
	void* InitRecognizer(const string& cfgname, const string& modelname, const string& mean_file)
	{
		// caffe setting
		FLAGS_minloglevel = 3; // close INFO and WARNING level log
		::google::InitGoogleLogging("SRZNFaceRecognitionLib");
		Classifier *ret_recognizer = new Classifier(cfgname, modelname, mean_file);
		return (void*)ret_recognizer;
	}
	vector<float> ExtractFaceFeatureFromImage(Mat face, Classifier *classifier)
	{
    return classifier->extract_layer_by_name(face, "prob");  
    //return classifier->extract_layer_by_name(face, "fc7");  
	}
	vector<float> ExtractFaceFeatureFromBuffer(void *rec, void *imgbuf, int w, int h)
	{
		Mat face_mat(h, w, CV_8UC3, imgbuf);
		Classifier *recognizer = (Classifier*)rec;
		return ExtractFaceFeatureFromImage(face_mat, recognizer);
	}
	float FaceVerification(const vector<float>& face1_feature, const vector<float>& face2_feature)
	{
		float sum = 0;
		for (int i = 0; i < face1_feature.size(); i++)
		{
			sum += (face1_feature[i] - face2_feature[i])*(face1_feature[i] - face2_feature[i]);
		}
		sum = sqrtf(sum);
    float best_distance = 0;
    float range = 0.f;
		if (face1_feature.size() == 160)
		{
      best_distance = 54.61f;
      range = 10.f;
		}
		else if (face1_feature.size() == 4096)
		{
      best_distance = 0;
      range = 100.f;
		}
		else
			return false;
		//if (sum <= min_distance)
			//return 1.0f;
		//else if (sum >= max_distance)
			//return 0.0f;
		//else
    float similarity = 0.9f - (sum-best_distance)/range;
    if(similarity<0.f)
      similarity = 0.f;
    else if(similarity>1.0f)
      similarity = 1.0f;
    return similarity;
	}
	void ReleaseRecognizer(void *rec)
	{
		Classifier *recognizer = (Classifier*)rec;
		delete recognizer;
	}
}
