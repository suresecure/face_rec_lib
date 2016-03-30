#ifndef HEADER_FACE_RECOGNITION
#define HEADER_FACE_RECOGNITION
//face recognition library
//author: xun changqing
//2016.3.28
//email: xunchangqing AT qq.com

#include <string>
#include <vector>

namespace face_rec_srzn{
	using namespace std;
	string GetVersion();
	void* InitRecognizer(const string& cfgname, const string& modelname, const string& mean_file);
	vector<float> ExtractFaceFeatureFromBuffer(void *recognizer, void *imgbuf, int w, int h);
	float FaceVerification(const vector<float>& face1_feature, const vector<float>& face2_feature);
	void ReleaseRecognizer(void *recognizer);
}
#endif
