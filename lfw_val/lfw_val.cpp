#include <time.h>
#include <stdio.h>

#include "opencv2/highgui/highgui.hpp"
#include "face_recognition.hpp"
#include "boost/filesystem.hpp"

using namespace cv;
using namespace std;
using namespace face_rec_srzn;
using namespace boost::filesystem;

int main( int argc, char** argv )
{
	//Init Recognizer
	void *recognizer = InitRecognizer("../../models/big/big.prototxt",
      "../../models/big/big.caffemodel", "");

	//Load face images
	//string face1_name = string("validation_faces/1.jpg");
	//string face2_name = string("validation_faces/2.jpg");
	//Mat face1 = imread(face1_name);
	//Mat face2 = imread(face2_name);
	//// compute frame per second (fps)
	////int64 start_tick = getTickCount();

	////Extract feature from images
	//vector<float> face1_feature = ExtractFaceFeatureFromBuffer(recognizer, face1.data, face1.cols, face1.rows);
	//vector<float> face2_feature = ExtractFaceFeatureFromBuffer(recognizer, face2.data, face2.cols, face2.rows);

	//double t = ((double)getTickCount() - start_tick) / getTickFrequency();//elapsed time
	//cout << "Feature extraction cost: " << t << " seconds" << endl;

	//float similarity_12 = FaceVerification(face1_feature, face2_feature);
	//float similarity_23 = FaceVerification(face2_feature, face3_feature);
	//string result = is_the_same?string("PASS"):string("FAILED");
	//cout << face_1_name << " and " << face_2_name << " verification " << result << endl;
	//imshow("face1", face1);
	//imshow("face2", face2);
	//imshow("face3", face3);
	//waitKey(0);
    return 0;
}
