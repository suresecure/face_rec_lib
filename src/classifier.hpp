#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

/* Pair (label, confidence) representing a prediction. */
// typedef std::pair<string, float> Prediction;
namespace face_rec_srzn {
using namespace caffe; // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

class Classifier {
public:
  Classifier(const string &model_file, const string &trained_file,
             const string &mean_file = string(), bool use_gpu = true);

  void extract_layer_by_name(const cv::Mat &img, const string &layer_name,
                             Mat &feature);
                             //vector<float> &feature);

private:
  void SetMean(cv::Scalar channel_mean); // set mean by value
  void SetMean(const string &mean_file);
  void WrapInputLayer(std::vector<cv::Mat> *input_channels);
  void Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels);

private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  // std::vector<string> labels_;
};
}
