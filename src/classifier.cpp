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

#include "classifier.hpp"

namespace face_rec_srzn {
using namespace caffe; // NOLINT(build/namespaces)
using namespace std;

Classifier::Classifier(const string &model_file, const string &trained_file,
                       const string &mean_file, bool use_gpu) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  if (use_gpu)
    Caffe::set_mode(Caffe::GPU);
  else
    Caffe::set_mode(Caffe::CPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float> *input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  if (!mean_file.empty())
    SetMean(mean_file);
  else {
    // cv::Scalar channel_mean(129.1863, 104.7624, 93.5940); // mean image
    // (constant) valuei per channel
    cv::Scalar channel_mean(
        93.5940, 104.7624,
        129.1863); // mean image (constant) valuei per channel
    SetMean(channel_mean);
  }
}

/* Set mean by meam value */
void Classifier::SetMean(cv::Scalar channel_mean) {
  mean_ = cv::Mat(input_geometry_, 21,
                  channel_mean); // ?? 21 = COLOR_BGR5652GRAY ?? is the value of
                                 // mean.type() in the original
                                 // Classifier::SetMean()
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string &mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float *data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  // cv::Scalar channel_mean = cv::mean(mean);
  // mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  mean_ = mean;
}

void Classifier::extract_layer_by_name(const cv::Mat &img,
                                       const string &layer_name,
                                       vector<float> &feature) {
  Blob<float> *input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_, input_geometry_.height,
                       input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();

  const boost::shared_ptr<Blob<float> > feature_blob =
      net_->blob_by_name(layer_name);
  const float *begin = feature_blob->cpu_data();
  const float *end = feature_blob->cpu_data() + feature_blob->count();
  feature = std::vector<float>(begin, end);
  return;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat> *input_channels) {
  Blob<float> *input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float *input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat &img,
                            std::vector<cv::Mat> *input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
 * input layer of the network because it is wrapped by the cv::Mat
 * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float *>(input_channels->at(0).data) ==
        net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}
}
