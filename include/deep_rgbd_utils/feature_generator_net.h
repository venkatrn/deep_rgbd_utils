#pragma once

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

using namespace caffe;
using std::string;

// Look at pixel_vertex_mapper.cpp for example usage of this class.
class FeatureGenerator {
 public:
  FeatureGenerator(const string& caffe_model_file,
             const string& trained_file);

  // Feature vectors for all pixels in the image, in row major order.
  // For RGB image. (Needs to be 0-255 range)
  std::vector<std::vector<float>> GetFeatures(const cv::Mat& img);
  // For RGB-D image. (RGB is [0-255], D is in millimiters -- the default
  // Kinect format)
  std::vector<std::vector<float>> GetFeatures(const cv::Mat& rgb_img, const cv::Mat& depth_image);
  // Ditto, but for a particular pixel in the image.
  std::vector<float> GetFeature(const cv::Mat& img, const cv::Point& point);

 private:
  void SetMean(const string& mean_file);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& rgb_img, const cv::Mat& depth_img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};
