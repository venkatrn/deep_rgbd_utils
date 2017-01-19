#pragma once

#include <opencv2/core/core.hpp>
#include <deep_rgbd_utils/helpers.h>

#include <vector>

namespace dru {

// A wrapper around an OpenCV image for convenient operations involving
// pixel-wise (i.e, dense) features.
class Image {
 public:
  Image() {
  }
  Image(const std::string &filename) {
    SetImage(filename);
  }

  Image(cv::Mat &image) {
    SetImage(image);
  }

  void SetImage(const std::string &filename);
  void SetImage(cv::Mat &image);

  const cv::Mat &image() const {
    return image_;
  }

  int cols() {
    return image_.cols;
  }

  int rows() {
    return image_.rows;
  }

  // Convenience wrapper for at method.
  template<typename T>
  T at(int row, int col) const {
    return image_.at<T>(row, col);
  }

  void SetFeatures(const std::vector<FeatureVector> &features) {
    features_ = features;
  }

  const FeatureVector& FeatureAt(int index) const {
    return features_.at(index);
  }
  const FeatureVector& FeatureAt(int x, int y) const {
    return FeatureAt(cv::Point(x, y));
  }
  const FeatureVector& FeatureAt(const cv::Point &point) const {
    return FeatureAt(PointToIndex(point, image_.rows, image_.cols));
  }

  // Return the k-nearest pixels in feature-space.
  std::vector<cv::Point> GetNearestPixels(int k,
                                          const FeatureVector &feature_vector);

  // Return the local maxima matches (sorted) for a given feature vector. This
  // first computes a distance map (as defined below) and then the k-largest maxima in
  // it.
  std::vector<PointValuePair> GetMaximaAndHeatmap(
    const FeatureVector &feature_vector, cv::Mat &heatmap,
    cv::Mat &raw_distance_map,
    double sigma = 1e-10, double alpha = 0.5);

  std::vector<PointValuePair> GetMaxima(
    const FeatureVector &feature_vector);
  // Get a heatmap representing the distance to each image pixel (in
  // feature space) for a given feature vector. We compute the distance as
  // exp((-sigma * dist(feature_vector - f)). The parameter alpha is used
  // to blend the heatmap color with the original image. A value of 0 means
  // the original image is returned, and 1.0 means the heatmap is
  // returned. Optionally, specify which dimensions should be used to compute
  // the distance.
  void GetHeatmap(const FeatureVector &feature_vector, cv::Mat &heatmap,
                  cv::Mat &raw_distance_map, std::vector<int> dim_idxs, 
                  double sigma = 1e-10, double alpha = 0.5) const;
  void GetHeatmap(const FeatureVector &feature_vector, cv::Mat &heatmap,
                  cv::Mat &raw_distance_map, 
                  double sigma = 1e-10, double alpha = 0.5) const;

 private:
  cv::Mat image_;
  int data_type_;
  std::unique_ptr<flann::Index<L2<float>>> kd_tree_index_;

  // The feature vector for pixel (x,y) is accessed as features_[x *
  // image_height + y] or equivalenty features_[col * image_rows + row].
  std::vector<FeatureVector> features_;

};

} // namespace
