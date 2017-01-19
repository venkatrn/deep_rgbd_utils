#include <deep_rgbd_utils/image.h>
#include <deep_rgbd_utils/helpers.h>

#include <flann/flann.h>

using namespace std;

namespace dru {

void Image::SetImage(const std::string &filename) {
  image_ = cv::imread(filename, -1);

  if (image_.empty()) {
    printf("Unable to read image %s\n", filename.c_str());
  }

  data_type_ = image_.type();
}

void Image::SetImage(cv::Mat &image) {
  image_ = image;
  data_type_ = image_.type();
}

std::vector<cv::Point> Image::GetNearestPixels(int num_neighbors,
                                               const FeatureVector &feature_vector) {
  if (kd_tree_index_ == nullptr) {
    auto shape_features = features_;
    const int num_dims = features_[0].size();
    for (size_t ii = 0; ii < features_.size(); ++ii) {
      for (size_t jj = num_dims - 3; jj < num_dims; ++jj) {
        shape_features[ii][jj] = 0;
      }
    }
    BuildKDTreeIndex(shape_features, kd_tree_index_);
  }

  flann::Matrix<float> query_matrix;
  query_matrix = VectorToFlann<float>(feature_vector);
  std::vector<std::vector<int>> k_indices(1);
  k_indices[0].resize(num_neighbors);
  std::vector<std::vector<float>> k_distances(1);
  k_distances[0].resize(num_neighbors);
  kd_tree_index_->knnSearch(query_matrix, k_indices, k_distances, num_neighbors,
                            flann::SearchParams(-1));

  vector<cv::Point> matches(num_neighbors);

  for (size_t ii = 0; ii < k_indices[0].size(); ++ii) {
    matches[ii] = IndexToPoint(k_indices[0][ii], image_.rows, image_.cols);
  }
}

std::vector<PointValuePair> Image::GetMaximaAndHeatmap(
  const FeatureVector &feature_vector, cv::Mat &heatmap,
  cv::Mat &distance_map,
  double sigma, double alpha) {

  GetHeatmap(feature_vector, heatmap, distance_map, sigma, alpha);
  const auto maxima = GetLocalMaxima(distance_map);
  return maxima;
}

std::vector<PointValuePair> Image::GetMaxima(
  const FeatureVector &feature_vector) {
  cv::Mat heatmap, distance_map;
  return GetMaximaAndHeatmap(feature_vector, heatmap, distance_map);
}

void Image::GetHeatmap(const FeatureVector &feature_vector, cv::Mat &heatmap,
                       cv::Mat &raw_distance_map, std::vector<int> dim_idxs,
                       double sigma, double alpha) const {
  cv::Mat distance_map;
  distance_map.create(image_.rows, image_.cols, CV_64FC1);

  // Generate distance_map.
  for (int row = 0; row < distance_map.rows; ++row) {
    auto row_ptr = distance_map.ptr<double>(row);

    for (int col = 0; col < distance_map.cols; ++col) {
      int pixel_idx = col * distance_map.rows + row;
      FeatureVector query_feature(dim_idxs.size());
      FeatureVector image_feature(dim_idxs.size());
      const FeatureVector& full_image_feature = FeatureAt(cv::Point(col, row));
      for (size_t ii = 0; ii < dim_idxs.size(); ++ii) {
        image_feature[ii] = full_image_feature[dim_idxs[ii]];
        query_feature[ii] = feature_vector[dim_idxs[ii]];
      }

      const float dist = EuclideanDist(image_feature,
                                       query_feature);
      row_ptr[col] = dist;
    }
  }

  // cv::Mat orig_heat;
  // distance_map.copyTo(orig_heat);
  cv::Mat normalized, img_color;
  cv::exp(-sigma * distance_map, distance_map);
  cv::normalize(distance_map, normalized, 0, 255, cv::NORM_MINMAX);
  normalized.convertTo(normalized, CV_8UC1);
  cv::applyColorMap(normalized, heatmap, cv::COLORMAP_JET);
  heatmap = alpha * heatmap + (1 - alpha) * image_;
  // printf("Val %f\n", orig_heat.at<double>(maxima[ii].first.y, maxima[ii].first.x));
}

void Image::GetHeatmap(const FeatureVector &feature_vector, cv::Mat &heatmap,
                       cv::Mat &distance_map,
                       double sigma, double alpha) const {
  distance_map.create(image_.rows, image_.cols, CV_64FC1);

  // Generate distance_map.
  for (int row = 0; row < distance_map.rows; ++row) {
    auto row_ptr = distance_map.ptr<double>(row);

    for (int col = 0; col < distance_map.cols; ++col) {
      int pixel_idx = col * distance_map.rows + row;
      const float dist = EuclideanDist(FeatureAt(cv::Point(col, row)),
                                       feature_vector);
      row_ptr[col] = dist;
    }
  }

  // cv::Mat orig_heat;
  // distance_map.copyTo(orig_heat);
  cv::Mat normalized, img_color;
  cv::exp(-sigma * distance_map, distance_map);
  cv::normalize(distance_map, normalized, 0, 255, cv::NORM_MINMAX);
  normalized.convertTo(normalized, CV_8UC1);
  cv::applyColorMap(normalized, heatmap, cv::COLORMAP_JET);
  heatmap = alpha * heatmap + (1 - alpha) * image_;
  // printf("Val %f\n", orig_heat.at<double>(maxima[ii].first.y, maxima[ii].first.x));
}
} // namespace
