#include <deep_rgbd_utils/image.h>
#include <deep_rgbd_utils/helpers.h>

#include <flann/flann.h>

using namespace std;

void balance_white(cv::Mat& mat) {
  if (mat.type() != CV_8UC3) {
    return;
  }
  double discard_ratio = 0.05;
  int hists[3][256];
  memset(hists, 0, 3*256*sizeof(int));

  for (int y = 0; y < mat.rows; ++y) {
    uchar* ptr = mat.ptr<uchar>(y);
    for (int x = 0; x < mat.cols; ++x) {
      for (int j = 0; j < 3; ++j) {
        hists[j][ptr[x * 3 + j]] += 1;
      }
    }
  }

  // cumulative hist
  int total = mat.cols*mat.rows;
  int vmin[3], vmax[3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 255; ++j) {
      hists[i][j + 1] += hists[i][j];
    }
    vmin[i] = 0;
    vmax[i] = 255;
    while (hists[i][vmin[i]] < discard_ratio * total)
      vmin[i] += 1;
    while (hists[i][vmax[i]] > (1 - discard_ratio) * total)
      vmax[i] -= 1;
    if (vmax[i] < 255 - 1)
      vmax[i] += 1;
  }


  for (int y = 0; y < mat.rows; ++y) {
    uchar* ptr = mat.ptr<uchar>(y);
    for (int x = 0; x < mat.cols; ++x) {
      for (int j = 0; j < 3; ++j) {
        int val = ptr[x * 3 + j];
        if (val < vmin[j])
          val = vmin[j];
        if (val > vmax[j])
          val = vmax[j];
        ptr[x * 3 + j] = static_cast<uchar>((val - vmin[j]) * 255.0 / (vmax[j] - vmin[j]));
      }
    }
  }
}

void auto_contrast(cv::Mat &bgr_image) {
  if (bgr_image.type() != CV_8UC3) {
    return;
  }
  cv::Mat lab_image;
    cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

   // convert back to RGB
   cv::Mat image_clahe;
   cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
   bgr_image = image_clahe;
}

void saturate(cv::Mat& mat, int delta) {
  if (mat.type() != CV_8UC3) {
    return;
  }
  cv::Mat hsv;
  cv::cvtColor(mat, hsv, CV_BGR2HSV);
  for (int y = 0; y < mat.rows; ++y) {
    uchar* ptr = hsv.ptr<uchar>(y);
    for (int x = 0; x < mat.cols; ++x) {
        int val = ptr[x * 3 + 1];
        val += delta;
        if (val < 0)
          val = 0;
        if (val > 255)
          val = 255;
        ptr[x*3 + 1] = (uchar)val;
      }
  }
  cv::cvtColor(hsv, mat, CV_HSV2BGR);
}

namespace dru {

void Image::SetImage(const std::string &filename) {
  image_ = cv::imread(filename, -1);
  // saturate(image_, 50);
  if (image_.empty()) {
    printf("Unable to read image %s\n", filename.c_str());
  }

  data_type_ = image_.type();
}

void Image::SetImage(cv::Mat &image) {
  image_ = image;
  // saturate(image_, 50);
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
                       cv::Mat &distance_map, std::vector<int> dim_idxs,
                       double sigma, double alpha) const {
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
