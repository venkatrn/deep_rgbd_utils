#pragma once

#include <opencv2/core/core.hpp>

#include <perception_utils/perception_utils.h>

#include <flann/flann.h>

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/common.h>
#include <pcl/console/print.h>

#include <vector>
#include <string>
#include <memory>

namespace dru {

// http://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
/*
   Return a RGB colour value given a scalar v in the range [vmin,vmax]
   In this case each colour component ranges from 0 (no contribution) to
   1 (fully saturated), modifications for other ranges is trivial.
   The colour is clipped at the end of the scales if v is outside
   the range [vmin,vmax]
*/

typedef struct {
    double r,g,b;
} Color;

// Apply jet coloring.
Color GetColor(double v,double vmin,double vmax);

// Read a model means binary file and unpack to points, features for each
// point, and number of observations for each point. 
bool ReadModelFeatures(const std::string& file, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
    std::vector<std::vector<float>>* feature_vectors, std::vector<int>* num_observations);

// Return the local maxima points for an image (assumed to be CV_64FC1) and
// their corresponding values. Maxima are sorted in descending order by
// default.
std::vector<std::pair<cv::Point, double>> GetLocalMaxima(const cv::Mat &image, bool sort=true);

// Construct a KDTree index for a set of features.
bool BuildKDTreeIndex(const std::vector<std::vector<float>>& feature_vectors, 
                      std::unique_ptr<flann::Index<L2<float>>>& index);

// This *does not* copy data, it simply wraps it. Therefore, changes to the returned cv::Mat
// will affect the vector2d array.
void WrapVector2DToCVMat(std::vector<std::vector<double>>& array2d, cv::Mat& mat) {
  if (array2d.empty()) {
    return;
  }
  mat.create(array2d.size(), array2d[0].size(), CV_64F);
  for(int ii = 0; ii < array2d.size(); ++ii) {
        mat.row(ii) = cv::Mat(array2d[ii]).t();
  }
}

// This copies data. Therefore, changes to the returned cv::Mat
// will affect the vector2d array.
void Vector2DToCVMat(const std::vector<std::vector<double>>& array2d, cv::Mat& mat) {
  if (array2d.empty()) {
    return;
  }
  mat.create(array2d.size(), array2d[0].size(), CV_64F);
  for(int ii = 0; ii < array2d.size(); ++ii) {
        mat.row(ii) = cv::Mat(array2d[ii]).t();
  }
}

template <class T>
void PCDToCVMat(const T cloud, cv::Mat &mat) {
  if (cloud->points.empty()) {
    return;
  }
  mat.create(cloud->points.size(), 3, CV_64F);
  for (size_t ii = 0; ii < cloud->points.size(); ++ii) {
    const auto& point = cloud->points[ii];
    mat.at<double>(ii, 0) = point.x;
    mat.at<double>(ii, 1) = point.y;
    mat.at<double>(ii, 2) = point.z;
  }
}

template <typename ValueType>
flann::Matrix<ValueType> VectorToFlann(
    const std::vector<std::vector<ValueType>>& v
)
{
    size_t rows = v.size();
    size_t cols = v[0].size();
    size_t size = rows*cols;
    flann::Matrix<ValueType>m(new ValueType[size], rows, cols);
 
    for(size_t n = 0; n < size; ++n){
        *(m.ptr()+n) = v[n/cols][n%cols];
    }
 
    return m;
}

template <typename ValueType>
flann::Matrix<ValueType> VectorToFlann(
    const std::vector<ValueType>& v
)
{
    std::vector<std::vector<ValueType>> vectors;
    vectors.push_back(v);
    return VectorToFlann<ValueType>(vectors);
}


template<typename T>
T EuclideanDist2(const std::vector<T>&v1, const std::vector<T>& v2) {
  T sq_sum = 0;
  for (size_t ii = 0; ii < v1.size(); ++ii) {
    const T diff = (v1[ii] - v2[ii]);
    sq_sum += diff * diff;
  }
  return sq_sum;
}

template<typename T>
T EuclideanDist(const std::vector<T>&v1, const std::vector<T>& v2) {
  return sqrt(EuclideanDist2<T>(v1, v2));
}

} // namespace dru
