#pragma once

#include <opencv2/core/core.hpp>

#include <perception_utils/perception_utils.h>

#include <flann/flann.h>

#include <vector>
#include <string>
#include <memory>

namespace dru {

using FeatureVector = std::vector<float>;
using PointValuePair = std::pair<cv::Point, double>;

// TODO: move to params struct.
// Old UW 
// constexpr double kFocalLengthColorX = 536.6984;
// constexpr double kFocalLengthColorY = 536.7606;
// constexpr double kPrincipalPointX = 319.5645;
// constexpr double kPrincipalPointY = 243.335;

// CMU Dataset
constexpr double kFocalLengthColorX = 1077.7606;
constexpr double kFocalLengthColorY = 1078.189;
constexpr double kPrincipalPointX = 323.7872;
constexpr double kPrincipalPointY = 279.6921;

// UW Dataset
// constexpr double kFocalLengthColorX = 1066.778;
// constexpr double kFocalLengthColorY = 1067.487;
// constexpr double kPrincipalPointX = 312.9869;
// constexpr double kPrincipalPointY = 241.3109;
// const static float K1 = 0.04112172;
// const static float K2 = -0.4798174;
// const static float K3 = 1.890084;


constexpr int kColorWidth = 640;
constexpr int kColorHeight = 480;
const cv::Mat kCameraIntrinsics =  (cv::Mat_<double>(3,3) << kFocalLengthColorX, 0, kPrincipalPointX, 
                                                0, kFocalLengthColorY, kPrincipalPointY, 
                                                0, 0, 1);

pcl::PointXYZ CamToWorld(int u, int v, float depth);

// Project world (x,y,z) point to camera (x,y) point. For OpenCV,
// note camera (x,y) is equivalent to (col,row)
void WorldToCam(float x, float y, float z, int& cam_x, int& cam_y);

void WorldToCam(const pcl::PointXYZ& point, int& cam_x, int& cam_y);

cv::Point IndexToPoint(int i, int rows, int cols);

int PointToIndex(cv::Point point, int rows, int cols);

typedef struct {
    double r,g,b;
} Color;


// http://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
/*
   Return a RGB colour value given a scalar v in the range [vmin,vmax]
   In this case each colour component ranges from 0 (no contribution) to
   1 (fully saturated), modifications for other ranges is trivial.
   The colour is clipped at the end of the scales if v is outside
   the range [vmin,vmax]
*/
// Apply jet coloring.
Color GetColor(double v,double vmin,double vmax);

// Return the local maxima points for an image (assumed to be CV_64FC1) and
// their corresponding values. Maxima are sorted in descending order by
// default.
std::vector<PointValuePair> GetLocalMaxima(const cv::Mat &image, bool sort=true);

// Construct a KDTree index for a set of features.
bool BuildKDTreeIndex(const std::vector<std::vector<float>>& feature_vectors, 
                      std::unique_ptr<flann::Index<L2<float>>>& index);

// This *does not* copy data, it simply wraps it. Therefore, changes to the returned cv::Mat
// will affect the vector2d array.
void WrapVector2DToCVMat(std::vector<std::vector<double>>& array2d, cv::Mat& mat);

// This copies data. Therefore, changes to the returned cv::Mat
// will affect the vector2d array.
void Vector2DToCVMat(const std::vector<std::vector<double>>& array2d, cv::Mat& mat);

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
