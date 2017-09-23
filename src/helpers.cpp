#include <deep_rgbd_utils/helpers.h>

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>

using namespace std;

namespace {
  const int kNumClasses = 22;
}

namespace dru {


pcl::PointXYZ CamToWorld(int u, int v, float depth) {
  pcl::PointXYZ point;
  point.x =  depth * (u - kPrincipalPointX) / kFocalLengthColorX;
  point.y =  depth * (v - kPrincipalPointY) / kFocalLengthColorY;
  point.z = depth;
  return point;
}

// Project world (x,y,z) point to camera (x,y) point. For OpenCV,
// note camera (x,y) is equivalent to (col,row)
void WorldToCam(float x, float y, float z, int& cam_x, int& cam_y) {
  cam_x = kFocalLengthColorX * (x/z) + kPrincipalPointX;
  cam_y = kFocalLengthColorY * (y/z) + kPrincipalPointY;
}

void WorldToCam(const pcl::PointXYZ& point, int& cam_x, int& cam_y) {
  WorldToCam(point.x, point.y, point.z, cam_x, cam_y);
}

cv::Point IndexToPoint(int i, int rows, int cols) {
  return cv::Point(i / rows, i & rows);
}

int PointToIndex(cv::Point point, int rows, int cols) {
  return point.x * rows + point.y;
}

Color GetColor(double v,double vmin,double vmax)
{
   Color c = {1.0,1.0,1.0}; // white
   double dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      c.r = 0;
      c.g = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      c.r = 0;
      c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      c.r = 4 * (v - vmin - 0.5 * dv) / dv;
      c.b = 0;
   } else {
      c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      c.b = 0;
   }
   return(c);
}

std::vector<std::pair<cv::Point, double>> GetLocalMaxima(const cv::Mat &image, bool sort_maxima) {
  // Ignore borders.
  typedef std::pair<cv::Point, double> MaximaPoint;
  std::vector<MaximaPoint> maxima;
  for (int row = 1; row < image.rows - 1; ++row) {
    for (int col = 1; col < image.cols - 1; ++col) {
        // TODO: check type at runtime
        const double value = image.at<double>(row, col);
        if (
            value > image.at<double>(row, col - 1) &&
            value > image.at<double>(row, col + 1) &&
            value > image.at<double>(row - 1, col) &&
            value > image.at<double>(row + 1, col) &&
            value > image.at<double>(row - 1, col - 1) &&
            value > image.at<double>(row - 1, col + 1) &&
            value > image.at<double>(row + 1, col - 1) &&
            value > image.at<double>(row + 1, col + 1)
           ) {
          maxima.push_back(std::make_pair(cv::Point(col, row), value));
        }
      }
  }
  if (sort_maxima) {
    std::sort(maxima.begin(), maxima.end(), [](const MaximaPoint &p1, const MaximaPoint &p2) {
        return p1.second > p2.second;
        });
  }
  return maxima;
}

bool BuildKDTreeIndex(const std::vector<std::vector<float>>& feature_vectors, 
                      std::unique_ptr<flann::Index<L2<float>>>& index) {
  flann::Matrix<float> feature_matrix;
  if (feature_vectors.empty()) {
    printf("No feature vectors to build index!!\n");
    return false;
  }
  printf("Building KNN Index: Num feature vectors : %zu, Num dimensions: %zu\n", feature_vectors.size(), feature_vectors[0].size());
  feature_matrix = VectorToFlann<float>(feature_vectors);
  index.reset(new flann::Index<L2<float>>(feature_matrix, flann::KDTreeSingleIndexParams(4, false)));
  index->buildIndex();
  return true;
}

void WrapVector2DToCVMat(std::vector<std::vector<double>>& array2d, cv::Mat& mat) {
  if (array2d.empty()) {
    return;
  }
  mat.create(array2d.size(), array2d[0].size(), CV_64F);
  for(int ii = 0; ii < array2d.size(); ++ii) {
        mat.row(ii) = cv::Mat(array2d[ii]).t();
  }
}

void Vector2DToCVMat(const std::vector<std::vector<double>>& array2d, cv::Mat& mat) {
  if (array2d.empty()) {
    return;
  }
  mat.create(array2d.size(), array2d[0].size(), CV_64F);
  for(int ii = 0; ii < array2d.size(); ++ii) {
        mat.row(ii) = cv::Mat(array2d[ii]).t();
  }
}

void GetLabelImage(const cv::Mat &obj_probs, cv::Mat &labels) {
  labels.create(kColorHeight, kColorWidth, CV_8UC1);

  #pragma omp parallel for
  for (int ii = 0; ii < kColorHeight; ++ii) {
    for (int jj = 0; jj < kColorWidth; ++jj) {
      typedef cv::Vec<float, kNumClasses> ProbVec;
      auto prob_vec = obj_probs.at<ProbVec>(ii, jj);
      std::vector<float> scores(prob_vec.val, prob_vec.val + kNumClasses);

      int argmax = 0;
      Argmax(scores, &argmax);
      labels.at<uchar>(ii, jj) = argmax;
    }
  }
}

bool SlicePrediction(const cv::Mat &obj_probs,
                                const cv::Mat &vertex_preds, int object_id, cv::Mat &sliced_probs,
                                cv::Mat &sliced_verts) {

  if (object_id < 0 || object_id >= obj_probs.channels()) {
    printf("Invalid object_id %d for prediction with num_channels: %d\n",
           object_id, obj_probs.channels());
    return false;
  }

  sliced_probs.create(kColorHeight, kColorWidth, CV_32FC1);
  sliced_verts.create(kColorHeight, kColorWidth, CV_32FC3);

  int from_to_probs[] = {object_id, 0};
  vector<cv::Mat> source_probs{obj_probs};
  vector<cv::Mat> dest_probs{sliced_probs};
  cv::mixChannels(source_probs, dest_probs, from_to_probs, 1);

  const int offset = object_id * 3;
  int from_to_verts[] = {offset, 0,  offset + 1, 1,  offset + 2, 2};
  vector<cv::Mat> source_verts{vertex_preds};
  vector<cv::Mat> dest_verts{sliced_verts};
  cv::mixChannels(source_verts, dest_verts, from_to_verts, 3);

  sliced_probs.convertTo(sliced_probs, CV_64FC1);
  return true;
}

void ColorizeProbabilityMap(const cv::Mat &probability_map,
                                       cv::Mat &colored_map) {
  cv::normalize(probability_map, colored_map, 0.0, 255.0, cv::NORM_MINMAX);
  colored_map.convertTo(colored_map, CV_8UC3);
  cv::applyColorMap(colored_map, colored_map, cv::COLORMAP_JET);
}
} // namespace dru
