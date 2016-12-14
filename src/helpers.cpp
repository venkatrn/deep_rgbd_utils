#include <deep_rgbd_utils/helpers.h>

#include <algorithm>

using namespace std;

namespace dru {

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

bool ReadModelFeatures(const string& file, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
    std::vector<std::vector<float>>* feature_vectors, std::vector<int>* num_observations) {

  cloud->points.clear();

  ifstream model_file;
  model_file.open (file.c_str(), ios::in | ios::binary); 
  if (model_file.is_open()) {
    int32_t num_points = 0;
    model_file.read(reinterpret_cast<char *>(&num_points), sizeof(num_points));
    printf("Num points: %d\n", num_points);
    cloud->points.resize(num_points);
    // Read the points.
    for (int ii = 0; ii < num_points; ++ii) {
      pcl::PointXYZ point;
      model_file.read(reinterpret_cast<char *>(&point.x), sizeof(float));
      model_file.read(reinterpret_cast<char *>(&point.y), sizeof(float));
      model_file.read(reinterpret_cast<char *>(&point.z), sizeof(float));
      cloud->points[ii] = point;
      // printf("Vertex: %f %f %f\n", point.x, point.y, point.z);
    }
    cloud->width = 1;
    cloud->height = num_points;
    cloud->is_dense = false;

    // Read the number of observations per point.
    num_observations->clear();
    num_observations->resize(num_points, 0);
    int n_obs = 0;
    for (int ii = 0; ii < num_points; ++ii) {
      model_file.read(reinterpret_cast<char *>(&n_obs), sizeof(int));
      num_observations->at(ii) = static_cast<int>(n_obs);
    }

    int feature_dim = 0;
    model_file.read(reinterpret_cast<char *>(&feature_dim), sizeof(int));
    printf("Feature dimensionality: %d\n", feature_dim);

    // Read the features corresponding to each point.
    feature_vectors->clear();
    feature_vectors->resize(num_points, vector<float>(feature_dim, 0.0));
    // vector<float> feature_vector(feature_dim);
    for (int ii = 0; ii < num_points; ++ii) {
      // for (int jj = 0; jj < feature_dim; ++jj) {
      //   model_file.read(reinterpret_cast<char *>(&feature_vectors->at(ii)[jj]), sizeof(float));
      // }
      // model_file.read(reinterpret_cast<char *>(feature_vector.data()), sizeof(feature_vector));
      // feature_vectors->at(ii) = feature_vector;
      model_file.read(reinterpret_cast<char *>(feature_vectors->at(ii).data()), feature_dim * sizeof(float));
      // for (int jj = 0; jj < feature_dim; ++jj) {
      //   printf("%f ", feature_vectors->at(ii)[jj]);
      // }
      // printf("\n");
    }
    model_file.close();
  } else {
    printf("Could not open model means file\n");
    return false;
  }

  return true;
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
} // namespace dru
