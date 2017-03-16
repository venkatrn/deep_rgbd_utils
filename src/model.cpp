#include <deep_rgbd_utils/model.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/console/print.h>

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;

namespace {
// Read a model means binary file and unpack to points, features for each
// point, and number of observations for each point. 
bool ReadModelFeatures(const string &file,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                       std::vector<dru::FeatureVector> *feature_vectors,
                       std::vector<int> *num_observations) {

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
      model_file.read(reinterpret_cast<char *>(feature_vectors->at(ii).data()),
                      feature_dim * sizeof(float));
      // for (int jj = 0; jj < feature_dim; ++jj) {
      //   printf("%f ", feature_vectors->at(ii)[jj]);
      // }
      // printf("\n");
      // TODO: setting last 3 dimensions to 0 since they are object-specific
      // means.
      // feature_vectors->at(ii)[feature_dim - 3] = 0.0;
      // feature_vectors->at(ii)[feature_dim - 2] = 0.0;
      // feature_vectors->at(ii)[feature_dim - 1] = 0.0;
    }

    model_file.close();
  } else {
    printf("Could not open model means file\n");
    return false;
  }

  return true;
}
} // namespace

namespace dru {
void Model::SetMeans(const std::string &means_file) {
  means_file_ = means_file;
  cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  vector<int> num_observations;
  if (!ReadModelFeatures(means_file, cloud_, &features_, &num_observations)) {
    printf("Could not read means file %s\n", means_file.c_str());
    return;
  }
  auto shape_features = features_;
  const int num_dims = features_[0].size();
  // TODO: clean up.
  for (size_t ii = 0; ii < features_.size(); ++ii) {
    for (size_t jj = num_dims - 3; jj < num_dims; ++jj) {
      shape_features[ii][jj] = 0;
    }
  }
  BuildKDTreeIndex(shape_features, kd_tree_index_);
  ComputeGMM();
  ComputeNormals();
}
std::vector<pcl::PointXYZ> Model::GetNearestPoints(int num_neighbors,
                                                   const FeatureVector &feature_vector,
                                                   std::vector<int>* model_indices) const {
  model_indices->clear();
  flann::Matrix<float> query_matrix;
  query_matrix = VectorToFlann<float>(feature_vector);
  std::vector<std::vector<int>> k_indices(1);
  k_indices[0].resize(num_neighbors);
  std::vector<std::vector<float>> k_distances(1);
  k_distances[0].resize(num_neighbors);
  kd_tree_index_->knnSearch(query_matrix, k_indices, k_distances,
                              num_neighbors, flann::SearchParams(-1));
  vector<pcl::PointXYZ> matches(num_neighbors);
  model_indices->resize(num_neighbors);

  for (size_t ii = 0; ii < num_neighbors; ++ii) {
    matches[ii] = cloud_->points[k_indices[0][ii]];
    model_indices->at(ii) = k_indices[0][ii];
  }
  return matches;
}

void Model::GetNearestPoints(int num_neighbors,
                                                   const std::vector<FeatureVector> &feature_vectors,
                                                   std::vector<std::vector<int>>* model_indices,
                                                   std::vector<std::vector<float>>* distances) const {
  model_indices->clear();
  distances->clear();
  flann::Matrix<float> query_matrix;
  query_matrix = VectorToFlann<float>(feature_vectors);
  std::vector<std::vector<int>> k_indices(feature_vectors.size(), std::vector<int>(num_neighbors, 0));
  std::vector<std::vector<float>> k_distances(feature_vectors.size(), std::vector<float>(num_neighbors, 0.0));
  kd_tree_index_->knnSearch(query_matrix, k_indices, k_distances,
                              num_neighbors, flann::SearchParams(-1));
  *model_indices = k_indices;
  *distances = k_distances;
}

PointCloudPtr Model::GetHeatmapCloud(const FeatureVector &feature_vector) {
  PointCloudPtr colored_cloud(new PointCloud);
  copyPointCloud(*cloud_, *colored_cloud);

  vector<double> dists(cloud_->points.size(), 0.0);
  double normalizer = 0.0;
  double min_dist = std::numeric_limits<double>::max();
  double max_dist = std::numeric_limits<double>::lowest();
  std::vector<float> zero_vec(feature_vector.size(), 0.0);

  for (size_t ii = 0; ii < cloud_->points.size(); ++ii) {
    dists[ii] = static_cast<double>(EuclideanDist<float>(feature_vector,
                                                         features_[ii]));
    normalizer += dists[ii];
    min_dist = std::min(min_dist, dists[ii]);
    max_dist = std::max(max_dist, dists[ii]);
  }

  if (normalizer < 1e-5) {
    printf("Normalizer is 0!\n");
    return colored_cloud;
  }

  for (size_t ii = 0; ii < cloud_->points.size(); ++ii) {
    Color color = GetColor(max_dist - dists[ii], 0.0, max_dist - min_dist);
    colored_cloud->points[ii].r = 255 * color.r;
    colored_cloud->points[ii].g = 255 * color.g;
    colored_cloud->points[ii].b = 255 * color.b;
  }
  return colored_cloud;
}

std::vector<float> Model::GetModelMeanFeature() const {
  if (features_.empty() || features_[0].empty()) {
    printf("No features for this model\n");
    return std::vector<float>();
  }
  const size_t num_dims = features_[0].size();
  std::vector<float> mean_feature(num_dims);
  for (size_t ii = 0; ii < features_.size(); ++ii) {
    for (size_t jj = 0; jj < num_dims; ++jj) {
      mean_feature[jj] += features_[ii][jj];
    }
  }
  for (size_t ii = 0; ii < num_dims; ++ii) {
    mean_feature[ii] /= features_.size();
  }
  return mean_feature;
}

void Model::ComputeNormals() {
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloud_);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod(tree);
  ne.setRadiusSearch (0.03);
  normals_.reset(new pcl::PointCloud<pcl::Normal>);
  ne.compute(*normals_);
}

Eigen::Vector3f Model::GetNormal(int point_idx) const {
  const auto &pcl_normal = normals_->points[point_idx];
  return Eigen::Vector3f(pcl_normal.normal_x, pcl_normal.normal_y, pcl_normal.normal_z);
}

double Model::GetGMMLikelihood(const FeatureVector& feature_vector) const {
  using namespace arma;
  vector<float> x = {feature_vector[29], feature_vector[30], feature_vector[31]};
  vec a = conv_to<vec>::from(x);
  // return static_cast<double>(gmm_model_.log_p(a));
  return mlpack_gmm_.Probability(a);
}

void Model::ComputeGMM() {
  using namespace arma;
  printf("Learning GMM for %s\n", means_file_.c_str());
  if (features_.empty()) {
    printf("No features for GMM\n");
    return;
  }
  const int num_points = features_.size();
  const int dims = 3;
  mat data(dims, num_points, fill::zeros);
  for (int ii = 0; ii < num_points; ++ii) {
    vector<float> x = {features_[ii][29], features_[ii][30], features_[ii][31]};
    vec a = conv_to<vec>::from(x);
    data.col(ii) = a;
  }

  // gmm_model_.learn(data, 3, maha_dist, random_subset, 10, 5, 1e-10, true);
  
  const int kNumGaussians = 3;
  mlpack_gmm_ = mlpack::gmm::GMM(kNumGaussians, 3);
  mlpack_gmm_.Train(data, 1);
}
}
