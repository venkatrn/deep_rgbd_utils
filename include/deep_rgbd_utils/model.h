#pragma once

#include <deep_rgbd_utils/helpers.h>

#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <mlpack/core.hpp>
#include <mlpack/methods/gmm/gmm.hpp>


namespace dru {

class Model {
 public:
  Model() {};
  Model(const std::string &means_file) : means_file_(means_file) {
    SetMeans(means_file);
  }
  void SetMeans(const std::string &means_file);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud() const {
    return cloud_;
  }
  pcl::PointCloud<pcl::Normal>::Ptr normals() const {
    return normals_;
  }

  std::vector<float> GetModelMeanFeature() const;
  double GetGMMLikelihood(const FeatureVector& feature_vector) const;

  std::vector<pcl::PointXYZ> GetNearestPoints(int num_neighbors,
                                              const FeatureVector &feature_vector,
                                              std::vector<int>* model_indices) const;

  void GetNearestPoints(int num_neighbors,
                                              const std::vector<FeatureVector> &feature_vectors,
                                              std::vector<std::vector<int>>* model_indices, std::vector<std::vector<float>>* distances) const;

  Eigen::Vector3f GetNormal(int point_idx) const;

  PointCloudPtr GetHeatmapCloud(const FeatureVector &feature_vector);

 private:
  std::unique_ptr<flann::Index<L2<float>>> kd_tree_index_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
  pcl::PointCloud<pcl::Normal>::Ptr normals_;
  // There is a one-to-one index correspondence between features_ and
  // cloud_;
  std::vector<FeatureVector> features_;
  void ComputeGMM();
  void ComputeNormals();

  arma::gmm_diag gmm_model_;
  mlpack::gmm::GMM mlpack_gmm_;
  std::string means_file_;
};
}
