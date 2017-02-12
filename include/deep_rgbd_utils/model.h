#pragma once

#include <deep_rgbd_utils/helpers.h>

#include <pcl/common/common.h>


namespace dru {
  
  class Model {
    public: 
      Model() {};
      Model(const std::string& means_file) {
        SetMeans(means_file);
      }
      void SetMeans(const std::string& means_file);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud() const {
        return cloud_;
      }

      std::vector<float> GetModelMeanFeature() const;

      std::vector<pcl::PointXYZ> GetNearestPoints(int num_neighbors,
                                          const FeatureVector &feature_vector) const;

    PointCloudPtr GetHeatmapCloud(const FeatureVector &feature_vector);

    private:
    std::unique_ptr<flann::Index<L2<float>>> kd_tree_index_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    // There is a one-to-one index correspondence between features_ and
    // cloud_;
    std::vector<FeatureVector> features_;
  };
}
