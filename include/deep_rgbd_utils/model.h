#pragma once

#include <deep_rgbd_utils/helpers.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/common.h>
#include <pcl/console/print.h>

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

namespace dru {
  
  class Model {
    public: 
      Model() {};
      Model(const std::string& means_file) {
        SetMeans(means_file);
      }
      void SetMeans(const std::string& means_file);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud() {
        return cloud_;
      }

      std::vector<float> GetModelMeanFeature();

      std::vector<pcl::PointXYZ> GetNearestPoints(int num_neighbors,
                                          const FeatureVector &feature_vector);

    PointCloudPtr GetHeatmapCloud(const FeatureVector &feature_vector);

    private:
    std::unique_ptr<flann::Index<L2<float>>> kd_tree_index_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    // There is a one-to-one index correspondence between features_ and
    // cloud_;
    std::vector<FeatureVector> features_;
  };
}
