#pragma once

// #include <deep_rgbd_utils/vertex_net.h>
#include <opencv2/core/core.hpp>

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

#include <util/dart_io.h>
#include <mesh/assimp_mesh_reader.h>

#include <df/camera/camera.h>
#include <df/camera/linear.h>
#include <df/camera/poly3.h>
#include <df/camera/rig.h>
#include <df/prediction/glRender.h>
#include <df/prediction/glRenderTypes.h>
#include <model/host_only_model.h>

#include <pangolin/pangolin.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace dru {

class PoseEstimator {
 public:
  PoseEstimator();
  ~PoseEstimator() = default;
  std::vector<Eigen::Matrix4f> GetObjectPoseCandidates(const std::string
                                                       &rgb_file, const std::string &depth_file, const std::string &model_name,
                                                       int num_candidates);

  void SetVerbose(const std::string &debug_dir, const std::string &prefix = "") {
    debug_dir_ = debug_dir;
    im_prefix_ = prefix;
  }
  void SetImageNum(const std::string &image_num) {
    im_num_ = image_num;
  }
  void UseDepth(bool use_depth) {
    using_depth_ = use_depth;
  }
  void DrawProjection(const cv::Mat &rgb_image, const std::string &model_name,
                      const Eigen::Matrix4f &transform, const std::string &im_name,
                      const std::string &text_on_im = "");

  bool Verbose() const {
    return !debug_dir_.empty();
  }
  std::string Prefix() const {
    if (im_prefix_.empty()) {
      return (debug_dir_ + "/");
    }

    return (debug_dir_ + "/" + im_prefix_ + "_");
  }

  void GetProjectedDepthImage(const std::string &model_name,
                              const Eigen::Matrix4f &pose, cv::Mat &depth_image);
  std::vector<double> GetRANSACScores() {
    return ransac_scores_;
  }

 private:

  std::unique_ptr<df::Rig<double>> rig_;
  Sophus::SE3d T_cd_;
  std::vector<double> ransac_scores_;

  std::unique_ptr<df::GLRenderer<df::DepthRenderType>> vertRenderer;
  std::map<std::string, dart::HostOnlyModel> models_;
  // std::map<std::string, int> model_name_to_idx_;
  // std::map<int, std::string> model_idx_to_name_;
  std::map<std::string, pangolin::GlBuffer> modelVertexBuffers_;
  std::map<std::string, pangolin::GlBuffer> modelCanonicalVertexBuffers_;
  bool ReadModels(const std::vector<std::string> &model_names);

  // Mapping from model name to means file.
  // std::unordered_map<std::string, Model> model_means_map_;
  // std::unique_ptr<dru::VertexNet> vertex_net_;
  std::string debug_dir_ = "";
  std::string im_prefix_ = "";
  std::string im_num_ = "";
  void GetTopPoses(const cv::Mat &rgb_image, const cv::Mat &depth_image,
                   const std::string &model_name, int num_poses,
                   std::vector<Eigen::Matrix4f> *poses, int num_trials = 1000);
  bool using_depth_ = false;

  // void GetGMMMask(const cv::Mat &rgb_image, const std::string &object_name,
  //                 cv::Mat &heatmap, cv::Mat &distance_map);

  double EvaluatePose(const Sophus::SE3d &T_sm,
                      const cv::Mat &filled_depth_image_m, const cv::Mat &object_probability_mask,
                      const cv::Mat &object_binary_mask,
                      const std::string &model_name);
  double EvaluatePoseSDF(const Sophus::SE3d &T_sm,
                      const cv::Mat &filled_depth_image_m, const cv::Mat &object_probability_mask,
                      const cv::Mat &object_binary_mask,
                      const std::string &model_name);

  std::map<std::string, int> kObjectNameToIdx = {
    {"background", 0},
    {"002_master_chef_can", 1},
    {"003_cracker_box", 2},
    {"004_sugar_box", 3},
    {"005_tomato_soup_can", 4},
    {"006_mustard_bottle", 5},
    {"007_tuna_fish_can", 6},
    {"008_pudding_box", 7},
    {"009_gelatin_box", 8},
    {"010_potted_meat_can", 9},
    {"011_banana", 10},
    {"019_pitcher_base", 11},
    {"021_bleach_cleanser", 12},
    {"024_bowl", 13},
    {"025_mug", 14},
    {"035_power_drill", 15},
    {"036_wood_block", 16},
    {"037_scissors", 17},
    {"040_large_marker", 18},
    {"051_large_clamp", 19},
    {"052_extra_large_clamp", 20},
    {"061_foam_brick", 21}
  };
};
} // namespace
