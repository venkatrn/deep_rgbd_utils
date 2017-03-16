#pragma once

#include <deep_rgbd_utils/image.h>
#include <deep_rgbd_utils/model.h>
#include <deep_rgbd_utils/feature_generator_net.h>

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

namespace dru {

class PoseEstimator {
 public:
  PoseEstimator();
  ~PoseEstimator() = default;
  std::vector<Eigen::Matrix4f> GetObjectPoseCandidates(const std::string
                                                       &rgb_file, const std::string &depth_file, const std::string &model_name,
                                                       int num_candidates);

  const Model& GetModel(const string& name) {
    // TODO: check existence.
    return model_means_map_[name];
  }
  void SetVerbose(const std::string& debug_dir, const std::string& prefix = "") {
    debug_dir_ = debug_dir;
    im_prefix_ = prefix;
  }
  void UseDepth(bool use_depth) {
    using_depth_ = use_depth;
  }
  void DrawProjection(const Image &rgb_image, const std::string& model_name, const Model &model,
                                   const Eigen::Matrix4f &transform, const string &im_name);

  bool Verbose() const {
    return !debug_dir_.empty();
  }
  std::string Prefix() const {
    if (im_prefix_.empty()) {
      return (debug_dir_ + "/");
    }
    return (debug_dir_ + "/" + im_prefix_ + "_"); 
  }

  void GetProjectedDepthImage(const std::string& model_name, const Eigen::Matrix4f& pose, cv::Mat& depth_image);

 private:

  std::unique_ptr<df::Rig<double>> rig_;
  Sophus::SE3d T_cd_;
  std::unique_ptr<df::GLRenderer<df::DepthRenderType>> vertRenderer;
  std::map<std::string, dart::HostOnlyModel> models_;
  // std::map<std::string, int> model_name_to_idx_;
  // std::map<int, std::string> model_idx_to_name_;
  std::map<std::string, pangolin::GlBuffer> modelVertexBuffers_;
  std::map<std::string, pangolin::GlBuffer> modelCanonicalVertexBuffers_;
  bool ReadModels(const std::vector<std::string>& model_names);

  // Mapping from model name to means file.
  std::unordered_map<std::string, Model> model_means_map_;
  std::unique_ptr<FeatureGenerator> generator_;
  std::string debug_dir_ = "";
  std::string im_prefix_ = "";
  void GetTransforms(const Image &rgb_image,
                     const Image &depth_image,
                     const Model &model, const std::string& model_name, std::vector<Eigen::Matrix4f> *transforms);
  void GetTopPoses(const Image &rgb_image, const Image &depth_image,
                 const Model &model, const string& model_name, int num_poses,
                 vector<Eigen::Matrix4f> *poses, int num_trials = 1000);
  bool using_depth_ = false;

  void GetGMMMask(const Image& rgb_image, const std::string& object_name, cv::Mat& heatmap, cv::Mat&distance_map);

  double EvaluatePose(const Sophus::SE3d &T_sm,
                    const cv::Mat &filled_depth_image_m, const cv::Mat& object_probability_mask, const std::string& model_name, const Model &model);
};
} // namespace
