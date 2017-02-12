#pragma once

#include <deep_rgbd_utils/image.h>
#include <deep_rgbd_utils/model.h>
#include <deep_rgbd_utils/feature_generator_net.h>

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

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
  static void DrawProjection(const Image &rgb_image, const Model &model,
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
 private:
  // Mapping from model name to means file.
  std::unordered_map<std::string, Model> model_means_map_;
  std::unique_ptr<FeatureGenerator> generator_;
  std::string debug_dir_ = "";
  std::string im_prefix_ = "";
  void GetTransforms(const Image &rgb_image,
                     const Image &depth_image,
                     const Model &model, std::vector<Eigen::Matrix4f> *transforms);
  bool using_depth_ = false;
};
} // namespace
