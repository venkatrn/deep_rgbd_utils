#pragma once

#include <opencv2/core/core.hpp>
#include <string>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/default_device.h"

// TODO: including tensorflow headers here messes up OpenGL context
// creation, weird. Investigate. Hence forward declaring.

// namespace tensorflow {
//   class Session;
//   class GraphDef;
// }


// #include <algorithm>
// #include <iosfwd>
// #include <memory>
// #include <utility>
// #include <vector>

// Look at pixel_vertex_mapper.cpp for example usage of this class.
namespace dru {
class VertexNet {
 public:
  VertexNet(const std::string &proto_file);
  ~VertexNet();

  // Get the object probabilities and vertex predictions for each pixel.
  bool Run(const cv::Mat &rgb_img, cv::Mat &obj_probs,
           cv::Mat &vertex_predictions);
  static void GetLabelImage(const cv::Mat &obj_probs, cv::Mat &labels);
  static bool SlicePrediction(const cv::Mat &obj_probs,
                              const cv::Mat &vertex_preds, int object_id, cv::Mat &sliced_probs,
                              cv::Mat &sliced_verts);
  static void ColorizeProbabilityMap(const cv::Mat& probability_map, cv::Mat& colored_map);

 private:
  int num_channels_;
  tensorflow::GraphDef graph_def_;
  tensorflow::Session *session_;
};
} // namespace
