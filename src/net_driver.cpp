#include <deep_rgbd_utils/vertex_net.h>

#include <opencv2/contrib/contrib.hpp>

using namespace std;
using namespace tensorflow;

int main(int argc, char** argv) {
  VertexNet vertex_net("/home/venkatrn/research/dense_features/tensorflow/lov_frozen_graph.pb");
  cv::Mat image1 = cv::imread("/home/venkatrn/indigo_workspace/src/deep_rgbd_utils/dataset_uw_test/0053/rgb/7.png");
  // cv::Mat image2 = cv::imread("/home/venkatrn/indigo_workspace/src/deep_rgbd_utils/dataset_uw_test/0053/rgb/6.png");
  // cv::Mat image3 = cv::imread("/home/venkatrn/indigo_workspace/src/deep_rgbd_utils/dataset_uw_test/0053/rgb/5.png");
  // cv::Mat image4 = cv::imread("/home/venkatrn/indigo_workspace/src/deep_rgbd_utils/dataset_uw_test/0053/rgb/4.png");
  cv::Mat obj_probs, vert_preds, obj_labels;


  vertex_net.Run(image1, obj_probs, vert_preds);
  vertex_net.GetLabelImage(obj_probs, obj_labels);

  cv::imwrite("labels.png", obj_labels);

  cv::Mat sliced_probs, sliced_verts;
  vertex_net.SlicePrediction(obj_probs, vert_preds, obj_labels.at<uchar>(240, 320), sliced_probs, sliced_verts);

  cv::log(sliced_probs, sliced_probs);
  cv::normalize(sliced_probs, sliced_probs, 0.0, 255.0, cv::NORM_MINMAX);
  sliced_probs.convertTo(sliced_probs, CV_8UC3);
  cv::normalize(sliced_verts, sliced_verts, 0.0, 255.0, cv::NORM_MINMAX);
  cv::applyColorMap(sliced_probs, sliced_probs, cv::COLORMAP_JET);
  cv::applyColorMap(sliced_verts, sliced_verts, cv::COLORMAP_JET);

  cv::imwrite("probs.png", sliced_probs);
  cv::imwrite("verts.png", sliced_verts);
}

