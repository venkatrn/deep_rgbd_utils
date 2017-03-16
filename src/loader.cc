#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/default_device.h"
// #include "tensorflow/cc/ops/standard_ops.h"

// #include "tensorflow/cc/ops/standard_ops.h"
//
#include <vector>

template<typename T>
bool Argmax(const std::vector<T>& v, int* argmax, T* max_value = nullptr) {
  auto max_val_it = std::max_element(v.begin(), v.end());
  if (max_val_it == v.end()) {
    return false;
  }
  *argmax = std::distance(v.begin(),max_val_it);
  if (max_value != nullptr) {
    *max_value = *max_val_it;
  }
}

template<typename T>
bool Argmin(const std::vector<T>& v, int* argmin, T* min_value = nullptr) {
  auto min_val_it = std::min_element(v.begin(), v.end());
  if (min_val_it == v.end()) {
    return false;
  }
  *argmin = std::distance(v.begin(), min_val_it);
  if (min_value != nullptr) {
    *min_value = *min_val_it;
  }
}


using namespace tensorflow;

int main(int argc, char* argv[]) {
  // Initialize a tensorflow session
  Session* session;
  std::cout << "xxxxx";
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  std::cout << "yyyyy";

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  std::cout << "aaaa";

  SessionOptions opts;
    graph::SetDefaultDevice("/gpu:0", &graph_def);
  opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
  opts.config.mutable_gpu_options()->set_allow_growth(true);

  // status = ReadBinaryProto(Env::Default(), "models/graph.pb", &graph_def);
  std::cout << std::flush;
  status = ReadBinaryProto(Env::Default(), "/home/venkatrn/research/dense_features/tensorflow/lov_frozen_graph.pb", &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  std::cout << "bbbb";
  std::cout << std::flush;
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  std::cout << "ccccc";
  std::cout << std::flush;
  
  for (size_t ii = 0; ii < 100; ++ii) {
  int height = 480, width = 640;
  cv::Mat image = cv::imread("/home/venkatrn/indigo_workspace/src/deep_rgbd_utils/dataset_uw_test/0053/rgb/7.png");
  image.convertTo(image, CV_32FC3);
  cv::Vec3f mean(102.9801, 115.9465, 122.7717);
  // cv::subtract(image, mean, image);

  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      image.at<cv::Vec3f>(ii, jj) -= mean;
    }
  }

  // Setup inputs and outputs:

  // Our graph doesn't require any inputs, since it specifies default values,
  // but we'll change an input to demonstrate.
  Tensor a(DT_FLOAT, TensorShape());
  a.scalar<float>()() = 3.0;

  Tensor b(DT_FLOAT, TensorShape());
  b.scalar<float>()() = 2.0;

  // std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
  //   { "a", a },
  //   { "b", b },
  // };
  // tensorflow::ops::Input::Initializer im({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}}});
  // Eigen::MatrixXf ones = Eigen::MatrixXf::Ones(1,25);
  tensorflow::Tensor im(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,height,width,3}));
  // std::iota(im.flat<float>().data(), im.flat<float>().data() + 25, 0);
  std::copy_n(reinterpret_cast<float*>(image.data), width*height*3,
      im.flat<float>().data());

  // im.matrix() = ones;
// a.vec<float>()(0) = 1.0f;
// a.vec<float>()(1) = 4.0f;
// a.vec<float>()(2) = 2.0f;
  std::vector<std::pair<string, Tensor>> inputs = {std::make_pair("fifo_queue_Dequeue", im)};

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  // Run the session, evaluating our "c" operation from the graph
  // status = session->Run(inputs, {"Div"}, {}, &outputs);
  std::cout << "running";
  std::cout << std::flush;
    status = session->Run(inputs, {"Div","vertex_pred/BiasAdd"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  std::cout << "ran";
  std::cout << std::flush;

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  auto output_c = outputs[0].tensor<float, 4>();

  cv::Mat outMat = cv::Mat::zeros(height, width, CV_8UC1);
  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      std::vector<float> scores(22);
      for (int kk = 0; kk < 22; ++kk) {
        scores[kk] = output_c(0, ii, jj, kk);
      }
      int argmax = 0;
      Argmax(scores, &argmax);
      outMat.at<uchar>(ii, jj) = argmax;
    }
  }
  outMat = outMat * (255 / 22);
  cv::applyColorMap(outMat, outMat, cv::COLORMAP_JET);
  cv::imwrite("labels.png", outMat);

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  // std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
  // std::cout << output_c.DebugString() << "\n"; // 30
  std::cout << outputs[1].DebugString() << "\n"; // 30
  }

  // Free any resources used by the session
  session->Close();
  return 0;
}
