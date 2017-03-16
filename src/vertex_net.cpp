#include <deep_rgbd_utils/vertex_net.h>
#include <deep_rgbd_utils/helpers.h>

#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



using namespace std;
using namespace tensorflow;

namespace  {
const cv::Vec3f kRGBMean(102.9801, 115.9465, 122.7717);
const int kNumClasses = 22;
}

namespace dru {
VertexNet::VertexNet(const string &proto_file) {

  SessionOptions opts;
  graph::SetDefaultDevice("/gpu:0", &graph_def_);
  opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
  opts.config.mutable_gpu_options()->set_allow_growth(true);
  Status status = NewSession(opts, &session_);

  // Status status = NewSession(SessionOptions(), &session_);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return;
  }

  cout << "Created tensorflow session\n";


  status = ReadBinaryProto(Env::Default(), proto_file, &graph_def_);

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return;
  }

  cout << "Read proto file\n";

  status = session_->Create(graph_def_);

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return;
  }

  cout << "Created graph\n";
}

VertexNet::~VertexNet() {
  session_->Close();
}

// Get the object probabilities and vertex predictions for each pixel.
bool VertexNet::Run(const cv::Mat &rgb_img, cv::Mat &obj_probs,
                    cv::Mat &vert_preds) {
  // rgb_image.convertTo(image, CV_32FC3);
  // TODO: don't assume they are same as input size
  const int num_data = kColorWidth * kColorHeight * 3;
  Tensor im(tensorflow::DT_FLOAT, TensorShape({1, kColorHeight, kColorWidth, 3}));

  uchar *data_it = rgb_img.data;
  float *tensor_data_it = im.flat<float>().data();
  int idx = 0;
  // TODO: make less cryptic?
  std::transform(data_it, data_it + num_data,
  tensor_data_it, [&idx](const uchar & pixel) {
    float pixel_f = static_cast<float>(pixel);
    pixel_f -= kRGBMean[idx % 3];
    ++idx;
    return pixel_f;
  });

  vector<pair<string, Tensor>> inputs = {std::make_pair("fifo_queue_Dequeue", im)};

  // The session will initialize the outputs
  vector<Tensor> outputs;

  Status status = session_->Run(inputs, {"Div", "vertex_pred/BiasAdd"}, {},
                                &outputs);

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return false;
  }

  cout << "Ran network\n";

  const auto &output_probs = outputs[0].tensor<float, 4>();
  const auto &output_verts = outputs[1].tensor<float, 4>();

  obj_probs.create(kColorHeight, kColorWidth, CV_32FC(kNumClasses));
  std::copy_n(outputs[0].flat<float>().data(),
              kColorWidth * kColorHeight * kNumClasses,
              reinterpret_cast<float *>(obj_probs.data));

  vert_preds.create(kColorHeight, kColorWidth, CV_32FC(kNumClasses * 3));
  std::copy_n(outputs[1].flat<float>().data(),
              kColorWidth * kColorHeight * kNumClasses * 3,
              reinterpret_cast<float *>(vert_preds.data));

  return true;
}

void VertexNet::GetLabelImage(const cv::Mat &obj_probs, cv::Mat &labels) {
  labels.create(kColorHeight, kColorWidth, CV_8UC1);

  for (int ii = 0; ii < kColorHeight; ++ii) {
    for (int jj = 0; jj < kColorWidth; ++jj) {
      typedef cv::Vec<float, kNumClasses> ProbVec;
      auto prob_vec = obj_probs.at<ProbVec>(ii, jj);
      std::vector<float> scores(prob_vec.val, prob_vec.val + kNumClasses);

      int argmax = 0;
      Argmax(scores, &argmax);
      labels.at<uchar>(ii, jj) = argmax;
    }
  }

  labels = labels * (255 / kNumClasses);
  cv::applyColorMap(labels, labels, cv::COLORMAP_JET);
}

bool VertexNet::SlicePrediction(const cv::Mat &obj_probs,
                                const cv::Mat &vertex_preds, int object_id, cv::Mat &sliced_probs,
                                cv::Mat &sliced_verts) {

  if (object_id < 0 || object_id >= obj_probs.channels()) {
    printf("Invalid object_id %d for prediction with num_channels: %d\n",
           object_id, obj_probs.channels());
    return false;
  }

  sliced_probs.create(kColorHeight, kColorWidth, CV_32FC1);
  sliced_verts.create(kColorHeight, kColorWidth, CV_32FC3);

  int from_to_probs[] = {object_id, 0};
  vector<cv::Mat> source_probs{obj_probs};
  vector<cv::Mat> dest_probs{sliced_probs};
  cv::mixChannels(source_probs, dest_probs, from_to_probs, 1);

  const int offset = object_id * 3;
  int from_to_verts[] = {offset, 0,  offset + 1, 1,  offset + 2, 2};
  vector<cv::Mat> source_verts{vertex_preds};
  vector<cv::Mat> dest_verts{sliced_verts};
  cv::mixChannels(source_verts, dest_verts, from_to_verts, 3);

  sliced_probs.convertTo(sliced_probs, CV_64FC1);
  return true;
}

void VertexNet::ColorizeProbabilityMap(const cv::Mat &probability_map,
                                       cv::Mat &colored_map) {
  cv::normalize(probability_map, colored_map, 0.0, 255.0, cv::NORM_MINMAX);
  colored_map.convertTo(colored_map, CV_8UC3);
  cv::applyColorMap(colored_map, colored_map, cv::COLORMAP_JET);
}
} // namespace
