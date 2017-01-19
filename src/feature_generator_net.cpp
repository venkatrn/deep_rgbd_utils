#include <deep_rgbd_utils/feature_generator_net.h>
#define CPU_ONLY 0

FeatureGenerator::FeatureGenerator(const string& caffe_model_file,
                       const string& trained_file) {
// #ifdef CPU_ONLY
  // Caffe::set_mode(Caffe::CPU);
// #else
//   Caffe::set_mode(Caffe::GPU);
// #endif

  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);

  // Caffe::set_mode(Caffe::CPU);

  /* Load the network. */
  net_.reset(new Net<float>(caffe_model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  printf("Finished copy\n");

  // CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  printf("Input layer\n");
  num_channels_ = input_layer->channels();
  printf("Num channels: %d\n", num_channels_);
  CHECK(num_channels_ == 3 || num_channels_ == 4)
    << "Input layer should have 3 or 4 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  printf("Net input dimensions: %d %d\n", input_geometry_.height, input_geometry_.width);


  // Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();


  // Blob<float>* output_layer = net_->output_blobs()[0];
  // CHECK_EQ(labels_.size(), output_layer->channels())
  //   << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

// Get the feature vector for a pixel.
void CollapseBlobAtPixel(const Blob<float>* blob, const cv::Point& point, std::vector<float>* output) {
  CHECK(point.x < blob->width() && point.y < blob->height())
    << "Point (" << point.x << ", " << point.y << ") " << "is out of blob (width, height) bounds "
    << " (" << blob->width() << ", " << blob->height() << ")";
  const int num_channels = blob->channels();
  if (output == nullptr || output->size() != num_channels) {
    *output = std::vector<float>(num_channels, 0.0);
  }
  for (int ii = 0; ii < num_channels; ++ii) {
  // for (int ii = num_channels - 3; ii < num_channels; ++ii) {
  // for (int ii = 0; ii < num_channels - 3; ++ii) {
    // OpenCV Point(x,y) -> (Width,Height)
    const int data_offset = blob->offset(0, ii, point.y, point.x);
    output->at(ii) = *(blob->cpu_data() + data_offset);
  }
}

std::vector<float> FeatureGenerator::GetFeature(const cv::Mat& img, const cv::Point& point) {
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  // Force copy from cpu->gpu
  Blob<float>* input_layer = net_->input_blobs()[0];
  auto data = input_layer->gpu_data();

  // Run the net.
  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  // Enforce copy from gpu->cpu
  data = output_layer->cpu_data();

  std::vector<float> feature_vector(output_layer->channels(), 0.0);
  
  CollapseBlobAtPixel(output_layer, point, &feature_vector);
}

std::vector<std::vector<float>> FeatureGenerator::GetFeatures(const cv::Mat& img) {
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  Preprocess(img, &input_channels);

  // Force copy from cpu->gpu
  Blob<float>* input_layer = net_->input_blobs()[0];
  auto data = input_layer->gpu_data();

  // Run the net.
  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  // Enforce copy from gpu->cpu
  data = output_layer->cpu_data();

  printf("RGB Input for CNN\n");
  printf("Image rows, cols: %d %d\n", img.rows, img.cols);
  
  std::vector<std::vector<float>> features(img.rows * img.cols, std::vector<float>(output_layer->channels()));
  #pragma omp parallel for
  for (size_t ii = 0; ii < features.size(); ++ii) {
    int col = ii / img.rows;
    int row = ii % img.rows;
    cv::Point point(col, row);
    CollapseBlobAtPixel(output_layer, point, &features[ii]);
  }
  
  return features;
}

std::vector<std::vector<float>> FeatureGenerator::GetFeatures(const cv::Mat& rgb_img, const cv::Mat& depth_img) {
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  Preprocess(rgb_img, depth_img, &input_channels);

  // Force copy from cpu->gpu
  Blob<float>* input_layer = net_->input_blobs()[0];
  auto data = input_layer->gpu_data();

  // Run the net.
  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  // Enforce copy from gpu->cpu
  data = output_layer->cpu_data();

  printf("RGB-D Input for CNN\n");
  printf("Image rows, cols: %d %d\n", rgb_img.rows, rgb_img.cols);
  
  std::vector<std::vector<float>> features(rgb_img.rows * rgb_img.cols, std::vector<float>(output_layer->channels()));
  #pragma omp parallel for
  for (size_t ii = 0; ii < features.size(); ++ii) {
    int col = ii / rgb_img.rows;
    int row = ii % rgb_img.rows;
    cv::Point point(col, row);
    CollapseBlobAtPixel(output_layer, point, &features[ii]);
  }
  
  return features;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void FeatureGenerator::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void FeatureGenerator::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */

  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img.clone();

  cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);

  // Transpose image if appropriate.
  // const auto sample_shape = sample.size();
  // if (sample_shape.width == input_geometry_.height && sample_shape.height ==
  //     input_geometry_.width) {
  //   printf("Transposing input image\n");
  //   cv::transpose(sample, sample);
  // }
  // cv::transpose(sample, sample);

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  sample_float.copyTo(sample_normalized);
  // cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  // Normalize to 0-1.
  for (size_t ii = 0; ii < input_channels->size(); ++ii) {
    input_channels->at(ii) = input_channels->at(ii) / 255.0;
  }

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

void FeatureGenerator::Preprocess(const cv::Mat& rgb_img,
                                  const cv::Mat& depth_img,
                                  std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */

  cv::Mat sample;
  if (rgb_img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(rgb_img, sample, cv::COLOR_BGR2GRAY);
  else if (rgb_img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(rgb_img, sample, cv::COLOR_BGRA2GRAY);
  else if (rgb_img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(rgb_img, sample, cv::COLOR_BGRA2BGR);
  else if (rgb_img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(rgb_img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = rgb_img.clone();

  cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);

  // Transpose image if appropriate.
  // const auto sample_shape = sample.size();
  // if (sample_shape.width == input_geometry_.height && sample_shape.height ==
  //     input_geometry_.width) {
  //   printf("Transposing input image\n");
  //   cv::transpose(sample, sample);
  // }
  // cv::transpose(sample, sample);

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  sample_float.copyTo(sample_normalized);
  // cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  std::vector<cv::Mat> rgb_channels(input_channels->begin(), input_channels->end() - 1);
  // std::vector<cv::Mat> rgb_channels();
  cv::split(sample_normalized, rgb_channels);
  // input_channels->at(0) = rgb_channels[0].clone();
  // input_channels->at(1) = rgb_channels[1].clone();
  // input_channels->at(2) = rgb_channels[2].clone();

  // Normalize to 0-1.
  for (size_t ii = 0; ii < 3; ++ii) {
    input_channels->at(ii) = input_channels->at(ii) / 255.0;
  }
  // Convert mm to m.
  // input_channels->at(3) = (depth_img.clone() / 1000.0);
  cv::Mat rescaled_depth = depth_img.clone();
  rescaled_depth.convertTo(rescaled_depth, CV_64FC1);
  rescaled_depth = rescaled_depth / 1000.0;
  rescaled_depth.copyTo(input_channels->at(3));

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
