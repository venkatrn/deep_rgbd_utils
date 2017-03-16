#include <deep_rgbd_utils/pose_estimator.h>
#include <deep_rgbd_utils/helpers.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/registration/transformation_estimation_svd.h>
// #include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <deep_rgbd_utils/correspondence_rejection_sample_consensus_multiple.h>
#include <l2s/image/depthFilling.h>
#include <dataset_manager.h>
#include <ycb_utils.h>

#include <iostream>
#include <fstream>
#include <cstdint>

#include <chrono>
#include <map>

using namespace std;
using namespace dru;
using namespace ycb;
using namespace flann;
using high_res_clock = std::chrono::high_resolution_clock;

namespace {
// TODO: receive from caller.
// std::vector<std::string> kYCBObjects = {"002_master_chef_can", "003_cracker_box",
//                                         "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "011_banana",  "019_pitcher_base", "021_bleach_cleanser", "024_bowl", "025_mug", "035_power_drill", "036_wood_block", "037_scissors", "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"
//                                        };

std::vector<std::string> kYCBObjects = {"024_bowl", "002_master_chef_can"};

// const string kCaffeWeights =
//   "/home/venkatrn/research/dense_features/denseCorrespondence-lov/snap_iter_19100.caffemodel";
// const string kCaffeProto =
//   "/home/venkatrn/research/dense_features/denseCorrespondence-lov/deploy_cpp.prototxt";
// const string kMeansFolder = "/home/venkatrn/research/dense_features/means/lov";

const string kCaffeWeights =
  "/home/venkatrn/research/dense_features/denseCorrespondence-lov-rgbd/snap_iter_35300.caffemodel";
const string kCaffeProto =
  "/home/venkatrn/research/dense_features/denseCorrespondence-lov-rgbd/deploy_cpp.prototxt";
const string kMeansFolder =
  "/home/venkatrn/research/dense_features/means/lov_rgbd";
const string kRigFile = "/home/venkatrn/research/ycb/asus-uw.json";

const uint16_t kMaxDepth = 2000;
const unsigned char kObjectMaskThresh = 10;
const double kBackgroundWeight = 0.2;
const double kBackgroundNoise = (1.0 / 20.0);


void GetGMMMask(const Image &rgb_image, const Model &model, cv::Mat &heatmap,
                cv::Mat &distance_map) {
  auto image = rgb_image.image();
  heatmap.create(image.rows, image.cols, CV_8UC3);
  distance_map.create(image.rows, image.cols, CV_64FC1);
  #pragma omp parallel for

  for (int row = 0; row < image.rows; ++row) {
    for (int col = 0; col < image.cols; ++col) {
      const auto &feature = rgb_image.FeatureAt(col, row);
      double log_ll = model.GetGMMLikelihood(feature);
      distance_map.at<double>(row, col) = log_ll;
    }
  }

  cv::Mat normalized, img_color;
  // cv::exp(-sigma * distance_map, distance_map);
  cv::normalize(distance_map, normalized, 0, 255, cv::NORM_MINMAX);
  normalized.convertTo(normalized, CV_8UC1);
  cv::applyColorMap(normalized, heatmap, cv::COLORMAP_JET);
  double alpha = 1.0;
  heatmap = alpha * heatmap + (1 - alpha) * image;
}

} // namespace

namespace dru {

PoseEstimator::PoseEstimator() :
  generator_(std::unique_ptr<FeatureGenerator>(new FeatureGenerator(kCaffeProto,
                                                                    kCaffeWeights))) {
  for (const string &object : kYCBObjects) {
    const string model_means_file = kMeansFolder + "/" + object + ".means";
    model_means_map_.insert(make_pair<string, Model>(object.c_str(), Model()));
    model_means_map_[object].SetMeans(model_means_file);
  }

  DatasetManager::ParseRigFile(kRigFile, rig_, T_cd_);
  const auto &colorCamera = rig_->camera(colorStreamIndex);
  const auto &depthCamera = rig_->camera(depthStreamIndex);

  pangolin::CreateWindowAndBind("ycb pose estimation - ", 1280, 960);
  dart::Model::initializeRenderer(new dart::AssimpMeshReader());

  vertRenderer.reset(new df::GLRenderer<df::DepthRenderType>
                     (depthCamera.width(), depthCamera.height()));
  vertRenderer->setCameraParams(colorCamera.params(), colorCamera.numParams());
  ReadModels(kYCBObjects);
}

void PoseEstimator::GetProjectedDepthImage(const std::string &model_name,
                                           const Eigen::Matrix4f &pose, cv::Mat &depth_image) {
  vector<vector<pangolin::GlBuffer *>> attributeBuffers(1);
  std::vector<pangolin::GlBuffer> modelIndexBuffers;

  const dart::HostOnlyModel &model = models_[model_name];
  const dart::Mesh &mesh = model.getMesh(model.getMeshNumber(0));

  modelIndexBuffers.push_back(pangolin::GlBuffer(
                                pangolin::GlElementArrayBuffer, mesh.nFaces * 3, GL_UNSIGNED_INT, 3,
                                GL_STATIC_DRAW));
  modelIndexBuffers.back().Upload(mesh.faces, mesh.nFaces * sizeof(int3));

  attributeBuffers[0].push_back(&modelVertexBuffers_[model_name]);
  // attributeBuffers[0].push_back(&modelCanonicalVertexBuffers_[model_name]);

  glClearColor(std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  glEnable(GL_DEPTH_TEST);
  // vector<Eigen::Matrix4f> object_poses = {pose};

  vertRenderer->setModelViewMatrix(pose);
  vertRenderer->render(attributeBuffers[0], modelIndexBuffers[0], GL_TRIANGLES);
  vertRenderer->texture(0).RenderToViewportFlipY();

  std::vector<float> depth_map(kColorWidth * kColorHeight);
  vertRenderer->texture(0).Download(depth_map.data(), GL_LUMINANCE, GL_FLOAT);

  cv::Mat depth_mat = cv::Mat(kColorHeight, kColorWidth,
                              CV_32FC1, depth_map.data());
  depth_mat.copyTo(depth_image);
}

vector<Eigen::Matrix4f> PoseEstimator::GetObjectPoseCandidates(
  const std::string &rgb_file, const std::string &depth_file,
  const std::string &model_name, int num_candidates) {

  Image rgb_img, depth_img;
  rgb_img.SetImage(rgb_file);
  depth_img.SetImage(depth_file);

  // if (Verbose()) {
  // }

  // Run the feature generator.
  vector<FeatureVector> features;

  if (using_depth_) {
    features = generator_->GetFeatures(rgb_img.image(), depth_img.image());

  } else {
    features = generator_->GetFeatures(rgb_img.image());
  }

  rgb_img.SetFeatures(features);



  auto model_it = model_means_map_.find(model_name);

  if (model_it == model_means_map_.end()) {
    cerr << "Model name " << model_name << " is unknown" << endl;
    return vector<Eigen::Matrix4f>();
  }

  std::vector<Eigen::Matrix4f> transforms;
  // GetTransforms(rgb_img, depth_img, model_it->second, model_name, &transforms);
  GetTopPoses(rgb_img, depth_img, model_it->second, model_name, num_candidates,
              &transforms);

  // if (Verbose()) {
  //   for (size_t ii = 0; ii < transforms.size(); ++ii) {
  //     string pose_im_name = "pose_" + to_string(ii) +
  //                           ".png";
  //     DrawProjection(rgb_img, model_it->second, transforms[ii], Prefix() + pose_im_name);
  //   }
  // }

  return transforms;
}

void PoseEstimator::DrawProjection(const Image &rgb_image,
                                   const std::string &model_name, const Model &model,
                                   const Eigen::Matrix4f &transform, const string &im_name) {
  cv::Mat disp_image;
  // disp_image = rgb_image.image().clone();
  rgb_image.image().copyTo(disp_image);
  // disp_image.setTo(0);

  pcl::PointCloud<pcl::PointXYZ> projected_cloud;
  pcl::transformPointCloud(*model.cloud(), projected_cloud, transform);
  vector<cv::Point> projected_points(projected_cloud.points.size(), cv::Point(0,
                                                                              0));

  for (size_t ii = 0; ii < projected_cloud.points.size(); ++ii) {
    const auto &point = projected_cloud.points[ii];
    WorldToCam(point.x, point.y, point.z, projected_points[ii].x,
               projected_points[ii].y);
  }

  cv::Mat blend;

  const int kOffset = 0; //50

  // for (size_t ii = 0; ii < projected_points.size(); ++ii) {
  //   if (projected_points[ii].x < 0 || projected_points[ii].y < 0 ||
  //       projected_points[ii].x >= rgb_image.image().cols ||
  //       projected_points[ii].y >= rgb_image.image().rows) {
  //     continue;
  //   }
  //
  //   auto &pixel =
  //     disp_image.at<cv::Vec3b>(projected_points[ii].y, projected_points[ii].x);
  //   pixel[0] = 0;
  //   pixel[1] = 255;
  //   pixel[2] = 255;
  // }

  cv::Mat projected_image;
  GetProjectedDepthImage(model_name, transform, projected_image);
  cv::Mat colored_depth;
  // colored_depth = projected_image * 255 / 5.0;
  cv::normalize(projected_image, colored_depth, 0.0, 255, cv::NORM_MINMAX);
  colored_depth.convertTo(colored_depth, CV_8UC1);
  cv::applyColorMap(255 - colored_depth, colored_depth, cv::COLORMAP_JET);

  // colored_depth.setTo(cv::Scalar(0,0,0), projected_image < 0.001);
  for (int row = 0; row < projected_image.rows; ++row) {
    for (int col = 0; col < projected_image.cols; ++col) {
      if (std::isnan(projected_image.at<float>(row, col))) {
        colored_depth.at<cv::Vec3b>(row, col) = disp_image.at<cv::Vec3b>(row, col);
      }
    }
  }

  disp_image = 0.5 * disp_image + 0.5 * colored_depth;

  // cv::imshow("image_2", blend);
  if (!im_name.empty()) {
    // cv::imwrite(Prefix() + im_name, blend);
    cv::imwrite(im_name, disp_image);
  }
}

void PoseEstimator::GetTransforms(const Image &rgb_image,
                                  const Image &depth_image,
                                  const Model &model, const string &model_name,
                                  std::vector<Eigen::Matrix4f> *transforms) {



  // cv::Mat filled_depth_image = depth_image.image().clone();
  // filled_depth_image.convertTo(filled_depth_image, CV_32FC1);
  // filled_depth_image = filled_depth_image / 1000.0;
  // l2s::ImageXYCf l2s_depth_image(filled_depth_image.cols, filled_depth_image.rows, 1, (float*)filled_depth_image.data);
  // l2s::fillDepthImageRecursiveMedianFilter(l2s_depth_image, 2);
  // filled_depth_image = filled_depth_image * 1000.0;
  // filled_depth_image.convertTo(filled_depth_image, CV_16UC1);


  high_res_clock::time_point routine_begin_time = high_res_clock::now();
  const auto &model_mean = model.GetModelMeanFeature();
  // TODO: clean up.
  std::vector<int> dims = {29, 30, 31};
  cv::Mat obj_mask, distance_map;
  // rgb_image.GetHeatmap(model_mean, obj_mask, distance_map, 1e-3, 1.0);
  // rgb_image.GetHeatmap(model_mean, obj_mask, distance_map, dims, 1e-3, 1.0);
  // GetGMMMask(rgb_image, model, obj_mask, distance_map);
  GetGMMMask(rgb_image, model_name, obj_mask, distance_map);

  // Binarize the mask
  cv::Mat obj_mask_binary;
  cv::normalize(distance_map, distance_map, 0, 255, cv::NORM_MINMAX);
  distance_map.convertTo(distance_map, CV_8UC1);
  cv::threshold(distance_map, obj_mask_binary, kObjectMaskThresh, 255,
                CV_THRESH_BINARY);

  // cv::imshow("obj_mask", obj_mask_binary);
  if (Verbose()) {
    cv::imwrite(Prefix() + "b_mask.png", obj_mask);
    // cv::imwrite(Prefix() + "bin_mask.png", obj_mask_binary);
  }

  // Form the sample consensus method
  pcl::PointCloud<pcl::PointXYZ>::Ptr image_cloud(new
                                                  pcl::PointCloud<pcl::PointXYZ>);
  // pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud(new
  //                                                 pcl::PointCloud<pcl::PointXYZ>);
  const auto &img = rgb_image.image();
  // int idx = 0;

  pcl::Correspondences correspondences;

  vector<FeatureVector> img_features;
  img_features.reserve(img.rows * img.cols);

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      if (obj_mask_binary.at<uchar>(row, col) == 0 ||
          depth_image.at<uint16_t>(row, col) == 0 ||
          depth_image.at<uint16_t>(row, col) > kMaxDepth) {
        continue;
      }

      auto img_feature = rgb_image.FeatureAt(col, row);
      pcl::PointXYZ img_point = CamToWorld(col, row,
                                           (float)depth_image.at<uint16_t>(row, col) / 1000.0);
      // TODO: clean up.
      img_feature[29] = 0;
      img_feature[30] = 0;
      img_feature[31] = 0;
      img_features.push_back(img_feature);
      image_cloud->points.push_back(img_point);

      // vector<int> nn_indices;
      // auto matches = model.GetNearestPoints(1, img_feature, &nn_indices);
      // pcl::PointXYZ model_point = matches[0];
      // int nearest_idx = nn_indices[0];

      // model_cloud->points.push_back(model_point);
      // TODO: the distance argument might be useful in other correspondence
      // routines.
      // correspondences.push_back(pcl::Correspondence(idx, idx, 0.0));

      // correspondences.push_back(pcl::Correspondence(idx, nearest_idx, 0.0));
      // ++idx;
    }
  }

  vector<vector<int>> nn_indices;
  vector<vector<float>> nn_distances;
  model.GetNearestPoints(1, img_features, &nn_indices, &nn_distances);

  std::unordered_map<int, float> model_idxs;
  std::unordered_map<int, int> model_to_img_idx;

  for (size_t ii = 0; ii < img_features.size(); ++ii) {
    int nearest_idx = nn_indices[ii][0];

    if (model_idxs.find(nearest_idx) != model_idxs.end() &&
        model_idxs[nearest_idx] < nn_distances[ii][0]) {
      continue;
    }

    model_idxs[nearest_idx] = nn_distances[ii][0];
    model_to_img_idx[nearest_idx] = ii;
    // correspondences.push_back(pcl::Correspondence(ii, nearest_idx, 0.0));
  }

  for (const auto &item : model_to_img_idx) {
    correspondences.push_back(pcl::Correspondence(item.first, item.second,
                                                  model_idxs[item.first]));
  }

  image_cloud->width = 1;
  image_cloud->height = img_features.size();
  image_cloud->is_dense = false;
  // model_cloud->width = 1;
  // model_cloud->height = idx;
  // model_cloud->is_dense = false;

  auto feature_time = std::chrono::duration_cast<std::chrono::duration<double>>
                      (high_res_clock::now() - routine_begin_time);
  cout << "Feature matching took " << feature_time.count() << endl;

  pcl::registration::CorrespondenceRejectorSampleConsensusMultiple<pcl::PointXYZ>
  sac;
  // sac.setInputSource(model_cloud);
  // sac.setInputTarget(image_cloud);
  // sac.setInputTarget(model_cloud);
  sac.setInputSource(model.cloud());
  sac.setInputTarget(image_cloud);
  // sac.setInputSource(image_cloud);
  // sac.setInputTarget(model.cloud());
  high_res_clock::time_point ransac_begin_time = high_res_clock::now();
  // sac.setMaximumIterations(10000);
  sac.setMaximumIterations(1000);
  sac.setInlierThreshold(0.02);
  sac.setRefineModel(false);
  // sac.setSaveInliers(true);

  pcl::Correspondences new_correspondences;
  sac.getRemainingCorrespondences(correspondences, new_correspondences);
  *transforms = sac.getBestTransformations();

  // matches.resize(new_correspondences.size());
  //
  // for (size_t ii = 0; ii < new_correspondences.size(); ++ii) {
  //   matches[ii] = std::make_pair(
  //                    image_cloud->points[new_correspondences[ii].index_query],
  //                    model_cloud->points[new_correspondences[ii].index_match]);
  // }

  // sac.getInliersIndices(inliers_);
  for (auto &transform : *transforms) {
    // Eigen::Affine3f affine_transform;
    // affine_transform.matrix() = transform;
    // transform = affine_transform.inverse().matrix();
    // cout << transform << endl << endl;
  }

  auto current_time = high_res_clock::now();
  auto routine_time = std::chrono::duration_cast<std::chrono::duration<double>>
                      (current_time - routine_begin_time);
  auto ransac_time = std::chrono::duration_cast<std::chrono::duration<double>>
                     (current_time - ransac_begin_time);
  cout << "Pose estimation took " << routine_time.count() << endl;
  cout << "RANSAC took " << ransac_time.count() << endl;
}

void PoseEstimator::GetGMMMask(const Image &rgb_image,
                               const std::string &object_name, cv::Mat &heatmap, cv::Mat &distance_map) {
  auto image = rgb_image.image();
  heatmap.create(image.rows, image.cols, CV_8UC3);
  distance_map.create(image.rows, image.cols, CV_64FC1);


  cv::Mat object_probs, object_labels;
  object_probs.create(image.rows, image.cols, CV_64FC1);
  object_labels.create(image.rows, image.cols, CV_8UC3);

  const int background_idx = model_means_map_.size();

  #pragma omp parallel for

  for (int row = 0; row < image.rows; ++row) {
    for (int col = 0; col < image.cols; ++col) {
      const auto &feature = rgb_image.FeatureAt(col, row);

      // Num models + background
      vector<double> pixel_object_probs(model_means_map_.size() + 1);

      int obj_idx = 0;
      int target_object_idx = -1;

      for (const auto &item : model_means_map_) {
        const string &name = item.first;
        const auto &model = item.second;
        const double object_prob = model.GetGMMLikelihood(feature);
        pixel_object_probs[obj_idx] = object_prob;

        if (name == object_name) {
          target_object_idx = obj_idx;
        }

        ++obj_idx;
      }

      // Add the background as a separate class.
      pixel_object_probs[background_idx] = kBackgroundNoise;

      if (target_object_idx == -1) {
        printf("Error: target object %s not found in database\n", object_name.c_str());
        continue;
      }

      // Set mask pixel probability to normalized target object probability.
      const double pixel_normalizer = std::accumulate(pixel_object_probs.begin(),
                                                      pixel_object_probs.end(), 0.0);
      int best_object_idx = 0;
      Argmax(pixel_object_probs, &best_object_idx);

      if (best_object_idx == background_idx) {
        object_labels.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);

      } else {
        const Color color = GetColor(best_object_idx, 0, model_means_map_.size());
        object_labels.at<cv::Vec3b>(row, col) = cv::Vec3b(255 * color.b, 255 * color.g,
                                                          255 * color.r);
      }

      object_probs.at<double>(row, col) = pixel_object_probs[best_object_idx];

      // Normalizer should never be zero because of finite background
      // probability.
      distance_map.at<double>(row,
                              col) = (pixel_object_probs[target_object_idx] / pixel_normalizer);
      object_probs.at<double>(row, col) /= pixel_normalizer;
    }
  }

  // Colorize object_labels according to best object for each pixel.
  cv::cvtColor(object_labels, object_labels, cv::COLOR_BGR2HSV);

  for (int row = 0; row < image.rows; ++row) {
    for (int col = 0; col < image.cols; ++col) {
      // object_labels.at<cv::Vec3b>(row, col)[2] *= object_probs.at<double>(row,col);
      // Set saturation to be proportional to probability^2, "whiter" meaning
      // more uncertain.
      object_labels.at<cv::Vec3b>(row, col)[1] *= (object_probs.at<double>(row,
                                                                           col) * object_probs.at<double>(row, col));
    }
  }

  cv::cvtColor(object_labels, object_labels, cv::COLOR_HSV2BGR);
  cv::imwrite(Prefix() + "b_labels.png", object_labels);


  cv::Mat normalized, img_color;


  // Normalize likelihood across image.
  cv::normalize(distance_map, distance_map, 1.0, 0.0, cv::NORM_L1);

  cv::normalize(distance_map, normalized, 0.0, 255.0, cv::NORM_MINMAX);
  normalized.convertTo(normalized, CV_8UC1);
  cv::applyColorMap(normalized, heatmap, cv::COLORMAP_JET);
  double alpha = 1.0;
  heatmap = alpha * heatmap + (1 - alpha) * image;
}

bool PoseEstimator::ReadModels(const std::vector<std::string> &model_names) {
  for (int i = 0; i < model_names.size(); ++i) {
    const string &model_name = model_names[i];
    auto &model = models_[model_name];
    std::string filename("/home/venkatrn/research/ycb/models/" + model_names[i] +
                         "/" + model_names[i] +
                         ".xml");
    dart::readModelXML(filename.c_str(), model);
    // model.computeStructure();

    const dart::Mesh &mesh = model.getMesh(model.getMeshNumber(0));
    modelVertexBuffers_.insert(std::make_pair(model_name,
                                              pangolin::GlBuffer(pangolin::GlArrayBuffer,
                                                                 mesh.nVertices, GL_FLOAT, 3, GL_STATIC_DRAW)));
    modelVertexBuffers_[model_name].Upload(mesh.vertices,
                                           mesh.nVertices * sizeof(float3));
    const float3 canonicalVertexOffset = make_float3(i, 0, 0);
    std::vector<float3> canonicalVerts(mesh.nVertices);
    std::memcpy(canonicalVerts.data(), mesh.vertices,
                mesh.nVertices * sizeof(float3));
    std::transform(canonicalVerts.begin(), canonicalVerts.end(),
    canonicalVerts.begin(), [canonicalVertexOffset](const float3 & v) {
      return v + canonicalVertexOffset;
    });
    modelCanonicalVertexBuffers_.insert(std::make_pair(model_name,
                                                       pangolin::GlBuffer(
                                                         pangolin::GlArrayBuffer, mesh.nVertices, GL_FLOAT, 3, GL_STATIC_DRAW)));
    modelCanonicalVertexBuffers_[model_name].Upload(canonicalVerts.data(),
                                                    mesh.nVertices * sizeof(float3));

  }
}
} // namespace
