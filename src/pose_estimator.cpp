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

#include <iostream>
#include <fstream>
#include <cstdint>

#include <chrono>

using namespace std;
using namespace dru;
using namespace flann;
using high_res_clock = std::chrono::high_resolution_clock;

namespace {
  // TODO: receive from caller.
std::vector<std::string> kYCBObjects = {"002_master_chef_can", "006_mustard_bottle", "008_pudding_box", "024_bowl", "052_extra_large_clamp"}; 

const string kCaffeWeights =
  "/home/venkatrn/research/dense_features/denseCorrespondence-lov/snap_iter_17500.caffemodel";
const string kCaffeProto =
  "/home/venkatrn/research/dense_features/denseCorrespondence-lov/deploy_cpp.prototxt";
const string kMeansFolder = "/home/venkatrn/research/dense_features/means/lov";

} // namespace

namespace dru {

PoseEstimator::PoseEstimator() :
  generator_(std::unique_ptr<FeatureGenerator>(new FeatureGenerator(kCaffeProto,
                                                                    kCaffeWeights))) {
  for (const string& object : kYCBObjects) {
    const string model_means_file = kMeansFolder + "/" + object + ".means";
    model_means_map_.insert(make_pair<string, Model>(object.c_str(), Model()));
    model_means_map_[object].SetMeans(model_means_file);
  }
}

vector<Eigen::Matrix4f> PoseEstimator::GetObjectPoseCandidates(
  const std::string &rgb_file, const std::string &depth_file,
  const std::string &model_name, int num_candidates) {

  Image rgb_img, depth_img;
  rgb_img.SetImage(rgb_file);
  depth_img.SetImage(depth_file);

  if (Verbose()) {
    cv::imwrite(Prefix() + "rgb.png", rgb_img.image());
  }

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
  GetTransforms(rgb_img, depth_img, model_it->second, &transforms);

  // if (Verbose()) {
  //   for (size_t ii = 0; ii < transforms.size(); ++ii) {
  //     string pose_im_name = "pose_" + to_string(ii) +
  //                           ".png";
  //     DrawProjection(rgb_img, model_it->second, transforms[ii], Prefix() + pose_im_name);
  //   }
  // }

  return transforms;
}

void PoseEstimator::DrawProjection(const Image &rgb_image, const Model &model,
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

  for (size_t ii = 0; ii < projected_points.size(); ++ii) {
    // cout << endl << center.y << " " << center.x;

    // cv::circle(disp_image, projected_points[ii], 1, cv::Scalar(0, 255, 255), -1);
    if (projected_points[ii].x < 0 || projected_points[ii].y < 0 ||
        projected_points[ii].x >= rgb_image.image().cols || projected_points[ii].y >= rgb_image.image().rows) {
      continue;
    }
    auto &pixel = 
    disp_image.at<cv::Vec3b>(projected_points[ii].y, projected_points[ii].x);
    pixel[0] = 0;
    pixel[1] = 255;
    pixel[2] = 255;
    // double alpha = 0.5;
    // cv::addWeighted(disp_image, alpha, rgb_image.image(), 1 - alpha, 0, blend);
  }

  // cv::imshow("image_2", blend);
  if (!im_name.empty()) {
    // cv::imwrite(Prefix() + im_name, blend);
    cv::imwrite(im_name, disp_image);
  }
}

void PoseEstimator::GetTransforms(const Image &rgb_image,
                                  const Image &depth_image,
                                  const Model &model, std::vector<Eigen::Matrix4f> *transforms) {



  high_res_clock::time_point routine_begin_time = high_res_clock::now();
  const auto &model_mean = model.GetModelMeanFeature();
  // TODO: clean up.
  std::vector<int> dims = {29, 30, 31};
  cv::Mat obj_mask, distance_map;
  rgb_image.GetHeatmap(model_mean, obj_mask, distance_map, 1e-3, 1.0);

  // Binarize the mask
  cv::Mat obj_mask_binary;
  cv::normalize(distance_map, distance_map, 0, 255, cv::NORM_MINMAX);
  distance_map.convertTo(distance_map, CV_8UC1);
  cv::threshold(distance_map, obj_mask_binary, 180, 255, CV_THRESH_BINARY);

  // cv::imshow("obj_mask", obj_mask_binary);
  if (Verbose()) {
    cv::imwrite(Prefix() + "mask.png", obj_mask);
    cv::imwrite(Prefix() + "bin_mask.png", obj_mask_binary);
  }

  // Form the sample consensus method
  pcl::PointCloud<pcl::PointXYZ>::Ptr image_cloud(new
                                                  pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud(new
                                                  pcl::PointCloud<pcl::PointXYZ>);
  const auto &img = rgb_image.image();
  int idx = 0;

  pcl::Correspondences correspondences;

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      if (obj_mask_binary.at<uchar>(row, col) == 0 ||
          depth_image.at<uint16_t>(row, col) == 0) {
        continue;
      }

      auto img_feature = rgb_image.FeatureAt(col, row);
      pcl::PointXYZ img_point = CamToWorld(col, row,
                                           (float)depth_image.at<uint16_t>(row, col) / 1000.0);
      // TODO: clean up.
      img_feature[29] = 0;
      img_feature[30] = 0;
      img_feature[31] = 0;
      auto matches = model.GetNearestPoints(1, img_feature);
      pcl::PointXYZ model_point = matches[0];

      image_cloud->points.push_back(img_point);
      model_cloud->points.push_back(model_point);
      // TODO: the distance argument might be useful in other correspondence
      // routines.
      correspondences.push_back(pcl::Correspondence(idx, idx, 0.0));
      ++idx;
    }
  }

  image_cloud->width = 1;
  image_cloud->height = idx;
  image_cloud->is_dense = false;
  model_cloud->width = 1;
  model_cloud->height = idx;
  model_cloud->is_dense = false;

  pcl::registration::CorrespondenceRejectorSampleConsensusMultiple<pcl::PointXYZ>
  sac;
  // sac.setInputSource(model_cloud);
  // sac.setInputTarget(image_cloud);
  sac.setInputSource(image_cloud);
  sac.setInputTarget(model_cloud);
  high_res_clock::time_point ransac_begin_time = high_res_clock::now();
  // sac.setMaximumIterations(10000);
  sac.setMaximumIterations(1000);
  sac.setInlierThreshold(0.01);
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
    Eigen::Affine3f affine_transform;
    affine_transform.matrix() = transform;
    transform = affine_transform.inverse().matrix();
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
} // namespace
