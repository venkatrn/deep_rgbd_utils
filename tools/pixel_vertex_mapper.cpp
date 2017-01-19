#include <deep_rgbd_utils/feature_generator_net.h>
#include <deep_rgbd_utils/helpers.h>
#include <deep_rgbd_utils/image.h>
#include <deep_rgbd_utils/model.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#include <iostream>
#include <fstream>
#include <cstdint>

#include <chrono>

using namespace std;
using namespace dru;
using namespace flann;
using high_res_clock = std::chrono::high_resolution_clock;


namespace {
const int kNumNeighbors = 5;
const int kMinObservations = 0;

const string kCoffeeMugMeans =
  "/home/venkatrn/research/dense_features/means/brick.means";
// const string kCoffeeMugMeans = "/home/venkatrn/research/dense_features/means/coffee_mug.means";

bool updated = false;

const int kNumMatchesForPnP = 3;
Eigen::Matrix4f best_transform;

} // namespace

Image img1, img2, depth_img;
Model coffee_model;
PointCloudPtr colored_cloud_(new PointCloud);
cv::Mat coffee_mug_points_;

int click_counter_ = 0;
cv::Mat object_points_ = cv::Mat::zeros(kNumMatchesForPnP, 3, CV_64FC1);
cv::Mat depth_points_ = cv::Mat::zeros(kNumMatchesForPnP, 3, CV_64FC1);
cv::Mat image_points_ = cv::Mat::zeros(kNumMatchesForPnP, 2, CV_64FC1);

// void GetObjectPose(const cv::Mat& object_points, const cv::Mat& image_points,
//                    const cv::Mat& points_to_project,
//                    cv::Mat& projected_points) {
//   cv::Mat rvec, tvec;
//   cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
//   distCoeffs.at<double>(0) = 0;
//   distCoeffs.at<double>(1) = 0;
//   distCoeffs.at<double>(2) = 0;
//   distCoeffs.at<double>(3) = 0;
//   // cv::solvePnP(object_points, image_points, kCameraIntrinsics, distCoeffs, rvec, tvec, false, CV_P3P);
//   cv::solvePnP(object_points, image_points, kCameraIntrinsics, distCoeffs, rvec, tvec, false);
//   cv::projectPoints(points_to_project, rvec, tvec, kCameraIntrinsics, distCoeffs, projected_points);
// }

//  Return the transform from model to image.
void GetTransform(const Image &rgb_image, const Image &depth_image,
                  const Model &model, Eigen::Matrix4f &transform) {
  high_res_clock::time_point routine_begin_time = high_res_clock::now();
  const auto &model_mean = coffee_model.GetModelMeanFeature();
  // TODO: clean up.
  std::vector<int> dims = {29, 30, 31};
  cv::Mat obj_mask, distance_map;
  rgb_image.GetHeatmap(model_mean, obj_mask, distance_map, 1e-3, 1.0);

  // Binarize the mask
  cv::Mat obj_mask_binary;
  cv::normalize(distance_map, distance_map, 0, 255, cv::NORM_MINMAX);
  distance_map.convertTo(distance_map, CV_8UC1);
  cv::threshold(distance_map, obj_mask_binary, 180, 255, CV_THRESH_BINARY);

  cv::imshow("obj_mask", obj_mask_binary);


  // Form the sample consensus method
  pcl::PointCloud<pcl::PointXYZ>::Ptr image_cloud(new
                                                  pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud(new
                                                  pcl::PointCloud<pcl::PointXYZ>);
  pcl::Correspondences correspondences;
  pcl::Correspondences new_correspondences;
  const auto &img = rgb_image.image();
  int idx = 0;

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      if (obj_mask_binary.at<uchar>(row, col) == 0 ||
          depth_image.at<uint16_t>(row, col) == 0) {
        continue;
      }

      auto img_feature = img2.FeatureAt(col, row);
      pcl::PointXYZ img_point = CamToWorld(col, row,
                                           (float)depth_image.at<uint16_t>(row, col) / 1000.0);
      // TODO: clean up.
      img_feature[29] = 0;
      img_feature[30] = 0;
      img_feature[31] = 0;
      auto matches = coffee_model.GetNearestPoints(1, img_feature);
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

  pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> sac;
  sac.setInputSource(model_cloud);
  sac.setInputTarget(image_cloud);
  high_res_clock::time_point ransac_begin_time = high_res_clock::now();
  sac.setMaximumIterations(1000);
  sac.setInlierThreshold(0.005);
  sac.getRemainingCorrespondences(correspondences, new_correspondences);
  sac.setRefineModel(true);
  transform = sac.getBestTransformation();
  cout << transform << endl;
  auto current_time = high_res_clock::now();
  auto routine_time = std::chrono::duration_cast<std::chrono::duration<double>>
                      (current_time - routine_begin_time);
  auto ransac_time = std::chrono::duration_cast<std::chrono::duration<double>>
                     (current_time - ransac_begin_time);
  cout << "Pose estimation took " << routine_time.count() << endl;
  cout << "RANSAC took " << ransac_time.count() << endl;

}

void GetObjectPose(const cv::Mat &object_points, const cv::Mat &depth_points,
                   const cv::Mat &points_to_project,
                   cv::Mat &projected_points) {
  pcl::PointCloud<pcl::PointXYZ> c1, c2;

  for (size_t ii = 0; ii < object_points.rows; ++ii) {
    pcl::PointXYZ p;
    p.x = object_points.at<double>(ii, 0);
    p.y = object_points.at<double>(ii, 1);
    p.z = object_points.at<double>(ii, 2);
    c1.points.push_back(p);
  }

  c1.width = 1;
  c1.height = c1.points.size();

  for (size_t ii = 0; ii < depth_points.rows; ++ii) {
    pcl::PointXYZ p;
    p.x = depth_points.at<double>(ii, 0);
    p.y = depth_points.at<double>(ii, 1);
    p.z = depth_points.at<double>(ii, 2);
    c2.points.push_back(p);
  }

  c2.width = 1;
  c2.height = c2.points.size();


  pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>::Ptr
  est;
  est.reset(new pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,
            pcl::PointXYZ>);
  est->estimateRigidTransformation(c1, c2, best_transform);
  cout << endl << best_transform << endl;
}

void CVCallback(int event, int x, int y, int flags, void *userdata) {
  if  ( event == cv::EVENT_LBUTTONDOWN ) {
    ++click_counter_;
    updated = false;
    cv::Mat disp_image1 = img1.image().clone();
    cv::Mat disp_image2 = img2.image().clone();

    cv::Point point(x, y);
    cv::circle(disp_image2, point, 5, cv::Scalar(0, 255, 0), -1);
    // printf("Feature vector for (%d,%d:)\n", x, y);
    auto feature = img2.FeatureAt(x, y);
    // TODO: clean up.
    feature[29] = 0;
    feature[30] = 0;
    feature[31] = 0;
    cv::Mat heatmap, distance_map;
    const auto &maxima = img1.GetMaximaAndHeatmap(feature, heatmap, distance_map,
                                                  1e-10, 0.5);
    printf("M2: %zu\n", maxima.size());
    cv::imshow("heatmap", heatmap);

    const int num_maxima = 5;

    for (size_t ii = 0; ii < num_maxima && ii < maxima.size(); ++ii) {
      const cv::Point center(maxima[ii].first.x, maxima[ii].first.y);
      cv::circle(disp_image1, center, num_maxima - ii + 2, cv::Scalar(255, 0, 0),
                 -1);
    }

    cv::imshow("image_1", disp_image1);
    cv::imshow("image_2", disp_image2);
    const double blend_alpha = 0.5;
    // cv::imshow("heatmap", blend_alpha * img_color + (1 - blend_alpha) * disp_image1);
    //

    // printf("Getting colored cloud\n");
    colored_cloud_ = coffee_model.GetHeatmapCloud(feature);
    auto closest_points = coffee_model.GetNearestPoints(1, feature);
    pcl::PointXYZ closest_model_point = closest_points[0];
    const int row_idx = click_counter_ % kNumMatchesForPnP;
    image_points_.at<double>(row_idx, 0) = x;
    image_points_.at<double>(row_idx, 1) = y;
    object_points_.at<double>(row_idx, 0) = closest_model_point.x;
    object_points_.at<double>(row_idx, 1) = closest_model_point.y;
    object_points_.at<double>(row_idx, 2) = closest_model_point.z;
    printf("Model point: %f %f %f\n", closest_model_point.x, closest_model_point.y,
           closest_model_point.z);
    pcl::PointXYZ depth_point = CamToWorld(x, y, (float)depth_img.at<uint16_t>(y,
                                                                               x) / 1000.0);
    printf("Camera point: %f %f %f\n", depth_point.x, depth_point.y,
           depth_point.z);
    cout << "Depth val: " << (float)depth_img.at<uint16_t>(x, y) / 1000.0 << endl;
    depth_points_.at<double>(row_idx, 0) = depth_point.x;
    depth_points_.at<double>(row_idx, 1) = depth_point.y;
    depth_points_.at<double>(row_idx, 2) = depth_point.z;
  } else if  ( event == cv::EVENT_RBUTTONDOWN ) {
  } else if  ( event == cv::EVENT_MBUTTONDOWN ) {
  } else if ( event == cv::EVENT_MOUSEMOVE ) {
  }
}

void UpdateProjection(const Eigen::Matrix4f &transform) {
  cv::Mat disp_image2;
  disp_image2 = img2.image().clone();
  cv::Mat projected_points;
  // GetObjectPose(object_points_, depth_points_,
  //               coffee_mug_points_,
  //               projected_points);

  pcl::PointCloud<pcl::PointXYZ> projected_cloud;
  pcl::transformPointCloud (*coffee_model.cloud(), projected_cloud, transform);
  projected_points.create(coffee_mug_points_.rows, coffee_mug_points_.cols,
                          CV_64FC1);

  for (size_t ii = 0; ii < projected_cloud.points.size(); ++ii) {
    const auto &point = projected_cloud.points[ii];
    int cam_x = 0;
    int cam_y = 0;
    WorldToCam(point.x, point.y, point.z, cam_x, cam_y);
    projected_points.at<double>(ii, 0) = static_cast<double>(cam_x);
    projected_points.at<double>(ii, 1) = static_cast<double>(cam_y);
  }

  cv::Mat blend;

  for (size_t ii = 0; ii < projected_points.rows; ++ii) {
    const cv::Point center(projected_points.at<double>(ii, 0),
                           projected_points.at<double>(ii, 1));
    // cout << endl << center.y << " " << center.x;
    cv::circle(disp_image2, center, 1, cv::Scalar(0, 255, 255), -1);
    double alpha = 0.5;
    // disp_image2 = disp_image2 * alpha + img2 * (1-alpha);
    cv::addWeighted(disp_image2, alpha, img2.image(), 1 - alpha, 0, blend);
  }

  for (size_t ii = 0; ii < image_points_.rows; ++ii) {
    const cv::Point center(image_points_.at<double>(ii, 0),
                           image_points_.at<double>(ii, 1));
    cv::circle(blend, center, 5, cv::Scalar(255, 0, 0), -1);
  }

  cv::imshow("image_2", blend);
  cv::waitKey(10);
}

int main(int argc, char **argv) {
  // if (argc != 5) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " img1.jpg img2.jpg img2_depth.jpg" <<  std::endl;
    return 1;
  }

  pcl::visualization::PCLVisualizer *viewer;
  // Bring up PCL viewer.
  viewer = new pcl::visualization::PCLVisualizer("PCL Viewer");
  viewer->removeAllPointClouds();
  viewer->removeAllShapes();

  string model_file   = argv[1];
  string trained_file = argv[2];
  unique_ptr<FeatureGenerator> generator;
  generator.reset(new FeatureGenerator(model_file, trained_file));
  std::cout << "Initialized network\n";

  string img1_file = argv[3];
  string img2_file = argv[4];
  string depth_file = argv[5];

  img1.SetImage(img1_file);
  img2.SetImage(img2_file);
  depth_img.SetImage(depth_file);
  coffee_model.SetMeans(kCoffeeMugMeans);
  PCDToCVMat(coffee_model.cloud(), coffee_mug_points_);

  // Run the feature generator.
  const auto &features1 = generator->GetFeatures(img1.image());
  const auto &features2 = generator->GetFeatures(img2.image());
  img1.SetFeatures(features1);
  img2.SetFeatures(features2);
  // features1 = generator->GetFeatures(img1, depth_img);
  // features2 = generator->GetFeatures(img2, depth_img);

  cv::namedWindow("image_1");
  cv::namedWindow("image_2");
  cv::namedWindow("heatmap");
  cv::namedWindow("obj_mask");

  const int im1_x = 100;
  const int im1_y = 100;
  const int padding = 30;
  cv::moveWindow("image_1", im1_x, im1_y);
  cv::moveWindow("image_2", im1_x + img1.cols() + padding, im1_y);
  cv::moveWindow("heatmap", im1_x + img1.cols() / 2 + padding / 2,
                 im1_y + img1.rows() + padding);

  cv::imshow("image_1", img1.image());
  cv::imshow("image_2", img2.image());
  cv::imshow("heatmap", img1.image());
  cv::imshow("obj_mask", img2.image());

  GetTransform(img2, depth_img, coffee_model, best_transform);
  UpdateProjection(best_transform);

  cv::setMouseCallback("image_2", CVCallback, NULL);

  while (1) {
    // if (!viewer->updatePointCloud(colored_cloud_, "colored_cloud")) {
    //   viewer->addPointCloud(colored_cloud_, "colored_cloud");
    //   viewer->setPointCloudRenderingProperties(
    //     pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "colored_cloud");
    // }
    //
    // if (click_counter_ != 0 && click_counter_ % kNumMatchesForPnP == 0 &&
    //     !updated) {
    //   UpdateProjection(best_transform);
    // }

    updated = true;
    viewer->spinOnce();
    cv::waitKey(10);
  }

  delete viewer;
  return 0;
}
