#include <deep_rgbd_utils/pose_estimator.h>
#include <deep_rgbd_utils/helpers.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/registration/transformation_estimation_svd.h>
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
std::vector<std::string> kYCBObjects = {"002_master_chef_can", "003_cracker_box",
                                        "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "011_banana",  "019_pitcher_base", "021_bleach_cleanser", "024_bowl", "025_mug", "035_power_drill", "036_wood_block", "037_scissors", "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"
                                       };

// std::vector<std::string> kYCBObjects = {"007_tuna_fish_can", "024_bowl", "025_mug", "035_power_drill"};


const string kTensorFlowProto =
  "/home/venkatrn/research/dense_features/tensorflow/lov_frozen_graph.pb";
const string kRigFile = "/home/venkatrn/research/ycb/asus-uw.json";
const string kMeansFolder =
  "/home/venkatrn/research/dense_features/means/lov_rgbd";
const float kSDFResolution = 0.0005;

const uint16_t kMaxDepth = 2000;
const unsigned char kObjectMaskThresh = 10;
const double kBackgroundWeight = 0.2;
const double kBackgroundNoise = (1.0 / 20.0);

} // namespace

namespace dru {

PoseEstimator::PoseEstimator() : PoseEstimator(true) {

}

PoseEstimator::PoseEstimator(bool visual_mode) : visual_mode_(visual_mode) {
  // for (const string &object : kYCBObjects) {
  //   const string model_means_file = kMeansFolder + "/" + object + ".means";
  //   model_means_map_.insert(make_pair<string, Model>(object.c_str(), Model()));
  //   model_means_map_[object].SetMeans(model_means_file);
  // }

  if (visual_mode_) {
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
  // vertex_net_.reset(new VertexNet(kTensorFlowProto));
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

  cv::Mat rgb_img, depth_img;
  rgb_img = cv::imread(rgb_file);
  depth_img = cv::imread(depth_file, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
  auto model_it = kObjectNameToIdx.find(model_name);

  if (model_it == kObjectNameToIdx.end()) {
    cerr << "Model name " << model_name << " is unknown" << endl;
    return vector<Eigen::Matrix4f>();
  }

  std::vector<Eigen::Matrix4f> transforms;
  GetTopPoses(rgb_img, depth_img, model_name, num_candidates, &transforms);

  // if (Verbose()) {
  //   for (size_t ii = 0; ii < transforms.size(); ++ii) {
  //     string pose_im_name = "pose_" + to_string(ii) +
  //                           ".png";
  //     DrawProjection(rgb_img, model_it->second, transforms[ii], Prefix() + pose_im_name);
  //   }
  // }

  return transforms;
}

std::vector<Eigen::Matrix4f> PoseEstimator::GetObjectPoseCandidates(cv::Mat rgb_img, cv::Mat depth_img, const std::string &model_name, int num_candidates) {
  auto model_it = kObjectNameToIdx.find(model_name);

  if (model_it == kObjectNameToIdx.end()) {
    cerr << "Model name " << model_name << " is unknown" << endl;
    return vector<Eigen::Matrix4f>();
  }

  std::vector<Eigen::Matrix4f> transforms;
  GetTopPoses(rgb_img, depth_img, model_name, num_candidates, &transforms);

  return transforms;
}

void PoseEstimator::DrawProjection(const cv::Mat &rgb_image,
                                   const std::string &model_name,
                                   const Eigen::Matrix4f &transform, const string &im_name, const string& text_on_im) {
  cv::Mat disp_image;
  rgb_image.copyTo(disp_image);

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

  if (!text_on_im.empty()) {
    int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 3;  
    cv::Point textOrg(250, 470);
    cv::putText(disp_image, text_on_im, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
  }

  // cv::Mat blend;
  // cv::imshow("image_2", blend);
  if (!im_name.empty()) {
    // cv::imwrite(Prefix() + im_name, blend);
    cv::imwrite(im_name, disp_image);
  }
}

// void PoseEstimator::GetGMMMask(const cv::Mat &rgb_image,
//                                const std::string &object_name, cv::Mat &heatmap, cv::Mat &probability_map) {
//
//   cv::Mat object_labels, obj_probs, vert_preds;
//   vertex_net_->Run(rgb_image, obj_probs, vert_preds);
//   vertex_net_->GetLabelImage(obj_probs, object_labels);
//
//   // int object_idx = ObjectNameToIdx(object_name);
//   int object_idx = 13;
//   cv::Mat sliced_verts;
//   vertex_net->SlidePrediction(obj_probs, vert_preds, object_idx, probability_map, sliced_verts);
//   
//   // Colorize object_labels according to best object for each pixel.
//   // cv::cvtColor(object_labels, object_labels, cv::COLOR_BGR2HSV);
//   //
//   // for (int row = 0; row < image.rows; ++row) {
//   //   for (int col = 0; col < image.cols; ++col) {
//   //     // object_labels.at<cv::Vec3b>(row, col)[2] *= object_probs.at<double>(row,col);
//   //     // Set saturation to be proportional to probability^2, "whiter" meaning
//   //     // more uncertain.
//   //     object_labels.at<cv::Vec3b>(row, col)[1] *= (object_probs.at<double>(row,
//   //                                                                          col) * object_probs.at<double>(row, col));
//   //   }
//   // }
//   //
//   // cv::cvtColor(object_labels, object_labels, cv::COLOR_HSV2BGR);
//   cv::imwrite(Prefix() + "b_labels.png", object_labels);
//
//   cv::Mat normalized, img_color;
//
//   // Normalize likelihood across image.
//   cv::normalize(probability_map, probability_map, 1.0, 0.0, cv::NORM_L1);
//
//   cv::normalize(probability_map, normalized, 0.0, 255.0, cv::NORM_MINMAX);
//   normalized.convertTo(normalized, CV_8UC1);
//   cv::applyColorMap(normalized, heatmap, cv::COLORMAP_JET);
//   double alpha = 1.0;
//   heatmap = alpha * heatmap + (1 - alpha) * image;
// }

bool PoseEstimator::ReadModels(const std::vector<std::string> &model_names) {
  if (!visual_mode_) {
    printf("Pose Estimator: Cannot read models in non-visual mode\n");
    return false;
  }
  for (int i = 0; i < model_names.size(); ++i) {
    const string &model_name = model_names[i];
    auto &model = models_[model_name];
    std::string filename("/home/venkatrn/research/ycb/models/" + model_names[i] +
                         "/" + model_names[i] +
                         ".xml");
    dart::readModelXML(filename.c_str(), model);
    model.computeStructure();
    model.voxelize(kSDFResolution, 0.05,"/tmp/" + model_name);
    std::cout << model.getNumSdfs() << std::endl;
    std::cout << model.getSdf(0).dim.x << " x " << model.getSdf(0).dim.y << " x " << model.getSdf(0).dim.z << std::endl;


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
