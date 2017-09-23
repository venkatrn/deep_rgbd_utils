#include <deep_rgbd_utils/pose_estimator.h>
#include <deep_rgbd_utils/helpers.h>
#include <deep_rgbd_utils/cv_serialization.h>

#include <pcl/common/transforms.h>

#include <vector>

#include <l2s/image/depthFilling.h>

#include <algorithm>
#include <chrono>

using namespace dru;
using namespace std;
using high_res_clock = std::chrono::high_resolution_clock;

struct PoseCandidate {
  Sophus::SE3d pose;
  double score = 0;
};

typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> Vec3f;
Vec3f PointToVec(const pcl::PointXYZ &p) {
  Vec3f vec;
  vec << p.x, p.y, p.z;
  return vec;
}

Vec3f PointToVec(const cv::Vec3f &p) {
  Vec3f vec;
  vec << p[0], p[1], p[2];
  return vec;
}

template <typename Scalar>
Sophus::SE3d ComputePose(const
                         std::vector<Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign>> &sourcePoints,
                         const std::vector<Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign>>
                         &targetPoints) {
  typedef Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign> Vec3;

  const Vec3 centroidSource = Scalar(1) / sourcePoints.size() * std::accumulate(
                                sourcePoints.begin(), sourcePoints.end(), Vec3(0, 0, 0));
  const Vec3 centroidTarget = Scalar(1) / targetPoints.size() * std::accumulate(
                                targetPoints.begin(), targetPoints.end(), Vec3(0, 0, 0));

  Eigen::Matrix<Scalar, 3, Eigen::Dynamic> recenteredSource(3,
                                                            sourcePoints.size());
  Eigen::Matrix<Scalar, 3, Eigen::Dynamic> recenteredTarget(3,
                                                            targetPoints.size());

  std::transform(sourcePoints.begin(), sourcePoints.end(),
                 reinterpret_cast<Vec3 *>(recenteredSource.data()), [&centroidSource](
  const Vec3 & vec) {
    return vec - centroidSource;
  });

  std::transform(targetPoints.begin(), targetPoints.end(),
                 reinterpret_cast<Vec3 *>(recenteredTarget.data()), [&centroidTarget](
  const Vec3 & vec) {
    return vec - centroidTarget;
  });

  const Eigen::Matrix3d M = (recenteredTarget *
                             recenteredSource.transpose()).template cast<double>();

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(M,
                                        Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3d R = svd.matrixV() * (svd.matrixU().transpose());

  // handle reflections
  if (R.determinant() < 0) {
    for (int i = 0; i < 3; ++i) {
      R(i, 2) *= -1;
    }
  }

  return Sophus::SE3d(Sophus::SO3d(R),
                      -R * (centroidTarget.template cast<double>()) + centroidSource.template
                      cast<double>());

}

template <typename Scalar>
std::vector<Sophus::SE3d> ComputePoses(
  const
  Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign> &s,
  const Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign> &sn,

  const
  Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign> &t,
  const Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign> &tn,
  int num_candidates) {

  typedef Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign> VecType;
  VecType sp, tp;

  if (sn[0] > 0.99) {
    sp = sn.cross(Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign>(0.0, 1.0, 0.0));
  } else {
    sp = sn.cross(Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign>(1.0, 0.0, 0.0));
  }

  if (tn[0] > 0.99) {
    tp = tn.cross(Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign>(0.0, 1.0, 0.0));
  } else {
    tp = tn.cross(Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign>(1.0, 0.0, 0.0));
  }

  vector<VecType> source_points = {s, s + sn, s + sp};
  vector<VecType> target_points = {t, t + tn, t + tp};
  const auto canonical_pose = ComputePose(source_points, target_points);


  const double discretization = M_PI / static_cast<double>(num_candidates);

  vector<Sophus::SE3d> candidate_poses(num_candidates);

  for (int ii = 0; ii < num_candidates; ++ii) {
    double rot_angle = static_cast<double>(ii) * discretization;
    Eigen::AngleAxis<Scalar> rot_matrix(rot_angle, sn);
    Sophus::SE3d new_pose = canonical_pose;
    Eigen::Matrix<Scalar, 3, 3, Eigen::DontAlign> new_rotation;
    new_rotation = canonical_pose.rotationMatrix().template cast<Scalar>();
    new_rotation = rot_matrix * new_rotation;
    new_pose.setRotationMatrix(new_rotation.template cast<double>());
    candidate_poses[ii] =  new_pose;
  }

  return candidate_poses;
}

// void GetNormalImage(const cv::Mat &input_depth_image_m, cv::Mat &normals) {
//   cv::Mat depth;
//   input_depth_image_m.convertTo(depth, CV_32FC1);
//   normals.create(depth.rows, depth.cols, CV_32FC3);
//
//   // TODO: fix for borders
//   for (int x = 1; x < depth.rows - 1; ++x) {
//     for (int y = 1; y < depth.cols - 1; ++y) {
//       // use float instead of double otherwise you will not get the correct result
//       // check my updates in the original post. I have not figure out yet why this
//       // is happening.
//       float dzdx = (depth.at<float>(x + 1, y) - depth.at<float>(x - 1, y)) / 2.0;
//       float dzdy = (depth.at<float>(x, y + 1) - depth.at<float>(x, y - 1)) / 2.0;
//
//       cv::Vec3f d(-dzdx, -dzdy, 1.0f);
//
//       cv::Vec3f n = cv::normalize(d);
//       normals.at<cv::Vec3f>(x, y) = n;
//     }
//   }
// }

void GetNormalImage(const cv::Mat &input_depth_image_m, cv::Mat &normals) {
  cv::Mat depth;
  input_depth_image_m.convertTo(depth,
                                CV_32FC1); // I do not know why it is needed to be transformed to 64bit image my input is 32bit
  normals.create(depth.rows, depth.cols, CV_32FC3);

  for (int x = 1; x < depth.cols - 1; ++x) {
    for (int y = 1; y < depth.rows - 1; ++y) {
      /*double dzdx = (depth(y, x+1) - depth(y, x-1)) / 2.0;
      double dzdy = (depth(y+1, x) - depth(y-1, x)) / 2.0;
      Vec3d d = (-dzdx, -dzdy, 1.0);*/
      cv::Vec3f t(x, y - 1, depth.at<float>(y - 1, x)/*depth(y-1,x)*/);
      cv::Vec3f l(x - 1, y, depth.at<float>(y, x - 1)/*depth(y,x-1)*/);
      cv::Vec3f c(x, y, depth.at<float>(y, x)/*depth(y,x)*/);
      cv::Vec3f d = (l - c).cross(t - c);
      cv::Vec3f n = cv::normalize(d);
      normals.at<cv::Vec3f>(y, x) = -n;
    }
  }
}


template <typename T>
T normal_pdf(T x, T m, T s) {
  static const T inv_sqrt_2pi = 0.3989422804014327;
  T a = (x - m) / s;

  return inv_sqrt_2pi / s * std::exp(-T(0.5) * a * a);
}

double SensorProb(const double d_exp, const double d_obs) {
  const double max_d = 20.0;

  const double occ_weight = 0.33;
  const double sigma = 0.03; // 2 c,

  double p_occ = 0.33;
  double d_thresh = d_exp + sigma;
  d_thresh = std::min(d_thresh, max_d - 1e-10);

  // if (d_obs < d_thresh || d_obs > max_d) {
  if (d_obs > d_thresh) {
    p_occ = 0.0;
  } else {
    // p_occ = 1.0 / (max_d - d_thresh);
    p_occ = 1.0 / (d_thresh);
  }

  double p_gauss = normal_pdf(d_obs, d_exp, sigma);
  double p_long_drop = normal_pdf(3 * sigma, 0.0, sigma);
  double p_short_drop = normal_pdf(sigma, 0.0, sigma);

  // const double p_final = occ_weight * p_occ + (1 - occ_weight) * p_gauss;
  double p_final = 0;

  if (d_obs <= d_exp) {
    p_final = std::max(p_gauss, p_short_drop);
  } else {
    p_final = std::max(p_gauss, p_long_drop);
  }

  return p_final;
}

double PoseEstimator::EvaluatePose(const Sophus::SE3d &T_sm,
                                   const cv::Mat &filled_depth_image_m,
                                   const cv::Mat &object_probability_mask, const cv::Mat &object_binary_mask,
                                   const string &model_name) {
  int rows = filled_depth_image_m.rows;
  int cols = filled_depth_image_m.cols;

  cv::Mat projected_image;
  Eigen::Matrix4f pose;
  pose = T_sm.matrix().cast<float>();
  GetProjectedDepthImage(model_name, pose, projected_image);
  double score = 0;
  double num_points = 0;

  for (int y = 0; y < projected_image.rows; ++y) {
    for (int x = 0; x < projected_image.cols; ++x) {
      double expected_depth = static_cast<double>(projected_image.at<float>(y, x));

      bool projected_pixel = expected_depth > 0 && !std::isnan(expected_depth);
      bool mask_pixel = object_binary_mask.at<uchar>(y, x) != 0;

      if (mask_pixel) {
        if (projected_pixel) {
          double observed_depth = static_cast<double>(filled_depth_image_m.at<float>(y,
                                                                                     x));
          const double error = (expected_depth - observed_depth);
          const double sq_error = error * error;
          score -= sq_error;
        } else {
          score -= 1.0;
        }
      }
    }
  }
  return score;
}

double PoseEstimator::EvaluatePoseSDF(const Sophus::SE3d &T_sm,
                                   const cv::Mat &filled_depth_image_m,
                                   const cv::Mat &object_probability_mask, const cv::Mat &object_binary_mask,
                                   const string &model_name) {
  int rows = filled_depth_image_m.rows;
  int cols = filled_depth_image_m.cols;

  Eigen::Matrix4f T_ms = T_sm.inverse().matrix().cast<float>();

  cv::Mat projected_image;
  Eigen::Matrix4f pose;
  pose = T_sm.matrix().cast<float>();
  GetProjectedDepthImage(model_name, pose, projected_image);
  double score = 0;
  double num_points = 0;

  pcl::PointCloud<pcl::PointXYZ> cloud, transformed_cloud;
  cloud.points.reserve(rows * cols);

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      bool mask_pixel = object_binary_mask.at<uchar>(y, x) != 0;
      if (mask_pixel) {
          double observed_depth = static_cast<double>(filled_depth_image_m.at<float>(y,
                                                                                     x));
    pcl::PointXYZ cam_point, world_point;

    cam_point = CamToWorld(x, y, filled_depth_image_m.at<float>(y, x));
    cloud.points.push_back(cam_point);
      }
    }
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = false;
  pcl::transformPointCloud(cloud, transformed_cloud, T_ms);

  if (cloud.points.empty()) {
    // TODO
    return -std::numeric_limits<double>::max();
  }

  const dart::Grid3D<float> & sdf = models_[model_name].getSdf(0);
  for (size_t ii = 0; ii < transformed_cloud.points.size(); ++ii) {
    const auto& point = transformed_cloud.points[ii];
    float3 p = make_float3(point.x, point.y, point.z);
    float3 grid_p = sdf.getGridCoords(p);
    float df_val = 0.0;
    if (!sdf.isInBoundsInterp(grid_p)) {
      // TODO: make principled
      df_val = fabs(sdf.getValue(make_int3(0, 0, 0)));
    } else {
      df_val = fabs(sdf.getValueInterpolated(grid_p));
    }
    // cout << sdf_val << endl;
    score -= static_cast<double>(df_val);
  }
  return score;
}

// double PoseEstimator::EvaluatePose(const Sophus::SE3d &T_sm,
//                                    const cv::Mat &filled_depth_image_m,
//                                    const cv::Mat &object_probability_mask, const cv::Mat& object_binary_mask,
//                                    const string &model_name, const Model &model) {
//   // pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new
//   //                                                       pcl::PointCloud<pcl::PointXYZ>);
//   // pcl::transformPointCloud(*model.cloud(), *transformed_cloud,
//   //                          T_sm.matrix().cast<float>());
//   int rows = filled_depth_image_m.rows;
//   int cols = filled_depth_image_m.cols;
//
//   // double score = 0.0;
//   // for (size_t ii = 0; ii < transformed_cloud->points.size(); ++ii) {
//   //   const auto &point = transformed_cloud->points[ii];
//   //   int cam_x = 0;
//   //   int cam_y = 0;
//   //   WorldToCam(point, cam_x, cam_y);
//   //
//   //   if (cam_x < 0 || cam_y < 0 || cam_x >= rows || cam_y >= cols) {
//   //     continue;
//   //   }
//   //
//   //   double expected_depth = point.z;
//   //   double observed_depth = static_cast<double>(filled_depth_image_m.at<float>(cam_y, cam_x));
//   //
//   //   // if (projected_image.at<double>(cam_y, cam_x) > observed_depth) {
//   //   //   projected_image.at<double>(cam_y, cam_x) = observed_depth;
//   //   // }
//   //
//   //   if (fabs(observed_depth - expected_depth) < 0.01) {
//   //     score = score + 1;
//   //   }
//   // }
//
//   cv::Mat projected_image;
//   Eigen::Matrix4f pose;
//   pose = T_sm.matrix().cast<float>();
//   GetProjectedDepthImage(model_name, pose, projected_image);
//   // cv::imwrite(Prefix() + "proj.png", 100 * projected_image);
//
//   double score = 0;
//
//   double num_points = 0;
//   #pragma omp parallel for
//
//   for (int y = 0; y < projected_image.rows; ++y) {
//     for (int x = 0; x < projected_image.cols; ++x) {
//       double expected_depth = static_cast<double>(projected_image.at<float>(y, x));
//
//       if (!(expected_depth <= 0 || std::isnan(expected_depth))) {
//         double observed_depth = static_cast<double>(filled_depth_image_m.at<float>(y,
//                                                                                    x));
//         // cout << expected_depth << endl;
//         // score += log(SensorProb(expected_depth, observed_depth));
//         double prob = log(SensorProb(expected_depth, observed_depth));
//         // TODO: get rid of arbitrary constant.
//         // prob += log(std::max(object_probability_mask.at<double>(y, x), 1e-5));
//         // if (fabs(observed_depth - expected_depth) < 0.01) {
//         //   score = score + 1;
//         // }
//         #pragma omp critical
//         {
//           score += prob;
//           num_points = num_points + 1;
//         }
//       }
//     }
//   }
//
//   if (num_points == 0) {
//     return -std::numeric_limits<double>::max();
//   }
//
//   return score / std::max(num_points, 1.0);
// }

int SampleFromCDF(const std::vector<double> &cdf) {
  if (cdf.empty()) {
    return -1;
  }

  double uniform_rand = ((double) rand() / (RAND_MAX));
  auto it = find_if(cdf.begin(), cdf.end(), [uniform_rand](double cdf_val) {
    return cdf_val >= uniform_rand;
  });

  if (it == cdf.end()) {
    printf("Error sampling from CDF: %f  %f\n", uniform_rand, cdf.back());
    return -1;
  }

  return distance(cdf.begin(), it);
}

void GenerateSample(const cv::Mat &integral_image,
                    int &xs, int &ys) {
  if (integral_image.rows == 0 || integral_image.cols == 0) {
    printf("Invalid integral image\n");
    return;
  }

  const int X = integral_image.cols - 1;
  const int Y = integral_image.rows - 1;
  std::vector<double> x_cdf(X + 1);
  std::vector<double> y_cdf(Y + 1);

  double cumsum = 0;

  for (int x = 0; x <= X; ++x) {
    x_cdf[x] = integral_image.at<double>(Y, x);

    if (x_cdf[x] < 0 || x_cdf[x] > 1 + 1e-10) {
      cout << "WTF_X    " << x_cdf[x] << endl;
    }
  }

  for (int x = 0; x < X; ++x) {
    x_cdf[x] /= x_cdf[X];
  }

  if (fabs(x_cdf[X] - 1.0) > 1e-5) {
    cout << "WTF_CUMX    " << x_cdf[X - 1] << endl;
  }

  // printf("X CDF\n\n");
  // for (int x = 0; x <= X; ++x) {
  //   cout << x_cdf[x] << " ";
  // }
  // cout << endl << endl;

  int x_sample = SampleFromCDF(x_cdf);

  y_cdf[0] = integral_image.at<double>(1, x_sample);

  if (x_sample > 0) {
    y_cdf[0] -= integral_image.at<double>(1, x_sample - 1);
  }

  for (int y = 1; y <= Y; ++y) {
    y_cdf[y] = integral_image.at<double>(y,
                                         x_sample);

    if (x_sample > 0) {
      y_cdf[y] -= integral_image.at<double>(y, x_sample - 1);
    }

    if (y_cdf[y] < 0 || y_cdf[y] > 1) {
      cout << "WTF_Y    " << y_cdf[y] << endl;
    }
  }

  if (y_cdf[Y] > 1e-50) {
    for (int y = 0; y < Y; ++y) {
      y_cdf[y] /= y_cdf[Y];
    }
  } else {
    for (int y = 0; y <= Y; ++y) {
      y_cdf[y] = static_cast<double>(y + 1) / static_cast<double>(Y + 1);
    }
  }

  int y_sample = SampleFromCDF(y_cdf);

  xs = std::max(0, x_sample - 1);
  ys = std::max(0, y_sample - 1);
}

void GenerateSample(const cv::Mat &integral_image,
                    std::vector<cv::Point> &samples, int num_samples) {
  samples.resize(num_samples);

  for (int ii = 0; ii < num_samples; ++ii) {
    GenerateSample(integral_image, samples[ii].x, samples[ii].y);
  }
}


void PoseEstimator::SetPrediction(const std::string &probs, const std::string &verts) {
    probs_mat_ = probs;
    verts_mat_ = verts;
    loadMat(obj_probs_, probs_mat_);
    loadMat(vert_preds_, verts_mat_);
    GetLabelImage(obj_probs_, obj_labels_);
}

void PoseEstimator::GetObjectMask(const std::string &model_name, cv::Mat& object_mask) {
  if (object_masks_.find(model_name) != object_masks_.end()) {
    object_masks_[model_name].copyTo(object_mask);
    return;
  }

  cv::Mat obj_labels, obj_probs;
  string prob_file = probs_mat_;

  // TODO: remove.
  if (prob_file == "") {
    prob_file = debug_dir_ + "/" + im_num_ + "_probs.mat";
    loadMat(obj_probs, prob_file) ;
    GetLabelImage(obj_probs, obj_labels);
  } else {
    obj_probs = obj_probs_;
    obj_labels = obj_labels_;
  }

  const int object_idx = kObjectNameToIdx[model_name];
  object_mask = cv::Mat::zeros(obj_labels.size(), CV_8UC1);
  object_mask.setTo(255, obj_labels == object_idx);
  object_mask.copyTo(object_masks_[model_name]);
  return;
}

void PoseEstimator::GetTopPoses(const cv::Mat &img,
                                const cv::Mat &depth_img,
                                const string &model_name, int num_poses,
                                vector<Eigen::Matrix4f> *poses, int num_trials) {

  high_res_clock::time_point routine_begin_time = high_res_clock::now();

  // Filter depth image.
  cv::Mat filled_depth_image_m;
  depth_img.convertTo(filled_depth_image_m, CV_32FC1);
  filled_depth_image_m = filled_depth_image_m / 1000.0;
  l2s::ImageXYCf l2s_depth_image(filled_depth_image_m.cols,
                                 filled_depth_image_m.rows, 1, (float *)filled_depth_image_m.data);
  l2s::fillDepthImageRecursiveMedianFilter(l2s_depth_image, 2);


  cv::Mat obj_labels, obj_probs, vert_preds, sliced_verts, heatmap,
  probability_map;

  string prob_file = probs_mat_;
  string vert_file = verts_mat_;

  // TODO: remove.
  if (prob_file == "") {
    prob_file = debug_dir_ + "/" + im_num_ + "_probs.mat";
    vert_file = debug_dir_ + "/" + im_num_ + "_verts.mat";
    loadMat(obj_probs, prob_file) ;
    loadMat(vert_preds, vert_file);
    GetLabelImage(obj_probs, obj_labels);
  } else {
    obj_probs = obj_probs_;
    vert_preds = vert_preds_;
    obj_labels = obj_labels_;
  }

  const int object_idx = kObjectNameToIdx[model_name];
  SlicePrediction(obj_probs, vert_preds, object_idx, probability_map,
                  sliced_verts);
  cv::Mat obj_mask = cv::Mat::zeros(obj_labels.size(), CV_8UC1);
  obj_mask.setTo(255, obj_labels == object_idx);
  obj_mask.copyTo(object_masks_[model_name]);

  // ColorizeProbabilityMap(probability_map, object_labels);
  // cv::imwrite(Prefix() + "b_labels.png", object_labels);

  cv::Mat normalized, img_color;

  // Normalize likelihood across image.
  cv::normalize(probability_map, probability_map, 1.0, 0.0, cv::NORM_L1);

  cv::normalize(probability_map, normalized, 0.0, 255.0, cv::NORM_MINMAX);
  normalized.convertTo(normalized, CV_8UC1);
  cv::applyColorMap(normalized, heatmap, cv::COLORMAP_JET);
  double alpha = 1.0;
  heatmap = alpha * heatmap + (1 - alpha) * img;

  // Use the binary mask as probability map.
  cv::Mat obj_mask_double;
  obj_mask.convertTo(obj_mask_double, CV_64FC1);
  cv::normalize(obj_mask_double, probability_map, 1.0, 0.0, cv::NORM_L1);
  cv::Mat integral_image;
  cv::integral(probability_map, integral_image);

  // cv::Mat normal_image;
  // GetNormalImage(filled_depth_image_m, normal_image);

  // cv::imwrite(Prefix() + "integral.png", 255 * integral_image);

  // Binarize the mask
  // cv::Mat obj_mask_binary;
  // cv::normalize(probability_map, probability_map, 0.0, 255.0, cv::NORM_MINMAX);
  // probability_map.convertTo(probability_map, CV_8UC1);
  // cv::threshold(probability_map, obj_mask_binary, 10, 255, CV_THRESH_BINARY);

  // cv::imshow("obj_mask", obj_mask_binary);
  if (Verbose()) {
    cv::imwrite(Prefix() + "a_rgb.png", img);
    // cv::imwrite(Prefix() + "b_depth.png", filled_depth_image_m * 1000);
    cv::imwrite(Prefix() + "c_mask.png", heatmap);
    // cv::imwrite(Prefix() + "d_normals.png", normal_image);
    // cv::imwrite(Prefix() + "bin_mask.png", obj_mask_binary);
  }

  cv::Mat smoothed_likelihood;
  cv::GaussianBlur(probability_map, smoothed_likelihood, cv::Size(15, 15), 0, 0);
  cv::normalize(smoothed_likelihood, smoothed_likelihood, 1.0, 0.0, cv::NORM_L1);

  // cv::namedWindow("ransac");
  cv::Mat disp_image;

  num_trials = 1000;
  // Generate candidate poses
  vector<Sophus::SE3d> candidate_poses(num_trials);

  for (int trial = 0; trial < num_trials; ++trial) {
    img.copyTo(disp_image);
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    int x3 = 0;
    int y3 = 0;
    GenerateSample(integral_image, x1, y1);
    GenerateSample(integral_image, x2, y2);
    GenerateSample(integral_image, x3, y3);

    // Eigen::Vector3f im_n2, im_n3;
    // Eigen::Vector3f im_n1(normal_image.at<cv::Vec3f>(y1, x1).val);

    // while (!(abs(x2 - x1) < 100 && abs(y2 - y1) < 100)) {
    //   GenerateSample(integral_image, x2, y2);
    // }
    //
    // while (!(abs(x3 - x1) < 100 && abs(y3 - y1) < 100)) {
    //   GenerateSample(integral_image, x3, y3);
    // }

    // printf("Sampl: (%d %d), (%d %d), (%d %d)\n", x1, y1, x2, y2, x3, y3);
    // cv::circle(disp_image, cv::Point(x1, y1), 4, cv::Scalar(0,255,0), -1);
    // cv::circle(disp_image, cv::Point(x2, y2), 4, cv::Scalar(0,255,0), -1);
    // cv::circle(disp_image, cv::Point(x3, y3), 4, cv::Scalar(0,255,0), -1);
    // cv::imshow("ransac", disp_image);
    // cv::waitKey(0);

    pcl::PointXYZ p1, p2, p3;
    p1 = CamToWorld(x1, y1, filled_depth_image_m.at<float>(y1, x1));
    p2 = CamToWorld(x2, y2, filled_depth_image_m.at<float>(y2, x2));
    p3 = CamToWorld(x3, y3, filled_depth_image_m.at<float>(y3, x3));
    Vec3f v1, v2, v3;
    v1 = PointToVec(p1);
    v2 = PointToVec(p2);
    v3 = PointToVec(p3);

    pcl::PointXYZ m1, m2, m3;
    auto cv_v1 = sliced_verts.at<cv::Vec3f>(y1, x1) / 10.0;
    auto cv_v2 = sliced_verts.at<cv::Vec3f>(y2, x2) / 10.0;
    auto cv_v3 = sliced_verts.at<cv::Vec3f>(y3, x3) / 10.0;

    Vec3f mv1, mv2, mv3;
    mv1 = PointToVec(cv_v1);
    mv2 = PointToVec(cv_v2);
    mv3 = PointToVec(cv_v3);
    // cout << "pred\n";
    // cout << mv1 << endl;
    // cout << mv2 << endl;
    // cout << mv3 << endl;
    // cout << "im point\n";
    // cout << v1 << endl;
    // cout << v2 << endl;
    // cout << v3 << endl;
    // cout << "im depth\n";
    // cout << filled_depth_image_m.at<float>(y1,x1) << endl;
    // cout << filled_depth_image_m.at<float>(y2,x2) << endl;
    // cout << filled_depth_image_m.at<float>(y3,x3) << endl;
    // cout << depth_img.at<uint16_t>(y1,x1) / 1000.0 << endl;
    // cout << depth_img.at<uint16_t>(y2,x2) / 1000.0 << endl;
    // cout << depth_img.at<uint16_t>(y3,x3) / 1000.0 << endl;

    // Sophus::SE3d pose = ComputePose(vector<Vec3f>({mv1, mv2, mv3}), vector<Vec3f>({v1, v2, v3}));
    Sophus::SE3d pose = ComputePose(vector<Vec3f>({v1, v2, v3}), vector<Vec3f>({mv1, mv2, mv3}));
    candidate_poses[trial] = pose;
  }


  // EXPERIMENTAL ///////////////////////////////////////////////////////////
  // num_trials = 55;
  // vector<Sophus::SE3d> candidate_poses;
  //
  // for (int trial = 0; trial < num_trials; ++trial) {
  //   img.copyTo(disp_image);
  //   int x1 = 0;
  //   int y1 = 0;
  //   int x2 = 0;
  //   int y2 = 0;
  //   int x3 = 0;
  //   int y3 = 0;
  //   GenerateSample(integral_image, x1, y1);
  //
  //   Vec3f im_n1(normal_image.at<cv::Vec3f>(y1, x1).val);
  //   cout << im_n1 << endl;
  //
  //   pcl::PointXYZ p1;
  //   auto im_f1 = rgb_image.FeatureAt(x1, y1);
  //
  //   // TODO: clean up.
  //   im_f1[29] = 0;
  //   im_f1[30] = 0;
  //   im_f1[31] = 0;
  //   vector<FeatureVector> img_features;
  //   img_features.push_back(im_f1);
  //
  //   p1 = CamToWorld(x1, y1, filled_depth_image_m.at<float>(y1, x1));
  //   Vec3f v1;
  //   v1 = PointToVec(p1);
  //
  //   vector<vector<int>> nn_indices;
  //   vector<vector<float>> nn_distances;
  //   model.GetNearestPoints(1, img_features, &nn_indices, &nn_distances);
  //
  //   pcl::PointXYZ m1;
  //   m1 = model.cloud()->points[nn_indices[0][0]];
  //   Vec3f mv1;
  //   mv1 = PointToVec(m1);
  //
  //   pcl::Normal mn;
  //   mn = model.normals()->points[nn_indices[0][0]];
  //   Vec3f model_normal(mn.normal_x, mn.normal_y, mn.normal_z);
  //
  //   // Sophus::SE3d pose = ComputePose(vector<Vec3f>({mv1, mv2, mv3}), vector<Vec3f>({v1, v2, v3}));
  //   // Sophus::SE3d pose = ComputePose(vector<Vec3f>({v1, v2, v3}), vector<Vec3f>({mv1, mv2, mv3}));
  //
  //   vector<Sophus::SE3d> poses = ComputePoses(v1, im_n1, mv1, model_normal, 18);
  //
  //   for (size_t jj = 0; jj < poses.size(); ++jj) {
  //     candidate_poses.push_back(poses[jj]);
  //     // cout << poses[jj].matrix();
  //   }
  // }
  // EXPERIMENTAL ///////////////////////////////////////////////////////////


  vector<PoseCandidate> candidates(candidate_poses.size());

  for (size_t ii = 0; ii < candidate_poses.size(); ++ii) {
    double score = 0;
    const auto &pose = candidate_poses[ii];
    candidates[ii].pose = pose;
    if (visual_mode_) {
      score = EvaluatePoseSDF(pose, filled_depth_image_m, smoothed_likelihood, obj_mask,
                           model_name);
      candidates[ii].score = score;
    } else {
      candidates[ii].score = 1.0;
    }
    // cout << pose.matrix() << endl;
    // cout << score;
  }

  sort(candidates.begin(), candidates.end(), [](const PoseCandidate & p1,
  const PoseCandidate & p2) {
    return p1.score > p2.score;
  });

  int num_poses_to_return = std::min(num_poses,
                                     static_cast<int>(candidates.size()));
  poses->resize(num_poses_to_return);
  std::transform(candidates.begin(), candidates.begin() + num_poses_to_return,
  poses->begin(), [](const PoseCandidate & p) {
    cout << p.score << endl;
    cout << p.pose.matrix() << endl;
    const auto pose = p.pose.matrix().cast<float>();
    return pose;
  });

  ransac_scores_.resize(candidate_poses.size());
  std::transform(candidates.begin(), candidates.begin() + num_poses_to_return,
  ransac_scores_.begin(), [](const PoseCandidate & p) {
    return p.score;
  });

  auto current_time = high_res_clock::now();
  auto routine_time = std::chrono::duration_cast<std::chrono::duration<double>>
                      (current_time - routine_begin_time);
  cout << "Pose estimation took " << routine_time.count() << endl;
}
