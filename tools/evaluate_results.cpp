#include <deep_rgbd_utils/pose_estimator.h>
#include <perception_utils/pose_evaluator.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/filesystem.hpp>

#include <gflags/gflags.h>

using namespace std;
using namespace dru;
using namespace perception_utils;

DEFINE_double(eta, 0.1, "Fraction parameter for AD evaluation criterion");
DEFINE_bool(perch, true, "Evaluate on PERCH if true, baseline otherwise");

namespace {
const string models_dir =
  "/home/venkatrn/indigo_workspace/src/perch/sbpl_perception/data/ycb_models/";
// const string models_dir = "/home/venkatrn/research/ycb/models/";
}

// TODO move to utils file
template <typename Derived>
inline std::istream &operator >>(std::istream &stream,
                                 Eigen::MatrixBase<Derived> &M) {
  for (int r = 0; r < M.rows(); ++r) {
    for (int c = 0; c < M.cols(); ++c) {
      if (! (stream >> M(r, c))) {
        return stream;
      }
    }
  }

  // Strip newline character.
  if (stream.peek() == 10) {
    stream.ignore(1, '\n');
  }

  return stream;
}

enum class SymMode {
  A, // none
  B, // ninety degrees
  C, // one eighty degrees
  D  // radially symmetric
};

std::map<std::string, std::vector<SymMode>> kObjectSymmetries = {
  // {"background", {SymMode::A, SymMode::A, SymMode::A}},
  {"002_master_chef_can", {SymMode::C, SymMode::C, SymMode::D}},
  {"003_cracker_box", {SymMode::C, SymMode::C, SymMode::C}},
  {"004_sugar_box", {SymMode::C, SymMode::C, SymMode::C}},
  {"005_tomato_soup_can", {SymMode::C, SymMode::C, SymMode::D}},
  {"006_mustard_bottle", {SymMode::A, SymMode::A, SymMode::C}},
  {"007_tuna_fish_can", {SymMode::C, SymMode::C, SymMode::D}},
  {"008_pudding_box", {SymMode::C, SymMode::C, SymMode::C}},
  {"009_gelatin_box", {SymMode::C, SymMode::C, SymMode::C}},
  {"010_potted_meat_can", {SymMode::C, SymMode::C, SymMode::C}},
  {"011_banana", {SymMode::A, SymMode::A, SymMode::A}},
  {"019_pitcher_base", {SymMode::A, SymMode::A, SymMode::A}},
  {"021_bleach_cleanser", {SymMode::A, SymMode::A, SymMode::A}},
  {"024_bowl", {SymMode::A, SymMode::A, SymMode::D}},
  {"025_mug", {SymMode::A, SymMode::A, SymMode::A}},
  {"035_power_drill", {SymMode::A, SymMode::A, SymMode::A}},
  {"036_wood_block", {SymMode::C, SymMode::C, SymMode::B}},
  {"037_scissors", {SymMode::A, SymMode::A, SymMode::A}},
  // TODO: allow 180?
  {"040_large_marker", {SymMode::A, SymMode::D, SymMode::A}},
  {"051_large_clamp", {SymMode::A, SymMode::C, SymMode::A}},
  {"052_extra_large_clamp", {SymMode::C, SymMode::A, SymMode::A}},
  // {"061_foam_brick", {SymMode::A, SymMode::A, SymMode::C}}
  {"061_foam_brick", {SymMode::C, SymMode::B, SymMode::C}}
};

double normalize_angle_positive(double angle, double symmetry_angle = M_PI) {
  return fmod(fmod(angle, 2.0 * symmetry_angle) + 2.0 * symmetry_angle,
              2.0 * symmetry_angle);
}

double normalize_angle(double angle, double symmetry_angle = M_PI) {
  double a = normalize_angle_positive(angle, symmetry_angle);

  if (a > symmetry_angle) {
    a -= 2.0 * symmetry_angle;
  }

  return a;
}

Eigen::Matrix3f rpy_to_rot(const Eigen::Vector3f &rpy) {
  Eigen::Matrix3f rot;
  rot = Eigen::AngleAxisf(rpy[0], Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(rpy[1], Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(rpy[2], Eigen::Vector3f::UnitZ());
  return rot;
}

void CanonicalizeRPY(Eigen::Vector3f &rpy,
                     const vector<SymMode> &symmetry_modes) {
  for (int ii = 0; ii < 3; ++ii) {
    rpy[ii] = normalize_angle_positive(rpy[ii]);

    switch (symmetry_modes[ii]) {
    case SymMode::A:
      break;

    case SymMode::B:
      rpy[ii] = fmod(rpy[ii], M_PI / 2.0);
      // rpy[ii] = normalize_angle(rpy[ii], M_PI / 4.0);
      break;

    case SymMode::C:
      cout << "SYMMMMM!  " << ii << "\n";
      cout << rpy[ii] << endl;
      rpy[ii] = fmod(rpy[ii], M_PI);
      // rpy[ii] = normalize_angle(rpy[ii], M_PI / 2.0);
      cout << rpy[ii] << endl;
      break;

    case SymMode::D:
      rpy[ii] = 0;
      break;

    default:
      break;
    }
  }
}

void SymmetricDiff(Eigen::Vector3f &rpy,
                   const vector<SymMode> &symmetry_modes) {
  for (int ii = 0; ii < 3; ++ii) {

    switch (symmetry_modes[ii]) {
    case SymMode::A:
      rpy[ii] = normalize_angle(rpy[ii], M_PI);
      break;

    case SymMode::B:
      rpy[ii] = normalize_angle(rpy[ii], M_PI / 4.0);
      break;

    case SymMode::C:
      cout << "Diff!  " << ii << "\n";
      cout << rpy[ii] << endl;
      rpy[ii] = normalize_angle(rpy[ii], M_PI / 2.0);
      cout << rpy[ii] << endl;
      break;

    case SymMode::D:
      rpy[ii] = 0;
      break;

    default:
      break;
    }
  }
}

void EvaluatePose(const Eigen::Matrix4f &gt_pose, const Eigen::Matrix4f &pose,
                  const std::string &model_name,
                  float *trans_error, float *rot_error) {
  Eigen::Vector3f t_diff = gt_pose.block<3, 1>(0, 3) - pose.block<3, 1>(0, 3);
  cout << t_diff.transpose() << endl;

  Eigen::Matrix3f R_gt = gt_pose.block<3, 3>(0, 0);
  Eigen::Matrix3f R_est = pose.block<3, 3>(0, 0);

  Eigen::Vector3f rpy_gt = R_gt.eulerAngles(0, 1, 2);
  Eigen::Vector3f rpy_est = R_est.eulerAngles(0, 1, 2);

  const auto &symmetry_modes = kObjectSymmetries[model_name];
  CanonicalizeRPY(rpy_gt, symmetry_modes);
  CanonicalizeRPY(rpy_est, symmetry_modes);

  Eigen::Vector3f rpy_diff = rpy_gt - rpy_est;
  SymmetricDiff(rpy_diff, symmetry_modes);

  Eigen::Matrix3f modified_gt_pose = rpy_to_rot(rpy_gt);
  Eigen::Matrix3f modified_est_pose = rpy_to_rot(rpy_est);
  Eigen::Matrix3f diff_pose = rpy_to_rot(rpy_diff);

  Eigen::Quaternionf q1(modified_gt_pose.block<3, 3>(0, 0));
  Eigen::Quaternionf q2(modified_est_pose.block<3, 3>(0, 0));
  Eigen::Quaternionf q3(diff_pose.block<3, 3>(0, 0));

  *trans_error = t_diff.norm();
  // *rot_error = q1.angularDistance(q2);
  Eigen::AngleAxisf aa(q3);
  *rot_error = fabs(normalize_angle(aa.angle()));
}

bool ReadGTFile(const std::string &gt_file, std::vector<string> *model_names,
                std::vector<Eigen::Matrix4f> *model_transforms) {
  std::ifstream gt_stream;
  gt_stream.open(gt_file, std::ofstream::in);

  if (!gt_stream) {
    return false;
  }

  int num_models = 0;
  gt_stream >> num_models;
  model_names->resize(num_models);
  model_transforms->resize(num_models);

  for (int ii = 0; ii < num_models; ++ii) {
    gt_stream >> model_names->at(ii);
    gt_stream >> model_transforms->at(ii);
  }

  gt_stream.close();
  return true;
}

bool ReadPredictionsFile(const std::string &predictions_file, int num_models,
                         std::vector<string> *model_names,
                         std::vector<std::vector<Eigen::Matrix4f>> *predicted_transforms) {
  std::ifstream predictions_stream;
  predictions_stream.open(predictions_file, std::ofstream::in);

  if (!predictions_stream) {
    return false;
  }

  // int num_models = model_;
  // predictions_stream >> num_models;
  model_names->resize(num_models);
  predicted_transforms->resize(num_models);

  int num_candidates = 0;

  for (int ii = 0; ii < num_models; ++ii) {
    predictions_stream >> model_names->at(ii);
    predictions_stream >> num_candidates;
    predicted_transforms->at(ii).resize(num_candidates);

    for (int jj = 0; jj < num_candidates; ++jj) {
      predictions_stream >> predicted_transforms->at(ii)[jj];
    }
  }

  predictions_stream.close();
  return true;
}


int main(int argc, char **argv) {
  if (argc < 3) {
    cerr << "Usage: ./evaluate_results <path_to_dataset_folder> <path_to_results_folder>"
         << endl;
    return -1;
  }

  gflags::ParseCommandLineFlags(&argc, &argv, true);


  boost::filesystem::path dataset_dir = argv[1];
  boost::filesystem::path output_dir = argv[2];

  if (!boost::filesystem::is_directory(dataset_dir)) {
    cerr << "Invalid dataset directory" << endl;
    return -1;
  }

  if (!boost::filesystem::is_directory(output_dir)) {
    cerr << "Invalid output directory" << endl;
    return -1;
  }

  boost::filesystem::directory_iterator dataset_it(dataset_dir), dataset_end;

  std::string scenes_file_name = output_dir.string() + "/scenes_file.txt";
  std::ofstream scenes_file;
  scenes_file.open(scenes_file_name, std::ofstream::out);

  // Read all models for evaluation tool.
  std::map<std::string, PoseEvaluator> evaluators;

  for (auto it = kObjectSymmetries.begin(); it != kObjectSymmetries.end();
       ++it) {
    const string name = it->first;
    const string model_file = models_dir + name + ".ply";
    // const string model_file = models_dir + name + "/textured.obj";
    cout << model_file << endl;
    PoseEvaluator evaluator(model_file);
    pair<string, PoseEvaluator> item(name, evaluator);
    evaluators.insert(item);
  }

  for (dataset_it; dataset_it != dataset_end; ++dataset_it) {
    // Skip non-video folders (assuming every folder that contains "00" is video folder).
    if (dataset_it->path().filename().string().find("00") == std::string::npos) {
      continue;
    }

    const string scene = dataset_it->path().stem().string();
    const string scene_dir = dataset_dir.string() + "/" + scene + "/rgb";

    if (!boost::filesystem::is_directory(scene_dir)) {
      cerr << "Invalid scene directory " << scene_dir << endl;
      return -1;
    }

    cout << "Video: " << scene << endl;

    boost::filesystem::directory_iterator scene_it(scene_dir), scene_end;

    for (scene_it; scene_it != scene_end; ++scene_it) {
      // Skip non-png files.
      if (scene_it->path().extension().string().compare(".png") != 0) {
        continue;
      }

      const string rgb_file = scene_it->path().string();

      string depth_file = rgb_file;
      depth_file.replace(depth_file.rfind("rgb"), 3, "depth");

      if (!boost::filesystem::is_regular_file(depth_file)) {
        cerr << "Nonexistent depth file " << depth_file << endl;
        return -1;
      }

      string gt_file = rgb_file;
      gt_file.replace(gt_file.rfind("rgb"), 3, "gt");
      gt_file.replace(gt_file.rfind("png"), 3, "txt");

      if (!boost::filesystem::is_regular_file(gt_file)) {
        cerr << "Nonexistent gt file " << gt_file << endl;
        return -1;
      }

      cout << rgb_file << endl;
      cout << depth_file << endl;
      cout << gt_file << endl << endl;

      vector<string> model_names;
      vector<Eigen::Matrix4f> model_transforms;

      if (!ReadGTFile(gt_file, &model_names, &model_transforms)) {
        cerr << "Invalid gt file " << gt_file << endl;
      }

      const string scene = dataset_it->path().stem().string();
      const string image_num = scene_it->path().stem().string();
      const string scene_output_dir = output_dir.string() + "/" + scene;
      string predictions_file;

      if (FLAGS_perch) {
        predictions_file = output_dir.string() + "/" + scene + "/" +
                           image_num + "_perch.txt";
      } else {
        predictions_file = output_dir.string() + "/" + scene + "/" +
                           image_num + "_predictions.txt";
      }

      if (!boost::filesystem::is_regular_file(predictions_file)) {
        cerr << "Nonexistent gt file " << predictions_file << endl;
        return -1;
      }

      vector<vector<Eigen::Matrix4f>> predicted_transforms;
      vector<string> predicted_model_names;

      if (!ReadPredictionsFile(predictions_file, model_names.size(),
                               &predicted_model_names,
                               &predicted_transforms)) {
        cerr << "Invalid predictions file " << gt_file << endl;
      }

      if (predicted_model_names.size() != model_names.size()) {
        cerr << "Num objects in predictions file " << predicted_model_names.size() <<
             "does not match with gt file objects: " << model_names.size() << endl;
      }

      scenes_file << scene << "_" << image_num << endl;

      for (size_t ii = 0; ii < model_names.size(); ++ii) {
        const string model_stats_file = output_dir.string() + "/" + model_names[ii] +
                                        "_stats.txt";
        std::ofstream file;
        file.open(model_stats_file, std::ofstream::out | std::ofstream::app);

        vector<float> trans_errors(predicted_transforms[ii].size(), 0);
        vector<float> rot_errors(predicted_transforms[ii].size(), 0);

        vector<float> errors(predicted_transforms[ii].size(), 0.0);
        vector<int> error_idxs(predicted_transforms[ii].size());
        std::iota(error_idxs.begin(), error_idxs.end(), 0);

        //  TODO: write object name to file
        const int cutoff = 1;
        const int num_poses = std::min(cutoff,
                                       static_cast<int>(predicted_transforms[ii].size()));
        double ad_error = 0.0;

        for (int jj = 0; jj < num_poses; ++jj) {
          EvaluatePose(model_transforms[ii], predicted_transforms[ii][jj],
                       model_names[ii],
                       &trans_errors[jj],
                       &rot_errors[jj]);
          const auto it = evaluators.find(model_names[ii]);
          const auto &evaluator = it->second;

          errors[jj] = evaluator.SymmetricError(predicted_transforms[ii][jj],
                                                model_transforms[ii]);
          const double diameter = 2.0 * evaluator.CircumscribedRadius();
          cout << diameter << " " << FLAGS_eta *diameter << endl;

          if (errors[jj] < FLAGS_eta * diameter) {
            errors[jj] = 0.0;
          } else {
            errors[jj] = 1.0;
          }

          // errors[jj] = 10 * trans_errors[jj] + rot_errors[jj];
        }

        // TODO: write trans and rot error to file

        // std::ofstream file;
        // file.open(model_stats_file, std::ofstream::out | std::ofstream::app);
        //
        file << scene << "_" << image_num << endl;

        for (int jj = 0; jj < num_poses; ++jj) {
          file << trans_errors[jj];
          // file << errors[jj];

          if (jj != predicted_transforms[ii].size() - 1) {
            file << " ";
          }
        }

        if (num_poses == 0) {
          file << 1000.0;
        }

        file << endl;

        for (int jj = 0; jj < num_poses; ++jj) {
          // file << rot_errors[jj];
          file << errors[jj];

          if (jj != predicted_transforms[ii].size() - 1) {
            file << " ";
          }
        }

        if (num_poses == 0) {
          file << 1000.0;
        }

        file << endl;
        file.close();
      }
    }
  }

  scenes_file.close();

  return 0;
}
