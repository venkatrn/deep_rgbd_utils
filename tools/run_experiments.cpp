#include <deep_rgbd_utils/pose_estimator.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <cstdint>

#include <Eigen/Core>

#include <boost/filesystem.hpp>

using namespace std;
using namespace dru;

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

void EvaluatePose(const Eigen::Matrix4f &gt_pose, const Eigen::Matrix4f &pose,
                  float *trans_error, float *rot_error) {
  Eigen::Vector3f t_diff = gt_pose.block<3, 1>(0, 3) - pose.block<3, 1>(0, 3);
  // cout << t_diff.transpose() << endl;

  Eigen::Quaternionf q1(gt_pose.block<3, 3>(0, 0));
  Eigen::Quaternionf q2(pose.block<3, 3>(0, 0));

  *trans_error = t_diff.norm();
  *rot_error = q1.angularDistance(q2);
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

int main(int argc, char **argv) {
  if (argc < 3) {
    cerr << "Usage: ./run_experiments <path_to_dataset_folder> <path_to_results_folder> <OPTIONAL: object_name>"
         << endl;
    return -1;
  }

  boost::filesystem::path dataset_dir = argv[1];
  boost::filesystem::path output_dir = argv[2];
  string target_object = "";
  if (argc > 3) {
    target_object = argv[3];
  }

  if (!boost::filesystem::is_directory(dataset_dir)) {
    cerr << "Invalid dataset directory" << endl;
    return -1;
  }

  if (!boost::filesystem::is_directory(output_dir)) {
    cerr << "Invalid output directory" << endl;
    return -1;
  }

  boost::filesystem::directory_iterator dataset_it(dataset_dir), dataset_end;

  PoseEstimator pose_estimator;
  pose_estimator.UseDepth(true);

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

      // Skip this scene if it doesn't contain the target object.
      if (!target_object.empty()) {
        bool scene_contains_target = false;
        for (const string& object : model_names) {
          if (object == target_object) {
            scene_contains_target = true;
            break;
          }
        }
        if (!scene_contains_target) {
          continue;
        }
      }

      const string scene = dataset_it->path().stem().string();
      const string image_num = scene_it->path().stem().string();
      const string scene_output_dir = output_dir.string() + "/" + scene;

      if (!boost::filesystem::is_directory(scene_output_dir)) {
        boost::filesystem::create_directory(scene_output_dir);
      }


      const int num_candidates = 500;
      std::vector<Eigen::Matrix4f> output_transforms;

      std::ofstream scene_file;
      const string scene_stats_file = output_dir.string() + "/" + scene + "/" + image_num + "_predictions.txt";
      scene_file.open(scene_stats_file, std::ofstream::out);
      for (size_t ii = 0; ii < model_names.size(); ++ii) {
        // cout << "true pose " << endl;
        // cout << model_transforms[ii] << endl;
        string im_prefix = image_num + "_" + model_names[ii] ;
        pose_estimator.SetVerbose(scene_output_dir, im_prefix);

        // If we are detecting a specific object, ignore others.
        if (!target_object.empty() && model_names[ii] != target_object) {
          continue;
        }
        output_transforms = pose_estimator.GetObjectPoseCandidates(rgb_file,
                                                                   depth_file, model_names[ii], num_candidates);

        const string model_stats_file = output_dir.string() + "/" + model_names[ii] +
                                        "_stats.txt";

        vector<float> trans_errors(output_transforms.size(), 0);
        vector<float> rot_errors(output_transforms.size(), 0);

        vector<float> errors(output_transforms.size(), 0.0);
        vector<int> error_idxs(output_transforms.size());
        std::iota(error_idxs.begin(), error_idxs.end(), 0);

        for (size_t jj = 0; jj < output_transforms.size(); ++jj) {
          EvaluatePose(model_transforms[ii], output_transforms[jj], &trans_errors[jj],
                       &rot_errors[jj]);
          errors[jj] = 10 * trans_errors[jj] + rot_errors[jj];
        }

        // std::ofstream file;
        // file.open(model_stats_file, std::ofstream::out | std::ofstream::app);

        // for (size_t jj = 0; jj < output_transforms.size(); ++jj) {
        //   file << trans_errors[jj];
        //
        //   if (jj != output_transforms.size() - 1) {
        //     file << " ";
        //   }
        // }
        //
        // file << endl;
        //
        // for (size_t jj = 0; jj < output_transforms.size(); ++jj) {
        //   file << rot_errors[jj];
        //
        //   if (jj != output_transforms.size() - 1) {
        //     file << " ";
        //   }
        //
        // }
        //

        // file << endl;
  
        scene_file << model_names[ii] << endl;
        scene_file << output_transforms.size() << endl;
        for (size_t jj = 0; jj < output_transforms.size(); ++jj) {
          scene_file << output_transforms[jj] << endl;
          // file << output_transforms[jj] << endl;
        }

        // file.close();

        std::sort(error_idxs.begin(), error_idxs.end(), [&errors](int i1, int i2) {
          return errors[i1] < errors[i2];
        });


        if (pose_estimator.Verbose()) {
          Image rgb_img;
          rgb_img.SetImage(rgb_file);
          // string prefix = pose_estimator.Prefix();
          // string pose_im_name = "pose_0.png" ;
          //   PoseEstimator::DrawProjection(rgb_img, pose_estimator.GetModel(model_names[ii]),
          //                                                                  model_transforms[ii],
          //                                                                  prefix + pose_im_name);
          //

          int gt_idx = error_idxs[0];
          string prefix = pose_estimator.Prefix();
          string pose_im_name = "gt.png";
          // cout << output_transforms[gt_idx];
          pose_estimator.DrawProjection(rgb_img, model_names[ii], pose_estimator.GetModel(model_names[ii]),
                                                                         output_transforms[gt_idx],
                                                                         prefix + pose_im_name);
          for (size_t jj = 0; jj < std::min((size_t)5, output_transforms.size()); ++jj) {
            int new_idx = jj;
            string prefix = pose_estimator.Prefix();
            string pose_im_name = "pose_" + to_string(jj) + "_rank_" + to_string(new_idx) +
                                  ".png";
            // cout << output_transforms[new_idx];
            pose_estimator.DrawProjection(rgb_img, model_names[ii], pose_estimator.GetModel(model_names[ii]),
                                                                           output_transforms[new_idx],
                                                                           prefix + pose_im_name);
          }
        }
      }
      scene_file.close();
    }

  }

  return 0;
}
