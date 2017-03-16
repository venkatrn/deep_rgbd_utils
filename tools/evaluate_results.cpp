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
  cout << t_diff.transpose() << endl;

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
      const string predictions_file = output_dir.string() + "/" + scene + "/" +
                                      image_num + "_predictions.txt";

      if (!boost::filesystem::is_regular_file(predictions_file)) {
        cerr << "Nonexistent gt file " << predictions_file << endl;
        return -1;
      }

      vector<vector<Eigen::Matrix4f>> predicted_transforms;
      vector<string> predicted_model_names;

      if (!ReadPredictionsFile(predictions_file, model_names.size(), &predicted_model_names,
                               &predicted_transforms)) {
        cerr << "Invalid predictions file " << gt_file << endl;
      }

      if (predicted_model_names.size() != model_names.size()) {
        cerr << "Num objects in predictions file " << predicted_model_names.size() <<
             "does not match with gt file objects: " << model_names.size() << endl;
      }

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

        for (size_t jj = 0; jj < predicted_transforms[ii].size(); ++jj) {
          EvaluatePose(model_transforms[ii], predicted_transforms[ii][jj],
                       &trans_errors[jj],
                       &rot_errors[jj]);
          errors[jj] = 10 * trans_errors[jj] + rot_errors[jj];
        }

        // TODO: write trans and rot error to file

        // std::ofstream file;
        // file.open(model_stats_file, std::ofstream::out | std::ofstream::app);
        //
        for (size_t jj = 0; jj < predicted_transforms[ii].size(); ++jj) {
          file << trans_errors[jj];

          if (jj != predicted_transforms[ii].size() - 1) {
            file << " ";
          }
        }

        file << endl;

        for (size_t jj = 0; jj < predicted_transforms[ii].size(); ++jj) {
          file << rot_errors[jj];

          if (jj != predicted_transforms[ii].size() - 1) {
            file << " ";
          }
        }

        file << endl;
        file.close();
      }
    }
  }

  return 0;
}
