#include <deep_rgbd_utils/vertex_net.h>
#include <deep_rgbd_utils/cv_serialization.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <cstdint>

#include <Eigen/Core>

#include <boost/filesystem.hpp>
#include <pangolin/pangolin.h>

#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace dru;

namespace {
map<string, int> kObjectNameToIdx = {
  {"background", 0},
  {"002_master_chef_can", 1},
  {"003_cracker_box", 2},
  {"004_sugar_box", 3},
  {"005_tomato_soup_can", 4},
  {"006_mustard_bottle", 5},
  {"007_tuna_fish_can", 6},
  {"008_pudding_box", 7},
  {"009_gelatin_box", 8},
  {"010_potted_meat_can", 9},
  {"011_banana", 10},
  {"019_pitcher_base", 11},
  {"021_bleach_cleanser", 12},
  {"024_bowl", 13},
  {"025_mug", 14},
  {"035_power_drill", 15},
  {"036_wood_block", 16},
  {"037_scissors", 17},
  {"040_large_marker", 18},
  {"051_large_clamp", 19},
  {"052_extra_large_clamp", 20},
  {"061_foam_brick", 21}
};
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

  const string  kTFProto =
    "/home/venkatrn/research/dense_features/tensorflow/lov_frozen_graph.pb";
  VertexNet vertex_net(kTFProto);

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

        for (const string &object : model_names) {
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


      const string scene_stats_file = output_dir.string() + "/" + scene + "/" +
                                      image_num + "_predictions.txt";
      const string prefix = output_dir.string() + "/" + scene + "/" + image_num +
                            "_";

      cv::Mat img;
      img = cv::imread(rgb_file);
      cv::Mat object_labels, obj_probs, vert_preds, heatmap, probability_map;
      vertex_net.Run(img, obj_probs, vert_preds);
      vertex_net.GetLabelImage(obj_probs, object_labels);

      // const int object_idx = kObjectNameToIdx[target_object];

      cv::imwrite(prefix + "labels.png", object_labels);
      saveMat(obj_probs, prefix + "probs.mat");
      saveMat(vert_preds, prefix + "verts.mat");

      // cv::Mat sliced_verts;
      // vertex_net.SlicePrediction(loaded_mat, vert_preds, object_idx, probability_map,
      //                            sliced_verts);
      // VertexNet::ColorizeProbabilityMap(probability_map, probability_map);
      // cv::imwrite(prefix + "probs.png", probability_map);
    }
  }
  return 0;
}
