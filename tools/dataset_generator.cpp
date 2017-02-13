#include <iostream>
#include <vector>

#include <cstdint>

#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <df/camera/rig.h>
#include <dataset_manager.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace std;

string rig_file = "";

// bool DatasetManager::ParseRigFile(const std::string &rig_file,
//                                   Sophus::SE3d &T_cd) {
//   std::ifstream rigStream(rig_file);
//   pangolin::json::value val;
//   rigStream >> val;
//
//   if (!val.contains("rig")) {
//     throw std::runtime_error("could not find rig");
//   }
//
//   pangolin::json::value rigVal = val["rig"];
//
//   if (!rigVal.contains("camera")) {
//     throw std::runtime_error("could not find camera");
//   }
//
//   rig.reset(new df::Rig<double>(rigVal));
//
//   if (rig->numCameras() != 2) {
//     throw std::runtime_error("expected a rig configuration with 2 cameras (RGB + depth)");
//   }
//
//
//   const Sophus::SE3d T_rd = rig->transformCameraToRig(depthStreamIndex);
//   const Sophus::SE3d T_rc = rig->transformCameraToRig(colorStreamIndex);
//   T_cd = T_rc * T_rd.inverse();
//   return true;
// }

// template <typename Derived>
// inline std::istream &operator >>(std::istream &stream,
//                                  Eigen::MatrixBase<Derived> &M) {
//   for (int r = 0; r < M.rows(); ++r) {
//     for (int c = 0; c < M.cols(); ++c) {
//       if (! (stream >> M(r, c))) {
//         return stream;
//       }
//     }
//   }
//
//   // Strip newline character.
//   if (stream.peek() == 10) {
//     stream.ignore(1, '\n');
//   }
//
//   return stream;
// }

// Save the depth and RGB frames from *.pango video to the respective
// directories, and the frame-wise ground truths in gt_dir. Frames will equispaced based on
// num_frames, if negative, we will use all frames.
bool SaveFrames(const string &video_file, const string &ground_truth,
                const string &rgb_dir, const string &depth_dir, const string &gt_dir,
                int num_frames_to_save = -1) {
  // TODO: read from rig
  Sophus::SE3d T_cd;
  std::unique_ptr<df::Rig<double>> rig;
  ycb::DatasetManager::ParseRigFile(rig_file, rig, T_cd);

  pangolin::VideoInput video(video_file);
  pangolin::VideoPlaybackInterface *playback =
    video.Cast<pangolin::VideoPlaybackInterface>();

  // check video
  if (video.Streams().size() != 2) {
    printf("Expected two-stream video (RGB + depth)");
    return false;
  }

  const int colorStreamIndex = 0;
  const int depthStreamIndex = 1;

  const pangolin::StreamInfo &colorStreamInfo =
    video.Streams()[colorStreamIndex];
  const pangolin::StreamInfo &depthStreamInfo =
    video.Streams()[depthStreamIndex];

  std::vector<unsigned char> videoBuffer(colorStreamInfo.SizeBytes() +
                                         depthStreamInfo.SizeBytes());

  const int total_frames = playback->GetTotalFrames();

  // Read ground truth file.
  std::ifstream stream(ground_truth);

  Eigen::Matrix<double, 3, 4> M;
  std::string model_name;

  std::vector<Sophus::SE3d> T_wd;
  std::vector<Sophus::SE3d> T_wo;
  std::vector<string> model_names;

  // bool model_poses_done = !(isalpha(stream.peek()));
  //
  // while (!model_poses_done) {
  //   stream >> model_name;
  //   stream >> M;
  //
  //   model_names.push_back(model_name);
  //   T_wo.emplace_back(Sophus::SO3d(M.block<3, 3>(0, 0)),
  //                     M.block<3, 1>(0, 3));
  //   model_poses_done = !(isalpha(stream.peek()));
  //   cout << model_name << endl << M << endl;
  // }

  while (stream >> model_name && model_name.compare("camera") != 0) {
    stream >> M;
    model_names.push_back(model_name);
    T_wo.emplace_back(Sophus::SO3d(M.block<3, 3>(0, 0)),
                      M.block<3, 1>(0, 3));
    // cout << model_name << endl << M << endl;
  }

  T_wd.reserve(total_frames);

  while (stream >> M) {
    T_wd.emplace_back(Sophus::SO3d(M.block<3, 3>(0, 0)),
                      M.block<3, 1>(0, 3));
  }

  if (T_wd.size() != total_frames) {
    printf("We have %zu transforms and %d frames -- mismatch\n", T_wd.size(),
           total_frames);
    return false;
  }

  const int num_models = model_names.size();
  // For each frame, we have num_object transforms.
  std::vector<std::vector<Eigen::Matrix4f>> T_co(total_frames,
                                                 vector<Eigen::Matrix4f>(num_models));

  for (int ii = 0; ii < total_frames; ++ii) {
    for (int jj = 0; jj < num_models; ++jj) {
      // const Sophus::SE3d pose = T_wc[ii].inverse() * T_wo[jj];
      // T_co[ii][jj] = pose.matrix().cast<float>();
      const Sophus::SE3d T_do = T_wd[ii].inverse() * T_wo[jj];
      const Sophus::SE3d pose = T_cd * T_do;
      // const Sophus::SE3d pose = T_do;
      T_co[ii][jj] = pose.matrix().cast<float>();
    }
  }

  // Write frame-wise images and ground truth to disk.

  if (num_frames_to_save < 0 || num_frames_to_save > total_frames) {
    num_frames_to_save = total_frames;
  }

  for (int ii = 0; ii < num_frames_to_save; ++ii) {
    int frame = (total_frames * ii) / num_frames_to_save;
    playback->Seek(frame);
    video.GrabNext(videoBuffer.data());
    unsigned char *raw_depth = videoBuffer.data() + (uint64_t)
                               depthStreamInfo.Offset();
    unsigned char *raw_rgb = videoBuffer.data() + (uint64_t)
                             colorStreamInfo.Offset();

    pangolin::Image<unsigned char> rgb_image(colorStreamInfo.Width(),
                                             colorStreamInfo.Height(), colorStreamInfo.Pitch(), raw_rgb);
    pangolin::Image<unsigned char> depth_image(depthStreamInfo.Width(),
                                             depthStreamInfo.Height(), depthStreamInfo.Pitch(), raw_depth);
    pangolin::SaveImage(rgb_image, colorStreamInfo.PixFormat(), string(rgb_dir +
                                                                       "/"
                                                                       + to_string(ii) + ".png"));
    pangolin::SaveImage(depth_image, depthStreamInfo.PixFormat(),
                        string(depth_dir +
                               "/"
                               + to_string(ii) + ".png"));
    string gt_file = gt_dir + "/" + to_string(ii) + ".txt";
    cout << gt_file;
    std::ofstream gt_stream;
    gt_stream.open(gt_file, std::ofstream::out | std::ofstream::trunc);
    gt_stream << num_models << endl;
    for (int jj = 0; jj < num_models; ++jj) {
      gt_stream << model_names[jj] << endl;
      gt_stream << T_co[frame][jj] << endl;
    }
    gt_stream.close();
  }
  return true;
}

bool SaveFrames(const string& video_file_str, const string& gt_file_str, const string& output_dir_str, int num_frames_per_video) {
  boost::filesystem::path video_file = video_file_str;
  boost::filesystem::path gt_file = gt_file_str;
  boost::filesystem::path output_dir = output_dir_str;

  if (!boost::filesystem::is_regular_file(video_file)) {
    cerr << "Invalid video file" << endl;
    return false;
  }

  if (!boost::filesystem::is_regular_file(gt_file)) {
    cerr << "Invalid poses file" << endl;
    return false;
  }

  if (!boost::filesystem::is_directory(output_dir)) {
    boost::filesystem::create_directory(output_dir);
  }
  output_dir = boost::filesystem::canonical(output_dir);

  const string rgb_dir = output_dir.string() + "/rgb";
  std::cout << rgb_dir << endl;
  if (!boost::filesystem::is_directory(rgb_dir)) {
    boost::filesystem::create_directory(rgb_dir);
  }

  const string depth_dir = output_dir.string() + "/depth";
  if (!boost::filesystem::is_directory(depth_dir)) {
    boost::filesystem::create_directory(depth_dir);
  }

  const string gt_dir = output_dir.string() + "/gt";
  if (!boost::filesystem::is_directory(gt_dir)) {
    boost::filesystem::create_directory(gt_dir);
  }

  return SaveFrames(video_file_str, gt_file_str, rgb_dir, depth_dir, gt_dir, num_frames_per_video);
}

int main (int argc, char **argv) {
  if (argc < 4) {
    cerr << "Usage: ./dataset_generator <path_to_videos_folder> <rig_file> <path_to_output_dir>"
         << endl;
    return -1;
  }
  boost::filesystem::path videos_dir = argv[1];
  rig_file = argv[2];
  boost::filesystem::path output_dir = argv[3];

  if (!boost::filesystem::is_directory(videos_dir)) {
    cerr << "Invalid videos directory" << endl;
    return -1;
  }
  if (!boost::filesystem::is_directory(output_dir)) {
    cerr << "Invalid output directory" << endl;
    return -1;
  }

  boost::filesystem::directory_iterator dir_it(videos_dir), dir_end;

  for (dir_it; dir_it != dir_end; ++dir_it) {
    // Skip non-video folders (assuming every folder that contains "00" is video folder).
    if (dir_it->path().filename().string().find("00") == std::string::npos) {
      continue;
    }
    const string scene = dir_it->path().stem().string();
    const string scene_output_dir = output_dir.string() + "/" + scene;
    const string scene_gt_file = dir_it->path().string() + "/state.txt";
    const string scene_video_file = dir_it->path().string() + "/video.pango";
    cout << scene << endl;
    cout << scene_output_dir << endl;
    cout << scene_gt_file << endl;
    cout << scene_video_file << endl;
    if (!SaveFrames(scene_video_file, scene_gt_file, scene_output_dir, 10)) {
      return -1;
    }
  }
  return 0;
}
