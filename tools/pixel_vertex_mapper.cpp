#include <deep_rgbd_utils/feature_generator_net.h>
#include <deep_rgbd_utils/helpers.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <cstdint>

using namespace std;
using namespace dru;
using namespace flann;

namespace {
const int kNumNeighbors = 5;
const int kMinObservations = 100;
// const string kCoffeeMugMeans = "/home/venkatrn/research/YCB/025_mug/coffee_mug_0004.means";
const string kCoffeeMugMeans = "/home/venkatrn/research/YCB/025_mug/coffee_mug.means";
} // namespace

// Yuck, create a class.
cv::Mat img1, img2, heatmap;
std::vector<std::vector<float>> features1, features2, coffee_mug_features;
std::vector<int> coffee_mug_counts;
std::unique_ptr<flann::Index<L2<float>>> image1_index;
std::unique_ptr<flann::Index<L2<float>>> coffee_mug_index;
pcl::PointCloud<pcl::PointXYZ>::Ptr coffee_mug_cloud(new pcl::PointCloud<pcl::PointXYZ>);
PointCloudPtr colored_cloud_(new PointCloud);
pcl::visualization::PCLVisualizer *viewer;


PointCloudPtr GetModelHeatmap(const std::vector<float>& feature, 
                              pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud, 
                              const std::vector<std::vector<float>>& model_features,
                              const std::vector<int>& num_observations) {
  PointCloudPtr colored_cloud(new PointCloud);
  copyPointCloud(*model_cloud, *colored_cloud);
  if (model_cloud->points.size() != model_features.size()) {
    printf("Number of model points is not identical to number of model features!\n");
    return colored_cloud;
  }
  if (model_cloud->points.size() != num_observations.size()) {
    printf("Number of model points is not identical to number of observations!\n");
    return colored_cloud;
  }

  L2<float> euclidean_dist;
  flann::Matrix<float> query_matrix;
  flann::Matrix<float> model_point_matrix;
  query_matrix = VectorToFlann<float>(feature);
  vector<double> dists(model_cloud->points.size(), 0.0);
  double normalizer = 0.0;
  double min_dist = std::numeric_limits<double>::max();
  double max_dist = std::numeric_limits<double>::lowest();
  std::vector<float> zero_vec(feature.size(), 0.0);
  for (size_t ii = 0; ii < model_cloud->points.size(); ++ii) {
      if (num_observations[ii] < kMinObservations) {
        // Hack.
        dists[ii] = 1e10;
      } else {
        model_point_matrix = VectorToFlann<float>(model_features[ii]);
        // dists[ii] = exp(-1e-2 * static_cast<double>(
        //       sqrt(euclidean_dist(query_matrix[0], model_point_matrix[0], features.size()))));
        // dists[ii] = exp(-1e-2 * static_cast<double>(EuclideanDist<float>(feature, model_features[ii])));
        dists[ii] = static_cast<double>(EuclideanDist<float>(feature, model_features[ii]));
        // const float f_norm = EuclideanDist(model_features[ii], zero_vec);
        // printf("FNorm: %f\n", f_norm);
        // printf("F1\n");
        // for (size_t jj = 0; jj < 10; ++jj) {
        //   printf("%f ,", model_features[ii][jj]);
        // }
        // printf("\nF2\n");
        // printf("\n");
        // for (size_t jj = 0; jj < 10; ++jj) {
        //   printf("%f ,", query_matrix[jj]);
        //   printf("%f ,", query_matrix[jj]);
        // }
        // printf("\n");
        // for (size_t jj = 0; jj < 10; ++jj) {
        //   printf("%f ,", feature[jj]);
        // }
        // printf("\n");
        // printf("Dist: %f\n", dists[ii]);
        normalizer += dists[ii];
        min_dist = std::min(min_dist, dists[ii]);
        max_dist = std::max(max_dist, dists[ii]);
      }
  }
  if (normalizer < 1e-5) {
    printf("Normalizer is 0!\n");
    return colored_cloud;
  }
  for (size_t ii = 0; ii < model_cloud->points.size(); ++ii) {
    // Color color = GetColor(dists[ii], min_dist, max_dist);
    // Color color = GetColor(max_dist - dists[ii], 0.0, max_dist - min_dist);
    Color color = GetColor(dists[ii], min_dist, max_dist);
    colored_cloud->points[ii].r = 255 * color.r;
    colored_cloud->points[ii].g = 255 * color.g;
    colored_cloud->points[ii].b = 255 * color.b;
    // printf("Color: %f %f %f\n", color.r, color.g, color.b);
  }
  return colored_cloud;
}

void CVCallback(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == cv::EVENT_LBUTTONDOWN )
     {
        cv::Mat disp_image1 = img1.clone();
        cv::Mat disp_image2 = img2.clone();

        cv::Point point(x,y);
        cv::circle(disp_image2, point, 5, cv::Scalar(0,255,0), -1);
        // printf("Feature vector for (%d,%d:)\n", x, y);
        int pixel_index = x * img1.rows + y;
        auto feature = features2[pixel_index];

        flann::Matrix<float> query_matrix;
        query_matrix = VectorToFlann<float>(feature);
        std::vector<std::vector<int>> k_indices(1);
        k_indices[0].resize(kNumNeighbors);
        std::vector<std::vector<float>> k_distances(1);
        k_distances[0].resize(kNumNeighbors);
        image1_index->knnSearch(query_matrix, k_indices, k_distances, kNumNeighbors, flann::SearchParams(-1));

        // for (size_t ii = 0; ii < feature.size(); ++ii) {
        //   cout << feature[ii] << " ";
        // }
        // cout << "\n";

        // Generate heatmap.
        for (int row = 0; row < heatmap.rows; ++row) {
          auto row_ptr = heatmap.ptr<double>(row);
          for (int col = 0; col < heatmap.cols; ++col) {
            int pixel_idx = col * heatmap.rows + row;
            const float dist = EuclideanDist(features1[pixel_idx], feature);
            row_ptr[col] = dist;
          }
        }
        // printf("Colorizing\n");
        cv::Mat normalized, img_color;
        cv::exp(-1e-10 * heatmap, heatmap);
        cv::normalize(heatmap, normalized, 0, 255, cv::NORM_MINMAX);
        normalized.convertTo(normalized, CV_8UC1);
        cv::applyColorMap(normalized, img_color, cv::COLORMAP_JET);

        // printf("Getting maxima\n");
        const auto& maxima = GetLocalMaxima(heatmap);
        // for (size_t ii = 0; ii < k_indices[0].size(); ++ii) {
        //   const cv::Point center(k_indices[0][ii] / img1.rows, k_indices[0][ii] % img1.rows);
        //   cv::circle(disp_image1, center, 5, cv::Scalar(255,0,0), -1);
        // }
        // printf("Drawing circles\n");
        const int num_maxima = 5;
        for (size_t ii = 0; ii < num_maxima && ii < maxima.size(); ++ii) {
          const cv::Point center(maxima[ii].first.x, maxima[ii].first.y);
          cv::circle(disp_image1, center, num_maxima - ii + 2, cv::Scalar(255,0,0), -1);
        }

        cv::imshow("image_1", disp_image1);
        cv::imshow("image_2", disp_image2);
        const double blend_alpha = 0.5;
        cv::imshow("heatmap", blend_alpha * img_color + (1 - blend_alpha) * disp_image1);

        // printf("Getting colored cloud\n");
        colored_cloud_ = GetModelHeatmap(feature, coffee_mug_cloud, coffee_mug_features, coffee_mug_counts);
        
     }
     else if  ( event == cv::EVENT_RBUTTONDOWN ) {
     }
     else if  ( event == cv::EVENT_MBUTTONDOWN ) {
     }
     else if ( event == cv::EVENT_MOUSEMOVE ) {
     }
}

int main(int argc, char** argv) {
   if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " img1.jpg img2.jpg" <<  std::endl;
    return 1;
  }


  // Bring up PCL viewer.
  viewer = new pcl::visualization::PCLVisualizer("PCL Viewer");
  viewer->removeAllPointClouds();
  viewer->removeAllShapes();

  // coffee_mug_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
  ReadModelFeatures(kCoffeeMugMeans, coffee_mug_cloud, &coffee_mug_features, &coffee_mug_counts);
  // Build NN index for features of the 3D model.
  // BuildKDTreeIndex(model_features, coffee_mug_index);

  string model_file   = argv[1];
  string trained_file = argv[2];
  unique_ptr<FeatureGenerator> generator;
  generator.reset(new FeatureGenerator(model_file, trained_file));

  string img1_file = argv[3];
  string img2_file = argv[4];

  std::cout << "---------- Prediction for "
            << img1_file << " ----------" << std::endl;

  img1 = cv::imread(img1_file, -1);
  CHECK(!img1.empty()) << "Unable to read image1 " << img1_file;
  img2 = cv::imread(img2_file, -1);
  CHECK(!img2.empty()) << "Unable to read image2 " << img1_file;
  
  heatmap.create(img1.rows, img1.cols, CV_64FC1);

  // Run the feature generator.
  features1 = generator->GetFeatures(img1);
  features2 = generator->GetFeatures(img2);

  // Build NN index for features of image 1.
  BuildKDTreeIndex(features1, image1_index);
  
  cv::namedWindow("image_1");
  cv::namedWindow("image_2");
  cv::namedWindow("heatmap");

  const int im1_x = 100;
  const int im1_y = 100;
  const int padding = 30;
  cv::moveWindow("image_1", im1_x, im1_y);
  cv::moveWindow("image_2", im1_x + img1.cols + padding, im1_y);
  cv::moveWindow("heatmap", im1_x + img1.cols / 2 + padding / 2, im1_y + img1.rows + padding);

  cv::imshow("image_1", img1);
  cv::imshow("image_2", img2);
  cv::imshow("heatmap", img1);

  cv::setMouseCallback("image_2", CVCallback, NULL);


  while (1) {
    if (!viewer->updatePointCloud(colored_cloud_, "colored_cloud")) {
      viewer->addPointCloud(colored_cloud_, "colored_cloud");
      viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "colored_cloud");
    }
    viewer->spinOnce();
    cv::waitKey(10);
  }

  delete viewer;
  return 0;
}
