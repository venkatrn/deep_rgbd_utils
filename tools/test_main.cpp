#include <deep_rgbd_utils/pose_estimator.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <cstdint>

#include <Eigen/Core>

#include <boost/filesystem.hpp>
#include <pangolin/pangolin.h>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/freeglut_std.h>

using namespace std;
using namespace dru;

int main(int argc, char **argv) {
  // pangolin::CreateWindowAndBind("aaa", 1280, 960);
  // cout << "Created context\n";
  // cout << flush;
  // for (int ii = 0; ii < 10000; ++ii) {
  //   cout << ii << endl;
  // }
  // glutInit(&argc, argv);
  // glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
  // glutCreateWindow("aaa");
  // pangolin::CreateWindowAndBind("ycb pose estimation - ", 1280, 960);

  PoseEstimator pose_estimator;
  // pose_estimator.UseDepth(true);
  cout << "created pest\n";
  int a;
  cout << "enter\n";
  cin >> a;
  // pangolin::CreateWindowAndBind("aaa", 1280, 960);
  // cout << "Created context\n";
  return 0;
}
