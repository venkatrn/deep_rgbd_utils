#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/utility.hpp>
#include <opencv2/core/core.hpp>

#include <string>
#include <fstream>

// http://stackoverflow.com/questions/4170745/serializing-opencv-mat-vec3f/6311896#6311896
namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive &ar, cv::Mat &mat, const unsigned int) {
  int cols, rows, type;
  bool continuous;

  if (Archive::is_saving::value) {
    cols = mat.cols;
    rows = mat.rows;
    type = mat.type();
    continuous = mat.isContinuous();
  }

  ar &cols &rows &type &continuous;

  if (Archive::is_loading::value) {
    mat.create(rows, cols, type);
  }

  if (continuous) {
    const unsigned int data_size = rows * cols * mat.elemSize();
    ar &boost::serialization::make_array(mat.ptr(), data_size);
  } else {
    const unsigned int row_size = cols * mat.elemSize();

    for (int i = 0; i < rows; i++) {
      ar &boost::serialization::make_array(mat.ptr(i), row_size);
    }
  }

}
} // namespace serialization
} // namespace boost

void saveMat(cv::Mat &m, std::string filename) {
  std::ofstream ofs(filename.c_str());
  boost::archive::binary_oarchive oa(ofs);
  //boost::archive::text_oarchive oa(ofs);
  oa << m;
}

void loadMat(cv::Mat &m, std::string filename) {
  std::ifstream ifs(filename.c_str());
  boost::archive::binary_iarchive ia(ifs);
  //boost::archive::text_iarchive ia(ifs);
  ia >> m;
}
