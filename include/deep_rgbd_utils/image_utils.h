#include <deep_rgbd_utils/helpers.h>

namespace dru {

void balance_white(cv::Mat &mat) {
  if (mat.type() != CV_8UC3) {
    return;
  }

  double discard_ratio = 0.05;
  int hists[3][256];
  memset(hists, 0, 3 * 256 * sizeof(int));

  for (int y = 0; y < mat.rows; ++y) {
    uchar *ptr = mat.ptr<uchar>(y);

    for (int x = 0; x < mat.cols; ++x) {
      for (int j = 0; j < 3; ++j) {
        hists[j][ptr[x * 3 + j]] += 1;
      }
    }
  }

  // cumulative hist
  int total = mat.cols * mat.rows;
  int vmin[3], vmax[3];

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 255; ++j) {
      hists[i][j + 1] += hists[i][j];
    }

    vmin[i] = 0;
    vmax[i] = 255;

    while (hists[i][vmin[i]] < discard_ratio * total) {
      vmin[i] += 1;
    }

    while (hists[i][vmax[i]] > (1 - discard_ratio) * total) {
      vmax[i] -= 1;
    }

    if (vmax[i] < 255 - 1) {
      vmax[i] += 1;
    }
  }

  for (int y = 0; y < mat.rows; ++y) {
    uchar *ptr = mat.ptr<uchar>(y);

    for (int x = 0; x < mat.cols; ++x) {
      for (int j = 0; j < 3; ++j) {
        int val = ptr[x * 3 + j];

        if (val < vmin[j]) {
          val = vmin[j];
        }

        if (val > vmax[j]) {
          val = vmax[j];
        }

        ptr[x * 3 + j] = static_cast<uchar>((val - vmin[j]) * 255.0 /
                                            (vmax[j] - vmin[j]));
      }
    }
  }
}

void auto_contrast(cv::Mat &bgr_image) {
  if (bgr_image.type() != CV_8UC3) {
    return;
  }

  cv::Mat lab_image;
  cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

  // Extract the L channel
  std::vector<cv::Mat> lab_planes(3);
  cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

  // apply the CLAHE algorithm to the L channel
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
  clahe->setClipLimit(4);
  cv::Mat dst;
  clahe->apply(lab_planes[0], dst);

  // Merge the the color planes back into an Lab image
  dst.copyTo(lab_planes[0]);
  cv::merge(lab_planes, lab_image);

  // convert back to RGB
  cv::Mat image_clahe;
  cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
  bgr_image = image_clahe;
}

void saturate(cv::Mat &mat, int delta) {
  if (mat.type() != CV_8UC3) {
    return;
  }

  cv::Mat hsv;
  cv::cvtColor(mat, hsv, CV_BGR2HSV);

  for (int y = 0; y < mat.rows; ++y) {
    uchar *ptr = hsv.ptr<uchar>(y);

    for (int x = 0; x < mat.cols; ++x) {
      int val = ptr[x * 3 + 1];
      val += delta;

      if (val < 0) {
        val = 0;
      }

      if (val > 255) {
        val = 255;
      }

      ptr[x * 3 + 1] = (uchar)val;
    }
  }

  cv::cvtColor(hsv, mat, CV_HSV2BGR);
}
};
