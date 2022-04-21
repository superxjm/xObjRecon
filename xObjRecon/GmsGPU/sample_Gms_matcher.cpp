#include "stdafx.h"

#if 0

#define USE_GPU

#include <cuda_runtime.h>
#include "GmsMatcherGPU.h"
#include <innoreal/utils/innoreal_timer.hpp>

using namespace std;
using namespace cv;

int main() {
//  cv::setUseOptimized(true);

#ifdef USE_GPU
  int flag = cuda::getCudaEnabledDeviceCount();
  if (flag != 0) {
    cuda::setDevice(0);
  }
#endif // USE_GPU

  cv::setUseOptimized(1);

  Mat img1 = imread("../data/0.png");
//  Mat img1 = imread("/home/vradmin/workspace/SimpleView_FetchFrame/cmake-build-release/save/color_20.png");
  Mat img2 = imread("../data/1.png");
//  Mat img2 = imread("/home/vradmin/workspace/SimpleView_FetchFrame/cmake-build-release/save/color_22.png");

  cv::Mat match_idx, show;

  innoreal::InnoRealTimer timer;
  timer.TimeStart();
  GmsMatcherGPU gms_matcher(img1.rows, img1.cols);
  timer.TimeEnd();
  printf("time of initializing GmsMatcherGPU: %.6f ms \n", timer.TimeGap_in_ms());

  timer.TimeStart();
  match_idx = gms_matcher.GetMatchesGPU(img1, img2);

  timer.TimeEnd();
  printf("time of get matches: %.6f ms \n", timer.TimeGap_in_ms());

  show = gms_matcher.DrawMatches(img1, img2, match_idx);
  cv::imshow("show matches", show);
  cv::waitKey();

  return 0;
}

#endif
