#pragma once

#include <opencv2/opencv.hpp>

#include "Helpers/xUtils.h"

void OrbBFMatch(int* dMatchesIdx,
                int* dMatchesDist,
                cv::cuda::GpuMat& src,
                cv::cuda::GpuMat& target);

void FPFHBFMatch(int* dMatchesIdx,
                 float* dMatchesDist,
                 cv::cuda::GpuMat& src,
                 cv::cuda::GpuMat& target);

void NormalizeFPFH(cv::cuda::GpuMat& src,
                   cv::cuda::GpuMat& target);

void CalcFPFHForOrbMatches(float* dMatchesDist,
						   int* dMatchesIdx,
						   int matehesNum,
						   cv::cuda::GpuMat& srcFPFHIdxMat,
						   cv::cuda::GpuMat& targetFPFHIdxMat,
						   cv::cuda::GpuMat& src,
						   cv::cuda::GpuMat& target);