#pragma once

#include "GPUTexture.h"
#include "Cuda/cudafuncs.cuh"
#include "OdometryProvider.h"
#include "GPUConfig.h"

#include <vector>
#include <vector_types.h>

class RGBDOdometry
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	RGBDOdometry(int width,
	             int height,
	             float cx, float cy, float fx, float fy,
	             float distThresh = 0.10f,
	             float angleThresh = sin(20.f * 3.14159254f / 180.f));

	virtual ~RGBDOdometry();

	void initICP(GPUTexture* filteredDepth, const float depthCutoff);

	void initICP(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff);

	void initICPModel(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff,
	                  const Eigen::Matrix4f& modelPose);

	void initRGB(GPUTexture* rgb);

	void initRGBModel(GPUTexture* rgb);

	void initFirstRGB(GPUTexture* rgb);

	void getIncrementalTransformation(Eigen::Vector3f& trans,
	                                  Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& rot,
	                                  const float& icpWeight);

	Eigen::MatrixXd getCovariance();

	float lastICPError;
	float lastICPCount;
	float lastRGBError;
	float lastRGBCount;
	float lastSO3Error;
	float lastSO3Count;

	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> lastA;
	Eigen::Matrix<double, 6, 1> lastb;

private:
	void populateRGBDData(GPUTexture* rgb,
	                      DeviceArray2D<float>* destDepths,
	                      DeviceArray2D<unsigned char>* destImages);

	std::vector<DeviceArray2D<unsigned short>> depth_tmp;

	DeviceArray<float> vmaps_tmp;
	DeviceArray<float> nmaps_tmp;

	std::vector<DeviceArray2D<float>> vmaps_g_prev_;
	std::vector<DeviceArray2D<float>> nmaps_g_prev_;

	std::vector<DeviceArray2D<float>> vmaps_curr_;
	std::vector<DeviceArray2D<float>> nmaps_curr_;

	CameraModel intr;

	DeviceArray<JtJJtrSE3> sumDataSE3;
	DeviceArray<JtJJtrSE3> outDataSE3;
	DeviceArray<int2> sumResidualRGB;

	DeviceArray<JtJJtrSO3> sumDataSO3;
	DeviceArray<JtJJtrSO3> outDataSO3;

	const int sobelSize;
	const float sobelScale;
	const float maxDepthDeltaRGB;
	const float maxDepthRGB;

	std::vector<int2> pyrDims;

	static const int NUM_PYRS = 3;

	DeviceArray2D<float> lastDepth[NUM_PYRS];
	DeviceArray2D<unsigned char> lastImage[NUM_PYRS];

	DeviceArray2D<float> nextDepth[NUM_PYRS];
	DeviceArray2D<unsigned char> nextImage[NUM_PYRS];
	DeviceArray2D<short> nextdIdx[NUM_PYRS];
	DeviceArray2D<short> nextdIdy[NUM_PYRS];

	DeviceArray2D<unsigned char> lastNextImage[NUM_PYRS];

	DeviceArray2D<DataTerm> corresImg[NUM_PYRS];

	DeviceArray2D<float3> pointClouds[NUM_PYRS];

	std::vector<int> iterations;
	std::vector<float> minimumGradientMagnitudes;

	float distThres_;
	float angleThres_;

	Eigen::Matrix<double, 6, 6> lastCov;

	const int width;
	const int height;
	const float cx, cy, fx, fy;
};


