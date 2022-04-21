#pragma once

#include <vector>
#include <vector_types.h>

#include "GPUTexture.h"
#include "Cuda/cudafuncs.cuh"
#include "OdometryProvider.h"
#include "GPUConfig.h"
#include "IntegrationBase.h"
#include "GmsGPU/GmsMatcher.h"

class RGBDOdometry
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	RGBDOdometry(int width,
	             int height,
	             float cx, float cy, float fx, float fy,
	             float distThresh = 0.02f,
	             float angleThresh = sin(30.f * 3.14159254f / 180.f));

	virtual ~RGBDOdometry();

	void initICP(GPUTexture* filteredDepth, const float depthCutoff);

	void initICP(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff);

	void initICPModel(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff,
	                  const Eigen::Matrix4f& modelPose);

	void initRGB(GPUTexture* filteredDepthFloat32, GPUTexture* rgb);

	void initRGBModel(GPUTexture* rgb);

	void initFirstRGB(GPUTexture* rgb);

	void saveForTest(Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RPrev,
	                 Eigen::Vector3f& tPrev,
	                 Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RCurr,
	                 Eigen::Vector3f& tCurr,
	                 const char* saveDir);

	void calcOrbFPFHFeatures(std::vector<float4>& compressedVMap,
	                         std::vector<float4>& compressedNMap,
	                         DeviceArray2D<float>& dDepthImg,
	                         DeviceArray2D<unsigned char>& dColorImg,
	                         DeviceArray2D<float>& dVMap,
													 DeviceArray2D<float>& dNMap, int n,int& vexNum);

	void getIncrementalTransformation(Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RPrev,
	                                  Eigen::Vector3f& tPrev,
	                                  Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RCurr,
	                                  Eigen::Vector3f& tCurr,
	                                  float icpWeight,
	                                  float rgbWeight,
	                                  float imuWeight,
	                                  Eigen::Vector3f& velocity,
	                                  Eigen::Vector3f& biasAcc,
	                                  Eigen::Vector3f& biasGyr,
	                                  ImuMeasurements& imuMeasurements,
	                                  Gravity& gravityW, bool hasBeenPaused,int& matchesNum);
	void getIncrementalTransformation2(Eigen::Vector3f& trans,
	                                   Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& rot,
	                                   const float& icpWeight,
	                                   const float& imuWeight,
	                                   Eigen::Vector3f& velocity,
	                                   Eigen::Vector3f& biasAcc,
	                                   Eigen::Vector3f& biasGyr,
	                                   ImuMeasurements& imuMeasurements,
	                                   Gravity& gravityW);
	void getIncrementalTransformationOrbFPFH(Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RPrev,
	                                         Eigen::Vector3f& tPrev,
	                                         Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RCurr,
	                                         Eigen::Vector3f& tCurr,
																					 int& vexNum,
	                                         std::vector<cv::Mat>* colorImgForVis = NULL);

	Eigen::MatrixXd getCovariance();

	float lastICPError;
	float lastICPCount;
	float lastRGBError;
	float lastRGBCount;
	float lastSO3Error;
	float lastSO3Count;

	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> lastA66;
	Eigen::Matrix<double, 6, 1> lastb61;

	Eigen::Matrix<double, 15, 15, Eigen::RowMajor> lastA1515;
	Eigen::Matrix<double, 15, 1> lastb151;

private:
	void populateRGBDData(GPUTexture* rgb,
	                      DeviceArray2D<float>* destDepths,
	                      DeviceArray2D<unsigned char>* destImages);
    void populateRGBDData2(GPUTexture* rgb,
                           DeviceArray2D<float>* destDepths,
                           DeviceArray2D<unsigned char>* destImages);

	std::vector<DeviceArray2D<unsigned short>> depth_tmp;

	DeviceArray<float> vmaps_tmp;
	DeviceArray<float> nmaps_tmp;

	std::vector<DeviceArray2D<float>> vmaps_g_prev_;
	std::vector<DeviceArray2D<float>> nmaps_g_prev_;

	std::vector<DeviceArray2D<float>> vmaps_curr_;
	std::vector<DeviceArray2D<float>> nmaps_curr_;

	float4* m_dCompressedVMap, *m_dCompressedNMap;
	int* m_dIndexBuf, *m_dKeyPoints, *m_keyPoints;
	pcl::PointCloud<pcl::PointXYZ>::Ptr m_pointCloud;
	pcl::PointCloud<pcl::Normal>::Ptr m_pointNormal;
	cv::cuda::GpuMat m_dGrayImg, m_dDepthImg, m_dMask;

	CameraModel intr;

	DeviceArray<JtJJtrSE3> sumDataSE3;
	DeviceArray<JtJJtrSE3> outDataSE3;
	DeviceArray<int2> sumResidualRGB;

	DeviceArray<JtJJtrSO3> sumDataSO3;
	DeviceArray<JtJJtrSO3> outDataSO3;

	DeviceArray<float2> sumDataFeatureEstimation;
	DeviceArray<float2> outDataFeatureEstimation;

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
	DeviceArray2D<short> lastdIdx[NUM_PYRS];
	DeviceArray2D<short> lastdIdy[NUM_PYRS];

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

	IntegrationBase m_integrationBase;
	GmsMatcherGPU m_gmsMatcher;
	std::vector<ushort> m_matchesVecOrb;
	std::vector<ushort> m_matchesVecFPFH;
	ushort* m_dMatches;
	ushort* m_dMatchesOrb;
	ushort* m_dMatchesFPFH;

	std::vector<float4> m_compressedVMapPrev, m_compressedNMapPrev;
	std::vector<float4> m_compressedVMapCurr, m_compressedNMapCurr;
};


