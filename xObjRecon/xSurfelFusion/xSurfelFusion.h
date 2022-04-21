#pragma once

#include <utility>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

#include "Helpers/xUtils.h"
#include "RGBDOdometry.h"
#include "GlobalModel.h"
#include "IndexMap.h"
#include "FillIn.h"
#include "GPUTexture.h"

#include "xSurfelFusion/Shaders.h"
#include "xSurfelFusion/ComputePack.h"
#include "xSurfelFusion/FeedbackBuffer.h"

class xSurfelFusion
{
public:
	xSurfelFusion(int64_t& timeStamp,
								int& fragIdx,
								const float icpWeight = 10.0,
								const float rgbWeight = 0.1,
								const float imuWeight = 5.0,
								const int timeDelta = 2000,
								const int countThresh = 35000,
								const float errThresh = 5e-05,
								const float covThresh = 1e-05,
								const float photoThresh = 115,
								const float confidence = 8,
								const float depthCut = 3);

	virtual ~xSurfelFusion();

	void clear();

	void processFrame2(cv::cuda::GpuMat& dRenderedDepthImg32F,
										 float& velocity,
										 cv::cuda::GpuMat& dRawDepthImg,
										 cv::cuda::GpuMat& dFilteredDepthImg,
										 cv::cuda::GpuMat& dRawDepthImg32F,
										 cv::cuda::GpuMat& dFilteredDepthImg32F,
										 cv::cuda::GpuMat& dColorImgRGBA,
										 cv::cuda::GpuMat& dColorImgRGB,
										 ImuMeasurements& imuMeasurements,
										 Gravity& gravityW,
										 bool objectDetected);

	void processFrame(cv::Mat& renderedDepthImg,
										float& velocity,
										cv::Mat& colorImg,
										cv::Mat& depthImg,
										ImuMeasurements& imuMeasurements,
										Gravity& gravityW,
										bool objectDetected);

	void computeFeedbackBuffers();

	Eigen::Matrix4f & getCurrPose();

	GlobalModel & getGlobalModel();

private:
	inline void createTextures();

	inline void createCompute();

	inline void createFeedbackBuffers();

	inline void trackCamera(ImuMeasurements& imuMeasurements,
													Gravity& gravityW);
	inline void trackCamera(ImuMeasurements& imuMeasurements,
													Gravity& gravityW,
													cv::Mat& colorImg);

	inline void dofusion(float weight, int isFrag);

private:
	cv::cuda::GpuMat m_dVMapFloat4;

	IndexMap indexMap;
	FillIn fillIn;
	RGBDOdometry frameToModel;
	GlobalModel globalModel;

	std::map<std::string, GPUTexture*> textures;
	std::map<std::string, ComputePack*> computePacks;
	std::map<std::string, FeedbackBuffer*> feedbackBuffers;

	void filterDepth();
	void metriciseDepth();
	void normaliseDepth(const float & minVal, const float & maxVal);
	Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);

	std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > poseGraph;

	const int timeDelta;
	const int icpCountThresh;
	const float icpErrThresh;
	const float covThresh;
	const float maxDepthProcessed;
	float confidenceThreshold;
	const float depthCutoff;

	float m_icpWeight;
	float m_rgbWeight;
	float m_imuWeight;

	cudaArray_t m_cudaArrayDepthRaw = NULL,
		m_cudaArrayDepthFiltered = NULL,
		m_cudaArrayDepthMetric = NULL,
		m_cudaArrayDepthMetricFiltered = NULL,
		m_cudaArrayRGB = NULL,
		m_cudaArrayVMapFloat4 = NULL;

	std::vector<cv::Mat> m_colorImgForVis;

public:
	Eigen::Matrix4f m_currPose;
	Eigen::Vector3f m_currVelocity, m_currBiasAcc, m_currBiasGyr;
	int64_t& m_timeStamp;
	int& m_fragIdx;

	struct cudaGraphicsResource *m_vboCudaRes;
	VBOType* m_dVboCuda;

	bool m_hasBeenPaused = false;
	bool m_emptyVBO = true;
};


