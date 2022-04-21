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
	              const int timeDelta = 2000,
	              const int countThresh = 35000,
	              const float errThresh = 5e-05,
	              const float covThresh = 1e-05,
	              const float photoThresh = 115,
	              const float confidence = 8,
	              const float depthCut = 3,
	              const float icpThresh = 10);

	virtual ~xSurfelFusion();

	void clear();

	void processFrame(cv::Mat& depthImg,
	                  cv::Mat& colorImg,
	                  cv::cuda::GpuMat& dRenderedDepthImg32F,
	                  float& velocity,
	                  cv::cuda::GpuMat& dRawDepthImg32F,
	                  cv::cuda::GpuMat& dFilteredDepthImg,
	                  cv::cuda::GpuMat& dFilteredDepthImg32F,
	                  cv::cuda::GpuMat& dColorImg,
	                  bool objectDetected);

	void computeFeedbackBuffers();

	Eigen::Matrix4f& getCurrPose();

	GlobalModel& getGlobalModel();

private:
	inline void createTextures();

	inline void createCompute();

	inline void createFeedbackBuffers();

	inline void trackCamera();

	inline void dofusion(float weight, int isFrag);

private:
	cv::Mat m_vertexImg;

	IndexMap indexMap;
	FillIn fillIn;
	RGBDOdometry frameToModel;
	GlobalModel globalModel;

	std::map<std::string, GPUTexture*> textures;
	std::map<std::string, ComputePack*> computePacks;
	std::map<std::string, FeedbackBuffer*> feedbackBuffers;

	void filterDepth();
	void metriciseDepth();
	void normaliseDepth(const float& minVal, const float& maxVal);
	Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);

	std::vector<std::pair<unsigned long long int, Eigen::Matrix4f>> poseGraph;

	const int timeDelta;
	const int icpCountThresh;
	const float icpErrThresh;
	const float covThresh;
	const float maxDepthProcessed;
	float icpWeight;
	float confidenceThreshold;
	float fernThresh;
	float depthCutoff;

public:
	Eigen::Matrix4f currPose;
	int64_t& m_timeStamp;
	int& m_fragIdx;

	struct cudaGraphicsResource* m_vboCudaRes;
	VBOType* m_dVboCuda;
	cudaArray* m_textPtr;
};


