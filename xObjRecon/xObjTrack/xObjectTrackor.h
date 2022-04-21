#pragma once

#include <vector>
#include <opencv2\opencv.hpp>

#include "Helpers/xUtils.h"
#include "xSiftGen.h"
#include "xEdgeGen.h"
#include "GmsGPU/GmsMatcher.h"
#include "xSurfelFusion/Cuda/cudafuncs.cuh"

#define MYEPS 1e-24

class xObjectTrackor
{
public:
	explicit xObjectTrackor(cv::cuda::GpuMat& dFilteredDepthImg32F,
	                        cv::cuda::GpuMat& dRenderedDepthImg32F,
	                        cv::Mat& colorImg,
	                        cv::cuda::GpuMat& dGrayImg);
	~xObjectTrackor();

	cv::cuda::GpuMat& getPruneMaskGpu();
	void detect(Box& objectBox, Box& centerBox, Gravity gravity);
	std::vector<Box>& getCandidatedBoxes();
	void track(Box& box, int64_t timeStamp, Gravity gravity);
	void trackOpt1(Box& box, int64_t timeStamp, Gravity gravity, float &meanDepth2);
	void trackOpt2(Box& box, int64_t timeStamp, Gravity gravity, float &meanDepth2);
	void trackOpt3(Box& box, int64_t timeStamp, Gravity gravity, float &meanDepth2);
	void estimateMinMaxDepth(float &minDepth, float &maxDepth, float &meanDepth, 
													 Box &box, int64_t timeStamp, Gravity gravity);
	void track(Box& box, int64_t timeStamp);
	void mapToGravityCoordinate(Gravity& gravity, float depthThreshold);
	void mapToGravityCoordinateForRenderedDepth(Gravity& gravity, float depthThreshold);
	inline void fitThe3DArea(Box&box, Gravity &gravity);
	inline void fitThe3DArea(Box &box);
	inline void savePly(cv::Mat &vmap, float centerX, float centerY, float centerZ,
		float radSquare,
		float normalX, float normalY, float normalZ,
		float3 axis1, float3 axis2, float3 axis3);
	inline void savePly(cv::Mat &vmap, float centerX, float centerY, float centerZ,
		float radSquare,
		float normalX, float normalY, float normalZ);

private:
	xEdgeGen m_edgeGen;	
	GmsMatcherGPU m_gmsMatcher;
	//xSiftGen m_siftGen;

	char *m_dResultBuf, *m_resultBuf;
	float3* m_dPlaneVertexBuf;
	int* m_dIndex;

	std::vector<cv::Mat> colorImgVec;
	
	cv::Mat& m_colorImg;
	cv::cuda::GpuMat &m_dFilteredDepthImg32F, &m_dRenderedDepthImg32F, &m_dGrayImg;
	cv::Mat m_colorImgVis;

	cv::cuda::GpuMat m_dPruneMat;
	cv::Mat m_pruneMat;
	cv::Mat m_dilateKernel;

	cv::cuda::GpuMat m_dVMap;
	cv::Mat m_vmap;

	std::vector<Box> m_candidateBoxVec;
	std::vector<ushort> m_matchesVec;

	// For visualization
	cv::Mat m_edgeVis, m_edgeVisNext;
	cv::Mat m_catImgVis;
	cv::Mat m_ccImgVis;

#if 1
	DeviceArray2D<unsigned short> m_depthDevice;
	DeviceArray2D<float> m_vmapDevice;
	DeviceArray2D<float> m_nmapDevice;
	std::vector<float> m_vmapVec, m_nmapVec;
	std::vector<float> m_xs, m_ys, m_zs;
	cv::cuda::GpuMat m_pruneMatDevice;
#endif

	float m_zThresh;

	//cv::Mat kernelEllipse;
	float m_meanMaxRadSquare = 0.0f;
	int m_cntMaxRad = 0;
};