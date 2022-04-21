#ifndef OBJ_TRACK_CUDA_CUDAFUNCS_CUH_
#define OBJ_TRACK_CUDA_CUDAFUNCS_CUH_

#include "opencv2/opencv.hpp"
#include "Helpers/containers/device_array.hpp"
#include "Helpers/xUtils.h"
#include "Helpers/UtilsMath.h"

void RemoveNonObjectPixels2(DeviceArray2D<unsigned short> &depthDevice, DeviceArray2D<float> &vmapDevice,
	cv::cuda::GpuMat &pruneMatDevice,
	float centerX, float centerY, float centerZ, float normalX, float normalY, float normalZ, float radSquareThres);

void RemoveNonObjectPixels(cv::cuda::GpuMat& dDepth,
                           cv::cuda::GpuMat& dVMap,
                           cv::cuda::GpuMat& dPruneMat,
                           float3& median, float radSquareThesh,
                           float3& zAxis, float& zThresh);

void RemoveNonObjectPixels(cv::cuda::GpuMat& dDepth,
                           cv::cuda::GpuMat& dVMap,
                           float3& median, float radSquareThesh,
                           float3& zAxis, float& zThresh);

void RemoveNonObjectPixels2(float middleX, float middleY, float middleZ,
                            int left, int right, int top, int bottom,
							cv::cuda::GpuMat& dDepth,
                            cv::cuda::GpuMat& dVMap,
                            float3& zAxis, float& zThresh);

void RemoveVerticesOutOfRange(cv::cuda::GpuMat& dVMap,
                              float medianX, float medianY, float medianZ,
                              float lowRadSquare, float hightRadSquare);

bool CalcMinMeanDepth(float& minDepth, float& meanDepth, cv::cuda::GpuMat& dVMapRoi, float depthThreshold,
                                 char* dResultBuf, char* resultBuf, bool withGravity);
#if 0
bool CalcMinMeanDepth(float& minDepth, float& meanDepth, cv::cuda::GpuMat& dDpethRoi,
                      char* dResultBuf, char* resultBuf);
#endif

#if 0
void CalcEdgeIdx(cv::cuda::GpuMat& dEdgeIdx, cv::cuda::GpuMat& dMask,
                 cv::cuda::GpuMat& dFiltereddepthImg32F, float minDepth, float maxDepth, float magThresholdSquare,
                 int validBoxLeft, int validBoxRight, int validBoxTop, int validBoxBottom);
#endif
void CalcEdgeIdx(cv::cuda::GpuMat& dEdgeIdx, cv::cuda::GpuMat& dEdge, cv::cuda::GpuMat& dMask,
                 cv::cuda::GpuMat& dFiltereddepthImg32F, cv::cuda::GpuMat& dVMap, Box& box,
                 float minDepth, float maxDepth, float magThresholdSquare,
                 int validBoxLeft, int validBoxRight, int validBoxTop, int validBoxBottom, bool withGravity);
void CalcEdgeIdxOptWithoutBox(cv::cuda::GpuMat& dEdgeIdx, cv::cuda::GpuMat& dEdge, cv::cuda::GpuMat& dMask,
							  cv::cuda::GpuMat& dFiltereddepthImg32F, cv::cuda::GpuMat& dVMap,
							  float minDepth, float maxDepth, float magThresholdSquare,
							  int validBoxLeft, int validBoxRight, int validBoxTop, int validBoxBottom);
void CalcEdgeIdxOptWithBox(cv::cuda::GpuMat& dEdgeIdx, cv::cuda::GpuMat& dEdge, cv::cuda::GpuMat& dMask,
						   cv::cuda::GpuMat& dFiltereddepthImg32F, cv::cuda::GpuMat& dVMap, Box& box,
						   float minDepth, float maxDepth, float magThresholdSquare);


void CreateVMap(cv::cuda::GpuMat& vmap, const cv::cuda::GpuMat& depth,
                float fx, float fy, float cx, float cy,
                const float depthCutoff, float3 RX, float3 RY, float3 RZ);

void RotateVertices(cv::cuda::GpuMat& vmap, float3 RX, float3 RY, float3 RZ);

void CalcMedianAndRadius(float3& median, float& radiusSquare, cv::cuda::GpuMat& dVMapRoi,
                         char* dResultBuf, char* resultBuf);

bool CalcAxisForPlane2(float& middleX, float& middleY, float& middleZ, float3& zAxis, float& zThresh,
											 cv::cuda::GpuMat& dVMap,
											 float minDepth, float maxDepth,
											 int left, int right, int top, int bottom,
											 float3* dPlaneVertexBuf, int* dIndex, bool withGravity);

bool CalcMedianOnlyZ(float3& median, cv::cuda::GpuMat& dVMapRoi,
                     char* dResultBuf, char* resultBuf, float minDepth, float maxDepth);

bool CalcAxisForPlane(float3 &zAxis, float &zThresh, cv::cuda::GpuMat& dVMap, float3 median,
                      float lowRadiusSquareThres,
                      float hightRadiusSquareThres,
                      float3* dPlaneVertexBuf, int* dIndex);

#endif /* CUDA_CUDAFUNCS_CUH_ */
