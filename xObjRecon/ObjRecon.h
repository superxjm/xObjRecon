#pragma once

#include <QtCore/QThread>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "Helpers/xUtils.h"

class xGUI;
class xSurfelFusion;
class xDeformation;
class xObjectTrackor;

void RegisteredImgs(cv::Mat& colorDepthImg, cv::Mat& colorImg, cv::Mat& depthImg);

class ObjRecon
{
public:
	ObjRecon();
	~ObjRecon();

	void processFrame(cv::Mat& rawDepthImg, cv::Mat& colorImg,
					  std::vector<ImuMsg>& imuMeasurements, double3& Gravity, int keyFrameIdxEachFrag);
	void processFrameOpt(cv::Mat& rawDepthImg, cv::Mat& colorImg,
						 std::vector<ImuMsg>& imuMeasurements, double3& Gravity, int keyFrameIdxEachFrag);
	void processFrame(cv::Mat& rawDepthImg, cv::Mat& colorImg, cv::Mat& fullColorImg,
					  std::vector<ImuMsg>& imuMeasurements, double3& Gravity, int keyFrameIdxEachFrag, int hasBeenPaused);

	cv::Mat getRenderedModelImg();

	void setIntrinExtrin(float* extrinR, float* extrinT, float* intrinColor, float* intrinDepth);
	void setIntrinExtrin(float fxDepth, float fyDepth, float cxDepth, float cyDepth,
	                     float3 depthToColorRX, float3 depthToColorRY, float3 depthToColorRZ, float3 depthToColort,
	                     float fxColor, float fyColor, float cxColor, float cyColor);
	void registerDepthImgCPU(cv::Mat &registeredDepthImg, cv::Mat &depthImg);

public:
	float m_dispScale;
	int64_t m_timeStamp;
	int64_t m_timeStampForDetect;
	int m_fragIdx;
	int m_fragNumInDetectStage;
	xGUI* m_pGui;
	xSurfelFusion* m_pFusion;
	xDeformation* m_pDeform;
	xObjectTrackor* m_pObjectTrackor;

	cv::Mat m_rawDepthImg,
		m_registeredRawDepthImg,
		m_renderedDepthImg,
		m_colorImg,
		m_grayImg,
		m_pruneMat;;
	cv::cuda::GpuMat m_dRawDepthImg,
		m_dRawDepthImgBuffer,
		m_dFilteredDepthImg,
		m_dRawDepthImg32F,
		m_dFilteredDepthImg32F,
		m_dRenderedDepthImg32F,
		m_dColorImgRGB,
		m_dColorImgRGBA,
		m_dGrayImg,
		m_dPruneMat;	

	Box m_objectBox;
	Box m_centerBox;
	std::vector<Box> m_boxVec;
	bool m_objectDetected;
	float m_velocity;
	float m_velocityThreshold;
	std::vector<float> m_velocityVec;

	float m_fxColor, m_fyColor, m_cxColor, m_cyColor;
	float m_fxDepth, m_fyDepth, m_cxDepth, m_cyDepth;
	float3 m_depthToColorRX, m_depthToColorRY, m_depthToColorRZ, m_depthToColort;
	ushort* m_registeredDepthImgBuffer = NULL;

	std::vector<Eigen::Matrix4f> m_cameraPoseVec;
	float m_meanDepth;
};

class xCapture;
class ObjRecon;

class ObjReconThread : public QThread
{
public:
	ObjReconThread(xCapture *sensor);
	~ObjReconThread();

	void setIntrinExtrin(float* extrinR, float* extrinT, float* intrinColor, float* intrinDepth);

	//void setSensor(RemoteStructureSeneorFromFile sensor)
	//{
		//m_sensor = sensor;
	//}
	
	void run() override;

private:
	xCapture *m_sensor;
	ObjRecon *m_objRecon;

	float m_fxColor, m_fyColor, m_cxColor, m_cyColor;
	float m_fxDepth, m_fyDepth, m_cxDepth, m_cyDepth;
	float3 m_depthToColorRX, m_depthToColorRY, m_depthToColorRZ, m_depthToColort;

public:
	volatile bool m_isFinish;
};