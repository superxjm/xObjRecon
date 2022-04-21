#include "stdafx.h"

#include "ObjRecon.h"

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Eigen/Eigen>
#include <QtWidgets/QApplication>

#include "xSurfelFusion/xSurfelFusion.h"
#include "xDeformation/xDeformation.h"
#include "xSensors/xCapture.hpp"
#include "xObjTrack/xObjectTrackor.h"
#include "MainGui.hpp"
#include "Helpers/xUtils.cuh"
#include "xSensors/xRemoteStructureSensorImporter.h"
#include "xSensors/xImageImporter.hpp"

#define DATA_TRANS_TEST 0

#if 0
void RenderScene(xSurfelFusion *pFusion, pangolin::OpenGlRenderState &cameraState, int timeStamp,
				 Box &objectBox, std::vector<Box> &boxVec, bool objectDetected, cv::Mat &colorImg, cv::Mat &depthImg);
void SaveModel(int fragNum, xDeformation *pDeform, float imgScale);
#endif

void RegisteredImgs(cv::Mat& colorDepthImg, cv::Mat& colorImg, cv::Mat& depthImg) {
	colorDepthImg.create(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3);

	cv::Mat grayImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1);
	cv::Mat grayDepthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1);
	depthImg.convertTo(grayDepthImg, CV_8UC1, 1.0f, 0);
	cv::cvtColor(colorImg, grayImg, CV_RGB2GRAY);
	for (int r = 0; r < colorDepthImg.rows; ++r)
	{
		for (int c = 0; c < colorDepthImg.cols; ++c)
		{
			cv::Vec3b& pixel = colorDepthImg.at<cv::Vec3b>(r, c);
			pixel[0] = 20;
			pixel[1] = grayImg.at<uchar>(r, c);
			pixel[2] = grayDepthImg.at<uchar>(r, c);
		}
	}
}

bool selectObject = false;
cv::Point origin;
cv::Mat img, imgBuf;
Box selection;

static void onMouse(int event, int x, int y, int, void*) {
	if (selectObject)
	{
		selection = Box(MIN(x, origin.x), MAX(x, origin.x), MIN(y, origin.y), MAX(y, origin.y));
		selection.m_left = clamp(selection.m_left, 0, img.cols - 1);
		selection.m_right = clamp(selection.m_right, 0, img.cols - 1);
		selection.m_top = clamp(selection.m_top, 0, img.rows - 1);
		selection.m_bottom = clamp(selection.m_bottom, 0, img.rows - 1);
		img = imgBuf.clone();
		cv::rectangle(img, cv::Rect(selection.m_left, selection.m_top,
									selection.m_right - selection.m_left, selection.m_bottom - selection.m_top),
					  cv::Scalar(255, 0, 0), 2);
		cv::imshow("input_color", img);
	}

	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:
		origin = cv::Point(x, y);
		selection = Box(x, x, y, y);
		selectObject = true;
		break;

	case cv::EVENT_LBUTTONUP:
		selectObject = false;
		break;
	}
}

static cv::Mat renderedImg, resizedRenderedImg;
static char renderedDir[256];

ObjRecon::ObjRecon()
	: m_rawDepthImg(Resolution::getInstance().height(),
					Resolution::getInstance().width(),
					CV_16UC1),
	m_renderedDepthImg(Resolution::getInstance().height(),
					   Resolution::getInstance().width(),
					   CV_16UC1),
	m_colorImg(Resolution::getInstance().height(),
			   Resolution::getInstance().width(),
			   CV_8UC3),
	m_grayImg(Resolution::getInstance().height(),
			  Resolution::getInstance().width(),
			  CV_8UC1),
	m_pruneMat(Resolution::getInstance().height(),
			   Resolution::getInstance().width(),
			   CV_8UC1),
	m_dRawDepthImg(Resolution::getInstance().height(),
				   Resolution::getInstance().width(),
				   CV_16UC1),
	m_dRawDepthImgBuffer(Resolution::getInstance().height(),
						 Resolution::getInstance().width(),
						 CV_16UC1),
	m_dFilteredDepthImg(Resolution::getInstance().height(),
						Resolution::getInstance().width(),
						CV_16UC1),
	m_dRawDepthImg32F(Resolution::getInstance().height(),
					  Resolution::getInstance().width(),
					  CV_32FC1),
	m_dFilteredDepthImg32F(Resolution::getInstance().height(),
						   Resolution::getInstance().width(),
						   CV_32FC1),
	m_dRenderedDepthImg32F(Resolution::getInstance().height(),
						   Resolution::getInstance().width(),
						   CV_32FC1),
	m_dColorImgRGB(Resolution::getInstance().height(),
				   Resolution::getInstance().width(),
				   CV_8UC3),
	m_dColorImgRGBA(Resolution::getInstance().height(),
					Resolution::getInstance().width(),
					CV_8UC4),
	m_dGrayImg(Resolution::getInstance().height(),
			   Resolution::getInstance().width(),
			   CV_8UC1),
	m_dPruneMat(Resolution::getInstance().height(),
				Resolution::getInstance().width(),
				CV_8UC1) {
	m_dispScale = 1.4f;

	m_timeStamp = 0;
	m_timeStampForDetect = 0;
	m_fragIdx = 0;
	m_pGui = new xGUI(m_dispScale);

	std::cout << "xSurfelFusion" << std::endl;
	m_pFusion = new xSurfelFusion(m_timeStamp, m_fragIdx);
	std::cout << "xDeformation" << std::endl;
	m_pDeform = new xDeformation(m_fragIdx, m_pFusion->m_dVboCuda);
	std::cout << "xObjectTrackor" << std::endl;
	m_pObjectTrackor = new xObjectTrackor(m_dFilteredDepthImg32F, m_dRenderedDepthImg32F,
										  m_colorImg, m_dGrayImg);

	m_objectBox = Box(0, 0, 0, 0);
	m_centerBox = Box(Resolution::getInstance().width() / 4, Resolution::getInstance().width() / 4 * 3,
					  Resolution::getInstance().height() / 4, Resolution::getInstance().height() / 4 * 3);
	std::vector<Box> boxVec;
	bool objectDetected = false;
	float velocity = 0.0f;
	//m_velocityThreshold = 0.005f;
	m_velocityThreshold = 0.01f;
	m_velocityVec = std::vector<float>(10, m_velocityThreshold + 1);

#if 1
	cv::namedWindow("input_depth", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("input_color", CV_WINDOW_AUTOSIZE);
	cv::setMouseCallback("input_color", onMouse, 0);
#endif	
}

ObjRecon::~ObjRecon() {
	std::cout << "delete m_pGui" << std::endl;
	delete m_pGui;
	std::cout << "delete m_pFusion" << std::endl;
	delete m_pFusion;
	std::cout << "delete m_pDeform" << std::endl;
	delete m_pDeform;
	std::cout << "delete m_pObjectTrackor" << std::endl;
	delete m_pObjectTrackor;
}

void ObjRecon::setIntrinExtrin(float* extrinR, float* extrinT, float* intrinColor, float* intrinDepth) {
	m_fxColor = intrinColor[0];
	m_fyColor = intrinColor[1];
	m_cxColor = intrinColor[2];
	m_cyColor = intrinColor[3];

	m_fxDepth = intrinDepth[0];
	m_fyDepth = intrinDepth[1];
	m_cxDepth = intrinDepth[2];
	m_cyDepth = intrinDepth[3];

	m_depthToColorRX.x = extrinR[0]; m_depthToColorRY.x = extrinR[1]; m_depthToColorRZ.x = extrinR[2]; m_depthToColort.x = -extrinT[0];
	m_depthToColorRX.y = extrinR[3]; m_depthToColorRY.y = extrinR[4]; m_depthToColorRZ.y = extrinR[5]; m_depthToColort.y = -extrinT[1];
	m_depthToColorRX.z = extrinR[6]; m_depthToColorRY.z = extrinR[7]; m_depthToColorRZ.z = extrinR[8]; m_depthToColort.z = -extrinT[2];
}

void ObjRecon::setIntrinExtrin(float fxDepth, float fyDepth, float cxDepth, float cyDepth,
							   float3 depthToColorRX, float3 depthToColorRY, float3 depthToColorRZ,
							   float3 depthToColort,
							   float fxColor, float fyColor, float cxColor, float cyColor) {
	m_fxDepth = fxDepth;
	m_fyDepth = fyDepth;
	m_cxDepth = cxDepth;
	m_cyDepth = cyDepth;
	m_depthToColorRX = depthToColorRX;
	m_depthToColorRY = depthToColorRY;
	m_depthToColorRZ = depthToColorRZ;
	m_depthToColort = depthToColort;
	m_fxColor = fxColor;
	m_fyColor = fyColor;
	m_cxColor = cxColor;
	m_cyColor = cyColor;
}

void ObjRecon::registerDepthImgCPU(cv::Mat &registeredDepthImg, cv::Mat &depthImg) {
	float3 pos;
	int u, v, idx;
	ushort depth;
	memset(registeredDepthImg.data, -1, registeredDepthImg.rows * registeredDepthImg.cols * sizeof(ushort));
	for (int r = 0; r < depthImg.rows; ++r)
	{
		for (int c = 0; c < depthImg.cols; ++c)
		{
			pos.z = depthImg.at<ushort>(r, c);
			pos.x = (c - m_cxDepth) / m_fxDepth * pos.z;
			pos.y = (r - m_cyDepth) / m_fyDepth * pos.z;
			pos = m_depthToColorRX * pos.x +
				m_depthToColorRY * pos.y +
				m_depthToColorRZ * pos.z +
				m_depthToColort;

			u = pos.x / pos.z * m_fxColor + m_cxColor + 0.5;
			v = pos.y / pos.z * m_fyColor + m_cyColor + 0.5;
			if (u >= 0 && u < registeredDepthImg.cols && v >= 0 && v < registeredDepthImg.rows)
			{
				depth = pos.z;

				if (depth < registeredDepthImg.at<ushort>(v, u))
				{
					registeredDepthImg.at<ushort>(v, u) = depth;
				}
			}
		}
	}
	for (int r = 0; r < registeredDepthImg.rows; ++r)
	{
		for (int c = 0; c < registeredDepthImg.cols; ++c)
		{
			if ((int)registeredDepthImg.at<ushort>(r, c) == 65535)
			{
				registeredDepthImg.at<ushort>(r, c) = 0;
			}
		}
	}
}

void ObjRecon::processFrame(cv::Mat& rawDepthImg, cv::Mat& colorImg,
							std::vector<ImuMsg>& imuMeasurements, double3& gravity, int keyFrameIdxEachFrag) {
	innoreal::InnoRealTimer timer;
	m_colorImg = colorImg;
	// the m_rawDepthImg is the registered depth img, same size as color img
#if CPU_REGISTER
	registerDepthImgCPU(m_rawDepthImg, rawDepthImg);
	cv::Mat colorDepthImg;
	RegisteredImgs(colorDepthImg, m_colorImg, m_rawDepthImg);
	cv::imshow("colorDepthImg", colorDepthImg);
	cv::waitKey(1);
	m_dRawDepthImg.upload(m_rawDepthImg);
#else
	timer.TimeStart();
	m_dRawDepthImg.upload(rawDepthImg);
	RegisterDepthImg(m_dRawDepthImg, m_dRawDepthImgBuffer,
					 m_cxDepth, m_cyDepth, m_fxDepth, m_fyDepth,
					 m_depthToColorRX, m_depthToColorRY, m_depthToColorRZ, m_depthToColort,
					 m_cxColor, m_cyColor, m_fxColor, m_fyColor);
	m_dRawDepthImgBuffer.download(m_rawDepthImg);
	for (int r = 0; r < m_rawDepthImg.rows; ++r)
	{
		for (int c = 0; c < m_rawDepthImg.cols; ++c)
		{
			if (m_rawDepthImg.at<ushort>(r, c) == 20000)
			{
				std::cout << m_rawDepthImg.at<ushort>(r, c) << std::endl;
			}
		}
	}
	timer.TimeEnd();
	std::cout << "register time: " << timer.TimeGap_in_ms() << std::endl;
#endif
	assert(m_colorImg.rows == m_rawDepthImg.rows && m_colorImg.cols == m_rawDepthImg.cols);

	double3 RImuCam[3] = { 0, -1, 0, -1, 0, 0, 0, 0, -1 };
	gravity = normalize(gravity);
	gravity = RImuCam[0] * gravity.x + RImuCam[1] * gravity.y + RImuCam[2] * gravity.z;
	for (int i = 0; i < imuMeasurements.size(); ++i)
	{
		ImuMsg& imuMsg = imuMeasurements[i];
		imuMsg.acc = RImuCam[0] * imuMsg.acc.x + RImuCam[1] * imuMsg.acc.y + RImuCam[2] * imuMsg.acc.z;
		imuMsg.gyr = RImuCam[0] * imuMsg.gyr.x + RImuCam[1] * imuMsg.gyr.y + RImuCam[2] * imuMsg.gyr.z;
	}
#if 0
	double tt = 0;
	for (int i = 0; i < imuMeasurements.size(); ++i)
	{
		ImuMsg& imuMsg = imuMeasurements[i];
		if (i == 0)
		{
			tt = imuMsg.timeStamp;
		}
		std::cout << "imuMsg.timeStamp: " << imuMsg.timeStamp - tt << std::endl;
		std::cout << "imuMsg.acc: " << imuMsg.acc.x << " : " << imuMsg.acc.y << " : " << imuMsg.acc.z << std::endl;
		std::cout << "imuMsg.gyr: " << imuMsg.gyr.x << " : " << imuMsg.gyr.y << " : " << imuMsg.gyr.z << std::endl;
		tt = imuMsg.timeStamp;
	}
#endif

	timer.TimeStart();
	BilateralFilter(m_dFilteredDepthImg, m_dRawDepthImg);
	m_dRawDepthImg.convertTo(m_dRawDepthImg32F, CV_32FC1, 1.0 / 1000.0, 0);
	m_dFilteredDepthImg.convertTo(m_dFilteredDepthImg32F, CV_32FC1, 1.0 / 1000.0, 0);
	m_dColorImgRGB.upload(m_colorImg);
	cv::cuda::cvtColor(m_dColorImgRGB, m_dGrayImg, CV_RGB2GRAY, 1);
	cv::cuda::cvtColor(m_dColorImgRGB, m_dColorImgRGBA, CV_RGB2RGBA, 4);
	timer.TimeEnd();
	std::cout << "upload download time: " << timer.TimeGap_in_ms() << std::endl;

#if 0
	if (m_objectDetected == false || imuMeasurements.size() == 0)
	{
		img = m_colorImg.clone();
		imgBuf = m_colorImg.clone();
		cv::imshow("input_color", m_colorImg);
		cv::imshow("input_depth", m_rawDepthImg * 30);
		cv::waitKey(0);
#if 0
		selection.m_top = 122;
		selection.m_bottom = 330;
		selection.m_left = 265;
		selection.m_right = 547;
#endif

		m_objectDetected = true;
		m_objectBox = selection;
#if 0
		m_timeStamp = 0;
		m_timeStampForDetect = 0;
		m_fragIdx = 0;
#endif
	}
#endif

#if 0
	if (m_objectDetected == false && m_timeStamp > 10)
	{
		++m_timeStampForDetect;
		timer.TimeStart();
		m_pObjectTrackor->detect(m_objectBox, m_centerBox, gravity);
		timer.TimeEnd();
		std::cout << "detect time: " << timer.TimeGap_in_ms() << std::endl;
		m_boxVec = m_pObjectTrackor->getCandidatedBoxes();
		m_velocityVec[(m_timeStampForDetect - 1) % m_velocityVec.size()] = m_velocity;
		m_objectDetected = true;
		std::cout << "--------------------" << std::endl;
		for (int i = 0; i < m_velocityVec.size(); ++i)
		{
			std::cout << "velocity: " << m_velocityVec[i] << std::endl;
			std::cout << "score: " << m_objectBox.m_score << std::endl;
			std::cout << "bottom: " << m_objectBox.m_bottom << std::endl;
			if (m_velocityVec[i] > m_velocityThreshold || m_objectBox.m_bottom == 0 || m_objectBox.m_score > 30)
			{
				m_objectDetected = false;
			}
		}
		std::cout << "m_objectDetected: " << m_objectDetected << std::endl;
		std::cout << "--------------------" << std::endl;
		//std::cout << "ObjectDetected: " << m_objectDetected << std::endl;
		//std::cout << "objectBox.m_bottom: " << m_objectBox.m_bottom << std::endl;
		if (m_objectDetected == true)
		{
			m_timeStamp = 0;
			m_timeStampForDetect = 0;
			m_fragIdx = 0;
			m_pFusion->clear();
			return;
		}
	}
#endif

#if 1
	if (m_objectDetected == true)
	{
		std::cout << "track process" << std::endl;
		timer.TimeStart();
		m_pObjectTrackor->track(m_objectBox, m_timeStamp, gravity);
		m_dPruneMat = m_pObjectTrackor->getPruneMaskGpu();
		PruneDepth(m_dRawDepthImg, m_dFilteredDepthImg, m_dRawDepthImg32F, m_dFilteredDepthImg32F, m_dPruneMat);
		timer.TimeEnd();
		std::cout << "track time: " << timer.TimeGap_in_ms() << std::endl;	
	}
#endif
	std::cout << "timestamp: " << m_timeStamp << std::endl;
	++m_timeStamp;

#if 0
	m_dRawDepthImg.download(m_rawDepthImg);
	cv::imshow("input_color", colorImg);
	cv::imshow("input_depth", m_rawDepthImg * 60);
	cv::waitKey(1);
#if 0
	std::vector<int> pngCompressionParams;
	pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	pngCompressionParams.push_back(0);
	char saveColorDir[256], saveDepthDir[256];
	sprintf(saveColorDir, "D:\\xjm\\result\\before_opt\\input_data\\frame-%06d.color.png", m_timeStamp - 1);
	sprintf(saveDepthDir, "D:\\xjm\\result\\before_opt\\input_data\\frame-%06d.depth.png", m_timeStamp - 1);
	cv::Mat resizeColor = colorImg, resizedDepth = m_rawDepthImg;
	//cv::resize(m_rawDepthImg, resizedDepth, cv::Size(640, 480));
	//resizedDepth = resizedDepth * 40;
	//cv::resize(colorImg, resizeColor, cv::Size(640, 480));
	cv::imwrite(saveColorDir, resizeColor, pngCompressionParams);
	cv::imwrite(saveDepthDir, resizedDepth, pngCompressionParams);
	cv::imshow("colorImg", colorImg);
	cv::imshow("rawDepthImg", rawDepthImg);
	cv::waitKey(1);
#endif
#endif
#if 0
	m_pFusion->processFrame(m_renderedDepthImg,
							m_velocity,
							m_colorImg,
							m_rawDepthImg,
							imuMeasurements,
							gravity,
							m_objectDetected);
#endif
#if 1
#if 1
	//std::cout << "start fusion" << std::endl;
	//timer.TimeStart();
	m_pFusion->processFrame2(m_dRenderedDepthImg32F,
							 m_velocity,
							 m_dRawDepthImg,
							 m_dFilteredDepthImg,
							 m_dRawDepthImg32F,
							 m_dFilteredDepthImg32F,
							 m_dColorImgRGBA,
							 m_dColorImgRGB,
							 imuMeasurements,
							 gravity,
							 m_objectDetected);	
	//timer.TimeEnd();
	//std::cout << "fusion time: " << timer.TimeGap_in_ms() << std::endl;
#endif

	if (m_objectDetected == true)
	{
		timer.TimeStart();
		m_pDeform->addDataWithKeyFrame(m_dGrayImg, m_pFusion->m_currPose.data());
		timer.TimeEnd();
		std::cout << "add data time: " << timer.TimeGap_in_ms() << std::endl;
	}

	if (IsFrag(m_timeStamp) == 1 && m_objectDetected == true)
	{
		m_pDeform->deform(reinterpret_cast<xMatrix4f *>(&m_pFusion->m_currPose),
						  m_pFusion->m_dVboCuda,
						  m_pFusion->getGlobalModel().lastCount(),
						  keyFrameIdxEachFrag);		
		++m_fragIdx;

		m_pGui->setPanelInfo(m_pDeform);
	}
#endif

#if 0
	//timer.TimeStart();
	m_pGui->renderScene(m_pFusion, m_pDeform, m_timeStamp, m_objectBox, m_boxVec, m_objectDetected, m_colorImg, m_rawDepthImg);
	xCheckGlDieOnError();
	//timer.TimeEnd();
	//std::cout << "render scene time: " << timer.TimeGap_in_ms() << std::endl;
#endif
#if 1
	if (m_pGui->m_debugButton->Get())
	{
		std::cout << "debug" << std::endl;
		m_pGui->renderSceneForDebug(m_pDeform);
	} else
	{
		m_pGui->renderScene(m_pFusion, m_pDeform, m_timeStamp, m_objectBox, m_boxVec, m_objectDetected, m_colorImg, m_rawDepthImg);
	}
#endif
}

void ObjRecon::processFrameOpt(cv::Mat& rawDepthImg, cv::Mat& colorImg,
															 std::vector<ImuMsg>& imuMeasurements, double3& gravity,
															 int keyFrameIdxEachFrag)
{
	innoreal::InnoRealTimer timer;

	m_colorImg = colorImg;

#if 0
	registerDepthImgCPU(m_rawDepthImg, rawDepthImg);
	m_dRawDepthImg.upload(m_rawDepthImg);
#else
	m_dRawDepthImg.upload(rawDepthImg);
	RegisterDepthImg(m_dRawDepthImg, m_dRawDepthImgBuffer,
									 m_cxDepth, m_cyDepth, m_fxDepth, m_fyDepth,
									 m_depthToColorRX, m_depthToColorRY, m_depthToColorRZ, m_depthToColort,
									 m_cxColor, m_cyColor, m_fxColor, m_fyColor);
	m_dRawDepthImg.download(m_rawDepthImg);
#endif
	if (GlobalState::getInstance().m_dataTransTest)
	{
		++m_timeStamp;
		cv::Mat colorDepthImg;
		RegisteredImgs(colorDepthImg, m_colorImg, m_rawDepthImg);
		cv::imshow("Data Trans Test", colorDepthImg);
		cv::waitKey(1);
		return;
	}
	if (!GlobalState::getInstance().m_doReconstruction)
	{
		return;
	}

	double3 RImuCam[3] = { 0, -1, 0, -1, 0, 0, 0, 0, -1 };
	gravity = normalize(gravity);
	gravity = RImuCam[0] * gravity.x + RImuCam[1] * gravity.y + RImuCam[2] * gravity.z;
	for (int i = 0; i < imuMeasurements.size(); ++i)
	{
		ImuMsg& imuMsg = imuMeasurements[i];
		imuMsg.acc = RImuCam[0] * imuMsg.acc.x + RImuCam[1] * imuMsg.acc.y + RImuCam[2] * imuMsg.acc.z;
		imuMsg.gyr = RImuCam[0] * imuMsg.gyr.x + RImuCam[1] * imuMsg.gyr.y + RImuCam[2] * imuMsg.gyr.z;
	}

	BilateralFilter(m_dFilteredDepthImg, m_dRawDepthImg);
	m_dRawDepthImg.convertTo(m_dRawDepthImg32F, CV_32FC1, 1.0 / 1000.0, 0);
	m_dFilteredDepthImg.convertTo(m_dFilteredDepthImg32F, CV_32FC1, 1.0 / 1000.0, 0);
	m_dColorImgRGB.upload(m_colorImg);
	cv::cuda::cvtColor(m_dColorImgRGB, m_dGrayImg, CV_RGB2GRAY, 1);
	cv::cuda::cvtColor(m_dColorImgRGB, m_dColorImgRGBA, CV_RGB2RGBA, 4);

#if 0
	std::cout << "testtest" << std::endl;
	cv::Mat testtest;
	m_dFilteredDepthImg32F.download(testtest);
	cv::imshow("testtest", testtest);
	cv::waitKey(1);
#endif
#if 0
	if (m_objectDetected == false || imuMeasurements.size() == 0)
	{
		img = m_colorImg.clone();
		imgBuf = m_colorImg.clone();
		cv::imshow("input_color", m_colorImg);
		cv::imshow("input_depth", m_rawDepthImg * 30);
		cv::waitKey(0);
#if 0
		selection.m_top = 122;
		selection.m_bottom = 330;
		selection.m_left = 265;
		selection.m_right = 547;
#endif

		m_objectDetected = true;
		m_objectBox = selection;
#if 0
		m_timeStamp = 0;
		m_timeStampForDetect = 0;
		m_fragIdx = 0;
#endif
	}
#endif
#if 0
	if (m_objectDetected == false)
	{
		img = m_colorImg.clone();
		imgBuf = m_colorImg.clone();
		cv::imshow("input_color", m_colorImg);
		cv::imshow("input_depth", m_rawDepthImg * 30);
		cv::waitKey(0);

		m_objectDetected = true;
		m_objectBox = selection;
	}
#endif

#if 1
#if 0
	FilterDepthUsingDepth(m_dRawDepthImg,
												m_dFilteredDepthImg,
												m_dRawDepthImg32F,
												m_dFilteredDepthImg32F,
												2000.0f);
#endif
#if 0
	if (m_objectDetected == false)
	{
		FilterDepthUsingDepth(m_dRawDepthImg,
							  m_dFilteredDepthImg,
							  m_dRawDepthImg32F,
							  m_dFilteredDepthImg32F,
							  2000.0f);
	}
#endif
	if (m_objectDetected == false && m_timeStamp > 100)
	{
		++m_timeStampForDetect;
		timer.TimeStart();
		m_pObjectTrackor->detect(m_objectBox, m_centerBox, gravity);
		timer.TimeEnd();
		printf("detect time: %f\n", timer.TimeGap_in_ms());
		m_boxVec = m_pObjectTrackor->getCandidatedBoxes();
		m_velocityVec[(m_timeStampForDetect - 1) % m_velocityVec.size()] = m_velocity;

		float score_threshold = min((m_objectBox.m_right + m_objectBox.m_left) / 2,
			(m_objectBox.m_top + m_objectBox.m_bottom) / 2) / 4.0f;

		m_objectDetected = true;
		for (int i = 0; i < m_velocityVec.size(); ++i)
		{
			if (m_velocityVec[i] > m_velocityThreshold || m_objectBox.m_bottom == 0 || m_objectBox.m_score > score_threshold)
			{
				m_objectDetected = false;
			}
		}
		if (m_objectDetected == true && IsFrag(m_timeStamp + 1))
		{
			m_fragNumInDetectStage = (m_timeStamp + 1) / FRAG_SIZE;
			m_timeStamp = 0;
			m_timeStampForDetect = 0;
			m_fragIdx = 0;
			m_pFusion->clear();

			//float minDepth, maxDepth, meanDepth;
			//m_pObjectTrackor->estimateMinMaxDepth(minDepth, maxDepth, meanDepth,
																						//m_objectBox, m_timeStamp, gravity);
#if 0
			std::cout << "minDepth: " << minDepth << std::endl;
			std::cout << "meanDepth: " << meanDepth << std::endl;
			std::cout << "maxDepth: " << maxDepth << std::endl;

			cv::Mat boxVis = m_colorImg.clone();
			cv::rectangle(boxVis,
										cv::Rect(m_objectBox.m_left, m_objectBox.m_top,
										m_objectBox.m_right - m_objectBox.m_left, m_objectBox.m_bottom - m_objectBox.m_top),
										cv::Scalar(0, 0, 255), 4);
			cv::namedWindow("candidate boxes");
			cv::imshow("candidate boxes", boxVis);
			cv::waitKey(0);
#endif

			return;
		}
		m_objectDetected = false;
	}
#endif

	if (m_objectDetected == true)
	{
		//timer.TimeStart();
		//std::cout << "track start" << std::endl;
		if (imuMeasurements.size() != 0)
		{
			//std::cout << "trackOpt start" << std::endl;
			m_pObjectTrackor->trackOpt1(m_objectBox, m_timeStamp, gravity, m_meanDepth);
			//std::cout << "trackOpt1" << std::endl;
		} 
		else
		{
			//std::cout << "trackOpt2 start" << std::endl;
			m_pObjectTrackor->trackOpt2(m_objectBox, m_timeStamp, gravity, m_meanDepth);
			//std::cout << "trackOpt2" << std::endl;
		}
		//std::cout << "track finish" << std::endl;
#if 0
		cv::Mat boxVis = m_colorImg.clone();
		cv::rectangle(boxVis,
					  cv::Rect(m_objectBox.m_left, m_objectBox.m_top,
					  m_objectBox.m_right - m_objectBox.m_left, m_objectBox.m_bottom - m_objectBox.m_top),
					  cv::Scalar(0, 0, 255), 4);
		cv::namedWindow("candidate boxes");
		cv::imshow("candidate boxes", boxVis);
		cv::waitKey(1);
#endif
#if 0
		std::vector<int> pngCompressionParams;
		pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
		pngCompressionParams.push_back(0);
		char renderedDir[256];
		sprintf(renderedDir, "D:\\xjm\\data_for_video\\first_section\\tracking\\%06d.png", m_timeStamp);
		cv::imwrite(renderedDir, boxVis, pngCompressionParams);
#endif
		m_dPruneMat = m_pObjectTrackor->getPruneMaskGpu();
		PruneDepth(m_dRawDepthImg, m_dFilteredDepthImg, m_dRawDepthImg32F, m_dFilteredDepthImg32F, m_dPruneMat);
		//timer.TimeEnd();
		//printf("track time: %f\n", timer.TimeGap_in_ms());
		//m_dRawDepthImg.download(m_rawDepthImg);
		//cv::imshow("input_depth", m_rawDepthImg * 30);
		//cv::waitKey(0);
#if 0
		m_dRawDepthImg.download(m_rawDepthImg);
		cv::imshow("input_depth", m_rawDepthImg * 30);
		cv::waitKey(1);
		std::vector<int> pngCompressionParams;
		pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
		pngCompressionParams.push_back(0);
		char renderedDir[256];
		sprintf(renderedDir, "D:\\xjm\\data_for_video\\first_section\\extracted_object\\%06d.png", m_timeStamp);
		cv::imwrite(renderedDir, m_rawDepthImg * 50, pngCompressionParams);
#endif
	}
	//std::cout << "timeStamp: " << m_timeStamp << std::endl;
	++m_timeStamp;	

#if 1
#if 1
	//timer.TimeStart();
	//std::cout << "processFrame2 start" << std::endl;
	m_pFusion->processFrame2(m_dRenderedDepthImg32F,
							 m_velocity,
							 m_dRawDepthImg,
							 m_dFilteredDepthImg,
							 m_dRawDepthImg32F,
							 m_dFilteredDepthImg32F,
							 m_dColorImgRGBA,
							 m_dColorImgRGB,
							 imuMeasurements,
							 gravity,
							 m_objectDetected);
	//std::cout << "processFrame2 finish" << std::endl;
	m_cameraPoseVec.emplace_back(m_pFusion->getCurrPose());
#if 0
	cv::Mat renderedDepthImg32F;
	m_dRenderedDepthImg32F.download(renderedDepthImg32F);
	cv::imshow("renderedDepthImg32F", renderedDepthImg32F / 1.5);
	cv::waitKey(1);
#endif
	//timer.TimeEnd();
	//printf("fusion time: %f\n", timer.TimeGap_in_ms());
#endif

	if (m_objectDetected == true)
	{
		//timer.TimeStart();
		m_pDeform->addDataWithKeyFrame(m_dGrayImg, m_pFusion->m_currPose.data());
		//timer.TimeEnd();
		//printf("add data with key frame time: %f\n", timer.TimeGap_in_ms());
	}
	if (IsFrag(m_timeStamp) == 1 && m_objectDetected == true)
	{
		//timer.TimeStart();
		m_pDeform->deform(reinterpret_cast<xMatrix4f *>(&m_pFusion->m_currPose),
						  m_pFusion->m_dVboCuda,
						  m_pFusion->getGlobalModel().lastCount(),
						  keyFrameIdxEachFrag);
		//timer.TimeEnd();
		//printf("deform time: %f\n", timer.TimeGap_in_ms());
		//Sleep(3000);
		++m_fragIdx;
		m_pGui->setPanelInfo(m_pDeform);
	}	
#endif

	//timer.TimeStart();
	//m_pGui->renderScene(m_pFusion, m_pDeform, m_timeStamp, m_objectBox, m_boxVec, m_objectDetected, m_colorImg, m_rawDepthImg);
	//xCheckGlDieOnError();
	//timer.TimeEnd();
	//std::cout << "render scene time: " << timer.TimeGap_in_ms() << std::endl;
}

void ObjRecon::processFrame(cv::Mat& rawDepthImg, cv::Mat& colorImg, cv::Mat& fullColorImg,
							std::vector<ImuMsg>& imuMeasurements, double3& gravity, int keyFrameIdxEachFrag, int hasBeenPaused) {
	m_colorImg = colorImg;
	// the m_rawDepthImg is the registered depth img, same size as color img
#if CPU_REGISTER
	registerDepthImgCPU(m_rawDepthImg, rawDepthImg);
	cv::Mat colorDepthImg;
	RegisteredImgs(colorDepthImg, m_colorImg, m_rawDepthImg);
	cv::imshow("colorDepthImg", colorDepthImg);
	cv::waitKey(1);

	m_dRawDepthImg.upload(m_rawDepthImg);
#else
	RegisterDepthImg(m_dRawDepthImg, m_dRawDepthImgBuffer,
					 m_cxDepth, m_cyDepth, m_fxDepth, m_fyDepth,
					 m_depthToColorRX, m_depthToColorRY, m_depthToColorRZ, m_depthToColort,
					 m_cxColor, m_cyColor, m_fxColor, m_fyColor);
	m_dRawDepthImg.download(m_rawDepthImg);
#endif
	assert(m_colorImg.rows == m_dRawDepthImg.rows && m_colorImg.cols == m_dRawDepthImg.cols);

	//std::cout << "processFrame" << std::endl;
	innoreal::InnoRealTimer timer;
	double3 RImuCam[3] = { 0, -1, 0, -1, 0, 0, 0, 0, -1 };

	gravity = normalize(gravity);
	gravity = RImuCam[0] * gravity.x + RImuCam[1] * gravity.y + RImuCam[2] * gravity.z;
	for (int i = 0; i < imuMeasurements.size(); ++i)
	{
		ImuMsg& imuMsg = imuMeasurements[i];
		imuMsg.acc = RImuCam[0] * imuMsg.acc.x + RImuCam[1] * imuMsg.acc.y + RImuCam[2] * imuMsg.acc.z;
		imuMsg.gyr = RImuCam[0] * imuMsg.gyr.x + RImuCam[1] * imuMsg.gyr.y + RImuCam[2] * imuMsg.gyr.z;
	}
#if 0
	double tt = 0;
	for (int i = 0; i < imuMeasurements.size(); ++i)
	{
		ImuMsg& imuMsg = imuMeasurements[i];
		if (i == 0)
		{
			tt = imuMsg.timeStamp;
		}
		std::cout << "imuMsg.timeStamp: " << imuMsg.timeStamp - tt << std::endl;
		std::cout << "imuMsg.acc: " << imuMsg.acc.x << " : " << imuMsg.acc.y << " : " << imuMsg.acc.z << std::endl;
		std::cout << "imuMsg.gyr: " << imuMsg.gyr.x << " : " << imuMsg.gyr.y << " : " << imuMsg.gyr.z << std::endl;
		tt = imuMsg.timeStamp;
	}
#endif

	//timer.TimeStart();
	BilateralFilter(m_dFilteredDepthImg, m_dRawDepthImg);
	m_dRawDepthImg.convertTo(m_dRawDepthImg32F, CV_32FC1, 1.0 / 1000.0, 0);
	m_dFilteredDepthImg.convertTo(m_dFilteredDepthImg32F, CV_32FC1, 1.0 / 1000.0, 0);
	m_dColorImgRGB.upload(m_colorImg);
	cv::cuda::cvtColor(m_dColorImgRGB, m_dGrayImg, CV_RGB2GRAY, 1);
	cv::cuda::cvtColor(m_dColorImgRGB, m_dColorImgRGBA, CV_RGB2RGBA, 4);
	//timer.TimeEnd();
	//std::cout << "upload download time: " << timer.TimeGap_in_ms() << std::endl;

#if 1
	if (hasBeenPaused == 1)
	{
		img = m_colorImg.clone();
		imgBuf = m_colorImg.clone();
		cv::imshow("input_color", m_colorImg);
		//cv::imshow("input_depth", m_rawDepthImg * 30);
		cv::waitKey(0);
#if 0
		selection.m_top = 122;
		selection.m_bottom = 330;
		selection.m_left = 265;
		selection.m_right = 547;
#endif

		m_objectDetected = true;
		m_objectBox = selection;
#if 0
		m_timeStamp = 0;
		m_timeStampForDetect = 0;
		m_fragIdx = 0;
#endif
	}
#endif
#if 0
	if (m_objectDetected == false && m_timeStamp > 10)
	{
		++m_timeStampForDetect;
		timer.TimeStart();
		m_pObjectTrackor->detect(m_objectBox, m_centerBox, gravity);
		timer.TimeEnd();
		std::cout << "detect time: " << timer.TimeGap_in_ms() << std::endl;
		m_boxVec = m_pObjectTrackor->getCandidatedBoxes();
		m_velocityVec[(m_timeStampForDetect - 1) % m_velocityVec.size()] = m_velocity;
		m_objectDetected = true;
		std::cout << "--------------------" << std::endl;
		for (int i = 0; i < m_velocityVec.size(); ++i)
		{
			std::cout << "velocity: " << m_velocityVec[i] << std::endl;
			std::cout << "score: " << m_objectBox.m_score << std::endl;
			std::cout << "bottom: " << m_objectBox.m_bottom << std::endl;
			if (m_velocityVec[i] > m_velocityThreshold || m_objectBox.m_bottom == 0 || m_objectBox.m_score > 30)
			{
				m_objectDetected = false;
			}
		}
		std::cout << "m_objectDetected: " << m_objectDetected << std::endl;
		std::cout << "--------------------" << std::endl;
		//std::cout << "ObjectDetected: " << m_objectDetected << std::endl;
		//std::cout << "objectBox.m_bottom: " << m_objectBox.m_bottom << std::endl;
		if (m_objectDetected == true)
		{
			m_timeStamp = 0;
			m_timeStampForDetect = 0;
			m_fragIdx = 0;
			m_pFusion->clear();
			return;
		}
	}
#endif

#if 1
	if (m_objectDetected == true)
	{
		timer.TimeStart();
		m_pObjectTrackor->track(m_objectBox, m_timeStamp);
		m_dPruneMat = m_pObjectTrackor->getPruneMaskGpu();
		PruneDepth(m_dRawDepthImg, m_dFilteredDepthImg, m_dRawDepthImg32F, m_dFilteredDepthImg32F, m_dPruneMat);
		timer.TimeEnd();
		std::cout << "track time: " << timer.TimeGap_in_ms() << std::endl;
	}
#endif
	std::cout << "timestamp: " << m_timeStamp << std::endl;
	++m_timeStamp;
	cv::cvtColor(m_colorImg, m_grayImg, CV_RGB2GRAY);

#if 1
	m_dRawDepthImg.download(m_rawDepthImg);
	cv::imshow("input_color", colorImg);
	cv::imshow("input_depth", m_rawDepthImg * 50);
	cv::waitKey(1);
#if 0
	std::vector<int> pngCompressionParams;
	pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	pngCompressionParams.push_back(0);
	char saveColorDir[256], saveDepthDir[256];
	sprintf(saveColorDir, "D:\\xjm\\result\\before_opt\\input_data\\frame-%06d.color.png", m_timeStamp - 1);
	sprintf(saveDepthDir, "D:\\xjm\\result\\before_opt\\input_data\\frame-%06d.depth.png", m_timeStamp - 1);
	cv::Mat resizeColor = colorImg, resizedDepth = m_rawDepthImg;
	//cv::resize(m_rawDepthImg, resizedDepth, cv::Size(640, 480));
	//resizedDepth = resizedDepth * 40;
	//cv::resize(colorImg, resizeColor, cv::Size(640, 480));
	cv::imwrite(saveColorDir, resizeColor, pngCompressionParams);
	cv::imwrite(saveDepthDir, resizedDepth, pngCompressionParams);
	cv::imshow("colorImg", colorImg);
	cv::imshow("rawDepthImg", rawDepthImg);
	cv::waitKey(1);
#endif
#endif
#if 0
	m_pFusion->processFrame(m_renderedDepthImg,
							m_velocity,
							m_colorImg,
							m_rawDepthImg,
							imuMeasurements,
							gravity,
							m_objectDetected);
#endif
#if 1
#if 1
	//timer.TimeStart();
	m_pFusion->processFrame2(m_dRenderedDepthImg32F,
							 m_velocity,
							 m_dRawDepthImg,
							 m_dFilteredDepthImg,
							 m_dRawDepthImg32F,
							 m_dFilteredDepthImg32F,
							 m_dColorImgRGBA,
							 m_dColorImgRGB,
							 imuMeasurements,
							 gravity,
							 m_objectDetected);
	//timer.TimeEnd();
	//std::cout << "fusion time: " << timer.TimeGap_in_ms() << std::endl;
#endif

	if (m_objectDetected == true)
	{
		//timer.TimeStart();
		m_pDeform->addData(m_colorImg, fullColorImg, m_grayImg, m_dGrayImg, m_pFusion->m_currPose.data());
		//timer.TimeEnd();
		//std::cout << "add data time: " << timer.TimeGap_in_ms() << std::endl;
	}

	if (IsFrag(m_timeStamp) == 1 && m_objectDetected == true)
	{
		m_pDeform->deform(reinterpret_cast<xMatrix4f *>(&m_pFusion->m_currPose),
						  m_pFusion->m_dVboCuda,
						  m_pFusion->getGlobalModel().lastCount(),
						  keyFrameIdxEachFrag);
		++m_fragIdx;
#if 0
		if (m_pDeform->getFragNum() == 21)
		{
			char fileDir[256];
#if 0
			for (int i = 0; i < m_pDeform->getFragNum(); ++i)
			{
				sprintf(fileDir, "D:\\xjm\\snapshot\\model_%06d_%d_snapshot.ply", static_cast<int>(m_timeStamp), i);
				std::cout << fileDir << std::endl;
				m_pDeform->savePly(fileDir, i);
			}
#endif
			sprintf(fileDir, "D:\\xjm\\snapshot\\model_%06d_%d_snapshot.ply", static_cast<int>(m_timeStamp), -1);
			std::cout << fileDir << std::endl;
			m_pDeform->savePly(fileDir, -1);
			std::exit(0);
		}
#endif
		m_pGui->setPanelInfo(m_pDeform);
	}
#endif

#if 1
	timer.TimeStart();
	m_pGui->renderScene(m_pFusion, m_pDeform, m_timeStamp, m_objectBox, m_boxVec, m_objectDetected, m_colorImg, m_rawDepthImg);
	xCheckGlDieOnError();
	timer.TimeEnd();
	std::cout << "render scene time: " << timer.TimeGap_in_ms() << std::endl;
#endif
#if 0
	if (pGui->m_debugButton->Get())
	{
		pGui->renderSceneForDebug(pDeform);
	} else
	{
		pGui->renderScene(pFusion, timeStamp, objectBox, boxVec, objectDetected, m_colorImg, m_rawDepthImg);
	}
#endif
}

cv::Mat ObjRecon::getRenderedModelImg() {
	return m_pGui->m_resizedGrayRenderedModelImg;
}

ObjReconThread::ObjReconThread(xCapture *sensor) : m_sensor(sensor)
{
	//m_objRecon = new ObjRecon();
}

ObjReconThread::~ObjReconThread() {
	delete m_objRecon;
}

void ObjReconThread::setIntrinExtrin(float* extrinR, float* extrinT, float* intrinColor, float* intrinDepth) {
	m_fxColor = intrinColor[0];
	m_fyColor = intrinColor[1];
	m_cxColor = intrinColor[2];
	m_cyColor = intrinColor[3];

	m_fxDepth = intrinDepth[0];
	m_fyDepth = intrinDepth[1];
	m_cxDepth = intrinDepth[2];
	m_cyDepth = intrinDepth[3];

	m_depthToColorRX.x = extrinR[0]; m_depthToColorRY.x = extrinR[1]; m_depthToColorRZ.x = extrinR[2]; m_depthToColort.x = -extrinT[0];
	m_depthToColorRX.y = extrinR[3]; m_depthToColorRY.y = extrinR[4]; m_depthToColorRZ.y = extrinR[5]; m_depthToColort.y = -extrinT[1];
	m_depthToColorRX.z = extrinR[6]; m_depthToColorRY.z = extrinR[7]; m_depthToColorRZ.z = extrinR[8]; m_depthToColort.z = -extrinT[2];
}

char TrackbarNameX[256];
char TrackbarNameY[256];
const int g_nMaxValue = 21;
const int g_nHalfMaxValue = (g_nMaxValue - 1) / 2;
int g_xOffset;
int g_yOffset;
cv::Mat g_rawDepthImg;
cv::Mat g_registeredDepthImg;
cv::Mat g_colorImg;
float gFxDepth, gFyDepth, gCxDepth, gCyDepth;
float gFxColor, gFyColor, gCxColor, gCyColor;
float3  gDepthToColorRX, gDepthToColorRY, gDepthToColorRZ, gDepthToColort;
float3 gOriDepthToColort;
void RegisterDepthImgCPU(cv::Mat &registeredDepthImg, cv::Mat &depthImg) {
	registeredDepthImg.create(depthImg.size(), depthImg.type());

	float3 pos;
	int u, v, idx;
	ushort depth;
	memset(registeredDepthImg.data, -1, registeredDepthImg.rows * registeredDepthImg.cols * sizeof(ushort));
	for (int r = 0; r < depthImg.rows; ++r)
	{
		for (int c = 0; c < depthImg.cols; ++c)
		{
			pos.z = depthImg.at<ushort>(r, c);
			pos.x = (c - gCxDepth) / gFxDepth * pos.z;
			pos.y = (r - gCyDepth) / gFyDepth * pos.z;
			pos = gDepthToColorRX * pos.x +
				gDepthToColorRY * pos.y +
				gDepthToColorRZ * pos.z +
				gDepthToColort;

			u = pos.x / pos.z * gFxColor + gCxColor;
			v = pos.y / pos.z * gFyColor + gCyColor;
			if (u >= 0 && u < registeredDepthImg.cols && v >= 0 && v < registeredDepthImg.rows)
			{
				depth = pos.z;

				if (depth < registeredDepthImg.at<ushort>(v, u))
				{
					registeredDepthImg.at<ushort>(v, u) = depth;
				}
			}
		}
	}
	for (int r = 0; r < registeredDepthImg.rows; ++r)
	{
		for (int c = 0; c < registeredDepthImg.cols; ++c)
		{
			if ((int)registeredDepthImg.at<ushort>(r, c) == 65535)
			{
				registeredDepthImg.at<ushort>(r, c) = 0;
			}
		}
	}
}
void onTrackbarX(int, void *) {
	gDepthToColort.x = gOriDepthToColort.x + g_xOffset - g_nHalfMaxValue;
	//std::cout << gDepthToColort.x << std::endl;
	RegisterDepthImgCPU(g_registeredDepthImg, g_rawDepthImg);
	cv::Mat colorDepthImg;
	RegisteredImgs(colorDepthImg, g_colorImg, g_registeredDepthImg);
	cv::imshow("colorDepthImg", colorDepthImg);
}
void onTrackbarY(int, void *) {
	gDepthToColort.y = gOriDepthToColort.y + g_yOffset - g_nHalfMaxValue;
	//std::cout << gDepthToColort.y << std::endl;
	RegisterDepthImgCPU(g_registeredDepthImg, g_rawDepthImg);
	cv::Mat colorDepthImg;
	RegisteredImgs(colorDepthImg, g_colorImg, g_registeredDepthImg);
	cv::imshow("colorDepthImg", colorDepthImg);
}

void ObjReconThread::run() {

	//ObjRecon objRecon;
	printf("New ObjRecon Thread\n");
	m_objRecon = new ObjRecon;
	m_isFinish = false;

	m_objRecon->setIntrinExtrin(m_fxDepth, m_fyDepth, m_cxDepth, m_cyDepth,
							  m_depthToColorRX, m_depthToColorRY, m_depthToColorRZ, m_depthToColort,
							  m_fxColor, m_fyColor, m_cxColor, m_cyColor);

	cv::Mat rawDepthImg(Resolution::getInstance().depthHeight(),
						Resolution::getInstance().depthWidth(),
						CV_16UC1);
	cv::Mat colorImg(Resolution::getInstance().height(),
					 Resolution::getInstance().width(),
					 CV_8UC3);
	cv::Mat fullColorImg(Resolution::getInstance().fullColorHeight(),
						 Resolution::getInstance().fullColorWidth(),
						 CV_8UC3);
	cv::Mat colorDepthImg(Resolution::getInstance().height(),
						  Resolution::getInstance().width(),
						  CV_8UC3);
	cv::Mat renderedModelImg;
	std::vector<ImuMsg> imuMeasurements;
	double3 gravity;
	int keyFrameIdxEachFrag;
	innoreal::InnoRealTimer timer;
	int cnt = 0;
	int cntcnt = 0;
	bool isFirstFrame = true;
	bool skip = false;
	while (!pangolin::ShouldQuit() && !m_isFinish)
	{
		if (pangolin::Pushed(*m_objRecon->m_pGui->m_saveButton))
		{
			std::ofstream fs;
			fs.open("C:\\xjm\\snapshot\\before_opt\\all_camera_pose.txt", std::ofstream::binary);
			fs.write((char *)m_objRecon->m_cameraPoseVec.data(), m_objRecon->m_cameraPoseVec.size() * sizeof(Eigen::Matrix4f));
			fs.close();
			std::cout << "frame size: " << m_objRecon->m_cameraPoseVec.size() << std::endl;
			m_objRecon->m_pDeform->saveModel(m_objRecon->m_fragNumInDetectStage);
			//std::exit(0);
		}

		if (!m_objRecon->m_pGui->m_pauseButton->Get() || pangolin::Pushed(*m_objRecon->m_pGui->m_stepButton))
		{
			timer.TimeStart();
#if USE_STRUCTURE_SENSOR
			bool isValid = true;
			//if (m_objRecon->m_timeStamp == 0)
			//{
				//isValid = m_sensor->next(rawDepthImg, colorImg, &imuMeasurements, &gravity, &keyFrameIdxEachFrag);
				//isValid = m_sensor->laterest(rawDepthImg, colorImg, &imuMeasurements, &gravity, &keyFrameIdxEachFrag);
			//}
			//else
			//{
				SkipFrame:
				isValid = m_sensor->next(rawDepthImg, colorImg, &imuMeasurements, &gravity, &keyFrameIdxEachFrag);
				//isValid = m_sensor->laterest(rawDepthImg, colorImg, &imuMeasurements, &gravity, &keyFrameIdxEachFrag);
				if (keyFrameIdxEachFrag == -1)
				{
					//printf("skip frame\n");
					skip = true;
					goto SkipFrame;
				}
			//}
			//printf("keyFrameIdxEachFrag: %d %d\n", m_objRecon->m_timeStamp, keyFrameIdxEachFrag);
			if (isValid == false)
			{
				//std::ofstream fs;
				//fs.open("C:\\xjm\\snapshot\\before_opt\\all_camera_pose.txt", std::ofstream::binary);
				//fs.write((char *)m_objRecon->m_cameraPoseVec.data(), m_objRecon->m_cameraPoseVec.size() * sizeof(Eigen::Matrix4f));
				//fs.close();
				m_objRecon->m_pDeform->saveModel(m_objRecon->m_fragNumInDetectStage);

				return;
			}
			timer.TimeEnd();
#if 1
			std::vector<int> pngCompressionParams;
			pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
			pngCompressionParams.push_back(0);
			char renderedDir1[256], renderedDir2[256];
			sprintf(renderedDir1, "C:\\xjm\\snapshot\\before_opt\\images\\%06d_depth.png", cntcnt);
			sprintf(renderedDir2, "C:\\xjm\\snapshot\\before_opt\\images\\%06d_color.png", cntcnt);
			cv::imwrite(renderedDir1, rawDepthImg * 50, pngCompressionParams);
			cv::imwrite(renderedDir2, colorImg, pngCompressionParams);
			++cntcnt;
#endif
#endif

#if USE_STRUCTURE_SENSOR
			if (GlobalState::getInstance().m_adjustExtrin)
			{
				isFirstFrame = false;
				g_rawDepthImg = rawDepthImg;
				g_colorImg = colorImg;
				RegisterDepthImgCPU(g_registeredDepthImg, rawDepthImg);

				cv::namedWindow("colorDepthImg");
				cv::imshow("colorDepthImg", colorDepthImg);
				sprintf(TrackbarNameX, "x offset: %d", g_nMaxValue);
				sprintf(TrackbarNameY, "y offset: %d", g_nMaxValue);

				gOriDepthToColort = m_depthToColort;
				gDepthToColort = m_depthToColort;
				gDepthToColorRX = m_depthToColorRX;
				gDepthToColorRY = m_depthToColorRY;
				gDepthToColorRZ = m_depthToColorRZ;
				gFxDepth = m_fxDepth, gFyDepth = m_fyDepth, gCxDepth = m_cxDepth, gCyDepth = m_cyDepth;
				gFxColor = m_fxColor, gFyColor = m_fyColor, gCxColor = m_cxColor, gCyColor = m_cyColor;
				cv::createTrackbar(TrackbarNameX, "colorDepthImg", &g_xOffset, g_nMaxValue, onTrackbarX);
				onTrackbarX(g_xOffset, 0);
				cv::createTrackbar(TrackbarNameY, "colorDepthImg", &g_yOffset, g_nMaxValue, onTrackbarY);
				onTrackbarY(g_yOffset, 0);

				cv::Mat colorDepthImg;
				RegisteredImgs(colorDepthImg, colorImg, g_registeredDepthImg);
				cv::imshow("colorDepthImg", colorDepthImg);
				cv::waitKey(0);
				m_depthToColort = gDepthToColort;
				m_objRecon->setIntrinExtrin(m_fxDepth, m_fyDepth, m_cxDepth, m_cyDepth,
																		m_depthToColorRX, m_depthToColorRY, m_depthToColorRZ, m_depthToColort,
																		m_fxColor, m_fyColor, m_cxColor, m_cyColor);
				std::cout << "x offset: " << g_xOffset - g_nHalfMaxValue << std::endl;
				std::cout << "y offset: " << g_yOffset - g_nHalfMaxValue << std::endl;
			}
#endif

			//std::cout << "keyFrameIdxEachFrag: " << keyFrameIdxEachFrag << std::endl;
			//if (keyFrameIdxEachFrag != 0)
			//{
				//std::cout << keyFrameIdxEachFrag << std::endl;
				//Sleep(3000);
			//}
#if 0
			std::cout << "sensor time: " << timer.TimeGap_in_ms() << std::endl;
			std::cout << "gravity: " << gravity.x << std::endl;
			std::cout << "gravity: " << gravity.y << std::endl;
			std::cout << "gravity: " << gravity.z << std::endl;
#endif
			//cv::imshow("rawDepthImg", rawDepthImg * 30);
			//cv::imshow("colorImg", colorImg);
			//cv::imshow("input_color", rawDepthImg * 30);
			//cv::imshow("input_depth", colorImg);
			//cv::waitKey(1);
			//RegisteredImgs(colorDepthImg, colorImg, rawDepthImg);

			//timer.TimeStart();
#if USE_STRUCTURE_SENSOR

			//objRecon->processFrame(rawDepthImg, colorImg, imuMeasurements, gravity, keyFrameIdxEachFrag);
			m_objRecon->processFrameOpt(rawDepthImg, colorImg, imuMeasurements, gravity, keyFrameIdxEachFrag);
			if (GlobalState::getInstance().m_useStructureSensor && !GlobalState::getInstance().m_loadFromBinFile) {
				cv::Mat& renderedModelImg = m_objRecon->getRenderedModelImg();
				memcpy(((RemoteStructureSeneor *)m_sensor)->m_feedBackBuffer.data(), renderedModelImg.data, renderedModelImg.rows * renderedModelImg.cols);
			}
#endif
#if USE_XTION
			objRecon->processFrame(rawDepthImg, colorImg, fullColorImg, imuMeasurements, gravity, keyFrameIdxEachFrag, hasBeenPaused);
#endif
			//timer.TimeEnd();
			//std::cout << "process frame time: " << timer.TimeGap_in_ms() << std::endl;
#if 0
			renderedModelImg = objRecon->getRenderedModelImg();
			cv::imshow("renderedModelImg", renderedModelImg);
			cv::waitKey(1);
#endif
		}
#if 1
		if (m_objRecon->m_pGui->m_debugButton->Get())
		{
			std::cout << "debug" << std::endl;
			m_objRecon->m_pGui->renderSceneForDebug(m_objRecon->m_pDeform);
		}
		else
		{
			m_objRecon->m_pGui->renderScene(m_objRecon->m_pFusion,
																			m_objRecon->m_pDeform,
																			m_objRecon->m_timeStamp,
																			m_objRecon->m_objectBox,
																			m_objRecon->m_boxVec,
																			m_objRecon->m_objectDetected,
																			m_objRecon->m_colorImg,
																			m_objRecon->m_rawDepthImg);
		}
#endif
		xCheckGlDieOnError();
		pangolin::FinishFrame();
		glFinish();
	}
}

void Test3() {
	char imgDir[256], outputDir[256];
	cv::Mat resizedColorImg;
	for (int i = 0; i < 32; ++i)
	{
		sprintf(imgDir, "D:\\xjm\\result\\before_opt\\%06d_key_frame.png", i);
		sprintf(outputDir, "D:\\xjm\\result\\before_opt\\%06d.color.png", i);
		cv::Mat colorImg = cv::imread(imgDir);
		cv::Rect validBox = cv::Rect(0, 32, 1280, 960);
		cv::resize(colorImg(validBox), resizedColorImg, cv::Size(640, 480));
		std::vector<int> pngCompressionParams;
		pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
		pngCompressionParams.push_back(0);
		cv::imwrite(outputDir, resizedColorImg, pngCompressionParams);
	}
	std::exit(0);
}

void test4()
{
	char fileDir[256];
	cv::Mat img, imgResized(240, 320, CV_8UC3), catImg(249 * 4, 329 * 5, CV_8UC3);
	cv::Mat black = cv::Mat::zeros(240, 320, CV_8UC3);
	int cnt = 0;
	for (int r = 0; r < 4; ++r)
	{
		for (int c = 0; c < 5; ++c)
		{
			/*
			if (cnt == 15)
			{
			++c;
			}
			if (cnt == 20)
			goto output;
			*/
			sprintf(fileDir, "D:\\xjm\\data_for_video\\data2\\StructureSensorData_20180531_170507\\keyframes\\%06d_key_frame.png", cnt + 2);
			if (cnt == 18)
			{
				sprintf(fileDir, "D:\\xjm\\data_for_video\\data2\\StructureSensorData_20180531_170507\\keyframes\\%06d_key_frame.png", 0);
			}
			if (cnt == 19)
			{
				sprintf(fileDir, "D:\\xjm\\data_for_video\\data2\\StructureSensorData_20180531_170507\\keyframes\\%06d_key_frame.png", 1);
			}
			img = cv::imread(fileDir, cv::IMREAD_COLOR);

			cv::resize(img, imgResized, cv::Size(imgResized.cols, imgResized.rows));
			cv::namedWindow("hehe");
			cv::imshow("hehe", imgResized);
			cv::waitKey(1);

			imgResized.copyTo(catImg(cv::Rect(c * 329, r * 249, 320, 240)));
			++cnt;
		}
	}
	cv::namedWindow("hehe");
	cv::imshow("hehe", catImg);
	cv::waitKey(0);
	std::vector<int> pngCompressionParams;
	pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	pngCompressionParams.push_back(0);
	cv::imwrite("D:\\xjm\\data_for_video\\data2\\catImg.png", catImg, pngCompressionParams);
	std::exit(0);
}

void Test5() {
	int frameNum = 499;
	int fragNum = frameNum / FRAG_SIZE;
	std::vector<Eigen::Matrix4f> frameCameraPoseVec(frameNum);
	std::ifstream fs;
	fs.open("C:\\xjm\\snapshot\\before_opt\\all_camera_pose.txt", std::ofstream::binary);
	fs.read((char *)frameCameraPoseVec.data(), frameCameraPoseVec.size() * sizeof(Eigen::Matrix4f));
	fs.close();

	std::vector<Eigen::Matrix4f> updatedFragCameraPoseVec(fragNum);
	fs.open("C:\\xjm\\snapshot\\before_opt\\camera_pose.txt", std::ofstream::binary);
	fs.read((char *)updatedFragCameraPoseVec.data(), updatedFragCameraPoseVec.size() * sizeof(Eigen::Matrix4f));
	fs.close();

	std::vector<Eigen::Matrix4f> originFragCameraPoseVec(fragNum);
	fs.open("C:\\xjm\\snapshot\\before_opt\\camera_pose_original.txt", std::ofstream::binary);
	fs.read((char *)originFragCameraPoseVec.data(), originFragCameraPoseVec.size() * sizeof(Eigen::Matrix4f));
	fs.close();

	//std::ofstream outfs;
	//outfs.open("C:\\xjm\\snapshot\\before_opt\\total_camera_poses_ascii.txt");
	std::vector<Matrix4f> pose_vec;
	for (int i = 0; i < frameCameraPoseVec.size(); ++i) {
		//outfs << i + 1 << std::endl;
		//outfs << updatedFragCameraPoseVec[i / 50] * originFragCameraPoseVec[i / 50].inverse() * frameCameraPoseVec[i] << std::endl;
		//outfs << frameCameraPoseVec[i] << std::endl;
		pose_vec.push_back(updatedFragCameraPoseVec[i / 50] * originFragCameraPoseVec[i / 50].inverse() * frameCameraPoseVec[i]);
		//std::cout << updatedFragCameraPoseVec[i / 50] << std::endl;
		//std::cout << originFragCameraPoseVec[i / 50] << std::endl;
		std::cout << frameCameraPoseVec[i] << std::endl;
	}
	std::exit(0);
	//outfs.close();

	std::string line_buffer;
	std::ifstream input_file;
	input_file.open("C:\\xjm\\snapshot\\before_opt\\mesh_poisson.obj");

	std::vector<float3> vertex_pos_vec;

	char str[256];
	float3 pos = { 0.0f, 0.0f, 0.0f };
	int3 vertex_index;
	while (std::getline(input_file, line_buffer)) {

		if (line_buffer[0] == 'v' && line_buffer[1] == ' ') {
			sscanf(line_buffer.data(), "%s %f %f %f", &str, &pos.x, &pos.y, &pos.z);
			vertex_pos_vec.emplace_back(pos);
			continue;
		}
	}

	input_file.close();
	std::cout << 1 << std::endl;

	std::ofstream output_file;
	output_file.open("C:\\xjm\\snapshot\\before_opt\\test.obj");

	for (int i = 0; i < vertex_pos_vec.size(); ++i) {
		float3 pos = vertex_pos_vec[i];
		output_file << "v" << " " 
		<< pos.x << " " << pos.y << " " << pos.z << " " 
		<< 0.8 << " " << 0.8 << " " << 0.8 << std::endl;
	}
	//output_file.close();
	//std::exit(0);
	float scale = 0.001;
	for (int i = 0; i < pose_vec.size(); ++i) {	
		std::cout << pose_vec[i] << std::endl;
		continue;
		for (int j = 0; j < 100; ++j) {
			Eigen::Vector4f pos = pose_vec[i].col(3) + pose_vec[i].col(0) * j * scale;
			output_file << "v" << " "
				<< pos.x() << " " << pos.y() << " " << pos.z() << " "
				<< 0.8 << " " << 0.0 << " " << 0.0 << std::endl;
		}
		for (int j = 0; j < 100; ++j) {
			Eigen::Vector4f pos = pose_vec[i].col(3) + pose_vec[i].col(1) * j * scale;
			output_file << "v" << " "
				<< pos.x() << " " << pos.y() << " " << pos.z() << " "
				<< 0.0 << " " << 0.8 << " " << 0.0 << std::endl;
		}
		for (int j = 0; j < 100; ++j) {
			Eigen::Vector4f pos = pose_vec[i].col(3) + pose_vec[i].col(2) * j * scale;
			output_file << "v" << " "
				<< pos.x() << " " << pos.y() << " " << pos.z() << " "
				<< 0.0 << " " << 0.0 << " " << 0.8 << std::endl;
		}
	}

	output_file.close();
}

#if 0
int main(int argc, char** argv)
{
#if 0
	Test5();
	std::exit(0);
#endif
#if 0
	Test3();
#endif
#if 0
	test4();
#endif
	float imgScale = 1.0f;
#if USE_STRUCTURE_SENSOR
	int depthWidth = 640, depthHeight = 480,
		colorWidth = 648, colorHeight = 484,
		fullColorWidth = 2592, fullColorHeight = 1936;
	double intrinScale = 648 / (double)640;
	float fx = 544.8898 * intrinScale,
		fy = 545.9078 * intrinScale,
		cx = 321.6016 * intrinScale,
		cy = 237.0330 * intrinScale;	
#endif
#if USE_XTION
	int depthWidth = 640, depthHeight = 480,
		colorWidth = 640, colorHeight = 480,
		fullColorWidth = 1280, fullColorHeight = 1024;
	float fx = 525.0f,
		fy = 525.0f,
		cx = 319.5f,
		cy = 239.5f;
#endif

	Resolution::getInstance(depthWidth, depthHeight,
							colorWidth, colorHeight,
							fullColorWidth, fullColorHeight, imgScale);
	Intrinsics::getInstance(fx, fy, cx, cy, imgScale);

#if USE_XTION
	xCapture* capture = new xImageImporter2(imgScale);
	//xCapture* capture = new xOniCapture(width, height, 30, width, height, 30);
	float intrinColor[4] = { fx, fy, cx, cy };
	float intrinDepth[4] = { fx, fy, cx, cy };
	float extrinT[3] = { 0.0, 0.0, 0.0 };
	float extrinR[9] = {
		1.0000, 0.0000, 0.0000,
		0.0000, 1.0000, 0.0000,
		0.0000, 0.0000, 1.0000
	};
#endif
#if USE_STRUCTURE_SENSOR
#if 1
	//RemoteStructureSeneorFromFile capture(
		//"C:\\xjm\\snapshot\\StructureSensorData_20180731_163658\\StructureSensorData_20180731_163658.bin");
	RemoteStructureSeneorFromFile *capture = new RemoteStructureSeneorFromFile(
		"C:\\xjm\\snapshot\\StructureSensorData_20181130_000441\\StructureSensorData_20181130_000441.bin");
	float intrinColor[4] = { fx, fy, cx, cy };
	float intrinDepth[4] = { 574.0135, 575.5523, 314.5388, 242.4793 };
	float extrinT[3] = { -41.1776, -4.3666, -34.8012 };
#if 1
	float extrinR[9] = {
		1.0000, -0.0040, -0.0029,
		0.0040, 0.9999, 0.0132,
		0.0028, -0.0132, 0.9999
	};
#endif
#if 0
	float extrinR[9] = {
		1.0000, -0.0000, -0.0000,
		0.0000, 1.0000, 0.0000,
		0.0000, -0.0000, 1.0000
	};
#endif
#endif
#if 0
	xCapture* capture = new xImageImporter(imgScale);
	float intrinColor[4] = { fx, fy, cx, cy };
	float intrinDepth[4] = { fx, fy, cx, cy };
	float extrinT[3] = { 0.0, 0.0, 0.0 };
	float extrinR[9] = {
		1.0000, 0.0000, 0.0000,
		0.0000, 1.0000, 0.0000,
		0.0000, 0.0000, 1.0000
	};
#endif
#endif

	ObjReconThread objReconThread(capture);
	objReconThread.setIntrinExtrin(extrinR, extrinT, intrinColor, intrinDepth);
	//objReconThread.setSensor(capture);
	objReconThread.start();
	while (true)
	{
		Sleep(1000);
	}

	return 0;
}
#endif

#if 0
int main(int argc, char** argv) {
	float imgScale = 1.0f;
	int width = 640, height = 480, widthFullColor = 1920, heightFullColor = 1080;
	float fx = 544.8898, fy = 545.9078, cx = 321.6016, cy = 237.0330;
	Resolution::getInstance(width, height, widthFullColor, heightFullColor, imgScale);
	Intrinsics::getInstance(fx, fy, cx, cy, imgScale);

	//xCapture capture(imgScale);
	//capture.start();	
	//xOniCapture capture(width, height, 30, width, height, 30);
	RemoteStructureSeneorFromFile capture("D:\\xjm\\snapshot\\StructureSensorData_20180417_152713.bin");
	//RemoteStructureSeneorFromFile capture("D:\\xjm\\snapshot\\StructureSensorData_20180409_160031.bin");	

	float intrinColor[4] = { 544.8898, 545.9078, 321.6016, 237.0330 };
	float intrinDepth[4] = { 574.0135, 575.5523, 314.5388, 242.4793 };
	float extrinT[3] = { -41.1776, -4.3666, -34.8012 };
	float extrinR[9] = {
		1.0000, -0.0040, -0.0029,
		0.0040, 0.9999, 0.0132,
		0.0028, -0.0132, 0.9999
	};
	capture.setIntrinExtrin(extrinR, extrinT, intrinColor, intrinDepth);

	ObjRecon* objRecon = new ObjRecon();

	cv::Mat rawDepthImg(Resolution::getInstance().height(),
						Resolution::getInstance().width(),
						CV_16UC1);
	cv::Mat colorImg(Resolution::getInstance().height(),
					 Resolution::getInstance().width(),
					 CV_8UC3);
	cv::Mat colorDepthImg(Resolution::getInstance().height(),
						  Resolution::getInstance().width(),
						  CV_8UC3);
	cv::Mat renderedModelImg;
	std::vector<ImuMsg> imuMeasurements;
	double3 gravity;
	while (!pangolin::ShouldQuit())
	{
		if (pangolin::Pushed(*objRecon->m_pGui->m_saveButton))
		{
			char fileDir[256];
			sprintf(fileDir, "D:\\xjm\\snapshot\\model_%06d_snapshot.ply", static_cast<int>(objRecon->m_timeStamp));
			objRecon->m_pDeform->savePly(fileDir, -1);
		}

		//if (!pGui->m_pauseButton->Get() || pangolin::Pushed(*pGui->m_stepButton))
		{
			// nextº¯ÊýÊÇ×èÈûµÄ
			capture.next(rawDepthImg, colorImg, imuMeasurements, gravity, true);
			RegesteredImgs(colorDepthImg, colorImg, rawDepthImg);

			objRecon->processFrame(rawDepthImg, colorImg, imuMeasurements, gravity);
			renderedModelImg = objRecon->getRenderedModelImg();
			cv::imshow("renderedModelImg", renderedModelImg);
			cv::waitKey(1);
		}
	}
	return 0;
}
#endif

#if 0
std::vector<Eigen::Matrix4f> originalCameraPoses;
std::vector<Eigen::Matrix4f> originalFragmentCameraPoses;

void RenderScene(xSurfelFusion *pFusion, pangolin::OpenGlRenderState &cameraState, int timeStamp,
				 Box &objectBox, std::vector<Box> &boxVec, bool objectDetected, cv::Mat &colorImg, cv::Mat &depthImg) {
#if 1
	pangolin::OpenGlMatrix mv;

	Eigen::Matrix4f currPose = pFusion->getCurrPose();
	Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

	Eigen::Quaternionf currQuat(currRot);
	Eigen::Vector3f forwardVector(0, 0, 1);
	Eigen::Vector3f upVector(0, -1, 0);

	Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
	Eigen::Vector3f up = (currQuat * upVector).normalized();

	Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

	//eye -= forward;

	Eigen::Vector3f at = eye + forward;

	Eigen::Vector3f z = (eye - at).normalized();  // Forward
	Eigen::Vector3f x = up.cross(z).normalized(); // Right
	Eigen::Vector3f y = z.cross(x);

	Eigen::Matrix4d m;
	m << x(0), x(1), x(2), -(x.dot(eye)),
		y(0), y(1), y(2), -(y.dot(eye)),
		z(0), z(1), z(2), -(z.dot(eye)),
		0, 0, 0, 1;

	memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

	cameraState.SetModelViewMatrix(mv);
#endif

	//glClearColor(0.05 * 1, 0.05 * 1, 0.3 * 1, 0.0f);
	glClearColor(0.0, 0.0, 0.0, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	pangolin::Display("cam").Activate(cameraState);
	pFusion->getGlobalModel().renderPointCloud(cameraState.GetProjectionModelViewMatrix(),
											   cameraState.GetModelViewMatrix(),
											   20,
											   true,
											   false,
											   true,
											   false,
											   false,
											   false,
											   timeStamp,
											   200);

#if 0
	Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
	K(0, 0) = Intrinsics::getInstance().fx();
	K(1, 1) = Intrinsics::getInstance().fy();
	K(0, 2) = Intrinsics::getInstance().cx();
	K(1, 2) = Intrinsics::getInstance().cy();

	Eigen::Matrix3f Kinv = K.inverse();

	glColor3f(0, 1, 1);
	glLineWidth(3);
	pangolin::glDrawFrustrum(Kinv,
							 Resolution::getInstance().width(),
							 Resolution::getInstance().height(),
							 pFusion->getCurrPose(),
							 0.05f);

	if (pFusion->m_pDeform->m_keyPoseVec.size() > 0)
		originalFragmentCameraPoses.push_back(pFusion->m_pDeform->m_keyPoseVec.back());

	if (pFusion->m_pDeform->m_poseVec.size() > 0)
		originalCameraPoses.push_back(pFusion->m_pDeform->m_poseVec.back());

	std::vector<Eigen::Vector3f> positions;
	for (size_t i = 0; i < originalCameraPoses.size(); i++)
	{
		Eigen::Vector3f val;
		val[0] = originalCameraPoses[i].col(3)[0];
		val[1] = originalCameraPoses[i].col(3)[1];
		val[2] = originalCameraPoses[i].col(3)[2];
		positions.push_back(val);
		/*
		pangolin::glDrawFrustrum(Kinv,
		Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		originalCameraPoses[i],
		0.005f);
		*/
	}

	glColor3f(0.8, 0.8, 0);
	pangolin::glDrawLineStrip(positions);

	for (size_t i = 0; i < pFusion->m_pDeform->m_keyPoseVec.size(); i++)
	{
		glColor3f(1, 0, 0);
		pangolin::glDrawFrustrum(Kinv,
								 Resolution::getInstance().width(),
								 Resolution::getInstance().height(),
								 pFusion->m_pDeform->m_keyPoseVec[i],
								 0.05f);
	}
#if 0
	for (size_t i = 0; i < originalFragmentCameraPoses.size(); i++)
	{
		glColor3f(0.7, 0.3, 0.3);
		pangolin::glDrawFrustrum(Kinv,
								 Resolution::getInstance().width(),
								 Resolution::getInstance().height(),
								 originalFragmentCameraPoses[i],
								 0.05f);
	}
#endif

#endif

	glReadPixels(0, 0, 2 * Resolution::getInstance().width(), 2 * Resolution::getInstance().height(), GL_RGB, GL_UNSIGNED_BYTE, renderedImg.data);
	//cv::resize(renderedImg, resizedRenderedImg, cv::Size(Resolution::getInstance().width(), Resolution::getInstance().height()));
	resizedRenderedImg = renderedImg;
	cv::flip(resizedRenderedImg, resizedRenderedImg, 0);
#if 0
	resizedRenderedImg = depthImg.clone();
	resizedRenderedImg = resizedRenderedImg * 40;
#endif
#if 1
	cv::cvtColor(resizedRenderedImg, resizedRenderedImg, CV_BGR2RGB);
	if (objectDetected == false)
	{
		cv::rectangle(resizedRenderedImg, cv::Rect(320 - 17, 240 - 13, 35, 27),
					  cv::Scalar(0, 255, 255), 4);
		cv::line(resizedRenderedImg, cv::Point(320 - 17, 240 - 13), cv::Point(320 + 17, 240 + 13),
				 cv::Scalar(0, 255, 255), 4);
		cv::line(resizedRenderedImg, cv::Point(320 - 17, 240 + 13), cv::Point(320 + 17, 240 - 13),
				 cv::Scalar(0, 255, 255), 4);
	}
#endif
#if 1
	if (objectDetected == false)
	{
		for (int i = 0; i < boxVec.size(); ++i)
		{
			Box &box = boxVec[i];
			cv::rectangle(resizedRenderedImg, cv::Rect(box.m_left, box.m_top, box.m_right - box.m_left, box.m_bottom - box.m_top),
						  cv::Scalar(188, 188, 0), 4);
		}
	}
#endif
#if 1
	if (objectDetected == false)
	{
		cv::rectangle(resizedRenderedImg, cv::Rect(objectBox.m_left, objectBox.m_top, objectBox.m_right - objectBox.m_left, objectBox.m_bottom - objectBox.m_top),
					  cv::Scalar(0, 0, 255), 4);
	}
#endif
	cv::namedWindow("rendered img");
	cv::imshow("rendered img", resizedRenderedImg);
	cv::waitKey(1);

#if 0
	std::vector<int> pngCompressionParams;
	pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	pngCompressionParams.push_back(0);
	sprintf(renderedDir, "D:\\xjm\\result\\for_demo\\new_new_data\\test3\\%04d.png", totalCntTmp);
	cv::imwrite(renderedDir, resizedRenderedImg, pngCompressionParams);
#endif

	pangolin::FinishFrame();

	glFinish();
#if 0
#if 1
	if (true)//objectDetected == true)
	{
		pangolin::OpenGlMatrix mv;

		Eigen::Matrix4f currPose = pFusion->getCurrPose();
		Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

		Eigen::Quaternionf currQuat(currRot);
		Eigen::Vector3f forwardVector(0, 0, 1);
		Eigen::Vector3f upVector(0, -1, 0);

		Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
		Eigen::Vector3f up = (currQuat * upVector).normalized();

		Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

#if 0
		eye -= forward;
#endif

		Eigen::Vector3f at = eye + forward;

		Eigen::Vector3f z = (eye - at).normalized();  // Forward
		Eigen::Vector3f x = up.cross(z).normalized(); // Right
		Eigen::Vector3f y = z.cross(x);

		Eigen::Matrix4d m;
		m << x(0), x(1), x(2), -(x.dot(eye)),
			y(0), y(1), y(2), -(y.dot(eye)),
			z(0), z(1), z(2), -(z.dot(eye)),
			0, 0, 0, 1;

		memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

		cameraState.SetModelViewMatrix(mv);
	} else
	{
		pangolin::OpenGlMatrix mv;

		Eigen::Matrix4f currPose = Eigen::Matrix4f::Identity();
		Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

		Eigen::Quaternionf currQuat(currRot);
		Eigen::Vector3f forwardVector(0, 0, 1);
		Eigen::Vector3f upVector(0, -1, 0);

		Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
		Eigen::Vector3f up = (currQuat * upVector).normalized();

		Eigen::Vector3f eye(0, 0, 0);

		if (objectDetected == true)
		{
			eye -= forward * 0.6f;
		}

		Eigen::Vector3f at = eye + forward;

		Eigen::Vector3f z = (eye - at).normalized();  // Forward
		Eigen::Vector3f x = up.cross(z).normalized(); // Right
		Eigen::Vector3f y = z.cross(x);

		Eigen::Matrix4d m;
		m << x(0), x(1), x(2), -(x.dot(eye)),
			y(0), y(1), y(2), -(y.dot(eye)),
			z(0), z(1), z(2), -(z.dot(eye)),
			0, 0, 0, 1;

		memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

		cameraState.SetModelViewMatrix(mv);
	}
#endif

	//glClearColor(0.05 * 1, 0.05 * 1, 0.3 * 1, 0.0f);
	glClearColor(0.0, 0.0, 0.0, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	pangolin::Display("cam").Activate(cameraState);
	pFusion->getGlobalModel().renderPointCloud(cameraState.GetProjectionModelViewMatrix(),
											   cameraState.GetModelViewMatrix(),
											   25,
											   true,
											   false,
											   true,
											   false,
											   false,
											   false,
											   timeStamp,
											   200);

	Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
	K(0, 0) = Intrinsics::getInstance().fx();
	K(1, 1) = Intrinsics::getInstance().fy();
	K(0, 2) = Intrinsics::getInstance().cx();
	K(1, 2) = Intrinsics::getInstance().cy();

	Eigen::Matrix3f Kinv = K.inverse();

#if 0
	if (objectDetected == true)
	{
		Eigen::Matrix4f pose = pFusion->getCurrPose();
		//pose.col(3).head(3) += forward * 0.1;
		glColor3f(0, 1, 1);
		glLineWidth(4);
		pangolin::glDrawFrustrum(Kinv,
								 Resolution::getInstance().width(),
								 Resolution::getInstance().height(),
								 pose,
								 0.05f);
	}

	if (objectDetected == true)
	{
		if (pFusion->m_pDeform->m_keyPoseVec.size() > 0)
			originalFragmentCameraPoses.push_back(pFusion->m_pDeform->m_keyPoseVec.back());

		if (pFusion->m_pDeform->m_poseVec.size() > 0)
			originalCameraPoses.push_back(pFusion->m_pDeform->m_poseVec.back());

		for (size_t i = 0; i < pFusion->m_pDeform->m_keyPoseVec.size(); i++)
		{
			glColor3f(1, 0, 0);
			glLineWidth(4);
			pangolin::glDrawFrustrum(Kinv,
									 Resolution::getInstance().width(),
									 Resolution::getInstance().height(),
									 pFusion->m_pDeform->m_keyPoseVec[i],
									 0.05f);
		}

		std::vector<Eigen::Vector3f> positions;
		for (size_t i = 0; i < originalCameraPoses.size(); i++)
		{
			Eigen::Vector3f val;
			val[0] = originalCameraPoses[i].col(3)[0];
			val[1] = originalCameraPoses[i].col(3)[1];
			val[2] = originalCameraPoses[i].col(3)[2];
			positions.push_back(val);
			/*
			pangolin::glDrawFrustrum(Kinv,
				Resolution::getInstance().width(),
				Resolution::getInstance().height(),
				originalCameraPoses[i],
				0.005f);
				*/
		}

		glColor3f(0.8, 0.8, 0);
		glLineWidth(4);
		pangolin::glDrawLineStrip(positions);
	}
#if 0
	for (size_t i = 0; i < originalFragmentCameraPoses.size(); i++)
	{
		glColor3f(0.7, 0.3, 0.3);
		pangolin::glDrawFrustrum(Kinv,
								 Resolution::getInstance().width(),
								 Resolution::getInstance().height(),
								 originalFragmentCameraPoses[i],
								 0.05f);
	}
#endif

#endif

#if 1
	glReadPixels(0, 0, 2 * Resolution::getInstance().width(), 2 * Resolution::getInstance().height(), GL_RGB, GL_UNSIGNED_BYTE, renderedImg.data);
	cv::resize(renderedImg, resizedRenderedImg, cv::Size(Resolution::getInstance().width(), Resolution::getInstance().height()));
	cv::flip(resizedRenderedImg, resizedRenderedImg, 0);
#if 0
	resizedRenderedImg = depthImg.clone();
	resizedRenderedImg = resizedRenderedImg * 40;
#endif
#if 1
	cv::cvtColor(resizedRenderedImg, resizedRenderedImg, CV_BGR2RGB);
	if (objectDetected == false)
	{
		cv::rectangle(resizedRenderedImg, cv::Rect(320 - 17, 240 - 13, 35, 27),
					  cv::Scalar(0, 255, 255), 4);
		cv::line(resizedRenderedImg, cv::Point(320 - 17, 240 - 13), cv::Point(320 + 17, 240 + 13),
				 cv::Scalar(0, 255, 255), 4);
		cv::line(resizedRenderedImg, cv::Point(320 - 17, 240 + 13), cv::Point(320 + 17, 240 - 13),
				 cv::Scalar(0, 255, 255), 4);
	}
#endif
#if 1
	if (objectDetected == false)
	{
		for (int i = 0; i < boxVec.size(); ++i)
		{
			Box &box = boxVec[i];
			cv::rectangle(resizedRenderedImg, cv::Rect(box.m_left, box.m_top, box.m_right - box.m_left, box.m_bottom - box.m_top),
						  cv::Scalar(188, 188, 0), 4);
		}
	}
#endif
#if 1
	if (objectDetected == false)
	{
		cv::rectangle(resizedRenderedImg, cv::Rect(objectBox.m_left, objectBox.m_top, objectBox.m_right - objectBox.m_left, objectBox.m_bottom - objectBox.m_top),
					  cv::Scalar(0, 0, 255), 4);
	}
#endif
	cv::namedWindow("rendered img");
	cv::imshow("rendered img", resizedRenderedImg);
	cv::waitKey(1);

#if 1
	std::vector<int> pngCompressionParams;
	pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	pngCompressionParams.push_back(0);
	sprintf(renderedDir, "D:\\xjm\\result\\for_demo\\new_new_data\\test3\\%04d.png", totalCntTmp);
	cv::imwrite(renderedDir, resizedRenderedImg, pngCompressionParams);
#endif
#endif

	pangolin::FinishFrame();

	glFinish();
#endif
}

void SaveModel(int fragNum, xDeformation *pDeform, float imgScale) {
	std::cout << "save model" << std::endl;
#if 0
	FetchColor(pDeform->m_vboDevice,
			   pDeform->m_fragNum,
			   pDeform->m_keyFullColorImgsDevice,
			   Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(), Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
			   pDeform->m_width, pDeform->m_height);
#endif
	int width = Resolution::getInstance().width();
	int height = Resolution::getInstance().height();
	cv::Mat keyColorImg(height, width, CV_8UC3), keyGrayImg(height, width, CV_8UC1);
	cv::Mat keyColorImgResized(height / imgScale, width / imgScale, CV_8UC3);
	std::vector<int> pngCompressionParams;
	pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	pngCompressionParams.push_back(0);
	std::ofstream fs;
	fs.open("D:\\xjm\\result\\before_opt\\camera_pose.txt", std::ofstream::binary);

	char plyDir[256], keyColorDir[256], keyFullColorDir[256], keyGrayDir[256], keyDepthDir[256];
	float4 camPose[4], invCamPose[4];
	for (int fragInd = 0; fragInd < fragNum; ++fragInd)
	{
		if (pDeform->m_isFragValid[fragInd] > 0)
		{
			checkCudaErrors(cudaMemcpy(camPose,
									   pDeform->m_updatedKeyPosesDevice + 4 * fragInd,
									   4 * sizeof(float4), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(invCamPose,
									   pDeform->m_updatedKeyPosesInvDevice + 4 * fragInd,
									   4 * sizeof(float4), cudaMemcpyDeviceToHost));
			std::cout << camPose[0].x << " " << camPose[0].y << " " << camPose[0].z << " " << camPose[0].w <<
				camPose[1].x << " " << camPose[1].y << " " << camPose[1].z << " " << camPose[1].w <<
				camPose[2].x << " " << camPose[2].y << " " << camPose[2].z << " " << camPose[2].w <<
				camPose[3].x << " " << camPose[3].y << " " << camPose[3].z << " " << camPose[3].w;
			fs.write((char *)camPose, 4 * sizeof(float4));

			sprintf(plyDir, "D:\\xjm\\result\\before_opt\\%d.ply", fragInd);
			sprintf(keyColorDir, "D:\\xjm\\result\\before_opt\\%06d.color.png", fragInd);
			sprintf(keyFullColorDir, "D:\\xjm\\result\\before_opt\\%06d.fullcolor.png", fragInd);
			//sprintf(keyGrayDir, "D:\\xjm\\result\\before_opt\\%06d.gray.png", fragInd);
//sprintf(keyDepthDir, "D:\\xjm\\result\\before_opt\\%06d.depth.png", fragInd);

			//pDeform->savePly(plyDir, fragInd, imgScale, invCamPose);
			checkCudaErrors(cudaMemcpy(keyColorImg.data,
									   pDeform->m_keyColorImgsDevice.first + width * height * 3 * fragInd,
									   width * height * 3, cudaMemcpyDeviceToHost));
			cv::resize(keyColorImg, keyColorImgResized, keyColorImgResized.size());
			/*
			checkCudaErrors(cudaMemcpy(keyGrayImg.data,
				pDeform->m_keyGrayImgsDevice.first + width * height * fragInd,
				width * height, cudaMemcpyDeviceToHost));
				*/
				/*
				cv::namedWindow("hehe");
				cv::imshow("hehe", keyColorImg);
				cv::waitKey(0);
				*/
				//cv::imwrite(keyColorDir, keyColorImgResized, pngCompressionParams);
				//cv::imwrite(keyFullColorDir, pDeform->m_keyFullColorImgVec[fragInd], pngCompressionParams);
				//cv::imwrite(keyDepthDir, pDeform->m_keyDepthImgVec[fragInd], pngCompressionParams);
							//cv::imwrite(keyGrayDir, keyGrayImg, pngCompressionParams);	

			std::cout << "save frag: " << fragInd << std::endl;
		}
	}
	fs.close();
	exit(0);
}
#endif
