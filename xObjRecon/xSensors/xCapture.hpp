#pragma once

#include <QtCore/QThread>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#if 0
#include <OpenNI.h>
#include <PS1080.h>
#endif
#include "Helpers/xUtils.h"

class xCapture
{
public:
	virtual bool next(cv::Mat& depthImg, cv::Mat& colorImg,
					  ImuMeasurements* imuMeasurements, Gravity* gravity,
					  int* keyFrameIdxEachFrag) = 0;
	virtual bool nextForXTION(cv::Mat& depthImg, cv::Mat& colorImg, cv::Mat& fullColorImg,
							  ImuMeasurements* imuMeasurements, Gravity* gravity,
							  int* keyFrameIdxEachFrag)
	{
		return false;
	}
	virtual bool laterest(cv::Mat& depthImg, cv::Mat& colorImg,
						  ImuMeasurements* imuMeasurements, Gravity* gravity,
						  int* keyFrameIdxEachFrag)
	{
		std::cout << "xCapture laterest" << std::endl;
		return false;
	}
};

#if 0

#define INPUT_DIR "D:\\xjm\\result\\ieee_vr_oni_dataset\\demo_31\\match\\"
#define INPUT_DIR_AFTER_TRACK "D:\\xjm\\result\\ieee_vr_oni_dataset\\demo_31\\aftertrack\\"
//#define INPUT_DIR "D:\\xjm\\result\\for_demo\\data\\"//"D:\\xjm\\result\\ieee_vr_oni_dataset\\demo_31\\match\\"
//#define INPUT_DIR_AFTER_TRACK "D:\\xjm\\result\\for_demo\\data\\"
//#define INPUT_DIR "D:\\xjm\\result\\for_demo\\data\\"

class xCapture : public QThread 
{
public:	
	explicit xCapture(float imgScale) : m_imgScale(imgScale)
	{
		m_depthVec.resize(m_bufLen);
		m_colorVec.resize(m_bufLen);
		m_fullColorVec.resize(m_bufLen);
		m_readInd = m_writeInd = 1;// 100;
	}

	virtual ~xCapture()
	{
		
	}

	void run()
	{
		cv::Mat depthImg, colorImg;// , fullColorImg;
		char depthPath[256], colorPath[256];// , fullColorPath[256];
		while (true)
		{	
			sprintf(depthPath, INPUT_DIR_AFTER_TRACK"frame-%06d.depth.png", m_writeInd);
			sprintf(colorPath, INPUT_DIR_AFTER_TRACK"frame-%06d.color.png", m_writeInd);
			//sprintf(fullColorPath, INPUT_DIR"frame-%06d.fullcolor.png", m_writeInd);
		
			//std::cout << depthPath << std::endl;
			//std::cout << colorPath << std::endl;
			//std::cout << fullColorPath << std::endl;

			depthImg = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
			colorImg = cv::imread(colorPath, cv::IMREAD_COLOR);
			//fullColorImg = cv::imread(fullColorPath, cv::IMREAD_COLOR);

			if (depthImg.rows == 0 || colorImg.rows == 0)
			{
				std::cout << "exit thread" << std::endl;
				quit();
			}

			//std::cout << "write ind: " << m_writeInd << std::endl;
			while ((m_writeInd % m_bufLen + 1) % m_bufLen == m_readInd % m_bufLen)
			{
			}
			m_depthVec[m_writeInd % m_bufLen] = depthImg.clone();
			m_colorVec[m_writeInd % m_bufLen] = colorImg.clone();
			//m_fullColorVec[m_writeInd % m_bufLen] = fullColorImg.clone();

			++m_writeInd;
		}
	}

	bool next(cv::Mat &depthImg, cv::Mat &colorImg)
	{
		while (m_writeInd % m_bufLen== m_readInd % m_bufLen)
		{
			if (isFinished())
			{
				std::cout << "thread finished" << std::endl;
				return false;
			}
		}
		//std::cout << "read ind: " << m_readInd << std::endl;	

		m_depthImg = m_depthVec[m_readInd % m_bufLen];
		m_colorImg = m_colorVec[m_readInd % m_bufLen];
		//fullColorImg = m_fullColorVec[m_readInd % m_bufLen];

		//resizeDepthImg(depthImg);
		//resizeColorImg(colorImg);
		depthImg = m_depthImg;
		colorImg = m_colorImg;
		
		++m_readInd;	

		return true;
	}

	void resizeDepthImg(cv::Mat &resizedDepthImg)
	{
		if (m_imgScale >= 1.0 - MYEPS && m_imgScale <= 1.0 - MYEPS)
		{
			resizedDepthImg = m_depthImg;
			return;
		}
		// We do not use the opencv resize because it will add a gaussian step
		ushort *rowPtrTarget, *rowPtrSrc;
		int row, col;
		for (int r = 0; r < resizedDepthImg.rows; ++r)
		{
			rowPtrTarget = (ushort *)resizedDepthImg.ptr(r);
			row = r / m_imgScale + 0.5f;
			if (row >= 0 && row < m_depthImg.rows)
			{
				rowPtrSrc = (ushort *)m_depthImg.ptr(row);
//#pragma omp for
				for (int c = 0; c < resizedDepthImg.cols; ++c)
				{
					col = c / m_imgScale + 0.5f;
					if (col >= 0 && col < m_depthImg.cols)
					{
						rowPtrTarget[c] = rowPtrSrc[col];
					}
				}
			}
		}
		/*
		cv::namedWindow("resized depth img");
		cv::imshow("resized depth img", resizedDepthImg * 10);
		cv::waitKey(0);
		*/
	}

	void resizeColorImg(cv::Mat &resizedColorImg)
	{
		if (m_imgScale >= 1.0 - MYEPS && m_imgScale <= 1.0 - MYEPS)
		{
			resizedColorImg = m_colorImg;
			return;
		}
		cv::resize(m_colorImg, resizedColorImg, resizedColorImg.size());
		/*
		cv::namedWindow("resized color img");
		cv::imshow("resized color img", resizedColorImg);
		cv::waitKey(0);
		*/
	}

	void resizeColorImgWithFullColor(cv::Mat &&colorImg, cv::Mat &fullColorImg, std::vector<float> &coordMapVec, float imgScale)
	{
		float xColor, yColor;
		cv::Mat_<int> yMat(colorImg.rows, colorImg.cols), xMat(colorImg.rows, colorImg.cols);
		for (int r = 0; r < colorImg.rows; ++r)
		{
			for (int c = 0; c < colorImg.cols; ++c)
			{	
				int ind;
				ind = r * colorImg.cols + c;
				xColor = coordMapVec[2 * ind] + 0.5;
				yColor = coordMapVec[2 * ind + 1] + 0.5;
				if (xColor > 0 && xColor < fullColorImg.cols && yColor > 0 && yColor < fullColorImg.rows
					&& !isnan(xColor) && !isnan(yColor))
				{	
					yMat(r, c) = yColor;
					xMat(r, c) = xColor;
				}
				else
				{
					yMat(r, c) = -1;
					xMat(r, c) = -1;
				}
			}
		}
		float v, u, v0, v1, u0, u1, coef;
		float xCent, yCent, xTop, xBottom, yTop, yBottom;
		for (int r = 0; r < colorImg.rows * 2; ++r)
		{
			for (int c = 0; c < colorImg.cols * 2; ++c)
			{
				v = r / 2.0;
				u = c / 2.0;
				v0 = (int)v;
				v1 = v0 + 1;
				u0 = (int)u;
				u1 = u0 + 1;
				if (yMat(v0, u0) >= 0 && xMat(v0, u0) >= 0 &&
					yMat(v0, u1) >= 0 && xMat(v0, u1) >= 0 &&
					yMat(v1, u0) >= 0 && xMat(v1, u0) >= 0 &&
					yMat(v1, u1) >= 0 && xMat(v1, u1) >= 0)
				{
					coef = (u1 - u) / (float)(u1 - u0);
					xTop = coef * xMat(v0, u0) +
						(1 - coef) * xMat(v0, u1);
					xBottom = coef * xMat(v1, u0) +
						(1 - coef) * xMat(v1, u1);
					coef = (v1 - v) / (float)(v1 - v0);
					xCent = coef * xTop + (1 - coef) * xBottom;

					coef = (u1 - u) / (float)(u1 - u0);
					yTop = coef * yMat(v0, u0) +
						(1 - coef) * yMat(v0, u1);
					yBottom = coef * yMat(v1, u0) +
						(1 - coef) * yMat(v1, u1);
					coef = (v1 - v) / (float)(v1 - v0);
					yCent = coef * yTop + (1 - coef) * yBottom;

					colorImg.at<cv::Vec3b>(r, c) = fullColorImg.at<cv::Vec3b>(yCent, xCent);
				}
			}
		}	
	}

private:
	std::vector<cv::Mat> m_depthVec, m_colorVec, m_fullColorVec;
	int m_bufLen = 200;
	volatile int m_readInd;
	volatile int m_writeInd;

	cv::Mat m_depthImg, m_colorImg, m_fullColorImg;
	float m_imgScale;
};

class xOniCapture
{
public:
	xOniCapture(int depthWidth, int depthHeight, int depthFps,
		int colorWidth, int colorHeight, int colorFps)
		: m_depthWidth(depthWidth), m_depthHeight(depthHeight),
		m_colorWidth(colorWidth), m_colorHeight(colorHeight),
		m_depthSize(depthWidth * depthHeight), m_colorSize(m_colorWidth * m_colorHeight)
	{
		openni::Status rc = openni::STATUS_OK;

		const char * deviceURI = openni::ANY_DEVICE; //"D:\\xjm\\result\\ieee_vr_oni_dataset\\ttt.oni"; //openni::ANY_DEVICE;

		rc = openni::OpenNI::initialize();

		std::string errorString(openni::OpenNI::getExtendedError());

		if (errorString.length() > 0)
		{
			std::cout << "oni error" << std::endl;
			exit(0);
		}

		rc = m_device.open(deviceURI);
		if (rc != openni::STATUS_OK)
		{
			std::cout << "oni error" << std::endl;
			exit(0);
		}	

		openni::VideoMode depthMode;
		depthMode.setFps(depthFps);
		depthMode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
		depthMode.setResolution(depthWidth, depthHeight);

		openni::VideoMode colorMode;
		colorMode.setFps(colorFps);
		colorMode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
		colorMode.setResolution(colorWidth, colorHeight);	

		rc = m_depthStream.create(m_device, openni::SENSOR_DEPTH);
		if (rc == openni::STATUS_OK)
		{
			m_depthStream.setVideoMode(depthMode);	
			//m_device.setDepthColorSyncEnabled(true);
			m_device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
			rc = m_depthStream.start();
			if (rc != openni::STATUS_OK)
			{
				m_depthStream.destroy();
				std::cout << "oni error" << std::endl;
				exit(0);
			}
		}
		else
		{
			std::cout << "oni error" << std::endl;
			exit(0);
		}	

		rc = m_rgbStream.create(m_device, openni::SENSOR_COLOR);
		if (rc == openni::STATUS_OK)
		{		
			if (rc != openni::STATUS_OK)
			{
				std::cout << "Can't change gain" << std::endl;
				return;
			}

			//m_rgbStream.getCameraSettings()->setAutoExposureEnabled(true);
			//m_rgbStream.getCameraSettings()->setAutoWhiteBalanceEnabled(true);

			m_rgbStream.setVideoMode(colorMode);
			
			rc = m_rgbStream.start();
			if (rc != openni::STATUS_OK)
			{
				m_rgbStream.destroy();
				std::cout << "oni error" << std::endl;
				exit(0);
			}	
		}
		else
		{
			std::cout << "oni error" << std::endl;
			exit(0);
		}

		if (!m_depthStream.isValid() || !m_rgbStream.isValid())
		{
			openni::OpenNI::shutdown();
			std::cout << "oni error" << std::endl;
			exit(0);
		}			
	}

	void start()
	{

	}

	bool next(cv::Mat &depthImg, cv::Mat &colorImg)
	{
		openni::Status rc = openni::STATUS_OK;
		
		rc = m_depthStream.readFrame(&m_depthFrame);
		if (rc != openni::STATUS_OK)
			return false;	
		rc = m_rgbStream.readFrame(&m_rgbFrame);
		if (rc != openni::STATUS_OK)
			return false;

		memcpy(depthImg.data, m_depthFrame.getData(), m_depthSize * 2);
		memcpy(colorImg.data, m_rgbFrame.getData(), m_colorSize * 3);	

		return true;
	}

private:
	openni::Device m_device;

	openni::VideoStream m_depthStream;
	openni::VideoStream m_rgbStream;

	openni::VideoFrameRef m_depthFrame;
	openni::VideoFrameRef m_rgbFrame;

	int m_depthWidth, m_depthHeight;
	int m_colorWidth, m_colorHeight;

	int m_depthSize, m_colorSize;
};

#endif