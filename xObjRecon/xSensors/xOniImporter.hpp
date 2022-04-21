#pragma once

#include <OpenNI.h>
#include <PS1080.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "Helpers/InnorealTimer.hpp"
#include "xSensors/xCapture.hpp"

class xOniImporter : public xCapture
{
public:
	xOniImporter(int depthWidth, int depthHeight, int depthFps,
		int colorWidth, int colorHeight, int colorFps)
		: m_depthWidth(depthWidth), m_depthHeight(depthHeight),
		m_colorWidth(colorWidth), m_colorHeight(colorHeight),
		m_depthPixelNum(depthWidth * depthHeight), m_colorPixelNum(m_colorWidth * m_colorHeight)
	{
		openni::Status rc = openni::STATUS_OK;

		const char * deviceURI = openni::ANY_DEVICE;
		//const char * deviceURI = "D:\\xjm\\result\\ieee_vr_oni_dataset\\ttt.oni";

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

    bool next(cv::Mat& depthImg, cv::Mat& colorImg,
              ImuMeasurements* imuMeasurements = NULL, Gravity* gravity = NULL,
              int* keyFrameIdxEachFrag = NULL)
	{
		m_timer.TimeStart();
		do
		{
			m_depthStatus = openni::STATUS_OK;
			m_colorStatus = openni::STATUS_OK;

			m_depthStatus = m_depthStream.readFrame(&m_depthFrame);
			m_colorStatus = m_rgbStream.readFrame(&m_rgbFrame);

			if (m_depthStatus == openni::STATUS_OK && m_colorStatus == openni::STATUS_OK)
			{
				memcpy(depthImg.data, m_depthFrame.getData(), m_depthPixelNum * 2);
				memcpy(colorImg.data, m_rgbFrame.getData(), m_colorPixelNum * 3);
			}

			m_timer.TimeEnd();
			if (m_timer.TimeGap_in_ms() > 100.0)
			{
				return false; // Time out
			}
		} while (m_depthStatus != openni::STATUS_OK || m_colorStatus != openni::STATUS_OK);

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
	int m_depthPixelNum, m_colorPixelNum;

	openni::Status m_depthStatus;
	openni::Status m_colorStatus;
	innoreal::InnoRealTimer m_timer;
};
