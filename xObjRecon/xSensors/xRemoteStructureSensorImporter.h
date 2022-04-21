#pragma once

#include <QtCore/QThread>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <Eigen/Eigen>
//#include <atltime.h>

#include "Helpers/InnorealTimer.hpp"
#include <condition_variable>
#include <future>
#include "Compress/ImageTrans.h"
#include "Helpers/xUtils.h"
#include "xSensors/xCapture.hpp"

extern std::mutex dataValidMtx, queueMtx;
extern std::condition_variable dataValidCondVar, dataInvalidCondVar, emptyCondVar, fullCondVar;

#define SAVE_TO_FILE 0

struct CirularQueue
{
	volatile int m_readIdx;
	volatile int m_writeIdx;
	int m_bufferSize;
	int m_elemSize;
	char *m_buffer;

	CirularQueue(int elemSize, int bufferSize)
		: m_elemSize(elemSize), m_bufferSize(bufferSize),
		m_readIdx(0), m_writeIdx(0)
	{
		m_buffer = (char *)malloc(m_elemSize * m_bufferSize);
	}

	~CirularQueue()
	{
		free(m_buffer);
	}

	void clear()
	{
		m_readIdx = 0;
		m_writeIdx = 0;
	}
};

#if 0
class RemoteStructureSeneor : public QThread, public xCapture
{
public:
	RemoteStructureSeneor(CirularQueue* cirularQueue, char* saveDataPath = NULL)
		: m_depthPixelNum(Resolution::getInstance().depthWidth() * Resolution::getInstance().depthHeight()),
		m_colorPixelNum(Resolution::getInstance().width() * Resolution::getInstance().height()),
		m_cirularQueue(cirularQueue),
		m_timeStamp(0), m_fragIdx(0),
		m_saveDataPath(saveDataPath)
	{
		m_colorImgYCbCr420Buffer = (char *)malloc(m_colorPixelNum * 3);;
		cudaMalloc((void**)&m_dColorImgYCbCr420Buffer, m_colorPixelNum * 3);
		m_colorImgCbCr420SplittedBuffer = (char *)malloc(m_colorPixelNum * 3);
		m_depthImgMsbLsbSplittedBuffer = (char *)malloc(m_depthPixelNum * 2);
		m_registeredDepthImgBuffer = (char *)malloc(m_depthPixelNum * 2);
		//m_depthColorFrameInfo = new DepthColorFrameInfo;

		m_feedBackBuffer.resize(m_colorPixelNum * 3);

#if SAVE_TO_FILE
		m_saveDataFile.open(m_saveDataPath, std::ios::binary);
		if (!m_saveDataFile.is_open())
		{
			std::cout << "Fail in Open Save File" << std::endl;
			std::exit(0);
		}
#endif
	}

	~RemoteStructureSeneor()
	{
		free(m_colorImgYCbCr420Buffer);
		free(m_colorImgCbCr420SplittedBuffer);
		free(m_depthImgMsbLsbSplittedBuffer);
		free(m_registeredDepthImgBuffer);
		//delete m_depthColorFrameInfo;

#if SAVE_TO_FILE
		m_saveDataFile.close();
#endif
	}

	void setDataBuffer(char *pColorData, char *pDepthData, int colorComressedFrameSize, int depthComressedFrameSize,
										 DepthColorFrameInfo *depthColorFrameInfo, ImuMeasurements *imuMeasurements, Gravity *gravity)
	{
#if 1
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while ((m_cirularQueue->m_writeIdx + 1) % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			fullCondVar.wait(queueLock);
		}
#endif

#if 1
		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_writeIdx %
																																									m_cirularQueue->m_bufferSize);
		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * imuMeasurements->size(),
			l3 = sizeof(Gravity);
		if ((l1 + l2 + l3 + colorComressedFrameSize + depthComressedFrameSize) > m_cirularQueue->m_elemSize)
		{
			std::cout << "CircularQueue Element Size Error" << std::endl;
			std::exit(0);
		}
		memcpy(pCurrentElem, depthColorFrameInfo, l1);
		memcpy(pCurrentElem + l1, imuMeasurements->data(), l2);
		memcpy(pCurrentElem + l1 + l2, gravity, l3);
		memcpy(pCurrentElem + l1 + l2 + l3, pColorData, colorComressedFrameSize);
		memcpy(pCurrentElem + l1 + l2 + l3 + colorComressedFrameSize, pDepthData, depthComressedFrameSize);
#endif

		++m_cirularQueue->m_writeIdx;
#if 1
		queueLock.unlock();
		emptyCondVar.notify_one();
#endif
	}

	void setEndScanFlag()
	{
#if 1
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while ((m_cirularQueue->m_writeIdx + 1) % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			fullCondVar.wait(queueLock);
		}
#endif
		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_writeIdx %
																																									m_cirularQueue->m_bufferSize);
		DepthColorFrameInfo endFlag;
		memset(pCurrentElem, 0, m_cirularQueue->m_elemSize);

		++m_cirularQueue->m_writeIdx;
#if 1
		queueLock.unlock();
		emptyCondVar.notify_one();
#endif
	}

	void decompressData(cv::Mat &colorImg, cv::Mat &depthImg, char* pColorBuffer, char* pDepthBuffer)
	{
#if 0
		memcpy(m_colorImgYCbCr420Buffer, pColorBuffer, m_colorPixelNum);

		ImageTrans::DecompressCbCr(m_colorImgYCbCr420Buffer + m_colorPixelNum, m_colorPixelNum / 2,
															 m_colorImgCbCr420SplittedBuffer,
															 pColorBuffer + m_colorPixelNum, m_pDepthColorFrameInfo->cbCompressedLength,
															 m_pDepthColorFrameInfo->crCompressedLength);
#if 0
		ImageTrans::DecodeYCbCr420SP((unsigned char*)colorImg.data, (const uchar *)m_colorImgYCbCr420Buffer,
																 Resolution::getInstance().width(), Resolution::getInstance().height());
#else
		m_dColorImg.create(colorImg.size(), colorImg.type());
		cudaMemcpy(m_dColorImgYCbCr420Buffer, m_colorImgYCbCr420Buffer, m_colorPixelNum * 1.5, cudaMemcpyHostToDevice);
		DecodeYCbCr420SPCUDA(m_dColorImg, (const uchar *)m_dColorImgYCbCr420Buffer);
		m_dColorImg.download(colorImg);
#endif

		ImageTrans::DecompressDepth((char *)depthImg.data, m_depthPixelNum * 2, m_depthImgMsbLsbSplittedBuffer,
																pDepthBuffer, m_pDepthColorFrameInfo->msbCompressedLength,
																m_pDepthColorFrameInfo->lsbCompressedLength);
		shift2depth((uint16_t *)depthImg.data, m_depthPixelNum);
#endif
	}

	void saveData(char* pBuffer, int len)
	{
		m_saveDataFile.write((const char *)pBuffer, len);
		m_saveDataFile.flush();
	}

	void setIntrinExtrin(float* extrinR, float* extrinT, float* intrinColor, float* intrinDepth)
	{
		m_fxColor = intrinColor[0];
		m_fyColor = intrinColor[1];
		m_cxColor = intrinColor[2];
		m_cyColor = intrinColor[3];

		m_fxDepth = intrinDepth[0];
		m_fyDepth = intrinDepth[1];
		m_cxDepth = intrinDepth[2];
		m_cyDepth = intrinDepth[3];

		m_depthToColor(0, 0) = extrinR[0]; m_depthToColor(0, 1) = extrinR[1]; m_depthToColor(0, 2) = extrinR[2]; m_depthToColor(0, 3) = -extrinT[0];
		m_depthToColor(1, 0) = extrinR[3]; m_depthToColor(1, 1) = extrinR[4]; m_depthToColor(1, 2) = extrinR[5]; m_depthToColor(1, 3) = -extrinT[1];
		m_depthToColor(2, 0) = extrinR[6]; m_depthToColor(2, 1) = extrinR[7]; m_depthToColor(2, 2) = extrinR[8]; m_depthToColor(2, 3) = -extrinT[2];
	}

	void registerDepthImg(cv::Mat &registeredDepthImg, cv::Mat &depthImg)
	{
		Eigen::Vector4f pos;
		int u, v, idx;
		ushort depth;
		memset(registeredDepthImg.data, -1, m_colorPixelNum * sizeof(ushort));
		for (int r = 0; r < depthImg.rows; ++r)
		{
			for (int c = 0; c < depthImg.cols; ++c)
			{
				pos.z() = depthImg.at<ushort>(r, c);
				pos.x() = (c - m_cxDepth) / m_fxDepth * pos.z();
				pos.y() = (r - m_cyDepth) / m_fyDepth * pos.z();
				pos.w() = 1.0f;
				pos = m_depthToColor * pos;

				u = pos.x() / pos.z() * m_fxColor + m_cxColor;
				v = pos.y() / pos.z() * m_fyColor + m_cyColor;
				if (u >= 0 && u < registeredDepthImg.cols && v >= 0 && v < registeredDepthImg.rows)
				{
					depth = pos.z();

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

	bool nextWithRegister(cv::Mat& registeredDepthImg,
												cv::Mat& depthImg,
												cv::Mat& colorImg)
	{
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while (m_cirularQueue->m_writeIdx % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			emptyCondVar.wait(queueLock);
		}

		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)pCurrentElem;

		if (m_pDepthColorFrameInfo->colorRows == 0 && m_pDepthColorFrameInfo->depthRows == 0)
		{
			// end of scan
			return false;
		}

		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);

		char* pColorBuffer = pCurrentElem + l1 + l2 + l3;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);

		registerDepthImg(registeredDepthImg, depthImg);

		++m_cirularQueue->m_readIdx;
		queueLock.unlock();
		fullCondVar.notify_one();

		return true;
	}

	void nextForSave(cv::Mat& depthImg, cv::Mat& colorImg,
									 ImuMeasurements& imuMeasurements, Gravity& gravity)
	{
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while (m_cirularQueue->m_writeIdx % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			emptyCondVar.wait(queueLock);
		}

		innoreal::InnoRealTimer timer;
		timer.TimeStart();

		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)pCurrentElem;
		//std::cout << "keyFrameIdxEachFrag: " << m_pDepthColorFrameInfo->keyFrameIdxEachFrag << std::endl;	

		int l = sizeof(DepthColorFrameInfo) +
			sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize +
			sizeof(Gravity);
		int len = l + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->
			crCompressedLength + m_pDepthColorFrameInfo->msbCompressedLength + m_pDepthColorFrameInfo->lsbCompressedLength;
		saveData(pCurrentElem, len);

		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);
		imuMeasurements.resize(m_pDepthColorFrameInfo->imuMeasurementSize);
		memcpy(imuMeasurements.data(), pCurrentElem + l1, l2);
		memcpy(&gravity, pCurrentElem + l1 + l2, l3);

		char* pColorBuffer = pCurrentElem + l1 + l2 + l3;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		timer.TimeEnd();
		std::cout << "copy time: " << timer.TimeGap_in_ms() << std::endl;

		timer.TimeStart();
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);
		timer.TimeEnd();
		std::cout << "decompress data time: " << timer.TimeGap_in_ms() << std::endl;

		++m_cirularQueue->m_readIdx;
		queueLock.unlock();
		fullCondVar.notify_one();
	}

	bool next(cv::Mat& depthImg, cv::Mat& colorImg,
						ImuMeasurements* imuMeasurements, Gravity* gravity,
						int* keyFrameIdxEachFrag)
	{
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while (m_cirularQueue->m_writeIdx % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			emptyCondVar.wait(queueLock);
		}

		//innoreal::InnoRealTimer timer;
		//timer.TimeStart();

		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)pCurrentElem;
		*keyFrameIdxEachFrag = m_pDepthColorFrameInfo->keyFrameIdxEachFrag;
		//std::cout << "keyFrameIdxEachFrag: " << m_pDepthColorFrameInfo->keyFrameIdxEachFrag << std::endl;	

		if (m_pDepthColorFrameInfo->colorRows == 0 && m_pDepthColorFrameInfo->depthRows == 0)
		{
			return false;
		}

#if SAVE_TO_FILE
		int l = sizeof(DepthColorFrameInfo) +
			sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize +
			sizeof(Gravity);
		int len = l + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->
			crCompressedLength + m_pDepthColorFrameInfo->msbCompressedLength + m_pDepthColorFrameInfo->lsbCompressedLength;
		saveData(pCurrentElem, len);
#endif

		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);
		imuMeasurements->resize(m_pDepthColorFrameInfo->imuMeasurementSize);
		memcpy(imuMeasurements->data(), pCurrentElem + l1, l2);
		memcpy(gravity, pCurrentElem + l1 + l2, l3);

		char* pColorBuffer = pCurrentElem + l1 + l2 + l3;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		//timer.TimeEnd();
		//std::cout << "copy time: " << timer.TimeGap_in_ms() << std::endl;

		//timer.TimeStart();
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);
		//timer.TimeEnd();
		//std::cout << "decompress data time: " << timer.TimeGap_in_ms() << std::endl;

		++m_cirularQueue->m_readIdx;
		queueLock.unlock();
		fullCondVar.notify_one();

		return true;
	}

	bool laterest(cv::Mat& depthImg, cv::Mat& colorImg,
								ImuMeasurements* imuMeasurements, Gravity* gravity,
								int* keyFrameIdxEachFrag)
	{
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while (m_cirularQueue->m_writeIdx % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			emptyCondVar.wait(queueLock);
		}

		m_cirularQueue->m_readIdx = (m_cirularQueue->m_writeIdx - 1 + m_cirularQueue->m_bufferSize)
			% m_cirularQueue->m_bufferSize;

		innoreal::InnoRealTimer timer;
		timer.TimeStart();

		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)pCurrentElem;
		*keyFrameIdxEachFrag = m_pDepthColorFrameInfo->keyFrameIdxEachFrag;
		//std::cout << "keyFrameIdxEachFrag: " << m_pDepthColorFrameInfo->keyFrameIdxEachFrag << std::endl;	

		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);
		imuMeasurements->resize(m_pDepthColorFrameInfo->imuMeasurementSize);
		memcpy(imuMeasurements->data(), pCurrentElem + l1, l2);
		memcpy(gravity, pCurrentElem + l1 + l2, l3);

		char* pColorBuffer = pCurrentElem + l1 + l2 + l3;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		timer.TimeEnd();
		//std::cout << "copy time: " << timer.TimeGap_in_ms() << std::endl;

		timer.TimeStart();
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);
		timer.TimeEnd();
		//std::cout << "decompress data time: " << timer.TimeGap_in_ms() << std::endl;

		++m_cirularQueue->m_readIdx;
		queueLock.unlock();
		fullCondVar.notify_one();

		return true;
	}

#if 0
	bool next(cv::Mat& depthImg, cv::Mat& colorImg,
						ImuMeasurements& imuMeasurements, Gravity& gravity)
	{
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while (m_cirularQueue->m_writeIdx % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			emptyCondVar.wait(queueLock);
		}

		if (m_pDepthColorFrameInfo->colorRows == 0 && m_pDepthColorFrameInfo->depthRows == 0)
		{
			// end of scan
			return false;
		}

		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)pCurrentElem;
		//std::cout << "keyFrameIdxEachFrag: " << m_pDepthColorFrameInfo->keyFrameIdxEachFrag << std::endl;	

		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);
		imuMeasurements.resize(m_pDepthColorFrameInfo->imuMeasurementSize);
		memcpy(imuMeasurements.data(), pCurrentElem + l1, l2);
		memcpy(&gravity, pCurrentElem + l1 + l2, l3);

		char* pColorBuffer = pCurrentElem + l1 + l2 + l3;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);

		++m_cirularQueue->m_readIdx;
		queueLock.unlock();
		fullCondVar.notify_one();

		return true;
	}
#endif

	// Just For Test
	void run() override
	{
		cv::Mat depthImg(Resolution::getInstance().depthHeight(), Resolution::getInstance().depthWidth(), CV_16UC1),
			registeredDepthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_16UC1),
			colorImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3),
			grayImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1),
			grayDepthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1),
			colorDepthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3);

		while (true)
		{
			ImuMeasurements imuMeasurements;
			Gravity gravity;
			int keyFrameIdxEachFrag;
			nextWithRegister(registeredDepthImg, depthImg, colorImg);
#if 0
			laterest(depthImg, colorImg,
							 &imuMeasurements, &gravity, &keyFrameIdxEachFrag);
#endif
			//std::cout << "colorImg size: " << colorImg.rows << " : " << colorImg.cols << std::endl;
			//std::cout << "registeredDepthImg size: " << registeredDepthImg.rows << " : " << registeredDepthImg.cols << std::endl;

#if 1
			registeredDepthImg.convertTo(grayDepthImg, CV_8UC1, 4.0, 0);
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
			cv::imshow("Input_Color_Depth", colorDepthImg);
#endif

			cv::imshow("Input_Color", colorImg);
			cv::imshow("Input_Depth", depthImg * 30);
			//cv::imshow("Input_ResigeredDepth", registeredDepthImg * 30);
			cv::waitKey(1);
		}
#if 0
		float fxColor = 530.8013, fyColor = 532.1928, cxColor = 318.4930, cyColor = 229.9720;
		float fxDepth = 574.8815, fyDepth = 576.8138, cxDepth = 315.6864, cyDepth = 247.4785;
		Eigen::Matrix4f depthToColor = Eigen::Matrix4f::Identity();

		float T[3] = { -40.5318, -10.6037, -22.3758 };
		float R[9] = { 1.0000, -0.0069, 0.0049,
			0.0068, 0.9998, 0.0176,
			-0.0050, -0.0175, 0.9998 };
#endif

#if 0
		float fxColor = 544.8898, fyColor = 545.9078, cxColor = 321.6016, cyColor = 237.0330;
		float fxDepth = 574.0135, fyDepth = 575.5523, cxDepth = 314.5388, cyDepth = 242.4793;
		Eigen::Matrix4f depthToColor = Eigen::Matrix4f::Identity();

		float T[3] = { -41.1776, -4.3666, -34.8012 };
		float R[9] = {
			1.0000, -0.0040, -0.0029,
			0.0040, 0.9999, 0.0132,
			0.0028, -0.0132, 0.9999
		};

		depthToColor << R[0], R[1], R[2], -T[0],
			R[3], R[4], R[5], -T[1],
			R[6], R[7], R[8], -T[2];
		std::cout << depthToColor << std::endl;

		while (true)
		{
			next(depthImg, colorImg);
			cv::cvtColor(colorImg, grayImg, CV_RGB2GRAY);

			Eigen::Vector4f pos;
			int u, v;
			ushort depth;
			memset(registeredDepthImg.data, -1, registeredDepthImg.rows * registeredDepthImg.cols * sizeof(ushort));
			for (int r = 0; r < depthImg.rows; ++r)
			{
				for (int c = 0; c < depthImg.cols; ++c)
				{
					pos.z() = depthImg.at<ushort>(r, c);
					pos.x() = (c - cxDepth) / fxDepth * pos.z();
					pos.y() = (r - cyDepth) / fyDepth * pos.z();
					pos.w() = 1.0f;
					pos = depthToColor * pos;

					u = pos.x() / pos.z() * fxColor + cxColor;
					v = pos.y() / pos.z() * fyColor + cyColor;
					if (u >= 0 && u < depthImg.cols && v >= 0 && v < depthImg.rows)
					{
						depth = ushort(pos.z() * 1000.0f);
						if (depth < registeredDepthImg.at<ushort>(v, u))
						{
							registeredDepthImg.at<ushort>(v, u) = depth;
						}
					}
				}
			}
			for (int r = 0; r < depthImg.rows; ++r)
			{
				for (int c = 0; c < depthImg.cols; ++c)
				{
					if ((int)registeredDepthImg.at<ushort>(r, c) == 65535)
					{
						registeredDepthImg.at<ushort>(r, c) = 0;
					}
				}
			}

			registeredDepthImg.convertTo(grayDepthImg, CV_8UC1, 4.0, 0);

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

			cv::imshow("Input_Color", grayImg);
			cv::imshow("Input_Depth", grayDepthImg);
			//cv::imshow("Input_Color_Depth", colorDepthImg);
			cv::waitKey(1);
		}
#endif
	}

public:
	int m_depthPixelNum, m_colorPixelNum;
	//xGUI *m_pGui;
	int64_t m_timeStamp;
	int m_fragIdx;
	CirularQueue *m_cirularQueue;
	innoreal::InnoRealTimer m_timer;

	char *m_colorImgYCbCr420Buffer, *m_colorImgCbCr420SplittedBuffer, *m_depthImgMsbLsbSplittedBuffer;
	char *m_dColorImgYCbCr420Buffer;
	cv::cuda::GpuMat m_dColorImg;
	char* m_registeredDepthImgBuffer;
	DepthColorFrameInfo *m_pDepthColorFrameInfo;
	ImuMeasurements *m_imuMeasurement;
	Gravity* m_graviry;

	Eigen::Matrix4f m_depthToColor;
	float m_fxDepth, m_fyDepth, m_cxDepth, m_cyDepth, m_fxColor, m_fyColor, m_cxColor, m_cyColor;

	std::ofstream m_saveDataFile;
	char* m_saveDataPath;

	std::vector<char> m_feedBackBuffer;
};
#endif
class RemoteStructureSeneor : public QThread, public xCapture
{
public:
	RemoteStructureSeneor(CirularQueue* cirularQueue, char* saveDataPath = NULL)
		: m_depthPixelNum(Resolution::getInstance().depthWidth() * Resolution::getInstance().depthHeight()),
		m_colorPixelNum(Resolution::getInstance().width() * Resolution::getInstance().height()),
		m_cirularQueue(cirularQueue),
		m_timeStamp(0), m_fragIdx(0)
	{
		memcpy(m_saveDataPath, saveDataPath, sizeof(m_saveDataPath));

		m_colorImgYCbCr420Buffer = (char *)malloc(m_colorPixelNum * 3);
		cudaMalloc((void**)&m_dColorImgYCbCr420Buffer, m_colorPixelNum * 3);
		m_colorImgCbCr420SplittedBuffer = (char *)malloc(m_colorPixelNum * 3);
		m_depthImgMsbLsbSplittedBuffer = (char *)malloc(m_depthPixelNum * 2);
		m_registeredDepthImgBuffer = (char *)malloc(m_depthPixelNum * 2);
		//m_depthColorFrameInfo = new DepthColorFrameInfo;

		m_feedBackBuffer.resize(m_colorPixelNum * 3);

		if (GlobalState::getInstance().m_saveToBinFile) {
			m_saveDataFile.open(m_saveDataPath, std::ios::binary);
			if (!m_saveDataFile.is_open())
			{
				std::cout << "Fail in Open Save File" << std::endl;
				std::exit(0);
			}
		}
	}

	~RemoteStructureSeneor()
	{
		free(m_colorImgYCbCr420Buffer);
		free(m_colorImgCbCr420SplittedBuffer);
		free(m_depthImgMsbLsbSplittedBuffer);
		free(m_registeredDepthImgBuffer);
		//delete m_depthColorFrameInfo;

		if (GlobalState::getInstance().m_saveToBinFile) {
			m_saveDataFile.close();
		}
	}

	void setDataBuffer(char *pColorData, char *pDepthData, int colorComressedFrameSize, int depthComressedFrameSize,
										 DepthColorFrameInfo *depthColorFrameInfo, ImuMeasurements *imuMeasurements, Gravity *gravity)
	{
#if 1
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while ((m_cirularQueue->m_writeIdx + 1) % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			fullCondVar.wait(queueLock);
		}
#endif

#if 1
		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_writeIdx %
																																									m_cirularQueue->m_bufferSize);
		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * imuMeasurements->size(),
			l3 = sizeof(Gravity);
		if ((l1 + l2 + l3 + colorComressedFrameSize + depthComressedFrameSize) > m_cirularQueue->m_elemSize)
		{
			std::cout << "CircularQueue Element Size Error" << std::endl;
			std::exit(0);
		}
		memcpy(pCurrentElem, depthColorFrameInfo, l1);
		memcpy(pCurrentElem + l1, imuMeasurements->data(), l2);
		memcpy(pCurrentElem + l1 + l2, gravity, l3);
		memcpy(pCurrentElem + l1 + l2 + l3, pColorData, colorComressedFrameSize);
		memcpy(pCurrentElem + l1 + l2 + l3 + colorComressedFrameSize, pDepthData, depthComressedFrameSize);
#endif

		++m_cirularQueue->m_writeIdx;
#if 1
		queueLock.unlock();
		emptyCondVar.notify_one();
#endif
	}

	void setEndScanFlag()
	{
#if 1
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while ((m_cirularQueue->m_writeIdx + 1) % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			fullCondVar.wait(queueLock);
		}
#endif
		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_writeIdx %
																																									m_cirularQueue->m_bufferSize);
		DepthColorFrameInfo endFlag;
		memset(pCurrentElem, 0, m_cirularQueue->m_elemSize);

		++m_cirularQueue->m_writeIdx;
#if 1
		queueLock.unlock();
		emptyCondVar.notify_one();
#endif
	}

	void decompressData(cv::Mat &colorImg, cv::Mat &depthImg, char* pColorBuffer, char* pDepthBuffer)
	{
		memcpy(m_colorImgYCbCr420Buffer, pColorBuffer, m_colorPixelNum);

		ImageTrans::DecompressCbCr(m_colorImgYCbCr420Buffer + m_colorPixelNum, m_colorPixelNum / 2,
															 m_colorImgCbCr420SplittedBuffer,
															 pColorBuffer + m_colorPixelNum, m_pDepthColorFrameInfo->cbCompressedLength,
															 m_pDepthColorFrameInfo->crCompressedLength);
#if 1
		ImageTrans::DecodeYCbCr420SP((unsigned char*)colorImg.data, (const uchar *)m_colorImgYCbCr420Buffer,
																 Resolution::getInstance().width(), Resolution::getInstance().height());
#else
		m_dColorImg.create(colorImg.size(), colorImg.type());
		cudaMemcpy(m_dColorImgYCbCr420Buffer, m_colorImgYCbCr420Buffer, m_colorPixelNum * 1.5, cudaMemcpyHostToDevice);
		DecodeYCbCr420SPCUDA(m_dColorImg, (const uchar *)m_dColorImgYCbCr420Buffer);
		m_dColorImg.download(colorImg);
#endif

		ImageTrans::DecompressDepth((char *)depthImg.data, m_depthPixelNum * 2, m_depthImgMsbLsbSplittedBuffer,
																pDepthBuffer, m_pDepthColorFrameInfo->msbCompressedLength,
																m_pDepthColorFrameInfo->lsbCompressedLength);
		shift2depth((uint16_t *)depthImg.data, m_depthPixelNum);
	}

	void saveData(char* pBuffer, int len)
	{
		m_saveDataFile.write((const char *)pBuffer, len);
		//m_saveDataFile.flush();
	}

	void setIntrinExtrin(float* extrinR, float* extrinT, float* intrinColor, float* intrinDepth)
	{
		m_fxColor = intrinColor[0];
		m_fyColor = intrinColor[1];
		m_cxColor = intrinColor[2];
		m_cyColor = intrinColor[3];

		m_fxDepth = intrinDepth[0];
		m_fyDepth = intrinDepth[1];
		m_cxDepth = intrinDepth[2];
		m_cyDepth = intrinDepth[3];

		m_depthToColor(0, 0) = extrinR[0]; m_depthToColor(0, 1) = extrinR[1]; m_depthToColor(0, 2) = extrinR[2]; m_depthToColor(0, 3) = -extrinT[0];
		m_depthToColor(1, 0) = extrinR[3]; m_depthToColor(1, 1) = extrinR[4]; m_depthToColor(1, 2) = extrinR[5]; m_depthToColor(1, 3) = -extrinT[1];
		m_depthToColor(2, 0) = extrinR[6]; m_depthToColor(2, 1) = extrinR[7]; m_depthToColor(2, 2) = extrinR[8]; m_depthToColor(2, 3) = -extrinT[2];
	}

	void registerDepthImg(cv::Mat &registeredDepthImg, cv::Mat &depthImg)
	{
		std::cout << m_depthToColor << std::endl;
#if 1
		Eigen::Vector4f pos;
		int u, v, idx;
		ushort depth;
		memset(registeredDepthImg.data, -1, m_colorPixelNum * sizeof(ushort));
		for (int r = 0; r < depthImg.rows; ++r)
		{
			for (int c = 0; c < depthImg.cols; ++c)
			{
				pos.z() = depthImg.at<ushort>(r, c);
				pos.x() = (c - m_cxDepth) / m_fxDepth * pos.z();
				pos.y() = (r - m_cyDepth) / m_fyDepth * pos.z();
				pos.w() = 1.0f;
				pos = m_depthToColor * pos;

				u = pos.x() / pos.z() * m_fxColor + m_cxColor;
				v = pos.y() / pos.z() * m_fyColor + m_cyColor;
				if (u >= 0 && u < registeredDepthImg.cols && v >= 0 && v < registeredDepthImg.rows)
				{
					depth = pos.z();

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
#endif
	}

	bool nextWithRegister(cv::Mat& registeredDepthImg,
												cv::Mat& depthImg,
												cv::Mat& colorImg)
	{
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while (m_cirularQueue->m_writeIdx % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			emptyCondVar.wait(queueLock);
		}

		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)pCurrentElem;

		if (m_pDepthColorFrameInfo->colorRows == 0 && m_pDepthColorFrameInfo->depthRows == 0)
		{
			// end of scan
			return false;
		}

		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);

		char* pColorBuffer = pCurrentElem + l1 + l2 + l3;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);

		registerDepthImg(registeredDepthImg, depthImg);

		++m_cirularQueue->m_readIdx;
		queueLock.unlock();
		fullCondVar.notify_one();

		return true;
	}

	void nextForSave(cv::Mat& depthImg, cv::Mat& colorImg,
									 ImuMeasurements& imuMeasurements, Gravity& gravity)
	{
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while (m_cirularQueue->m_writeIdx % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			emptyCondVar.wait(queueLock);
		}

		innoreal::InnoRealTimer timer;
		timer.TimeStart();

		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)pCurrentElem;
		//std::cout << "keyFrameIdxEachFrag: " << m_pDepthColorFrameInfo->keyFrameIdxEachFrag << std::endl;	

		int l = sizeof(DepthColorFrameInfo) +
			sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize +
			sizeof(Gravity);
		int len = l + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->
			crCompressedLength + m_pDepthColorFrameInfo->msbCompressedLength + m_pDepthColorFrameInfo->lsbCompressedLength;
		saveData(pCurrentElem, len);

		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);
		imuMeasurements.resize(m_pDepthColorFrameInfo->imuMeasurementSize);
		memcpy(imuMeasurements.data(), pCurrentElem + l1, l2);
		memcpy(&gravity, pCurrentElem + l1 + l2, l3);

		char* pColorBuffer = pCurrentElem + l1 + l2 + l3;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		timer.TimeEnd();
		std::cout << "copy time: " << timer.TimeGap_in_ms() << std::endl;

		timer.TimeStart();
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);
		timer.TimeEnd();
		std::cout << "decompress data time: " << timer.TimeGap_in_ms() << std::endl;

		++m_cirularQueue->m_readIdx;
		queueLock.unlock();
		fullCondVar.notify_one();
	}

	bool next(cv::Mat& depthImg, cv::Mat& colorImg,
						ImuMeasurements* imuMeasurements, Gravity* gravity,
						int* keyFrameIdxEachFrag)
	{
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while (m_cirularQueue->m_writeIdx % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			emptyCondVar.wait(queueLock);
		}

		//innoreal::InnoRealTimer timer;
		//timer.TimeStart();

		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)pCurrentElem;
		*keyFrameIdxEachFrag = m_pDepthColorFrameInfo->keyFrameIdxEachFrag;
		//std::cout << "keyFrameIdxEachFrag: " << m_pDepthColorFrameInfo->keyFrameIdxEachFrag << std::endl;	

		if (m_pDepthColorFrameInfo->colorRows == 0 && m_pDepthColorFrameInfo->depthRows == 0)
		{
			return false;
		}

		if (GlobalState::getInstance().m_saveToBinFile) {
			int l = sizeof(DepthColorFrameInfo) +
				sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize +
				sizeof(Gravity);
			int len = l + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->
				crCompressedLength + m_pDepthColorFrameInfo->msbCompressedLength + m_pDepthColorFrameInfo->lsbCompressedLength;
			saveData(pCurrentElem, len);
		}

		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);
		imuMeasurements->resize(m_pDepthColorFrameInfo->imuMeasurementSize);
		memcpy(imuMeasurements->data(), pCurrentElem + l1, l2);
		memcpy(gravity, pCurrentElem + l1 + l2, l3);

		char* pColorBuffer = pCurrentElem + l1 + l2 + l3;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		//timer.TimeEnd();
		//std::cout << "copy time: " << timer.TimeGap_in_ms() << std::endl;

		//timer.TimeStart();
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);
		//timer.TimeEnd();
		//std::cout << "decompress data time: " << timer.TimeGap_in_ms() << std::endl;	

		++m_cirularQueue->m_readIdx;
		queueLock.unlock();
		fullCondVar.notify_one();

		return true;
	}

	bool laterest(cv::Mat& depthImg, cv::Mat& colorImg,
								ImuMeasurements* imuMeasurements, Gravity* gravity,
								int* keyFrameIdxEachFrag)
	{
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while (m_cirularQueue->m_writeIdx % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			emptyCondVar.wait(queueLock);
		}


		m_cirularQueue->m_readIdx = (m_cirularQueue->m_writeIdx - 1 + m_cirularQueue->m_bufferSize)
			% m_cirularQueue->m_bufferSize;

		//innoreal::InnoRealTimer timer;
		//timer.TimeStart();

		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)pCurrentElem;
		*keyFrameIdxEachFrag = m_pDepthColorFrameInfo->keyFrameIdxEachFrag;
		//std::cout << "keyFrameIdxEachFrag: " << m_pDepthColorFrameInfo->keyFrameIdxEachFrag << std::endl;	

		if (m_pDepthColorFrameInfo->colorRows == 0 && m_pDepthColorFrameInfo->depthRows == 0)
		{
			return false;
		}

#if SAVE_TO_FILE
		int l = sizeof(DepthColorFrameInfo) +
			sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize +
			sizeof(Gravity);
		int len = l + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->
			crCompressedLength + m_pDepthColorFrameInfo->msbCompressedLength + m_pDepthColorFrameInfo->lsbCompressedLength;
		saveData(pCurrentElem, len);
#endif

		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);
		imuMeasurements->resize(m_pDepthColorFrameInfo->imuMeasurementSize);
		memcpy(imuMeasurements->data(), pCurrentElem + l1, l2);
		memcpy(gravity, pCurrentElem + l1 + l2, l3);

		char* pColorBuffer = pCurrentElem + l1 + l2 + l3;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		//timer.TimeEnd();
		//std::cout << "copy time: " << timer.TimeGap_in_ms() << std::endl;

		//timer.TimeStart();
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);
		//timer.TimeEnd();
		//std::cout << "decompress data time: " << timer.TimeGap_in_ms() << std::endl;

		++m_cirularQueue->m_readIdx;
		queueLock.unlock();
		fullCondVar.notify_one();

		return true;
	}

#if 0
	bool next(cv::Mat& depthImg, cv::Mat& colorImg,
						ImuMeasurements& imuMeasurements, Gravity& gravity)
	{
		std::unique_lock<std::mutex> queueLock(queueMtx);
		while (m_cirularQueue->m_writeIdx % m_cirularQueue->m_bufferSize
					 == m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize)
		{
			emptyCondVar.wait(queueLock);
		}

		if (m_pDepthColorFrameInfo->colorRows == 0 && m_pDepthColorFrameInfo->depthRows == 0)
		{
			// end of scan
			return false;
		}

		char* pCurrentElem = m_cirularQueue->m_buffer + m_cirularQueue->m_elemSize * (m_cirularQueue->m_readIdx % m_cirularQueue->m_bufferSize);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)pCurrentElem;
		//std::cout << "keyFrameIdxEachFrag: " << m_pDepthColorFrameInfo->keyFrameIdxEachFrag << std::endl;	

		int l1 = sizeof(DepthColorFrameInfo),
			l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);
		imuMeasurements.resize(m_pDepthColorFrameInfo->imuMeasurementSize);
		memcpy(imuMeasurements.data(), pCurrentElem + l1, l2);
		memcpy(&gravity, pCurrentElem + l1 + l2, l3);

		char* pColorBuffer = pCurrentElem + l1 + l2 + l3;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);

		++m_cirularQueue->m_readIdx;
		queueLock.unlock();
		fullCondVar.notify_one();

		return true;
	}
#endif

	// Just For Test
	void run() override
	{
		cv::Mat depthImg(Resolution::getInstance().depthHeight(), Resolution::getInstance().depthWidth(), CV_16UC1),
			registeredDepthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_16UC1),
			colorImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3),
			grayImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1),
			grayDepthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1),
			colorDepthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3);

		while (true)
		{
			ImuMeasurements imuMeasurements;
			Gravity gravity;
			int keyFrameIdxEachFrag;
			nextWithRegister(registeredDepthImg, depthImg, colorImg);
			//nextForSave(depthImg, colorImg,
			//			imuMeasurements, gravity);
			//std::cout << "colorImg size: " << colorImg.rows << " : " << colorImg.cols << std::endl;
			//std::cout << "registeredDepthImg size: " << registeredDepthImg.rows << " : " << registeredDepthImg.cols << std::endl;

#if 1
			registeredDepthImg.convertTo(grayDepthImg, CV_8UC1, 1.0, 0);
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
			cv::imshow("Input_Color_Depth", colorDepthImg);
#endif

			cv::imshow("Input_Color", colorImg);
			cv::imshow("Input_Depth", registeredDepthImg * 30);
			cv::waitKey(1);
		}
#if 0
		float fxColor = 530.8013, fyColor = 532.1928, cxColor = 318.4930, cyColor = 229.9720;
		float fxDepth = 574.8815, fyDepth = 576.8138, cxDepth = 315.6864, cyDepth = 247.4785;
		Eigen::Matrix4f depthToColor = Eigen::Matrix4f::Identity();

		float T[3] = { -40.5318, -10.6037, -22.3758 };
		float R[9] = { 1.0000, -0.0069, 0.0049,
			0.0068, 0.9998, 0.0176,
			-0.0050, -0.0175, 0.9998 };
#endif

#if 0
		float fxColor = 544.8898, fyColor = 545.9078, cxColor = 321.6016, cyColor = 237.0330;
		float fxDepth = 574.0135, fyDepth = 575.5523, cxDepth = 314.5388, cyDepth = 242.4793;
		Eigen::Matrix4f depthToColor = Eigen::Matrix4f::Identity();

		float T[3] = { -41.1776, -4.3666, -34.8012 };
		float R[9] = {
			1.0000, -0.0040, -0.0029,
			0.0040, 0.9999, 0.0132,
			0.0028, -0.0132, 0.9999
		};

		depthToColor << R[0], R[1], R[2], -T[0],
			R[3], R[4], R[5], -T[1],
			R[6], R[7], R[8], -T[2];
		std::cout << depthToColor << std::endl;

		while (true)
		{
			next(depthImg, colorImg);
			cv::cvtColor(colorImg, grayImg, CV_RGB2GRAY);

			Eigen::Vector4f pos;
			int u, v;
			ushort depth;
			memset(registeredDepthImg.data, -1, registeredDepthImg.rows * registeredDepthImg.cols * sizeof(ushort));
			for (int r = 0; r < depthImg.rows; ++r)
			{
				for (int c = 0; c < depthImg.cols; ++c)
				{
					pos.z() = depthImg.at<ushort>(r, c);
					pos.x() = (c - cxDepth) / fxDepth * pos.z();
					pos.y() = (r - cyDepth) / fyDepth * pos.z();
					pos.w() = 1.0f;
					pos = depthToColor * pos;

					u = pos.x() / pos.z() * fxColor + cxColor;
					v = pos.y() / pos.z() * fyColor + cyColor;
					if (u >= 0 && u < depthImg.cols && v >= 0 && v < depthImg.rows)
					{
						depth = ushort(pos.z() * 1000.0f);
						if (depth < registeredDepthImg.at<ushort>(v, u))
						{
							registeredDepthImg.at<ushort>(v, u) = depth;
						}
					}
				}
			}
			for (int r = 0; r < depthImg.rows; ++r)
			{
				for (int c = 0; c < depthImg.cols; ++c)
				{
					if ((int)registeredDepthImg.at<ushort>(r, c) == 65535)
					{
						registeredDepthImg.at<ushort>(r, c) = 0;
					}
				}
			}

			registeredDepthImg.convertTo(grayDepthImg, CV_8UC1, 4.0, 0);

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

			cv::imshow("Input_Color", grayImg);
			cv::imshow("Input_Depth", grayDepthImg);
			//cv::imshow("Input_Color_Depth", colorDepthImg);
			cv::waitKey(1);
		}
#endif
	}

public:
	int m_depthPixelNum, m_colorPixelNum;
	//xGUI *m_pGui;
	int64_t m_timeStamp;
	int m_fragIdx;
	CirularQueue *m_cirularQueue;
	innoreal::InnoRealTimer m_timer;

	char *m_colorImgYCbCr420Buffer, *m_colorImgCbCr420SplittedBuffer, *m_depthImgMsbLsbSplittedBuffer;
	char *m_dColorImgYCbCr420Buffer;
	cv::cuda::GpuMat m_dColorImg;
	char* m_registeredDepthImgBuffer;
	DepthColorFrameInfo *m_pDepthColorFrameInfo;
	ImuMeasurements *m_imuMeasurement;
	Gravity* m_graviry;

	Eigen::Matrix4f m_depthToColor;
	float m_fxDepth, m_fyDepth, m_cxDepth, m_cyDepth, m_fxColor, m_fyColor, m_cxColor, m_cyColor;

	std::ofstream m_saveDataFile;
	char m_saveDataPath[512];

	std::vector<char> m_feedBackBuffer;
};

class RemoteStructureSeneorFromFile : public QThread, public xCapture
{
public:
	RemoteStructureSeneorFromFile(const char *fileDir)
		: m_depthPixelNum(Resolution::getInstance().depthWidth() * Resolution::getInstance().depthHeight()),
		m_colorPixelNum(Resolution::getInstance().width() * Resolution::getInstance().height())
	{
		m_colorImgYCbCr420Buffer = (char *)malloc(m_colorPixelNum * 3);;
		m_colorImgCbCr420SplittedBuffer = (char *)malloc(m_colorPixelNum * 3);
		m_depthImgMsbLsbSplittedBuffer = (char *)malloc(m_depthPixelNum * 2);
		m_compressedColorDepthBuffer = (char *)malloc(m_colorPixelNum * 3 + m_depthPixelNum * 2 + HEAD_SIZE);
		m_registeredDepthImgBuffer = (char *)malloc(m_depthPixelNum * 2);

		m_saveDataFile.open(fileDir, std::ios::binary);
		if (!m_saveDataFile.is_open())
		{
			std::cout << "Import .bin File Path Error" << std::endl;
			std::exit(0);
		}

		m_depthToColor = Eigen::Matrix4f::Identity();
	}

	~RemoteStructureSeneorFromFile()
	{
		free(m_colorImgYCbCr420Buffer);
		free(m_colorImgCbCr420SplittedBuffer);
		free(m_depthImgMsbLsbSplittedBuffer);
		free(m_compressedColorDepthBuffer);
		free(m_registeredDepthImgBuffer);

		m_saveDataFile.close();
	}

	void decompressData(cv::Mat &colorImg, cv::Mat &depthImg, char* pColorBuffer, char* pDepthBuffer)
	{
		//m_timer.TimeStart();
		memcpy(m_colorImgYCbCr420Buffer, pColorBuffer, m_colorPixelNum);
		ImageTrans::DecompressCbCr(m_colorImgYCbCr420Buffer + m_colorPixelNum, m_colorPixelNum / 2, m_colorImgCbCr420SplittedBuffer,
															 pColorBuffer + m_colorPixelNum, m_pDepthColorFrameInfo->cbCompressedLength, m_pDepthColorFrameInfo->crCompressedLength);
		ImageTrans::DecodeYCbCr420SP((unsigned char*)colorImg.data, (const uchar *)m_colorImgYCbCr420Buffer, Resolution::getInstance().width(), Resolution::getInstance().height());
		ImageTrans::DecompressDepth((char *)depthImg.data, m_depthPixelNum * 2, m_depthImgMsbLsbSplittedBuffer,
																pDepthBuffer, m_pDepthColorFrameInfo->msbCompressedLength, m_pDepthColorFrameInfo->lsbCompressedLength);
		shift2depth((uint16_t *)depthImg.data, m_depthPixelNum);
		//m_timer.TimeEnd();
		//std::cout << "decompress time: " << m_timer.TimeGap_in_ms() << std::endl;
	}

#if 0
	void setIntrinExtrin(float* extrinR, float* extrinT, float* intrinColor, float* intrinDepth)
	{
		m_fxColor = intrinColor[0];
		m_fyColor = intrinColor[1];
		m_cxColor = intrinColor[2];
		m_cyColor = intrinColor[3];

		m_fxDepth = intrinDepth[0];
		m_fyDepth = intrinDepth[1];
		m_cxDepth = intrinDepth[2];
		m_cyDepth = intrinDepth[3];

		m_depthToColor(0, 0) = extrinR[0]; m_depthToColor(0, 1) = extrinR[1]; m_depthToColor(0, 2) = extrinR[2]; m_depthToColor(0, 3) = -extrinT[0];
		m_depthToColor(1, 0) = extrinR[3]; m_depthToColor(1, 1) = extrinR[4]; m_depthToColor(1, 2) = extrinR[5]; m_depthToColor(1, 3) = -extrinT[1];
		m_depthToColor(2, 0) = extrinR[6]; m_depthToColor(2, 1) = extrinR[7]; m_depthToColor(2, 2) = extrinR[8]; m_depthToColor(2, 3) = -extrinT[2];
	}
#endif

	void registerDepthImg(cv::Mat &depthImg)
	{
		Eigen::Vector4f pos;
		int u, v, idx;
		ushort depth;
		memset(m_registeredDepthImgBuffer, -1, m_depthPixelNum * sizeof(ushort));
		ushort* pRegisteredDepth = (ushort*)m_registeredDepthImgBuffer;
		for (int r = 0; r < depthImg.rows; ++r)
		{
			for (int c = 0; c < depthImg.cols; ++c)
			{
				pos.z() = depthImg.at<ushort>(r, c);
				pos.x() = (c - m_cxDepth) / m_fxDepth * pos.z();
				pos.y() = (r - m_cyDepth) / m_fyDepth * pos.z();
				pos.w() = 1.0f;
				pos = m_depthToColor * pos;

				u = pos.x() / pos.z() * m_fxColor + m_cxColor;
				v = pos.y() / pos.z() * m_fyColor + m_cyColor;
				idx = v * depthImg.cols + u;
				if (u >= 0 && u < depthImg.cols && v >= 0 && v < depthImg.rows)
				{
					depth = pos.z();

					if (depth < pRegisteredDepth[idx])
					{
						pRegisteredDepth[idx] = depth;
					}
				}
			}
		}
		for (int r = 0; r < depthImg.rows; ++r)
		{
			for (int c = 0; c < depthImg.cols; ++c)
			{
				idx = r * depthImg.cols + c;
				if ((int)pRegisteredDepth[idx] == 65535)
				{
					pRegisteredDepth[idx] = 0;
				}
			}
		}
		memcpy(depthImg.data, pRegisteredDepth, m_depthPixelNum * sizeof(ushort));
	}

	bool next(cv::Mat& depthImg, cv::Mat& colorImg,
						ImuMeasurements* imuMeasurements = NULL, Gravity* gravity = NULL,
						int* keyFrameIdxEachFrag = NULL)
	{
		int l1 = sizeof(DepthColorFrameInfo);

		m_saveDataFile.read(m_compressedColorDepthBuffer, l1);
		m_pDepthColorFrameInfo = (DepthColorFrameInfo *)m_compressedColorDepthBuffer;

		if (m_pDepthColorFrameInfo->colorRows == 0 && m_pDepthColorFrameInfo->depthRows == 0)
		{
			// end of scan
			return false;
		}
#if 0
		std::cout << m_pDepthColorFrameInfo->keyFrameIdxEachFrag << std::endl;
		std::cout << m_pDepthColorFrameInfo->colorRows << std::endl;
		std::cout << m_pDepthColorFrameInfo->colorCols << std::endl;
		std::cout << m_pDepthColorFrameInfo->depthRows << std::endl;
		std::cout << m_pDepthColorFrameInfo->depthCols << std::endl;
		std::cout << m_pDepthColorFrameInfo->cbCompressedLength << std::endl;
		std::cout << m_pDepthColorFrameInfo->crCompressedLength << std::endl;
		std::cout << m_pDepthColorFrameInfo->msbCompressedLength << std::endl;
		std::cout << m_pDepthColorFrameInfo->lsbCompressedLength << std::endl;
		std::cout << m_pDepthColorFrameInfo->frameIdx << std::endl;
		std::cout << m_pDepthColorFrameInfo->imuMeasurementSize << std::endl;
#endif

		*keyFrameIdxEachFrag = m_pDepthColorFrameInfo->keyFrameIdxEachFrag;
		int l2 = sizeof(ImuMsg) * m_pDepthColorFrameInfo->imuMeasurementSize,
			l3 = sizeof(Gravity);
		int compressedDepthColorLength = m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength + m_pDepthColorFrameInfo->msbCompressedLength + m_pDepthColorFrameInfo->lsbCompressedLength;
		m_saveDataFile.read(m_compressedColorDepthBuffer + l1, l2 + l3 + compressedDepthColorLength);

		imuMeasurements->resize(m_pDepthColorFrameInfo->imuMeasurementSize);
		memcpy(imuMeasurements->data(), m_compressedColorDepthBuffer + l1, l2);
		memcpy(gravity, m_compressedColorDepthBuffer + l1 + l2, l3);
		//std::cout << gravity->x << " : " << gravity->y << " : " << gravity->z << std::endl;

		char *pContent = m_compressedColorDepthBuffer + l1 + l2 + l3;
		char* pColorBuffer = pContent;
		char* pDepthBuffer = pColorBuffer + m_colorPixelNum + m_pDepthColorFrameInfo->cbCompressedLength + m_pDepthColorFrameInfo->crCompressedLength;
		decompressData(colorImg, depthImg, pColorBuffer, pDepthBuffer);

		return true;
#if 0
		if (isRegister)
		{
			registerDepthImg(depthImg);
		}
#endif	
	}

	// Just For Test
	void run() override
	{

	}

private:
	int m_depthPixelNum, m_colorPixelNum;
	innoreal::InnoRealTimer m_timer;

	char *m_colorImgYCbCr420Buffer, *m_colorImgCbCr420SplittedBuffer, *m_depthImgMsbLsbSplittedBuffer;
	char *m_compressedColorDepthBuffer;
	char* m_registeredDepthImgBuffer;
	DepthColorFrameInfo *m_pDepthColorFrameInfo;

	Eigen::Matrix4f m_depthToColor;
	float m_fxDepth, m_fyDepth, m_cxDepth, m_cyDepth, m_fxColor, m_fyColor, m_cxColor, m_cyColor;

	std::ifstream m_saveDataFile;
};

