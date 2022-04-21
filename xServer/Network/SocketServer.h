#ifndef SOCKETSERVER_H
#define SOCKETSERVER_H

#include <QtNetwork\QtNetwork>
#include <QtNetwork\QTcpServer>
#include <QtCore\QBuffer>
#include <assert.h>
#include <opencv2/opencv.hpp>

#include"Compress/ImageTrans.h"
#include"Helpers/InnorealTimer.hpp"
#include"Helpers/xUtils.h"
#include "xSensors/xRemoteStructureSensorImporter.h"

enum HEADTYPE
{
	TEST_TYPE = 0,
	COLOR_FRAME_TYPE = 1,
	DEPTH_FRAME_TYPE = 2,
	DEPTH_COLOR_FRAME_TYPE = 3,
	IR_COLOR_FRAME_TYPE = 4,
	FULL_COLOR_FRAME_TYPE = 5,
};

class ReaderThread;

class SendMsg : public QThread
{
public:
	SendMsg(QObject* parent) : QThread(parent)
	{
		srand(0);
	}

	void run() override
	{
		char str[256];
		cv::Mat grayImg(480, 640, CV_8UC1);	
		while (true)
		{	
			int val = rand() % 480;
			memset(grayImg.data, 0, val * 640);
			memset(grayImg.data + val * 640, 200, 640 * 480 - val * 640);
			//sprintf(str, "%s", "1234hehehe\r\n");
			m_tcpSocket->write((const char *)grayImg.data, 640 * 480);
			m_tcpSocket->flush();
			//std::cout << str << std::endl;
			Sleep(20);
		}
	}

	QTcpSocket* m_tcpSocket;
};

class TcpServer : public QObject
{
Q_OBJECT

public:
	explicit TcpServer(RemoteStructureSeneor* remoteStructureSeneor, char* saveDataPath = NULL);
	TcpServer::~TcpServer();

public slots:
	void slotNewConnection();
	void slotReadData();
	void slotDisconnectSocket();

public:
	void sendToClient(const char* data, int size);

private:
	void readColorFrame(const char* headContent);
	void readDepthFrame(const char* headContent);
	void readDepthAndColorFrame(const char *headContent);
	void readIrAndColorFrame(const char *headContent);
	void readFullColorFrame(const char *headContent);
	void readMessage(const char* headContent);

	void readMax(QIODevice* io, int n);
	bool isWaittingForHead;
	QTcpServer* m_tcpServer;
	/*struct ColorFrameInfo {
		int rows;
		int bytesPerRow;
		int cbCompressedLength;
		int crCompressedLength;
	};*/
	QTcpSocket* m_tcpSocket;
	ColorFrameInfo* m_colorFrameInfo;
	DepthFrameInfo* m_depthFrameInfo;
	DepthColorFrameInfo* m_depthColorFrameInfo;	
	ImuMeasurements* m_imuMeasurements;
	Gravity *m_gravity;
	cv::Mat m_colorImg;
	cv::Mat m_grayImg;
	cv::Mat m_depthImg;
	std::vector<char> m_colorImgCbCr420SplittedBuffer;
	std::vector<char> m_colorImgYCbCr420Buffer;	
	std::vector<char> m_depthImgMsbLsbSplittedBuffer;

	FullColorFrameInfo* m_fullColorFrameInfo;
	cv::Mat m_fullColorImg;
	std::vector<char> m_fullColorImgYCbCr420Buffer;
	std::vector<cv::Mat> m_keyFrameVec;
	std::vector<FullColorFrameInfo> m_keyFrameInfoVec;
	std::vector<int> m_pngCompressionParams;	

	innoreal::InnoRealTimer m_timer;
	innoreal::InnoRealTimer m_process_timer;
	int m_frameIdx;
	int m_keyFrameIdx;

	RemoteStructureSeneor* m_remoteStructureSeneor;
	SendMsg m_sendMsg;

  char m_saveDataPath[512];
};

#endif // SOCKETSERVER_H
