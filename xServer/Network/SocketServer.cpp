#include"SocketServer.h"

#include <Mmsystem.h>

#include "Helpers/xUtils.h"
#include <chrono>

#pragma comment(lib, "winmm.lib") 

TcpServer::TcpServer(RemoteStructureSeneor* remoteStructureSeneor, char* saveDataPath)
	: m_frameIdx(0),
	m_keyFrameIdx(0),
	m_remoteStructureSeneor(remoteStructureSeneor),
	m_sendMsg(this)	
{
	memcpy(m_saveDataPath, saveDataPath, sizeof(m_saveDataPath));

	m_colorImg = cv::Mat(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_8UC3);
	m_fullColorImg = cv::Mat(Resolution::getInstance().fullColorRows(), Resolution::getInstance().fullColorCols(), CV_8UC3);
	m_grayImg = cv::Mat(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_8UC1);
	m_depthImg = cv::Mat(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_16UC1);
	m_colorImgCbCr420SplittedBuffer.resize(Resolution::getInstance().rows() * Resolution::getInstance().cols() * 3);
	m_colorImgYCbCr420Buffer.resize(Resolution::getInstance().rows() * Resolution::getInstance().cols() * 3);
	m_fullColorImgYCbCr420Buffer.resize(Resolution::getInstance().fullColorRows() * Resolution::getInstance().fullColorCols() * 3);
	m_depthImgMsbLsbSplittedBuffer.resize(Resolution::getInstance().rows() * Resolution::getInstance().cols() * 2);

	m_pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	m_pngCompressionParams.push_back(0);

	m_tcpServer = new QTcpServer(this);
	connect(m_tcpServer, SIGNAL(newConnection()), this, SLOT(slotNewConnection()));
	m_colorFrameInfo = new ColorFrameInfo;
	m_depthFrameInfo = new DepthFrameInfo;
	m_depthColorFrameInfo = new DepthColorFrameInfo;
	m_fullColorFrameInfo = new FullColorFrameInfo;
	m_imuMeasurements = new ImuMeasurements;
	m_gravity = new Gravity;
	//std::cout << "m_saveDataPath: " << m_saveDataPath << std::endl;
	if (!m_tcpServer->listen(QHostAddress::Any, 6000)) {
		//if (!m_tcpServer->listen(QHostAddress::LocalHost, 6000)) {
		qDebug() << "server is not started";
	}
	else {
		qDebug() << "server is started";
	}
}

TcpServer::~TcpServer()
{
	delete m_tcpServer;
	delete m_colorFrameInfo;
	delete m_depthFrameInfo;
	delete m_depthColorFrameInfo;
	delete m_fullColorFrameInfo;
	delete m_imuMeasurements;
	delete m_gravity;
}

void TcpServer::slotNewConnection() {
	m_tcpSocket = m_tcpServer->nextPendingConnection();
	std::cout << "New connection come" << std::endl;
	//in.setDevice(mTcpSocket);
	//in.setVersion(QDataStream::Qt_5_6);*/	
	connect(m_tcpSocket, SIGNAL(readyRead()), this, SLOT(slotReadData()));
	connect(m_tcpSocket, SIGNAL(disconnected()), this, SLOT(slotDisconnectSocket()));
}

void TcpServer::sendToClient(const char* data, int size)
{
#if 0
	m_sendMsg.m_tcpSocket = m_tcpSocket;
	m_sendMsg.start();
#endif
#if 0
	cv::Mat grayImg(480, 640, CV_8UC1);
	//memset(grayImg.data, 200, 640 * 480);
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
#endif
	m_tcpSocket->write(data, size);
	m_tcpSocket->flush();
}

inline const char *writeData(const char* dst, const char* src, int size)
{
	memcpy((void *)dst, src, size);

	return dst + size;
}
inline const char *readData(const char* dst, const char* src, int size)
{
	memcpy((void *)dst, src, size);

	return src + size;
}

void TcpServer::readColorFrame(const char *headContent)
{
#if 1
	sscanf(headContent, "%d%d%d%d%d", &m_colorFrameInfo->rows,
				 &m_colorFrameInfo->cols,
				 &m_colorFrameInfo->cbCompressedLength,
				 &m_colorFrameInfo->crCompressedLength,
				 &m_colorFrameInfo->frameIdx);
#if 0
	std::cout << m_colorFrameInfo->rows << std::endl;
	std::cout << m_colorFrameInfo->cols << std::endl;
	std::cout << m_colorFrameInfo->cbCompressedLength << std::endl;
	std::cout << m_colorFrameInfo->crCompressedLength << std::endl;
	std::cout << m_colorFrameInfo->frameIdx << std::endl;
#endif
	int colorPixelNum = m_colorFrameInfo->rows*m_colorFrameInfo->cols;
	int comressedFrameSize = colorPixelNum + m_colorFrameInfo->cbCompressedLength + m_colorFrameInfo->crCompressedLength;

	while (m_tcpSocket->bytesAvailable() < comressedFrameSize)
	{
		m_tcpSocket->waitForReadyRead();
	}

	QByteArray in(m_tcpSocket->read(comressedFrameSize));
#if 1
	//std::cout << m_colorImgYUV420Buf.size() << std::endl;
	//std::cout << m_colorFrameInfo->rows*m_colorFrameInfo->cols << std::endl;
	memcpy(m_colorImgYCbCr420Buffer.data(), in.data(), colorPixelNum);
	ImageTrans::DecompressCbCr(m_colorImgYCbCr420Buffer.data() + colorPixelNum, colorPixelNum / 2, m_colorImgCbCr420SplittedBuffer.data(),
														 in.data() + colorPixelNum, m_colorFrameInfo->cbCompressedLength, m_colorFrameInfo->crCompressedLength);
	//m_grayImg.data = (uchar *)m_colorImgYCbCr420Buffer.data();
	//cv::imshow("input_color", m_grayImg);
	ImageTrans::DecodeYCbCr420SP(m_colorImg.data, (const uchar *)m_colorImgYCbCr420Buffer.data(), m_colorFrameInfo->cols, m_colorFrameInfo->rows);
	//cv::cvtColor(m_colorImg, m_colorImg, cv::COLOR_YCrCb2BGR);
	//cv::imshow("input_color", m_colorImg);
	//cv::waitKey(1);
#endif
#endif

	m_timer.TimeEnd();
	double timeSum = m_timer.TimeGap_in_ms();
	std::cout << "Average: " << timeSum / (m_frameIdx + 1) << std::endl;
	if (m_frameIdx == 1000) {
		std::cout << "Average: " << timeSum / (m_frameIdx + 1) << std::endl;
		std::exit(0);
	}
	++m_frameIdx;
#if 0
	ImageTrans::decompress_CbCr(compressedColorImgYUV420 + m_colorFrameInfo->rows * m_colorFrameInfo->cols,
															m_colorFrameInfo->cbCompressedLength + m_colorFrameInfo->crCompressedLength,
															m_colorImgYUV420Buf.data() + m_colorFrameInfo->rows * m_colorFrameInfo->cols,
															m_colorFrameInfo->rows * m_colorFrameInfo->cols / 2,
															m_colorFrameInfo->cbCompressedLength,
															m_colorFrameInfo->crCompressedLength);
	decodeYUV420SP((int*)m_colorImg.data, m_colorImgYUV420Buf.data(), m_colorFrameInfo->cols, m_colorFrameInfo->rows);
	cv::cvtColor(color_bgr, color_rgb, CV_BGRA2RGB);
#endif

	//cv::imshow("Color", color_rgb);
	//cv::waitKey(1);
#if 0

		/*if (cnt_time == 1) {
		reader_thread = new ReaderThread();
		assert(1);
		}*/
		/*if (cnt < 100 && ++cnt == 100) {
		reader_thread = new ReaderThread();
		}*/
		//reader_thread->start();
		/*while (!color_buffer.readable)
		Sleep(1);*/
	color_buffer.readable = false;
	if (!needNewThread) {
		reader_thread->mtx.lock();
		//std::cout << "Main thread lock" << std::endl;
	}
	if (!color_buffer.isFull) {
		color_buffer.color_queue[color_buffer.rear % color_buffer.queueSize] = color_rgb.clone();
		color_buffer.rear = (color_buffer.rear + 1) % color_buffer.queueSize;
		//std::cout << color_buffer.front << " " << color_buffer.rear << std::endl;
		if (color_buffer.front == color_buffer.rear)
			color_buffer.isFull = true;
	}
	else {
		std::cout << "Is full! " << color_buffer.front << " " << color_buffer.rear << std::endl;
	}
	if (!needNewThread) {
		reader_thread->mtx.unlock();
		std::cout << "Main thread unlock" << std::endl;
	}
	//color_buffer.readable = true;
	if (needNewThread) {
		reader_thread = new ReaderThread();
		needNewThread = false;
	}
#endif
#if 0

	int sleep_ms = 33 - sleepTimer.TimeGap_in_ms();
	if (sleep_ms > 0) {
		std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_ms * 1000000));
		printf("sleep_ms=%f\n", sleepTimer.TimeGap_in_ms());
	}
#endif	
}

void TcpServer::readDepthFrame(const char *headContent)
{
	sscanf(headContent, "%d%d%d%d%d", &m_depthFrameInfo->rows,
				 &m_depthFrameInfo->cols,
				 &m_depthFrameInfo->msbCompressedLength,
				 &m_depthFrameInfo->lsbCompressedLength,
				 &m_depthFrameInfo->frameIdx);
#if 0
	std::cout << m_depthFrameInfo->rows << std::endl;
	std::cout << m_depthFrameInfo->cols << std::endl;
	std::cout << m_depthFrameInfo->msbCompressedLength << std::endl;
	std::cout << m_depthFrameInfo->lsbCompressedLength << std::endl;
	std::cout << m_depthFrameInfo->frameIdx << std::endl;
#endif
	int depthPixelNum = m_depthImg.rows * m_depthImg.cols;
	int comressedFrameSize = m_depthFrameInfo->msbCompressedLength + m_depthFrameInfo->lsbCompressedLength;

	while (m_tcpSocket->bytesAvailable() < comressedFrameSize)
	{
		m_tcpSocket->waitForReadyRead();
	}

	QByteArray in(m_tcpSocket->read(comressedFrameSize));
#if 1
	ImageTrans::DecompressDepth((char *)m_depthImg.data, depthPixelNum * 2, m_depthImgMsbLsbSplittedBuffer.data(),
															in.data(), m_depthFrameInfo->msbCompressedLength, m_depthFrameInfo->msbCompressedLength);
#endif
#if 0
	memcpy(m_depthImg.data, in.data(), depthPixelNum * 2);
#endif
#if 0
	char *pMsbData = in.data(), *pLsbData = in.data() + depthPixelNum;
	char *pDepthImgData = (char *)m_depthImg.data;
	for (int i = 0; i < depthPixelNum; ++i)
	{
		*(pDepthImgData++) = *(pLsbData++);
		*(pDepthImgData++) = *(pMsbData++);
	}
#endif
	shift2depth((uint16_t *)m_depthImg.data, depthPixelNum);
#if 0
	for (int i = 0; i < depthPixelNum; ++i)
	{
		std::cout << *((uint16_t *)m_depthImg.data + i) << ", ";
	}
	std::cout << std::endl;
#endif
	cv::imshow("input_depth", m_depthImg * 30);
	cv::waitKey(1);

#if 0
	m_timer.TimeEnd();
	double timeSum = m_timer.TimeGap_in_ms();
	std::cout << "Average: " << timeSum / (m_frameIdx + 1) << std::endl;
	if (m_frameIdx == 1000) {
		std::cout << "Average: " << timeSum / (m_frameIdx + 1) << std::endl;
		std::exit(0);
	}
	++m_frameIdx;
#endif
}

void TcpServer::readDepthAndColorFrame(const char *headContent)
{
#if 0
	sscanf(headContent, "%d%d%d%d%d%d%d", &m_depthColorFrameInfo->rows,
				 &m_depthColorFrameInfo->cols,
				 &m_depthColorFrameInfo->cbCompressedLength,
				 &m_depthColorFrameInfo->crCompressedLength,
				 &m_depthColorFrameInfo->msbCompressedLength,
				 &m_depthColorFrameInfo->lsbCompressedLength,
				 &m_depthColorFrameInfo->frameIdx);
#endif
#if 1
	const char* pCur = headContent;
	pCur = readData((char *)&m_depthColorFrameInfo->keyFrameIdxEachFrag, pCur, sizeof(m_depthColorFrameInfo->keyFrameIdxEachFrag));
	pCur = readData((char *)&m_depthColorFrameInfo->colorRows, pCur, sizeof(m_depthColorFrameInfo->colorRows));
	pCur = readData((char *)&m_depthColorFrameInfo->colorCols, pCur, sizeof(m_depthColorFrameInfo->colorCols));
	pCur = readData((char *)&m_depthColorFrameInfo->depthRows, pCur, sizeof(m_depthColorFrameInfo->depthRows));
	pCur = readData((char *)&m_depthColorFrameInfo->depthCols, pCur, sizeof(m_depthColorFrameInfo->depthCols));
	pCur = readData((char *)&m_depthColorFrameInfo->cbCompressedLength, pCur, sizeof(m_depthColorFrameInfo->cbCompressedLength));
	pCur = readData((char *)&m_depthColorFrameInfo->crCompressedLength, pCur, sizeof(m_depthColorFrameInfo->crCompressedLength));
	pCur = readData((char *)&m_depthColorFrameInfo->msbCompressedLength, pCur, sizeof(m_depthColorFrameInfo->msbCompressedLength));
	pCur = readData((char *)&m_depthColorFrameInfo->lsbCompressedLength, pCur, sizeof(m_depthColorFrameInfo->lsbCompressedLength));
	pCur = readData((char *)&m_depthColorFrameInfo->frameIdx, pCur, sizeof(m_depthColorFrameInfo->frameIdx));

#if 1
	int gravityElemSize, measurementElemSize;
	pCur = readData((char *)&gravityElemSize, pCur, sizeof(gravityElemSize));
	pCur = readData((char *)m_gravity, pCur, gravityElemSize);
	assert(sizeof(Gravity) == gravityElemSize);

	pCur = readData((char *)&measurementElemSize, pCur, sizeof(measurementElemSize));
	pCur = readData((char *)&m_depthColorFrameInfo->imuMeasurementSize, pCur, sizeof(m_depthColorFrameInfo->imuMeasurementSize));
	m_imuMeasurements->resize(m_depthColorFrameInfo->imuMeasurementSize);
	if (m_imuMeasurements->size() > 0)
		assert(sizeof(m_imuMeasurements->at(0)) == measurementElemSize);
	pCur = readData((char *)m_imuMeasurements->data(), pCur, measurementElemSize * m_depthColorFrameInfo->imuMeasurementSize);
#endif
#endif
#if 0
	std::cout << m_depthColorFrameInfo->rows << std::endl;
	std::cout << m_depthColorFrameInfo->cols << std::endl;
	std::cout << m_depthColorFrameInfo->cbCompressedLength << std::endl;
	std::cout << m_depthColorFrameInfo->crCompressedLength << std::endl;
	std::cout << m_depthColorFrameInfo->msbCompressedLength << std::endl;
	std::cout << m_depthColorFrameInfo->lsbCompressedLength << std::endl;
	std::cout << m_depthColorFrameInfo->frameIdx << std::endl;
	std::cout << "gravityElemSize:" << gravityElemSize << std::endl;
	std::cout << "sizeof(gravity): " << sizeof(*m_gravity) << std::endl;
	std::cout << "measurementElemSize: " << measurementElemSize << std::endl;
	std::cout << "measurementSize:" << m_imuMeasurements->size() << std::endl;
	std::cout << m_gravity->x << std::endl;
	std::cout << m_gravity->y << std::endl;
	std::cout << m_gravity->z << std::endl;
#endif
	int colorComressedFrameSize = m_depthColorFrameInfo->colorRows*m_depthColorFrameInfo->colorCols + m_depthColorFrameInfo->cbCompressedLength + m_depthColorFrameInfo->crCompressedLength;
	int depthComressedFrameSize = m_depthColorFrameInfo->msbCompressedLength + m_depthColorFrameInfo->lsbCompressedLength;
	//m_process_timer.TimeStart();
	while (m_tcpSocket->bytesAvailable() < colorComressedFrameSize)
	{
		m_tcpSocket->waitForReadyRead();
	}
	QByteArray &inColor = m_tcpSocket->read(colorComressedFrameSize);
	while (m_tcpSocket->bytesAvailable() < depthComressedFrameSize)
	{
		m_tcpSocket->waitForReadyRead();
	}
	QByteArray &inDepth = m_tcpSocket->read(depthComressedFrameSize);
	//m_process_timer.TimeEnd();
	//std::cout << "socket time: " << m_process_timer.TimeGap_in_ms() << std::endl;

#if 1
	m_remoteStructureSeneor->setDataBuffer(inColor.data(), inDepth.data(), colorComressedFrameSize, depthComressedFrameSize, m_depthColorFrameInfo, m_imuMeasurements, m_gravity);
#endif
#if 0
	char* colorData = inColor.data();
	char* depthData = inDepth.data();
	int colorPixelNum = m_colorFrameInfo->rows*m_colorFrameInfo->cols;
	int depthPixelNum = colorPixelNum;
	m_process_timer.TimeStart();
	memcpy(m_colorImgYCbCr420Buffer.data(), colorData, colorPixelNum);
	ImageTrans::DecompressCbCr(m_colorImgYCbCr420Buffer.data() + colorPixelNum, colorPixelNum / 2, m_colorImgCbCr420SplittedBuffer.data(),
														 colorData + colorPixelNum, m_colorFrameInfo->cbCompressedLength, m_colorFrameInfo->crCompressedLength);
	ImageTrans::DecodeYCbCr420SP((unsigned char*)m_colorImg.data, (const uchar *)m_colorImgYCbCr420Buffer.data(), m_colorFrameInfo->cols, m_colorFrameInfo->rows);

	ImageTrans::DecompressDepth((char *)m_depthImg.data, depthPixelNum * 2, m_depthImgMsbLsbSplittedBuffer.data(),
															depthData, m_depthFrameInfo->msbCompressedLength, m_depthFrameInfo->msbCompressedLength);
	shift2depth((uint16_t *)m_depthImg.data, depthPixelNum);
	m_process_timer.TimeEnd();
	std::cout << "decompress time: " << m_process_timer.TimeGap_in_ms() << std::endl;
#endif
#if 0
	cv::imshow("input_color", m_colorImg);
	cv::imshow("input_depth", m_depthImg * 30);
	cv::waitKey(1);
#endif

	m_timer.TimeEnd();
	double timeSum = m_timer.TimeGap_in_ms();
	printf("Average: %f\n", timeSum / (m_frameIdx + 1));
#if 0
	if (m_frameIdx == 1000) {
		std::cout << "Average: " << timeSum / (m_frameIdx + 1) << std::endl;
		std::exit(0);
	}
#endif
	++m_frameIdx;
}

void TcpServer::readIrAndColorFrame(const char *headContent)
{
	const char* pCur = headContent;
	pCur = readData((char *)&m_depthColorFrameInfo->keyFrameIdxEachFrag, pCur, sizeof(m_depthColorFrameInfo->keyFrameIdxEachFrag));
	pCur = readData((char *)&m_depthColorFrameInfo->colorRows, pCur, sizeof(m_depthColorFrameInfo->colorRows));
	pCur = readData((char *)&m_depthColorFrameInfo->colorCols, pCur, sizeof(m_depthColorFrameInfo->colorCols));
	pCur = readData((char *)&m_depthColorFrameInfo->depthRows, pCur, sizeof(m_depthColorFrameInfo->depthRows));
	pCur = readData((char *)&m_depthColorFrameInfo->depthCols, pCur, sizeof(m_depthColorFrameInfo->depthCols));
	pCur = readData((char *)&m_depthColorFrameInfo->cbCompressedLength, pCur, sizeof(m_depthColorFrameInfo->cbCompressedLength));
	pCur = readData((char *)&m_depthColorFrameInfo->crCompressedLength, pCur, sizeof(m_depthColorFrameInfo->crCompressedLength));
	pCur = readData((char *)&m_depthColorFrameInfo->msbCompressedLength, pCur, sizeof(m_depthColorFrameInfo->msbCompressedLength));
	pCur = readData((char *)&m_depthColorFrameInfo->lsbCompressedLength, pCur, sizeof(m_depthColorFrameInfo->lsbCompressedLength));
	pCur = readData((char *)&m_depthColorFrameInfo->frameIdx, pCur, sizeof(m_depthColorFrameInfo->frameIdx));

	int gravityElemSize, measurementElemSize;
	pCur = readData((char *)&gravityElemSize, pCur, sizeof(gravityElemSize));
	pCur = readData((char *)&m_gravity, pCur, gravityElemSize);
	assert(sizeof(Gravity) == gravityElemSize);

	pCur = readData((char *)&measurementElemSize, pCur, sizeof(measurementElemSize));
	pCur = readData((char *)&m_depthColorFrameInfo->imuMeasurementSize, pCur, sizeof(m_depthColorFrameInfo->imuMeasurementSize));
	m_imuMeasurements->resize(m_depthColorFrameInfo->imuMeasurementSize);
	if (m_imuMeasurements->size() > 0)
		assert(sizeof(m_imuMeasurements->at(0)) == measurementElemSize);
	pCur = readData((char *)m_imuMeasurements->data(), pCur, measurementElemSize * m_depthColorFrameInfo->imuMeasurementSize);

	int colorComressedFrameSize = m_depthColorFrameInfo->colorRows*m_depthColorFrameInfo->colorCols + m_depthColorFrameInfo->cbCompressedLength + m_depthColorFrameInfo->crCompressedLength;
	int depthComressedFrameSize = m_depthColorFrameInfo->msbCompressedLength + m_depthColorFrameInfo->lsbCompressedLength;
	m_process_timer.TimeStart();
	while (m_tcpSocket->bytesAvailable() < colorComressedFrameSize)
	{
		m_tcpSocket->waitForReadyRead();
	}
	QByteArray &inColor = m_tcpSocket->read(colorComressedFrameSize);
	while (m_tcpSocket->bytesAvailable() < depthComressedFrameSize)
	{
		m_tcpSocket->waitForReadyRead();
	}
	QByteArray &inDepth = m_tcpSocket->read(depthComressedFrameSize);
	m_process_timer.TimeEnd();
	std::cout << "socket time: " << m_process_timer.TimeGap_in_ms() << std::endl;

	m_remoteStructureSeneor->setDataBuffer(inColor.data(), inDepth.data(), colorComressedFrameSize, depthComressedFrameSize, m_depthColorFrameInfo, m_imuMeasurements, m_gravity);

	m_timer.TimeEnd();
	double timeSum = m_timer.TimeGap_in_ms();
	std::cout << "Average: " << timeSum / (m_frameIdx + 1) << std::endl;
#if 0
	if (m_frameIdx == 1000) {
		std::cout << "Average: " << timeSum / (m_frameIdx + 1) << std::endl;
		std::exit(0);
	}
#endif
	++m_frameIdx;
}

void TcpServer::readFullColorFrame(const char *headContent)
{
	const char* pCur = headContent;
	pCur = readData((char *)&m_fullColorFrameInfo->keyFrameNum, pCur, sizeof(m_fullColorFrameInfo->keyFrameNum));
	pCur = readData((char *)&m_fullColorFrameInfo->frameIdx, pCur, sizeof(m_fullColorFrameInfo->frameIdx));
	pCur = readData((char *)&m_fullColorFrameInfo->colorRows, pCur, sizeof(m_fullColorFrameInfo->colorRows));
	pCur = readData((char *)&m_fullColorFrameInfo->colorCols, pCur, sizeof(m_fullColorFrameInfo->colorCols));
	pCur = readData((char *)&m_fullColorFrameInfo->colorBytePerRow, pCur, sizeof(m_fullColorFrameInfo->colorBytePerRow));

#if 1
	std::cout << "key frame: " << std::endl;
	std::cout << m_fullColorFrameInfo->frameIdx << std::endl;
	std::cout << m_fullColorFrameInfo->colorRows << std::endl;
	std::cout << m_fullColorFrameInfo->colorCols << std::endl;
	std::cout << m_fullColorFrameInfo->colorBytePerRow << std::endl;
#endif
	int frameSize = 1.5 * m_fullColorFrameInfo->colorRows*m_fullColorFrameInfo->colorBytePerRow;
	m_process_timer.TimeStart();
	while (m_tcpSocket->bytesAvailable() < frameSize)
	{
		m_tcpSocket->waitForReadyRead();
	}
	QByteArray &inColor = m_tcpSocket->read(frameSize);
	m_process_timer.TimeEnd();
	std::cout << "socket time: " << m_process_timer.TimeGap_in_ms() << std::endl;

#if 0
	m_remoteStructureSeneor->setDataBuffer(inColor.data(), inDepth.data(), colorComressedFrameSize, depthComressedFrameSize, m_depthColorFrameInfo, m_imuMeasurements, m_gravity);
	ImageTrans::DecodeYCbCr420SP((unsigned char*)m_colorImg.data, (const uchar *)m_colorImgYCbCr420Buffer.data(), m_colorFrameInfo->cols, m_colorFrameInfo->rows);
#endif
	cudaMemcpy2D(m_fullColorImgYCbCr420Buffer.data(), m_fullColorFrameInfo->colorCols,
							 inColor.data(), m_fullColorFrameInfo->colorBytePerRow, m_fullColorFrameInfo->colorCols, m_fullColorFrameInfo->colorRows * 1.5,
							 cudaMemcpyHostToHost);
	ImageTrans::DecodeYCbCr420SP((unsigned char*)m_fullColorImg.data, (const uchar *)m_fullColorImgYCbCr420Buffer.data(), m_fullColorFrameInfo->colorCols, m_fullColorFrameInfo->colorRows);
	m_keyFrameVec.push_back(m_fullColorImg.clone());
	m_keyFrameInfoVec.push_back(*m_fullColorFrameInfo);

	m_timer.TimeEnd();
	double timeSum = m_timer.TimeGap_in_ms();
	std::cout << "Average: " << timeSum / (m_frameIdx + 1) << std::endl;

	++m_frameIdx;
}

void TcpServer::slotReadData() {
	//QDataStream readInStream(mTcpSocket);
	static uint next_block_size = HEAD_SIZE;
	static int frameID = -1;
	volatile static bool needNewThread = true;
	::timeBeginPeriod(1);
	bool isFirstFrame = true, isFirstKeyFrame = true;
	while (true)
	{
		while (m_tcpSocket->bytesAvailable() < HEAD_SIZE)
		{
			m_tcpSocket->waitForReadyRead();
		}
#if 1 
		sendToClient(m_remoteStructureSeneor->m_feedBackBuffer.data(), m_remoteStructureSeneor->m_feedBackBuffer.size() / 3);
#endif
		QByteArray in(m_tcpSocket->read(HEAD_SIZE));
		char* pPackage = in.data();
		isWaittingForHead = false;
		uint8_t headType = *pPackage;
		//qDebug() << tag;
		char* headContent;
		static int preFrameID = -1;
		uint64_t pkgTime;
		uint64_t timestamp;
		std::chrono::time_point<std::chrono::system_clock> timepoint;
		switch (headType)
		{
		case TEST_TYPE:
			std::cout << "hello" << std::endl;
			char str[256];
			sscanf(pPackage + 1, "%s", str);
			std::cout << str << std::endl;
			break;
		case COLOR_FRAME_TYPE:
			readColorFrame(pPackage + 1);
#if 0
			pkgTime = *(uint64_t*)(pPackage + sizeof(headType));
			timepoint = std::chrono::system_clock::now();
			timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(timepoint.time_since_epoch()).count();
			std::cout << "Delte T " << (int)(timestamp - pkgTime) << std::endl;
#endif
			break;
		case DEPTH_FRAME_TYPE:
			readDepthFrame(pPackage + 1);
			//std::exit(0);
			break;
		case DEPTH_COLOR_FRAME_TYPE:
			if (isFirstFrame)
			{
				isFirstFrame = false;
				m_frameIdx = 0;
				m_timer.TimeStart();
			}
			readDepthAndColorFrame(pPackage + 1);
			break;
		case IR_COLOR_FRAME_TYPE:
			if (isFirstFrame)
			{
				isFirstFrame = false;
				m_frameIdx = 0;
				m_timer.TimeStart();
			}
			readDepthAndColorFrame(pPackage + 1);
			break;
		case FULL_COLOR_FRAME_TYPE:
			// the scan is end, add a end flag to the buffer
			m_remoteStructureSeneor->setEndScanFlag();

			if (isFirstKeyFrame)
			{
				isFirstKeyFrame = false;
				m_keyFrameIdx = 0;
			}
			readFullColorFrame(pPackage + 1);
			++m_keyFrameIdx;
			if (m_keyFrameIdx == m_fullColorFrameInfo->keyFrameNum)
			{
				goto SAVE_KEY_FRAMES;
			}
			break;
		default:
			qDebug() << "Can not recongize headType! " << headType;
			break;
		}
	}
	::timeEndPeriod(1);
SAVE_KEY_FRAMES:
	char saveDir[256];
	for (int i = 0; i < m_keyFrameVec.size(); ++i)
	{
		sprintf(saveDir, "%s\\%06d_key_frame.png", m_saveDataPath, i);
		std::cout << "save key frame path: " << saveDir << std::endl;
		std::cout << "save key frame " << m_keyFrameInfoVec[i].frameIdx << std::endl;

		cv::imwrite(saveDir, m_keyFrameVec[i], m_pngCompressionParams);
	}
}

void TcpServer::slotDisconnectSocket() {
	std::cout << "disconnection" << std::endl;
	m_tcpSocket->close();
}

void TcpServer::readMax(QIODevice *io, int n)
{
#if 0
	while (buffer.size() < n) {
		if (!io->bytesAvailable()) {
			io->waitForReadyRead(1);
		}
		buffer.write(io->read(n - buffer.size()));
	}
#endif
}

