#include "Helpers/xUtils.h"
#include "ObjRecon.h"
#include "MainWindow.h"

#include <opencv2/opencv.hpp>
#include <QtNetwork\QtNetwork>
#include <QtWidgets\QApplication>
#include <QtNetwork\QTcpServer>
#include <condition_variable>
#include <direct.h> 

#include "Network/SocketServer.h"
#include "GlobalWindowLayout.h"	

ViewWindow::ViewWindow(QWidget *parent) :
	QWidget(parent) {

	int depthWidth = 640, depthHeight = 480,
		colorWidth = 648, colorHeight = 484,
		fullColorWidth = 2592, fullColorHeight = 1936;
	double intrinScale = 648 / (double)640;
	float intrinColor[4] = { 544.8898 * intrinScale, 545.9078 * intrinScale,
		321.6016 * intrinScale, 237.0330 * intrinScale };
	float intrinDepth[4] = { 574.0135, 575.5523, 314.5388, 242.4793 };
	float extrinT[3] = { -41.1776, -4.3666, -34.8012 };
	float extrinR[9] = {
		1.0000, -0.0040, -0.0029,
		0.0040, 0.9999, 0.0132,
		0.0028, -0.0132, 0.9999
	};
#if 0
	float extrinT[3] = { -41.1776, -4.3666, -34.8012 };
	float extrinR[9] = {
		1.0000, -0.0040, -0.0029,
		0.0040, 0.9999, 0.0132,
		0.0028, -0.0132, 0.9999 };
	float scale = 0.25;
	float intrinColor[4] = { 2116.1 * scale, 2108.7 * scale, 1286.8 * scale, 937.4 * scale };
	float intrinDepth[4] = { 569.71, 572.04, 309.91, 247.86 };
	intrinDepth[3] -= 4; // IR图高度是488，depth图高度是480，假设位于中间
#endif

	GlobalState::getInstance().m_depthFx = intrinDepth[0];
	GlobalState::getInstance().m_depthFy = intrinDepth[1];
	GlobalState::getInstance().m_depthCx = intrinDepth[2];
	GlobalState::getInstance().m_depthCy = intrinDepth[3];
	GlobalState::getInstance().m_depthWidth = depthWidth;
	GlobalState::getInstance().m_depthHeight = depthHeight;
	GlobalState::getInstance().m_colorFx = intrinDepth[0];
	GlobalState::getInstance().m_colorFy = intrinDepth[1];
	GlobalState::getInstance().m_colorCx = intrinDepth[2];
	GlobalState::getInstance().m_colorCy = intrinDepth[3];
	GlobalState::getInstance().m_colorWidth = depthWidth;
	GlobalState::getInstance().m_colorHeight = depthHeight;
	memcpy(GlobalState::getInstance().m_extrinT, extrinT, sizeof(float) * 3);
	memcpy(GlobalState::getInstance().m_extrinR, extrinR, sizeof(float) * 9);

	QRect window_geometry = GlobalWindowLayout::getInstance().getGeometry("ViewWindow");
	setGeometry(window_geometry);
	image_widget_width_ = window_geometry.width();

	setWindowFlags(this->windowFlags() | Qt::FramelessWindowHint);
	setWindowFlags(this->windowFlags() | Qt::WindowStaysOnTopHint);

	mainWidget_ = new QWidget();

	mainScrollArea_ = new QScrollArea();
	mainScrollArea_->setWidget(mainWidget_);
	mainScrollArea_->setWidgetResizable(true);
	mainScrollArea_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	//mainScrollArea_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

	mainLayout_ = new QVBoxLayout();
	mainLayout_->setContentsMargins(0, 0, 0, 0);
	mainLayout_->setSpacing(0);
	mainLayout_->setMargin(0);
	mainWidget_->setLayout(mainLayout_);

	stackLayout_ = new QStackedLayout(this);
	stackLayout_->addWidget(mainScrollArea_);
	stackLayout_->setCurrentIndex(0);
	stackLayout_->setContentsMargins(0, 0, 0, 0);

	int y_step = 20, curr_y = 10;
	int y_height = 15;
	m_dataTransTestCheckBox = new QCheckBox("data trans test", this);
	m_dataTransTestCheckBox->setGeometry(QRect(QPoint(10, curr_y),
																			 QSize(300, y_height)));
	curr_y += y_step;
	m_useStructureSensorCheckBox = new QCheckBox("use structure sensor", this);
	m_useStructureSensorCheckBox->setGeometry(QRect(QPoint(10, curr_y),
																						QSize(300, y_height)));
	curr_y += y_step;
	m_doReconstructionCheckBox = new QCheckBox("do reconstruction", this);
	m_doReconstructionCheckBox->setGeometry(QRect(QPoint(10, curr_y),
																					QSize(300, y_height)));
	curr_y += y_step;
	m_saveToBinFileCheckBox = new QCheckBox("save to .bin file", this);
	m_saveToBinFileCheckBox->setGeometry(QRect(QPoint(10, curr_y),
																			 QSize(300, y_height)));
	curr_y += y_step;
	m_loadFromBinFileCheckBox = new QCheckBox("load from .bin file", this);
	m_loadFromBinFileCheckBox->setGeometry(QRect(QPoint(10, curr_y),
																				 QSize(300, y_height)));
	curr_y += y_step;
	m_adjustExtrinCheckBox = new QCheckBox("adjust extrin", this);
	m_adjustExtrinCheckBox->setGeometry(QRect(QPoint(10, curr_y),
																			QSize(300, y_height)));
	curr_y += y_step;
	m_debugTrackingCheckBox = new QCheckBox("debug tracking", this);
	m_debugTrackingCheckBox->setGeometry(QRect(QPoint(10, curr_y),
																			 QSize(300, y_height)));
	curr_y += y_step;
	m_useKNNCheckBox = new QCheckBox("use knn (slow)", this);
	m_useKNNCheckBox->setGeometry(QRect(QPoint(10, curr_y),
																QSize(300, y_height)));
	curr_y += y_step;
	m_withNonRigidRegistrationCheckBox = new QCheckBox("non rigid registration", this);
	m_withNonRigidRegistrationCheckBox->setGeometry(QRect(QPoint(10, curr_y),
																									QSize(300, y_height)));
	curr_y += y_step;
	curr_y += y_step;

	y_step = 30;
	y_height = 25;
	QPushButton *m_openFileButton = new QPushButton("open .bin file", this);
	m_openFileButton->setGeometry(QRect(QPoint(10, curr_y),
																QSize(100, y_height)));
	curr_y += y_step;
	QPushButton *m_startServerButton = new QPushButton("start server", this);
	m_startServerButton->setGeometry(QRect(QPoint(10, curr_y),
																	 QSize(100, y_height)));
	curr_y += y_step;
	QPushButton *m_stopServerButton = new QPushButton("stop server", this);
	m_stopServerButton->setGeometry(QRect(QPoint(10, curr_y),
																	QSize(100, y_height)));
	curr_y += y_step;
	
	QLabel *depthIntrinLabel = new QLabel(this);
	depthIntrinLabel->setText("Depth Intrinsics (fx,fy,cx,cy,w,h): ");
	depthIntrinLabel->setGeometry(QRect(QPoint(10, curr_y),
																 QSize(220, y_height)));
	curr_y += y_step;
	QPlainTextEdit *depthIntrinTextEdit = new QPlainTextEdit(this);
	depthIntrinTextEdit->setGeometry(QRect(QPoint(10, curr_y),
																	 QSize(220, y_height * 1.5)));
	curr_y += y_step;
	char str[1024];
	sprintf(str, "%.3f, %.3f, %.3f, %.3f, %d, %d", 
					GlobalState::getInstance().m_depthFx,
					GlobalState::getInstance().m_depthFy,
					GlobalState::getInstance().m_depthCx,
					GlobalState::getInstance().m_depthCy,
					GlobalState::getInstance().m_depthWidth,
					GlobalState::getInstance().m_depthHeight);
	depthIntrinTextEdit->setPlainText(str);
	QLabel *colorIntrinLabel = new QLabel(this);
	colorIntrinLabel->setText("Color Intrinsics (fx,fy,cx,cy,w,h): ");
	colorIntrinLabel->setGeometry(QRect(QPoint(10, curr_y),
																 QSize(220, y_height)));
	curr_y += y_step;
	QPlainTextEdit *colorIntrinTextEdit = new QPlainTextEdit(this);
	colorIntrinTextEdit->setGeometry(QRect(QPoint(10, curr_y),
																	 QSize(220, y_height * 1.5)));
	curr_y += y_step;
	sprintf(str, "%.3f, %.3f, %.3f, %.3f, %d, %d",
					GlobalState::getInstance().m_colorFx,
					GlobalState::getInstance().m_colorFy,
					GlobalState::getInstance().m_colorCx,
					GlobalState::getInstance().m_colorCy,
					GlobalState::getInstance().m_colorWidth,
					GlobalState::getInstance().m_colorHeight);
	colorIntrinTextEdit->setPlainText(str);
	QLabel *colorToDepthLabel = new QLabel(this);
	colorToDepthLabel->setText("Color To Depth: ");
	colorToDepthLabel->setGeometry(QRect(QPoint(10, curr_y),
																 QSize(220, y_height)));
	curr_y += y_step;
	QPlainTextEdit *colorToDepthTextEdit = new QPlainTextEdit(this);
	colorToDepthTextEdit->setGeometry(QRect(QPoint(10, curr_y),
																	 QSize(220, y_height * 2.0)));
	sprintf(str, "%.3f, %.3f, %.3f, %.3f\n %.3f, %.3f, %.3f, %.3f\n %.3f, %.3f, %.3f, %.3f\n",
					GlobalState::getInstance().m_extrinR[0], GlobalState::getInstance().m_extrinR[3], GlobalState::getInstance().m_extrinR[6],
					GlobalState::getInstance().m_extrinT[0],
					GlobalState::getInstance().m_extrinR[1], GlobalState::getInstance().m_extrinR[4], GlobalState::getInstance().m_extrinR[7],
					GlobalState::getInstance().m_extrinT[1],
					GlobalState::getInstance().m_extrinR[2], GlobalState::getInstance().m_extrinR[5], GlobalState::getInstance().m_extrinR[8], 
					GlobalState::getInstance().m_extrinT[2]);
	colorToDepthTextEdit->setPlainText(str);
	curr_y += y_step;
	curr_y += y_step;

	y_step = 20;
	y_height = 15;
	QLabel *depthRangeLabel = new QLabel(this);
	depthRangeLabel->setText("Depth Range [min, max] (mm): ");
	depthRangeLabel->setGeometry(QRect(QPoint(10, curr_y),
															 QSize(400, y_height)));
	curr_y += y_step;
	int nMin = 0;
	int nMax = 10000;
	int nSingleStep = 500;
	m_minDepthSpinBox = new QSpinBox(this);
	m_minDepthSpinBox->setMinimum(nMin);
	m_minDepthSpinBox->setMaximum(nMax);
	m_minDepthSpinBox->setSingleStep(nSingleStep);
	m_minDepthSpinBox->setGeometry(QRect(QPoint(10, curr_y),
																 QSize(50, y_height)));
	QSlider *minDepthSlider = new QSlider(this);
	minDepthSlider->setOrientation(Qt::Horizontal);
	minDepthSlider->setMinimum(nMin);
	minDepthSlider->setMaximum(nMax);
	minDepthSlider->setSingleStep(nSingleStep);
	minDepthSlider->setGeometry(QRect(QPoint(65, curr_y),
															QSize(180, y_height)));
	curr_y += y_step;
	connect(m_minDepthSpinBox, SIGNAL(valueChanged(int)), minDepthSlider, SLOT(setValue(int)));
	connect(minDepthSlider, SIGNAL(valueChanged(int)), m_minDepthSpinBox, SLOT(setValue(int)));
	m_minDepthSpinBox->setValue(0);
	m_maxDepthSpinBox = new QSpinBox(this);
	m_maxDepthSpinBox->setMinimum(nMin);
	m_maxDepthSpinBox->setMaximum(nMax);
	m_maxDepthSpinBox->setSingleStep(nSingleStep);
	m_maxDepthSpinBox->setGeometry(QRect(QPoint(10, curr_y),
																 QSize(50, y_height)));
	QSlider *m_maxDepthSlider = new QSlider(this);
	m_maxDepthSlider->setOrientation(Qt::Horizontal);
	m_maxDepthSlider->setMinimum(nMin);
	m_maxDepthSlider->setMaximum(nMax);
	m_maxDepthSlider->setSingleStep(nSingleStep);
	m_maxDepthSlider->setGeometry(QRect(QPoint(65, curr_y),
																QSize(180, y_height)));
	curr_y += y_step;
	connect(m_maxDepthSpinBox, SIGNAL(valueChanged(int)), m_maxDepthSlider, SLOT(setValue(int)));
	connect(m_maxDepthSlider, SIGNAL(valueChanged(int)), m_maxDepthSpinBox, SLOT(setValue(int)));
	m_maxDepthSpinBox->setValue(1500);
	connect(m_startServerButton, SIGNAL(clicked()), this, SLOT(startServer()));
	connect(m_stopServerButton, SIGNAL(clicked()), this, SLOT(stopServer()));
	connect(m_openFileButton, SIGNAL(clicked()), this, SLOT(loadBinFileSlot()));

	QLabel *depthThreshForEdgeLabel = new QLabel(this);
	depthThreshForEdgeLabel->setText("Depth Threshold for Edge (mm): ");
	depthThreshForEdgeLabel->setGeometry(QRect(QPoint(10, curr_y),
																			 QSize(400, y_height)));
	curr_y += y_step;
	nMin = 0;
	nMax = 200;
	nSingleStep = 5;
	m_depthThreshForEdgeSpinBox = new QSpinBox(this);
	m_depthThreshForEdgeSpinBox->setMinimum(nMin);
	m_depthThreshForEdgeSpinBox->setMaximum(nMax);
	m_depthThreshForEdgeSpinBox->setSingleStep(nSingleStep);
	m_depthThreshForEdgeSpinBox->setGeometry(QRect(QPoint(10, curr_y),
																					 QSize(50, y_height)));
	QSlider *depthThreshForEdgeSlider = new QSlider(this);
	depthThreshForEdgeSlider->setOrientation(Qt::Horizontal);
	depthThreshForEdgeSlider->setMinimum(nMin);
	depthThreshForEdgeSlider->setMaximum(nMax);
	depthThreshForEdgeSlider->setSingleStep(nSingleStep);
	depthThreshForEdgeSlider->setGeometry(QRect(QPoint(65, curr_y),
																				QSize(180, y_height)));
	curr_y += y_step;
	connect(m_depthThreshForEdgeSpinBox, SIGNAL(valueChanged(int)), depthThreshForEdgeSlider, SLOT(setValue(int)));
	connect(depthThreshForEdgeSlider, SIGNAL(valueChanged(int)), m_depthThreshForEdgeSpinBox, SLOT(setValue(int)));
	m_depthThreshForEdgeSpinBox->setValue(10);

	QLabel *nodeNumEachFragLabel = new QLabel(this);
	nodeNumEachFragLabel->setText("Number of Node Each Frag: ");
	nodeNumEachFragLabel->setGeometry(QRect(QPoint(10, curr_y),
																		QSize(400, y_height)));
	curr_y += y_step;
	nMin = 8;
	nMax = 512;
	nSingleStep = 8;
	m_nodeNumEachFragSpinBox = new QSpinBox(this);
	m_nodeNumEachFragSpinBox->setMinimum(nMin);
	m_nodeNumEachFragSpinBox->setMaximum(nMax);
	m_nodeNumEachFragSpinBox->setSingleStep(nSingleStep);
	m_nodeNumEachFragSpinBox->setGeometry(QRect(QPoint(10, curr_y),
																				QSize(50, y_height)));
	QSlider *nodeNumEachFragSpinBoxSlider = new QSlider(this);
	nodeNumEachFragSpinBoxSlider->setOrientation(Qt::Horizontal);
	nodeNumEachFragSpinBoxSlider->setMinimum(nMin);
	nodeNumEachFragSpinBoxSlider->setMaximum(nMax);
	nodeNumEachFragSpinBoxSlider->setSingleStep(nSingleStep);
	nodeNumEachFragSpinBoxSlider->setGeometry(QRect(QPoint(65, curr_y),
																						QSize(180, y_height)));
	curr_y += y_step;
	connect(m_nodeNumEachFragSpinBox, SIGNAL(valueChanged(int)), nodeNumEachFragSpinBoxSlider, SLOT(setValue(int)));
	connect(nodeNumEachFragSpinBoxSlider, SIGNAL(valueChanged(int)), m_nodeNumEachFragSpinBox, SLOT(setValue(int)));
	nodeNumEachFragSpinBoxSlider->setValue(16);

	m_useStructureSensorCheckBox->setChecked(true);
	m_doReconstructionCheckBox->setChecked(true);
	m_withNonRigidRegistrationCheckBox->setChecked(true);

	GlobalState::getInstance().m_dataTransTest = m_dataTransTestCheckBox->isChecked();
	GlobalState::getInstance().m_useStructureSensor = m_useStructureSensorCheckBox->isChecked();
	GlobalState::getInstance().m_doReconstruction = m_doReconstructionCheckBox->isChecked();
	GlobalState::getInstance().m_saveToBinFile = m_saveToBinFileCheckBox->isChecked();
	GlobalState::getInstance().m_loadFromBinFile = m_loadFromBinFileCheckBox->isChecked();
	GlobalState::getInstance().m_adjustExtrin = m_adjustExtrinCheckBox->isChecked();
	GlobalState::getInstance().m_debugTracking = m_debugTrackingCheckBox->isChecked();
	GlobalState::getInstance().m_useKNN = m_useKNNCheckBox->isChecked();
	GlobalState::getInstance().m_doNonrigidRegistration = m_withNonRigidRegistrationCheckBox->isChecked();
	GlobalState::getInstance().m_minDepthValue = m_minDepthSpinBox->value();
	GlobalState::getInstance().m_maxDepthValue = m_maxDepthSpinBox->value();
	GlobalState::getInstance().m_depthThresForEdge = m_depthThreshForEdgeSpinBox->value();
	GlobalState::getInstance().m_nodeNumEachFrag = m_nodeNumEachFragSpinBox->value();

	connect(m_dataTransTestCheckBox, SIGNAL(stateChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_useStructureSensorCheckBox, SIGNAL(stateChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_doReconstructionCheckBox, SIGNAL(stateChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_saveToBinFileCheckBox, SIGNAL(stateChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_loadFromBinFileCheckBox, SIGNAL(stateChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_adjustExtrinCheckBox, SIGNAL(stateChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_debugTrackingCheckBox, SIGNAL(stateChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_useKNNCheckBox, SIGNAL(stateChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_withNonRigidRegistrationCheckBox, SIGNAL(stateChanged(int)), this, SLOT(statusChangedSlot(int)));

	connect(m_minDepthSpinBox, SIGNAL(valueChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_maxDepthSpinBox, SIGNAL(valueChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_depthThreshForEdgeSpinBox, SIGNAL(valueChanged(int)), this, SLOT(statusChangedSlot(int)));
	connect(m_nodeNumEachFragSpinBox, SIGNAL(valueChanged(int)), this, SLOT(statusChangedSlot(int)));

	GlobalParameter::SetNodeNumEachFrag(GlobalState::getInstance().m_nodeNumEachFrag);
	if (GlobalState::getInstance().m_doNonrigidRegistration) {
		GlobalParameter::SetSampledVertexNumEachFrag(1024);
	}
	else
	{
		GlobalParameter::SetSampledVertexNumEachFrag(512);
	}
}

ViewWindow::~ViewWindow() {

}

void ViewWindow::setImages(std::vector<cv::Mat> &image_vec) {


}

void ViewWindow::sellectView(int sellected_idx) {

}

void ViewWindow::startServer() {

	float imgScale = 1.0f;
	int depthWidth = 640, depthHeight = 480,
		colorWidth = 648, colorHeight = 484,
		//colorWidth = 640, colorHeight = 480,
		fullColorWidth = 2592, fullColorHeight = 1936;
	double intrinScale = 648 / (double)640;
	float fx = 544.8898 * intrinScale,
		fy = 545.9078 * intrinScale,
		cx = 321.6016 * intrinScale,
		cy = 237.0330 * intrinScale;

	Resolution::getInstance(depthWidth, depthHeight,
													colorWidth, colorHeight,
													fullColorWidth, fullColorHeight, imgScale);
	Intrinsics::getInstance(fx, fy, cx, cy, imgScale);

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

	char keyFramePath[256], binPath[256];
	//if (m_saveToBinFile) {
		char name[256], rootPath[256];
		std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
		time_t rawtime = std::chrono::system_clock::to_time_t(now);
		struct tm* timeinfo;
		timeinfo = localtime(&rawtime);
		strftime(name, sizeof(name),
						 "StructureSensorData_%Y%m%d_%H%M%S", timeinfo);
		sprintf(rootPath, "C:\\xjm\\snapshot\\%s\\", name);
		mkdir(rootPath);
		std::cout << "data path: " << rootPath << std::endl;
		sprintf(binPath, "%s\\%s.bin", rootPath, name);
		sprintf(keyFramePath, "%s\\keyframes\\", rootPath);
		mkdir(keyFramePath);
	//}

#if 0
	CirularQueue* cirularQueue = new CirularQueue(
		Resolution::getInstance().width() * Resolution::getInstance().height() * 3.5, 150);

	RemoteStructureSeneor *capture = new RemoteStructureSeneor(cirularQueue, binPath);
	//remoteStructureSeneor.start();

	TcpServer server(capture, keyFramePath);
#else
	if (GlobalState::getInstance().m_loadFromBinFile)
	{
#if 0
		m_captureFile = new RemoteStructureSeneorFromFile(
			"C:\\xjm\\snapshot\\StructureSensorData_20181214_205549_1\\StructureSensorData_20181214_205549.bin");
#endif
		m_captureFile = new RemoteStructureSeneorFromFile(m_binFilePath.c_str());
		m_objReconThread = new ObjReconThread(m_captureFile);
		m_objReconThread->setIntrinExtrin(extrinR, extrinT, intrinColor, intrinDepth);
		m_objReconThread->start();
	}
	else
	{
		CirularQueue* cirularQueue = new CirularQueue(
			Resolution::getInstance().width() * Resolution::getInstance().height() * 3.5, 200);

		m_capture = new RemoteStructureSeneor(cirularQueue, binPath);

		m_server = new TcpServer(m_capture, keyFramePath);
		m_objReconThread = new ObjReconThread(m_capture);
		m_objReconThread->setIntrinExtrin(extrinR, extrinT, intrinColor, intrinDepth);
		m_objReconThread->start();
	}
#endif
}

void ViewWindow::stopServer() {

	m_objReconThread->m_isFinish = true;
	Sleep(1000);

	m_objReconThread->quit();
	m_objReconThread->terminate();
	m_objReconThread->destroyed();

	if (m_objReconThread) {
		delete m_objReconThread;
	}
	if (m_captureFile) {
		delete m_captureFile;
	}
	if (m_capture) {
		delete m_capture;
	}
	if (cirularQueue) {
		delete cirularQueue;
	}
	if (m_server) {
		delete m_server;
	}
	std::cout << "finish thread success" << std::endl;
}

void ViewWindow::statusChangedSlot(int state) {

	GlobalState::getInstance().m_dataTransTest = m_dataTransTestCheckBox->isChecked();
	GlobalState::getInstance().m_useStructureSensor = m_useStructureSensorCheckBox->isChecked();
	GlobalState::getInstance().m_doReconstruction = m_doReconstructionCheckBox->isChecked();
	GlobalState::getInstance().m_saveToBinFile = m_saveToBinFileCheckBox->isChecked();
	GlobalState::getInstance().m_loadFromBinFile = m_loadFromBinFileCheckBox->isChecked();
	GlobalState::getInstance().m_adjustExtrin = m_adjustExtrinCheckBox->isChecked();
	GlobalState::getInstance().m_debugTracking = m_debugTrackingCheckBox->isChecked();
	GlobalState::getInstance().m_useKNN = m_useKNNCheckBox->isChecked();
	GlobalState::getInstance().m_doNonrigidRegistration = m_withNonRigidRegistrationCheckBox->isChecked();
	GlobalState::getInstance().m_minDepthValue = m_minDepthSpinBox->value();
	GlobalState::getInstance().m_maxDepthValue = m_maxDepthSpinBox->value();
	GlobalState::getInstance().m_depthThresForEdge = m_depthThreshForEdgeSpinBox->value();
	GlobalState::getInstance().m_nodeNumEachFrag = m_nodeNumEachFragSpinBox->value();

	GlobalParameter::SetNodeNumEachFrag(GlobalState::getInstance().m_nodeNumEachFrag);
	if (GlobalState::getInstance().m_doNonrigidRegistration) {
		GlobalParameter::SetSampledVertexNumEachFrag(1024);
	}
	else
	{
		GlobalParameter::SetSampledVertexNumEachFrag(256);
	}
}

void ViewWindow::loadBinFileSlot() {

	QString fileName = QFileDialog::getOpenFileName(this, tr("import bin file"), ".", "Bin File(*.bin)", 0);
	m_binFilePath = fileName.toStdString();
}