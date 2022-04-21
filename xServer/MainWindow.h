#pragma once

#include <QDesktopWidget>
#include <QApplication>
#include <QRect>
#include <map>
#include <string>
#include <iostream>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QStackedLayout>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QSlider>
#include <QFileDialog>
#include <QSpinBox>
#include <QPlainTextEdit>
#include <QMouseEvent>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

class ViewWindow;
class ObjReconThread;
class RemoteStructureSeneorFromFile;
class RemoteStructureSeneor;
class CirularQueue;
class TcpServer;

class ViewWindow : public QWidget {
	Q_OBJECT

public:
	ViewWindow(QWidget *parent = NULL);
	~ViewWindow();

	void setImages(std::vector<cv::Mat> &image_vec_);
	void sellectView(int sellected_idx);

public slots:
	void startServer();
	void stopServer();
	void statusChangedSlot(int state);
	void loadBinFileSlot();

signals:
	void sellectViewSignal(int sellected_idx);

private:
	QWidget *mainWidget_;
	QScrollArea *mainScrollArea_;
	QVBoxLayout *mainLayout_;
	QStackedLayout *stackLayout_;

	QCheckBox *m_dataTransTestCheckBox;
	QCheckBox *m_useStructureSensorCheckBox;
	QCheckBox *m_doReconstructionCheckBox;
	QCheckBox *m_saveToBinFileCheckBox;
	QCheckBox *m_loadFromBinFileCheckBox;
	QCheckBox *m_adjustExtrinCheckBox;
	QCheckBox *m_debugTrackingCheckBox;
	QCheckBox *m_useKNNCheckBox;
	QCheckBox *m_withNonRigidRegistrationCheckBox;
	QSpinBox *m_minDepthSpinBox;
	QSpinBox *m_maxDepthSpinBox;
	QSpinBox *m_depthThreshForEdgeSpinBox;
	QSpinBox *m_nodeNumEachFragSpinBox;
	std::string m_binFilePath;

	bool m_saveToBinFile = NULL;
	ObjReconThread *m_objReconThread = NULL;
	RemoteStructureSeneorFromFile *m_captureFile = NULL;
	RemoteStructureSeneor *m_capture = NULL;
	CirularQueue* cirularQueue = NULL;
	TcpServer *m_server = NULL;

	int image_widget_width_, image_widget_height_;
public:
	int sellected_idx_ = -1;
};
