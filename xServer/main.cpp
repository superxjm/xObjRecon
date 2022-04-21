#include "ObjRecon.h"

#include <opencv2/opencv.hpp>
#include <QtNetwork\QtNetwork>
#include <QtWidgets\QApplication>
#include <QtNetwork\QTcpServer>
#include <condition_variable>
#include <direct.h> 
#include <Eigen/Eigen>
#include <iostream>
#include <pangolin/pangolin.h>

#include "Helpers/xUtils.h"
#include "Network/SocketServer.h"
#include "MainWindow.h"
#include "GlobalWindowLayout.h"

#define TEST_DATA_TRANSFORM 1

void SavePly(cv::Mat &depthImg, cv::Mat &colorImg, Gravity &gravity, const char* fileDir)
{
	float fx = Intrinsics::getInstance().fx();
	float fy = Intrinsics::getInstance().fy();
	float cx = Intrinsics::getInstance().cx();
	float cy = Intrinsics::getInstance().cy();
	std::vector<float> vertices;
	std::vector<uchar> colors;
	float x, y, z;
	float middleX = 0.0f, middleY = 0.0f, middleZ = 0.0f;
	for (int r = 0; r < depthImg.rows; ++r)
	{
		for (int c = 0; c < depthImg.cols; ++c)
		{
			if (depthImg.at<ushort>(r, c) > 0)
			{
				z = depthImg.at<ushort>(r, c) / 1000.0f;
				x = (c - cx) / fx * z;
				y = (r - cy) / fy * z;
				middleX += x;
				middleY += y;
				middleZ += z;
				vertices.push_back(x);
				vertices.push_back(y);
				vertices.push_back(z);
				colors.push_back(colorImg.at<cv::Vec3b>(r, c)[2]);
				colors.push_back(colorImg.at<cv::Vec3b>(r, c)[1]);
				colors.push_back(colorImg.at<cv::Vec3b>(r, c)[0]);
			}
		}
	}
	int verticeNum = vertices.size() / 3;
	middleX /= verticeNum;
	middleY /= verticeNum;
	middleZ /= verticeNum;
#if 0
	Eigen::Matrix3d RImuCam;
	RImuCam << 0, -1, 0, -1, 0, 0, 0, 0, -1;
	Eigen::Vector3d gravityCam = RImuCam * gravity;
	for (int i = -200; i < 200; ++i)
	{
		x = middleX + gravityCam.x() * i * 0.001;
		y = middleY + gravityCam.y() * i * 0.001;
		z = middleZ + gravityCam.z() * i * 0.001;
		vertices.push_back(x);
		vertices.push_back(y);
		vertices.push_back(z);
		colors.push_back(255);
		colors.push_back(0);
		colors.push_back(0);
	}
#endif
	std::ofstream fs;
	fs.open(fileDir);

	fs << "ply";
	fs << "\nformat " << "ascii" << " 1.0";
	fs << "\nelement vertex " << vertices.size() / 3;
	fs << "\nproperty float x"
		"\nproperty float y"
		"\nproperty float z";
	fs << "\nproperty uchar red"
		"\nproperty uchar green"
		"\nproperty uchar blue";
	fs << "\nend_header\n";

	for (unsigned int i = 0; i < vertices.size() / 3; i++)
	{
		fs << vertices[3 * i] << " " << vertices[3 * i + 1] << " " << vertices[3 * i + 2] << " "
			<< (int)colors[3 * i] << " " << (int)colors[3 * i + 1] << " " << (int)colors[3 * i + 2] << " "
			<< std::endl;
	}

	fs.close();
}

void ConvertIrToRGB(cv::Mat& coloredIrImg, cv::Mat& irImg)
{
	for (size_t i = 0; i < irImg.rows * irImg.cols; ++i)
	{
		ushort data = *((ushort *)(irImg.data) + i);

		// Use the upper byte of the linearized shift value to choose a base color
		// Base colors range from: (closest) White, Red, Orange, Yellow, Green, Cyan, Blue, Black (farthest)
		int lowerByte = (data & 0xff);

		// Use the lower byte to scale between the base colors
		int upperByte = (data >> 8);

		switch (upperByte)
		{
		case 0:
			coloredIrImg.data[3 * i + 0] = 255;
			coloredIrImg.data[3 * i + 1] = 255 - lowerByte;
			coloredIrImg.data[3 * i + 2] = 255 - lowerByte;
			break;
		case 1:
			coloredIrImg.data[3 * i + 0] = 255;
			coloredIrImg.data[3 * i + 1] = lowerByte;
			coloredIrImg.data[3 * i + 2] = 0;
			break;
		case 2:
			coloredIrImg.data[3 * i + 0] = 255 - lowerByte;
			coloredIrImg.data[3 * i + 1] = 255;
			coloredIrImg.data[3 * i + 2] = 0;
			break;
		case 3:
			coloredIrImg.data[3 * i + 0] = 0;
			coloredIrImg.data[3 * i + 1] = 255;
			coloredIrImg.data[3 * i + 2] = lowerByte;
			break;
		case 4:
			coloredIrImg.data[3 * i + 0] = 0;
			coloredIrImg.data[3 * i + 1] = 255 - lowerByte;
			coloredIrImg.data[3 * i + 2] = 255;
			break;
		case 5:
			coloredIrImg.data[3 * i + 0] = 0;
			coloredIrImg.data[3 * i + 1] = 0;
			coloredIrImg.data[3 * i + 2] = 255 - lowerByte;
			break;
		default:
			coloredIrImg.data[3 * i + 0] = 0;
			coloredIrImg.data[3 * i + 1] = 0;
			coloredIrImg.data[3 * i + 2] = 0;
			break;
		}
	}
}

std::mutex dataValidMtx, queueMtx;
std::condition_variable dataValidCondVar, dataInvalidCondVar, emptyCondVar, fullCondVar;

void setImageData(unsigned char * imageArray, int size) {
	for (int i = 0; i < size; i++) {
		imageArray[i] = (unsigned char)(rand() / (RAND_MAX / 255.0));
	}
}

#if 0
int main(/*int argc, char* argv[]*/)
{
	// Create OpenGL window in single line
	pangolin::CreateWindowAndBind("Main", 640, 480);

	xCheckGlDieOnError();

	// 3D Mouse handler requires depth testing to be enabled
	glEnable(GL_DEPTH_TEST);

	// Issue specific OpenGl we might need
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Define Camera Render Object (for view / scene browsing)
	pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000);
	pangolin::OpenGlRenderState s_cam(proj, pangolin::ModelViewLookAt(1, 0.5, -2, 0, 0, 0, pangolin::AxisY));
	pangolin::OpenGlRenderState s_cam2(proj, pangolin::ModelViewLookAt(0, 0, -2, 0, 0, 0, pangolin::AxisY));

	// Add named OpenGL viewport to window and provide 3D Handler
	pangolin::View& d_cam1 = pangolin::Display("cam1")
		.SetAspect(640.0f / 480.0f)
		.SetHandler(new pangolin::Handler3D(s_cam));

	pangolin::View& d_cam2 = pangolin::Display("cam2")
		.SetAspect(640.0f / 480.0f)
		.SetHandler(new pangolin::Handler3D(s_cam2));

	pangolin::View& d_cam3 = pangolin::Display("cam3")
		.SetAspect(640.0f / 480.0f)
		.SetHandler(new pangolin::Handler3D(s_cam));

	pangolin::View& d_cam4 = pangolin::Display("cam4")
		.SetAspect(640.0f / 480.0f)
		.SetHandler(new pangolin::Handler3D(s_cam2));

	pangolin::View& d_img1 = pangolin::Display("img1")
		.SetAspect(640.0f / 480.0f);

	pangolin::View& d_img2 = pangolin::Display("img2")
		.SetAspect(640.0f / 480.0f);

	// LayoutEqual is an EXPERIMENTAL feature - it requires that all sub-displays
	// share the same aspect ratio, placing them in a raster fasion in the
	// viewport so as to maximise display size.
	pangolin::Display("multi")
		.SetBounds(0.0, 1.0, 0.0, 1.0)
		.SetLayout(pangolin::LayoutEqual)
		.AddDisplay(d_cam1)
		.AddDisplay(d_img1)
		.AddDisplay(d_cam2)
		.AddDisplay(d_img2)
		.AddDisplay(d_cam3)
		.AddDisplay(d_cam4);

	xCheckGlDieOnError();

	const int width = 64;
	const int height = 48;
	unsigned char* imageArray = new unsigned char[3 * width*height];
	pangolin::GlTexture imageTexture(width, height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

	// Default hooks for exiting (Esc) and fullscreen (tab).
	while (!pangolin::ShouldQuit())
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Generate random image and place in texture memory for display
		setImageData(imageArray, 3 * width*height);
		imageTexture.Upload(imageArray, GL_RGB, GL_UNSIGNED_BYTE);

		glColor3f(1.0, 1.0, 1.0);

		d_cam1.Activate(s_cam);
		pangolin::glDrawColouredCube();

		d_cam2.Activate(s_cam2);
		pangolin::glDrawColouredCube();

		d_cam3.Activate(s_cam);
		pangolin::glDrawColouredCube();

		d_cam4.Activate(s_cam2);
		pangolin::glDrawColouredCube();

		d_img1.Activate();
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		imageTexture.RenderToViewport();

		d_img2.Activate();
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		imageTexture.RenderToViewport();

		// Swap frames and Process Events
		pangolin::FinishFrame();
	}

	delete[] imageArray;

	return 0;
}
#endif

#if 1
int main(int argc, char *argv[])
{	
#if 0
	cv::Mat mask_image = cv::imread("D:\\xjm\\dataset\\yundongxie\\StructureSensorData_20180517_142525\\before_opt\\000010.prune.png", 0);
	cv::Mat rgb_image = cv::imread("D:\\xjm\\dataset\\yundongxie\\StructureSensorData_20180517_142525\\before_opt\\000010_key_frame.png");
	cv::imshow("mask_image", mask_image);
	cv::waitKey(0);
	for (int row = 0; row < mask_image.rows; ++row) {
		for (int col = 0; col < mask_image.cols; ++col) {
			if (mask_image.at<uchar>(row, col) > 0) {
				rgb_image.at<cv::Vec3b>(row, col)[0] = 0;
				rgb_image.at<cv::Vec3b>(row, col)[1] = 0;
				rgb_image.at<cv::Vec3b>(row, col)[2] = 255;
			}
		}
	}
	cv::imshow("rgb_image", rgb_image);
	cv::waitKey(0);
	cv::imwrite("C:\\xjm\\snapshot\\before_opt\\object_extraction.png", rgb_image);
	std::exit(0);
#endif
#if 1
	QApplication app(argc, argv);

	GlobalWindowLayout::getInstance();
	ViewWindow w;
	w.show();

	app.exec();
#endif

#if 0
	int fragNum = 41;
	std::vector<Eigen::Matrix4f> frameCameraPoseVec(fragNum);
	std::ifstream fs;
	fs.open("C:\\xjm\\snapshot\\before_opt\\camera_pose.txt", std::ofstream::binary);
	int fragIdx;
	Eigen::Matrix4f cameraPose;
	while (fs.read((char *)&fragIdx, sizeof(int))) {
		std::cout << "frag idx: " << fragIdx << std::endl;
		fs.read((char *)cameraPose.data(), sizeof(Eigen::Matrix4f));
		std::cout << cameraPose << std::endl;
	}
	fs.close();
	std::exit(0);

	for (int i = 0; i < fragNum; ++i) {
		std::cout << frameCameraPoseVec[i] << std::endl;
	}

	std::exit(0);
#endif

#if 0
	std::vector<int> pngCompressionParams;
	pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	pngCompressionParams.push_back(0);
	char path[256];
	char path2[256];
	for (int i = 90; i >= 1; --i)
	{
		sprintf(path, "D:\\xjm\\data_for_video\\rendered_results_maya\\results\\result_for_ppt\\xiaoma_result_move_to_center\\rendered_with_texture\\rendered.%d.png", i);
		sprintf(path2, "D:\\xjm\\data_for_video\\rendered_results_maya\\results\\result_for_ppt\\xiaoma_result_move_to_center\\rendered_with_texture_rgb\\rendered.%d.png", 91 - i);
		std::cout << path << std::endl;
		cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
		cv::Mat rgb = cv::Mat(img.rows, img.cols, CV_8UC3);
		std::cout << img.rows << " : " << img.cols << std::endl;
		std::cout << img.type() << std::endl;
		for (int row = 0; row < img.rows; ++row)
		{
			//std::cout << row << " : " << 0 << std::endl;
			for (int col = 0; col < img.cols; ++col)
			{
				//std::cout << (int)img.at<cv::Vec4b>(row, col)[3] << std::endl;
				//img.at<cv::Vec4b>(row, col)[0] = 0;
				rgb.at<cv::Vec3b>(row, col)[0] = img.at<cv::Vec4b>(row, col)[0];
				rgb.at<cv::Vec3b>(row, col)[1] = img.at<cv::Vec4b>(row, col)[1];
				rgb.at<cv::Vec3b>(row, col)[2] = img.at<cv::Vec4b>(row, col)[2];
				if (img.at<cv::Vec4b>(row, col)[3] == 0)
				{
					rgb.at<cv::Vec3b>(row, col)[0] = 255;
					rgb.at<cv::Vec3b>(row, col)[1] = 255;
					rgb.at<cv::Vec3b>(row, col)[2] = 255;
				}
			}
		}
		cv::imshow("img", rgb);
		cv::waitKey(1);
		cv::imwrite(path2, rgb, pngCompressionParams);
	}
	for (int i = 91; i <= 180; ++i)
	{
		sprintf(path, "D:\\xjm\\data_for_video\\rendered_results_maya\\results\\result_for_ppt\\xiaoma_result_move_to_center\\rendered_with_texture\\rendered.%d.png", i);
		sprintf(path2, "D:\\xjm\\data_for_video\\rendered_results_maya\\results\\result_for_ppt\\xiaoma_result_move_to_center\\rendered_with_texture_rgb\\rendered.%d.png", i);
		std::cout << path << std::endl;
		cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
		cv::Mat rgb = cv::Mat(img.rows, img.cols, CV_8UC3);
		std::cout << img.rows << " : " << img.cols << std::endl;
		std::cout << img.type() << std::endl;
		for (int row = 0; row < img.rows; ++row)
		{
			//std::cout << row << " : " << 0 << std::endl;
			for (int col = 0; col < img.cols; ++col)
			{
				//std::cout << (int)img.at<cv::Vec4b>(row, col)[3] << std::endl;
				//img.at<cv::Vec4b>(row, col)[0] = 0;
				rgb.at<cv::Vec3b>(row, col)[0] = img.at<cv::Vec4b>(row, col)[0];
				rgb.at<cv::Vec3b>(row, col)[1] = img.at<cv::Vec4b>(row, col)[1];
				rgb.at<cv::Vec3b>(row, col)[2] = img.at<cv::Vec4b>(row, col)[2];
				if (img.at<cv::Vec4b>(row, col)[3] == 0)
				{
					rgb.at<cv::Vec3b>(row, col)[0] = 255;
					rgb.at<cv::Vec3b>(row, col)[1] = 255;
					rgb.at<cv::Vec3b>(row, col)[2] = 255;
				}
			}
		}
		cv::imshow("img", rgb);
		cv::waitKey(1);
		cv::imwrite(path2, rgb, pngCompressionParams);
	}
	std::exit(0);
#endif
#if 0
#define USE_STRUCTURE_SENSOR 1
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
	RemoteStructureSeneorFromFile capture(
		"C:\\xjm\\snapshot\\StructureSensorData_20180530_200131\\StructureSensorData_20180530_200131.bin");
	float intrinColor[4] = { fx, fy, cx, cy };
	float intrinDepth[4] = { 574.0135, 575.5523, 314.5388, 242.4793 };
	float extrinT[3] = { -41.1776, -4.3666, -34.8012 };
#if 0
	float extrinR[9] = {
		1.0000, -0.0040, -0.0029,
		0.0040, 0.9999, 0.0132,
		0.0028, -0.0132, 0.9999
	};
#endif
#if 1
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
	objReconThread.start();
	while (true)
	{
		Sleep(1000);
	}

	return 0;
#endif
#if 0
	char name[256], rootPath[256], binPath[256], keyFramePath[256];
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

#if 1
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

	//cv::namedWindow("InputColor", CV_WINDOW_AUTOSIZE);
	//cv::namedWindow("InputDepth", CV_WINDOW_AUTOSIZE);

	QApplication app(argc, argv);

	CirularQueue* cirularQueue = new CirularQueue(
		Resolution::getInstance().width() * Resolution::getInstance().height() * 3.5, 150);
#if 1
	RemoteStructureSeneor *capture = new RemoteStructureSeneor(cirularQueue, binPath);
	//remoteStructureSeneor.start();

	TcpServer server(capture, keyFramePath);
#else
	RemoteStructureSeneorFromFile *capture = new RemoteStructureSeneorFromFile(
		"C:\\xjm\\snapshot\\StructureSensorData_20181214_205549_1\\StructureSensorData_20181214_205549.bin");
#endif

#if 0
	//cv::namedWindow("input_depth", CV_WINDOW_AUTOSIZE);
	//cv::namedWindow("input_color", CV_WINDOW_AUTOSIZE);

	ObjReconThread objReconThread(capture);
	//objReconThread.setSensor(remoteStructureSeneor);
	objReconThread.setIntrinExtrin(extrinR, extrinT, intrinColor, intrinDepth);
	objReconThread.start();
#endif

	GlobalWindowLayout::getInstance();
	ViewWindow w;
	w.show();

	app.exec();
#endif
#endif

#if 0
	//xCapture capture(imgScale);
	//capture.start();	
	//xOniCapture capture(width, height, 30, width, height, 30);
	RemoteStructureSeneorFromFile* capture = new RemoteStructureSeneorFromFile(
		"D:\\xjm\\snapshot\\StructureSensorData_20180423_224856.bin");
	//RemoteStructureSeneorFromFile capture("D:\\xjm\\snapshot\\StructureSensorData_20180409_160031.bin");	
	float intrinColor[4] = { 544.8898, 545.9078, 321.6016, 237.0330 };
	float intrinDepth[4] = { 574.0135, 575.5523, 314.5388, 242.4793 };
	float extrinT[3] = { -41.1776, -4.3666, -34.8012 };
	float extrinR[9] = {
		1.0000, -0.0040, -0.0029,
		0.0040, 0.9999, 0.0132,
		0.0028, -0.0132, 0.9999
	};
	capture->setIntrinExtrin(extrinR, extrinT, intrinColor, intrinDepth);

	ObjReconThread objReconThread;
	objReconThread.setSeneor(capture);
	objReconThread.start();
	while (true)
	{
		Sleep(1000);
	}
#endif

#if 0
	QApplication app(argc, argv);

	CirularQueue* cirularQueue = new CirularQueue(
		Resolution::getInstance().width() * Resolution::getInstance().height() * 3, 120);
	RemoteStructureSeneor* remoteStructureSeneor = new RemoteStructureSeneor(cirularQueue);
	float intrinColor[4] = { 544.8898, 545.9078, 321.6016, 237.0330 };
	float intrinDepth[4] = { 574.0135, 575.5523, 314.5388, 242.4793 };
	float extrinT[3] = { -41.1776, -4.3666, -34.8012 };
	float extrinR[9] = {
		1.0000, -0.0040, -0.0029,
		0.0040, 0.9999, 0.0132,
		0.0028, -0.0132, 0.9999
	};
	remoteStructureSeneor->setIntrinExtrin(extrinR, extrinT, intrinColor, intrinDepth);
	remoteStructureSeneor->start();
	TcpServer server(remoteStructureSeneor);

	app.exec();
#endif
#if 0
	RemoteStructureSeneorFromFileForCelab *remoteStructureSeneorFromFile = new RemoteStructureSeneorFromFile("D:\\xjm\\snapshot\\StructureSensorData_celab_20180330_210926.bin");
	cv::Mat depthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_16UC1),
		irImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_16UC1),
		coloredIrImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3),
		colorImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3);
	ImuMeasurements imuMeasurements;
	Gravity gravity;
	int mycnt = 0;
	while (true)
	{
		//remoteStructureSeneorFromFile->nextDepthColor(depthImg, colorImg, imuMeasurements, gravity);
		remoteStructureSeneorFromFile->nextIrColor(irImg, colorImg, imuMeasurements, gravity);

		ConvertIrToRGB(coloredIrImg, irImg);

		std::cout << "gravity: " << gravity.x() << " : " << gravity.y() << " : " << gravity.z() << std::endl;
		//SavePly(depthImg, colorImg, gravity, "D:\\xjm\\snapshot\\model.ply");
		cv::imshow("InputColor", colorImg);
		cv::imshow("InputDepth", depthImg * 30);
		cv::imshow("InputIr", irImg * 60);
		cv::waitKey(1);

#if 1
		std::vector<int> pngCompressionParams;
		pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
		pngCompressionParams.push_back(0);
		char mydir[256];
		//sprintf(mydir, "D:\\xjm\\snapshot\\7\\frame-%06d.depth.png", mycnt);
		//cv::imwrite(mydir, depthImg, pngCompressionParams);
		sprintf(mydir, "D:\\xjm\\snapshot\\9\\frame-%06d.ir.png", mycnt);
		cv::imwrite(mydir, irImg, pngCompressionParams);
		sprintf(mydir, "D:\\xjm\\snapshot\\9\\frame-%06d.color.png", mycnt);
		cv::imwrite(mydir, colorImg, pngCompressionParams);
		mycnt++;
#endif
		//std::exit(0);
	}
#endif

#if 0
	int64_t timeStamp = 0;
	int fragIdx = 0;
	xGUI *pGui = new xGUI(dispScale);

	cv::namedWindow("input_depth", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("input_color", CV_WINDOW_AUTOSIZE);

	//xCapture capture(imgScale);
	//xOniCapture capture(width, height, 30, width, height, 30);
	//capture.start();
	xRemoteStructureCapture capture(width, height, 30, width, height, 30);
	capture.startListening();

	cv::Mat depthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_16UC1),
		colorImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3);

#if 0
	while (!pangolin::ShouldQuit())
	{
		if (pangolin::Pushed(*pGui->m_saveButton))
		{

		}

		if (!pGui->m_pauseButton->Get() || pangolin::Pushed(*pGui->m_stepButton))
		{
#if 0
			// nextº¯ÊýÊÇ×èÈûµÄ
			const bool valid = capture.next(depthImg, colorImg);
			if (!valid)
			{
				std::cout << "invalid data" << std::endl;
				std::exit(0);
			}
#endif
		}
		Sleep(30);

		pangolin::FinishFrame();
		glFinish();
	}
#endif
	while (true)
	{
	}
#endif

	return 0;
}
#endif

#if 0
std::vector<Eigen::Matrix4f> originalCameraPoses;
std::vector<Eigen::Matrix4f> originalFragmentCameraPoses;

void RenderScene(xSurfelFusion *pFusion, pangolin::OpenGlRenderState &cameraState, int timeStamp,
								 Box &objectBox, std::vector<Box> &boxVec, bool objectDetected, cv::Mat &colorImg, cv::Mat &depthImg)
{
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
	}
	else
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

void SaveModel(int fragNum, xDeformation *pDeform, float imgScale)
{
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
