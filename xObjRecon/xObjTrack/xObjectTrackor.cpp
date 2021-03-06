#include "stdafx.h"

#include "xObjectTrackor.h"

#include <opencv2/opencv.hpp>
//#include <Eigen/Eigenvalues> 

#include "Helpers/xUtils.h"
#include "Helpers/UtilsMath.h"
#include "Helpers/InnorealTimer.hpp"
//#include "xSurfelFusion/Cuda/cudafuncs.cuh"
#include "xObjTrack/Cuda/xObjTrackCudaFuncs.cuh"

//#define VISIALIZTION

xObjectTrackor::xObjectTrackor(cv::cuda::GpuMat& dFilteredDepthImg32F,
															 cv::cuda::GpuMat& dRenderedDepthImg32F,
															 cv::Mat& colorImg,
															 cv::cuda::GpuMat& dGrayImg)
	: m_gmsMatcher(),
	m_edgeGen(),
	m_dFilteredDepthImg32F(dFilteredDepthImg32F),
	m_dRenderedDepthImg32F(dRenderedDepthImg32F),
	m_colorImg(colorImg),
	m_dGrayImg(dGrayImg)
{
	m_catImgVis = cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width() * 2, CV_8UC3);
	m_ccImgVis = cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3);

	m_depthDevice.create(Resolution::getInstance().height(), Resolution::getInstance().width());
	m_vmapDevice.create(Resolution::getInstance().height() * 3, Resolution::getInstance().width());
	m_nmapDevice.create(Resolution::getInstance().height() * 3, Resolution::getInstance().width());

	m_dVMap = cv::cuda::GpuMat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_32FC3);

	m_vmapVec.resize(Resolution::getInstance().numPixels() * 3);
	m_nmapVec.resize(Resolution::getInstance().numPixels() * 3);

	m_xs.resize(Resolution::getInstance().numPixels());
	m_ys.resize(Resolution::getInstance().numPixels());
	m_zs.resize(Resolution::getInstance().numPixels());

	//m_kernelEllipse = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));

	checkCudaErrors(cudaMalloc((void **)&m_dResultBuf, (sizeof(float) * 2 + sizeof(int)) * 10));
	m_resultBuf = (char*)malloc((sizeof(float) * 2 + sizeof(int)) * 10);
	checkCudaErrors(cudaMalloc((void **)&m_dPlaneVertexBuf, sizeof(float3) * Resolution::getInstance().numPixels()));
	checkCudaErrors(cudaMalloc((void **)&m_dIndex, sizeof(int)));

	m_dPruneMat = cv::cuda::GpuMat(Resolution::getInstance().height(),
																 Resolution::getInstance().width(),
																 CV_8UC1);
	m_pruneMat = cv::Mat(Resolution::getInstance().height(),
											 Resolution::getInstance().width(),
											 CV_8UC1);
	m_dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
}

xObjectTrackor::~xObjectTrackor()
{
	cudaFree(m_dResultBuf);
	cudaFree(m_dPlaneVertexBuf);
	cudaFree(m_dIndex);
	free(m_resultBuf);
}

void xObjectTrackor::detect(Box &objectBox, Box &centerBox, Gravity gravity)
{
	const float magThreshold = GlobalState::getInstance().m_depthThresForEdge / 1000.0f,
		depthThreshold = GlobalState::getInstance().m_maxDepthValue / 1000.0f;
#if 0
	cv::Mat renderedDepthImg32F;
	m_dRenderedDepthImg32F.download(renderedDepthImg32F);
	cv::imshow("renderedDepthImg32F", renderedDepthImg32F * 30);
	cv::waitKey(1);
#endif

#if 1
	mapToGravityCoordinateForRenderedDepth(gravity, depthThreshold);

	float minDepth = 0.0f, maxDepth = 1.0e24, meanDepth;
	cv::Rect validBox = cv::Rect(centerBox.m_left, centerBox.m_top, centerBox.m_right - centerBox.m_left, centerBox.m_bottom - centerBox.m_top)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
	cv::cuda::GpuMat& dVMapRoi = m_dVMap(validBox);
	CalcMinMeanDepth(minDepth, meanDepth, dVMapRoi, depthThreshold, m_dResultBuf, m_resultBuf, true);
	maxDepth = meanDepth + 1.0 * (meanDepth - minDepth);

	//std::cout << "minDepth: " << minDepth << std::endl;
	//std::cout << "maxDepth: " << maxDepth << std::endl;

#if 0
	cv::Mat renderedDepthImg32F2;
	m_dRenderedDepthImg32F.download(renderedDepthImg32F2);
	for (int rr = 0; rr < renderedDepthImg32F2.rows; ++rr)
	{
		for (int cc = 0; cc < renderedDepthImg32F2.cols; ++cc)
		{
			if (renderedDepthImg32F2.at<float>(rr, cc) <= minDepth || renderedDepthImg32F2.at<float>(rr, cc) >= maxDepth)
				renderedDepthImg32F2.at<float>(rr, cc) = 0.0f;
		}
	}
	cv::imshow("renderedDepthImg32F2", renderedDepthImg32F2 / 1.5);
	cv::waitKey(1);
#endif

	m_edgeGen.calcEdge(m_dRenderedDepthImg32F, m_dVMap, centerBox, minDepth, maxDepth, magThreshold, true);
	//m_edgeGen.calcEdge(m_dRenderedDepthImg32F, centerBox, true);
	m_edgeGen.getBoxes(m_colorImg, m_candidateBoxVec);
	m_edgeGen.getObjectBox(objectBox, m_candidateBoxVec);
#endif
}

std::vector<Box>& xObjectTrackor::getCandidatedBoxes()
{
	return m_candidateBoxVec;
}

static inline void Rodrigues(float3 *mat, float3 axis, double s, double c)
{
	double rx = axis.x;
	double ry = axis.y;
	double rz = axis.z;
	const double I[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

	double c1 = 1.0 - c;
	double rrt[] = { rx*rx, rx*ry, rx*rz, rx*ry, ry*ry, ry*rz, rx*rz, ry*rz, rz*rz };
	double _r_x_[] = { 0, -rz, ry, rz, 0, -rx, -ry, rx, 0 };
	double R[9];
	for (int k = 0; k < 9; k++)
	{
		R[k] = c*I[k] + c1*rrt[k] + s*_r_x_[k];
	}

	mat[0].x = R[0];
	mat[1].x = R[1];
	mat[2].x = R[2];

	mat[0].y = R[3];
	mat[1].y = R[4];
	mat[2].y = R[5];

	mat[0].z = R[6];
	mat[1].z = R[7];
	mat[2].z = R[8];
}
void xObjectTrackor::mapToGravityCoordinate(Gravity& gravity, float depthThreshold)
{
	float3 gravityCam = normalize(make_float3(gravity.x, gravity.y, gravity.z)), gravityW = { 0.0, 0.0, 1.0 };
	float3 axis = cross(gravityCam, gravityW);
	float theta = asin(norm(axis)), sintheta = sin(theta), costheta = cos(theta);
	float3 RWCam[3];
	Rodrigues(RWCam, normalize(axis), sintheta, costheta);
	//timer.TimeEnd();
#if 0
	std::cout << "time1: " << timer.TimeGap_in_ms() << std::endl;
	std::cout << "theta: " << theta / 3.1415 * 180 << std::endl;
	std::cout << "gravity: " << gravity.x << " : " << gravity.y << " : " << gravity.z << std::endl;
	std::cout << "gravityCam: " << gravityCam.x << " : " << gravityCam.y << " : " << gravityCam.z << std::endl;
	std::cout << "gravityW: " << gravityW.x << " : " << gravityW.y << " : " << gravityW.z << std::endl;
	float3 gravityW2 = RWCam[0] * gravityCam.x + RWCam[1] * gravityCam.y + RWCam[2] * gravityCam.z;
	std::cout << "gravityW2: " << gravityW2.x << " : " << gravityW2.y << " : " << gravityW2.z << std::endl;
#endif

	CreateVMap(m_dVMap, m_dFilteredDepthImg32F,
						 Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(),
						 Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
						 depthThreshold, RWCam[0], RWCam[1], RWCam[2]);

#if 0
	cv::Mat vMap;
	m_dVMap.download(vMap);
	std::vector<float3> planeVertexVec;
	for (int r = 0; r < vMap.rows; ++r)
	{
		for (int c = 0; c < vMap.cols; ++c)
		{
			if (!isnan(vMap.at<float3>(r, c).x))
			{
				planeVertexVec.push_back(vMap.at<float3>(r, c));
			}
		}
	}

	std::ofstream fs;
	fs.open("D:\\xjm\\snapshot\\test3_.ply");

	// Write header
	fs << "ply";
	fs << "\nformat " << "ascii" << " 1.0";

	// Vertices
	fs << "\nelement vertex " << planeVertexVec.size();
	fs << "\nproperty float x"
		"\nproperty float y"
		"\nproperty float z";

	fs << "\nproperty uchar red"
		"\nproperty uchar green"
		"\nproperty uchar blue";

	fs << "\nproperty float nx"
		"\nproperty float ny"
		"\nproperty float nz";

	fs << "\nend_header\n";

	for (int i = 0; i < planeVertexVec.size(); ++i)
	{
		fs << planeVertexVec[i].x << " " << planeVertexVec[i].y << " " << planeVertexVec[i].z << " "
			<< (int)240 << " " << (int)240 << " " << (int)240 << " "
			<< 1 << " " << 0 << " " << 0
			<< std::endl;
	}

	// Close file
	fs.close();
	//std::exit(0);
#endif
}
void xObjectTrackor::mapToGravityCoordinateForRenderedDepth(Gravity& gravity, float depthThreshold)
{
	float3 gravityCam = normalize(make_float3(gravity.x, gravity.y, gravity.z)), gravityW = { 0.0, 0.0, 1.0 };
	float3 axis = cross(gravityCam, gravityW);
	float theta = asin(norm(axis)), sintheta = sin(theta), costheta = cos(theta);
	float3 RWCam[3];
	Rodrigues(RWCam, normalize(axis), sintheta, costheta);
	//timer.TimeEnd();
#if 0
	std::cout << "time1: " << timer.TimeGap_in_ms() << std::endl;
	std::cout << "theta: " << theta / 3.1415 * 180 << std::endl;
	std::cout << "gravity: " << gravity.x << " : " << gravity.y << " : " << gravity.z << std::endl;
	std::cout << "gravityCam: " << gravityCam.x << " : " << gravityCam.y << " : " << gravityCam.z << std::endl;
	std::cout << "gravityW: " << gravityW.x << " : " << gravityW.y << " : " << gravityW.z << std::endl;
	float3 gravityW2 = RWCam[0] * gravityCam.x + RWCam[1] * gravityCam.y + RWCam[2] * gravityCam.z;
	std::cout << "gravityW2: " << gravityW2.x << " : " << gravityW2.y << " : " << gravityW2.z << std::endl;
#endif

	CreateVMap(m_dVMap, m_dRenderedDepthImg32F,
						 Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(),
						 Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
						 depthThreshold, RWCam[0], RWCam[1], RWCam[2]);
}

cv::cuda::GpuMat& xObjectTrackor::getPruneMaskGpu()
{
	m_edgeGen.m_dEdge.download(m_pruneMat);
	cv::dilate(m_pruneMat, m_pruneMat, m_dilateKernel);
	m_dPruneMat.upload(m_pruneMat);

	return m_dPruneMat;
}

void xObjectTrackor::track(Box &box, int64_t timeStamp)
{
	if (timeStamp <= 1)
	{
		cv::Mat tmpDepthImg;
		m_dFilteredDepthImg32F.download(tmpDepthImg);
		m_edgeGen.calcValidBox(tmpDepthImg, 20, -10, -10);
	}

	const float magThreshold = 5.0f / 1000.0f, depthThreshold = 1.5f;

	float3 RWCam[3] = { { 1.0f, 0.0f, 0.0f },{ 0.0f, 1.0f, 0.0f } ,{ 0.0f, 0.0f, 1.0f } };
	CreateVMap(m_dVMap, m_dFilteredDepthImg32F,
						 Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(),
						 Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
						 depthThreshold, RWCam[0], RWCam[1], RWCam[2]);

	float minDepth = 0.0f, maxDepth = 1.0e24, meanDepth;
	cv::Rect validBox = cv::Rect(box.m_left - 10, box.m_top - 10, box.m_right - box.m_left + 20, box.m_bottom - box.m_top + 20)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
	cv::cuda::GpuMat& dVMapRoi = m_dVMap(validBox);
	CalcMinMeanDepth(minDepth, meanDepth, dVMapRoi, depthThreshold, m_dResultBuf, m_resultBuf, false);
	maxDepth = meanDepth + 1.5 * (meanDepth - minDepth);

	std::cout << "minDepth: " << minDepth << std::endl;
	std::cout << "maxDepth: " << maxDepth << std::endl;

	//timer.TimeStart();
	float3 zAxis = { 0.0f, 0.0f, 0.0f };
	float middleX = 0.0f, middleY = 0.0f, middleZ = 0.0f;
	CalcAxisForPlane2(middleX, middleY, middleZ, zAxis, m_zThresh, m_dVMap,
										minDepth, maxDepth,
										validBox.x, validBox.x + validBox.width, validBox.y, validBox.y + validBox.height,
										m_dPlaneVertexBuf, m_dIndex, false);
	//timer.TimeEnd();
	//std::cout << "CalcAxisForPlane2 time: " << timer.TimeGap_in_ms() << std::endl;
	std::cout << "zAxis: " << zAxis.x << " : " << zAxis.y << " : " << zAxis.z << std::endl;

#if 0
	//if (timeStamp == 20)
	{
		m_dVMap.download(m_vmap);
		savePly(m_vmap, middleX, middleY, middleZ,
						0.0,
						0, 0, -1, make_float3(0.0, 0.0, 0.0), make_float3(0.0, 0.0, 0.0), -zAxis);
		exit(0);
	}
#endif

#if 0
	cv::Mat test;
	m_dFilteredDepthImg32F.download(test);
	cv::imshow("test", test);
	cv::waitKey(0);
#endif

	//std::cout << "zAxis: " << zAxis.x << " : " << zAxis.y << " : " << zAxis.z << std::endl;
	//timer.TimeStart();
	RemoveNonObjectPixels2(middleX, middleY, middleZ,
												 validBox.x, validBox.x + validBox.width, validBox.y, validBox.y + validBox.height,
												 m_dFilteredDepthImg32F,
												 m_dVMap,
												 zAxis, m_zThresh);
	//timer.TimeEnd();
	//std::cout << "RemoveNonObjectPixels2 time: " << timer.TimeGap_in_ms() << std::endl;

#if 0
	if (timeStamp == 8)
	{
		m_dVMap.download(m_vmap);
		//float centerX, float centerY, float centerZ
		savePly(m_vmap, middleX, middleY, middleZ,
						0.0,
						zAxis.x, zAxis.y, zAxis.z, make_float3(1.0, 0.0, 0.0), make_float3(0.0, 1.0, 0.0), make_float3(0.0, 0.0, 1.0));
		exit(0);
	}
#endif
#if 0
	cv::Mat filteredDepthImg32F;
	m_dFilteredDepthImg32F.download(filteredDepthImg32F);
	cv::namedWindow("hehe3");
	cv::imshow("hehe3", filteredDepthImg32F * 30);
	cv::waitKey(1);
#endif
#if 0
	float3 median;
	float radiusSquare = 0;
	CalcMedianOnlyZ(median, dVMapRoi, m_dResultBuf, m_resultBuf, minDepth, maxDepth);
	median.z += 100;
#endif
	//std::cout << "median.z: " << median.z << std::endl;
	//timer.TimeStart(); 
	m_edgeGen.calcEdge(m_dFilteredDepthImg32F, m_dVMap, box, minDepth, maxDepth, magThreshold, false);
	//timer.TimeEnd();
	//std::cout << "calcEdge time: " << timer.TimeGap_in_ms() << std::endl;

	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	//cv::dilate(pruneMat, pruneMat, kernel);

	//timer.TimeStart();
	//std::cout << "median.z: " << median.z << std::endl;
	//m_edgeGen.calcEdge(m_dFilteredDepthImg32F, m_dVMap, median.z, box, true);
	//timer.TimeEnd();
	//std::cout << "time edge: " << timer.TimeGap_in_ms() << std::endl;
	//timer.TimeStart();

	//timer.TimeStart();
	m_gmsMatcher.computeOnlyOrbFeatures(m_dGrayImg, m_edgeGen.getMaskGpu());
	//timer.TimeEnd();
	//std::cout << "computeFeatures time: " << timer.TimeGap_in_ms() << std::endl;
	//timer.TimeEnd();
	//std::cout << "time feature: " << timer.TimeGap_in_ms() << std::endl;
	//std::exit(0);	

	if (timeStamp == 0)
	{
		//timer.TimeStart();
		//fitThe3DArea(box, gravity);
		//fitThe3DArea(box);
		//timer.TimeEnd();
		//std::cout << "time fitThe3DArea: " << timer.TimeGap_in_ms() << std::endl;
#if 0
		cv::Mat filteredDepthImg32F;
		m_dFilteredDepthImg32F.download(filteredDepthImg32F);
		cv::imshow("filteredDepthImg32F", filteredDepthImg32F);
		cv::waitKey(0);
#endif

		//std::exit(0);
		//std::exit(0);
		/*
		m_colorImgVis = m_colorImg.clone();
		cv::rectangle(m_colorImgVis,
		cv::Rect(box.m_left, box.m_top, box.m_right - box.m_left, box.m_bottom - box.m_top),
		cv::Scalar(255, 0, 0), 2);
		cv::imshow("tracking_win", m_colorImgVis);
		*/
		return;
	}

	//timer.TimeStart();	
	m_gmsMatcher.getMatches(m_matchesVec, colorImgVec);
	//timer.TimeEnd();
	//std::cout << "time get matches: " << timer.TimeGap_in_ms() << std::endl;

#if 0
	cv::Mat tmp(m_edgeGen.m_edgeIdx.size(), CV_8UC1);
	for (int r = 0; r < tmp.rows; ++r)
	{
		for (int c = 0; c < tmp.cols; ++c)
		{
			tmp.at<uchar>(r, c) = 0;
			if (m_edgeGen.m_edgeIdx.at<int>(r, c) == -1)
			{
				tmp.at<uchar>(r, c) = 255;
			}
		}
	}
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::dilate(tmp, tmp, kernel);
	cv::imshow("tmp", tmp);
	cv::waitKey(0);
	for (int r = 0; r < tmp.rows; ++r)
	{
		for (int c = 0; c < tmp.cols; ++c)
		{
			m_edgeGen.m_edgeIdx.at<int>(r, c) = 0;
			if (tmp.at<uchar>(r, c) > 0)
			{
				m_edgeGen.m_edgeIdx.at<int>(r, c) = -1;
			}
		}
	}
#endif

	//timer.TimeStart();
	m_edgeGen.getBoxes(m_colorImg, m_candidateBoxVec);
	//timer.TimeEnd();
	//std::cout << "time get boxes: " << timer.TimeGap_in_ms() << std::endl;

#if 0
	cv::Mat ccImgVis = m_colorImg.clone();// cv::Mat::zeros(m_edgeGen.m_edgeIdx.size(), CV_8UC3);
	for (int i = 0; i < m_gmsMatcher.m_keyPoint[m_gmsMatcher.m_prev].size(); ++i)
	{
		if (i % 2 == 0)
		{
			cv::circle(ccImgVis, m_gmsMatcher.m_keyPoint[m_gmsMatcher.m_prev][i].pt, 1, cv::Scalar(0, 255, 255), 1);
		}
	}
	for (int r = 0; r < ccImgVis.rows; r++)
	{
		for (int c = 0; c < ccImgVis.cols; c++)
		{
			int i = m_edgeGen.m_edgeIdx(r, c);
			if (i > 0)
			{
				ccImgVis.at<cv::Vec3b>(r, c)[0] = i <= 0 ? 1 : (123 * i + 128) % 255;
				ccImgVis.at<cv::Vec3b>(r, c)[1] = i <= 0 ? 1 : (79 * (i - 1) + 3) % 255;
				ccImgVis.at<cv::Vec3b>(r, c)[2] = i <= 0 ? 1 : 255;// (1174 * i + 80) % 255;
			}
		}
	}
	cv::imshow("object detect cc", ccImgVis);
	cv::waitKey(0);
#endif

	//timer.TimeStart();
	std::vector<int> cntVec(m_candidateBoxVec.size(), 0);
	std::vector<int> areaVec(m_candidateBoxVec.size(), 0);
	int srcR, srcC, targetR, targetC;
	int featureNumInBox = 0;
	for (int i = 0; i < m_matchesVec.size() / 4; ++i)
	{
		ushort *elem = m_matchesVec.data() + 4 * i;
		srcR = (int)elem[0];
		srcC = (int)elem[1];
		targetR = (int)elem[2];
		targetC = (int)elem[3];

		if (srcR > box.m_top && srcR < box.m_bottom && srcC > box.m_left && srcC < box.m_right)
		{
			++featureNumInBox;
			for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
			{
				Box &candidateBox = m_candidateBoxVec[candiIdx];
				if (targetR > candidateBox.m_top && targetR < candidateBox.m_bottom
						&& targetC > candidateBox.m_left && targetC < candidateBox.m_right)
				{
					++cntVec[candiIdx];
				}
			}
		}
	}
	//std::cout << "area: " << std::endl;
	for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
	{
		Box &candidateBox = m_candidateBoxVec[candiIdx];
		areaVec[candiIdx] = (candidateBox.m_bottom - candidateBox.m_top) * (candidateBox.m_right - candidateBox.m_left);
		//std::cout << areaVec[candiIdx] << std::endl;
	}

	int maxCCNum = 0, maxCCIdx = -1;
	for (int i = 0; i < cntVec.size(); ++i)
	{
		if (cntVec[i] > maxCCNum)
		{
			maxCCNum = cntVec[i];
			maxCCIdx = i;
		}
	}
	if (maxCCIdx == -1) // no matching features, choose the largest
	{
		int maxArea = 0;
		for (int i = 0; i < areaVec.size(); ++i)
		{
			if (areaVec[i] > maxArea)
			{
				maxArea = areaVec[i];
				maxCCIdx = i;
			}
		}
		if (maxArea == 0)
		{
			std::cout << "tracking fail" << std::endl;
			std::exit(0);
		}
	}
	box = Box(Resolution::getInstance().width(), 0, Resolution::getInstance().height(), 0);
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
	{
		if (maxCCIdx == i || (cntVec[i] > 5))// && cntVec[i] > featureNumInBox / 20))
		{
			if (m_candidateBoxVec[i].m_left < box.m_left)
				box.m_left = m_candidateBoxVec[i].m_left;
			if (m_candidateBoxVec[i].m_right > box.m_right)
				box.m_right = m_candidateBoxVec[i].m_right;
			if (m_candidateBoxVec[i].m_top < box.m_top)
				box.m_top = m_candidateBoxVec[i].m_top;
			if (m_candidateBoxVec[i].m_bottom > box.m_bottom)
				box.m_bottom = m_candidateBoxVec[i].m_bottom;
		}
	}

	box.m_top -= 20;
	box.m_bottom += 20;
	box.m_left -= 20;
	box.m_right += 20;

	box.m_left = clamp(box.m_left, 0, Resolution::getInstance().width() - 1);
	box.m_right = clamp(box.m_right, 0, Resolution::getInstance().width() - 1);
	box.m_top = clamp(box.m_top, 0, Resolution::getInstance().height() - 1);
	box.m_bottom = clamp(box.m_bottom, 0, Resolution::getInstance().height() - 1);

#if 0
	m_colorImgVis = m_colorImg.clone();
#if 1
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
	{
		cv::rectangle(m_colorImgVis,
									cv::Rect(m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_top,
									m_candidateBoxVec[i].m_right - m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_bottom - m_candidateBoxVec[i].m_top),
									cv::Scalar(128, 0, 128), 2);
	}
#endif
	cv::rectangle(m_colorImgVis,
								cv::Rect(box.m_left, box.m_top,
								box.m_right - box.m_left, box.m_bottom - box.m_top),
								cv::Scalar(0, 0, 255), 5);
	cv::namedWindow("candidate boxes");
	cv::imshow("candidate boxes", m_colorImgVis);
	cv::waitKey(0);
#endif
}

void xObjectTrackor::estimateMinMaxDepth(float &minDepth, float &maxDepth, float &meanDepth,
																				 Box &box, int64_t timeStamp, Gravity gravity)
{
	const float depthThreshold = 1.0e24f;

	//timer.TimeStart();
	mapToGravityCoordinate(gravity, depthThreshold);
	//timer.TimeEnd();
	//std::cout << "1 time: " << timer.TimeGap_in_ms() << std::endl;

	//timer.TimeStart();
	minDepth = 0.0f, maxDepth = 1.0e24;
	cv::Rect validBox1 = cv::Rect(box.m_left - 5, box.m_top - 5, box.m_right - box.m_left + 10, box.m_bottom - box.m_top + 10)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
#if 0
	cv::Rect validBox2 = cv::Rect(box.m_left - 10, box.m_top - 10, box.m_right - box.m_left + 20, box.m_bottom - box.m_top + 20
	)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
#endif
	cv::Rect validBox2 = validBox1;
	cv::cuda::GpuMat& dVMapRoi = m_dVMap(validBox1);
	CalcMinMeanDepth(minDepth, meanDepth, dVMapRoi, depthThreshold, m_dResultBuf, m_resultBuf, true);
	maxDepth = meanDepth + 1.5 * (meanDepth - minDepth);
}

void xObjectTrackor::trackOpt1(Box &box, int64_t timeStamp, Gravity gravity, float &meanDepth2)
{
	//colorImgVec.push_back(m_colorImg.clone());
	//innoreal::InnoRealTimer timer;	
	const float magThreshold = GlobalState::getInstance().m_depthThresForEdge / 1000.0f,
		depthThreshold = GlobalState::getInstance().m_maxDepthValue / 1000.0f;

	//timer.TimeStart();
	mapToGravityCoordinate(gravity, depthThreshold);
	//timer.TimeEnd();
	//std::cout << "1 time: " << timer.TimeGap_in_ms() << std::endl;

	//timer.TimeStart();
	float minDepth = 0.0f, maxDepth = 1.0e24, meanDepth;
	cv::Rect validBox1 = cv::Rect(box.m_left - 5, box.m_top - 5, box.m_right - box.m_left + 10, box.m_bottom - box.m_top + 10)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
#if 0
	cv::Rect validBox2 = cv::Rect(box.m_left - 10, box.m_top - 10, box.m_right - box.m_left + 20, box.m_bottom - box.m_top + 20
	)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
#endif
	cv::Rect validBox2 = validBox1;
	cv::cuda::GpuMat& dVMapRoi = m_dVMap(validBox1);
	CalcMinMeanDepth(minDepth, meanDepth, dVMapRoi, depthThreshold, m_dResultBuf, m_resultBuf, true);
	maxDepth = meanDepth + 1.5 * (meanDepth - minDepth);
	//timer.TimeEnd();
	//std::cout << "2 time: " << timer.TimeGap_in_ms() << std::endl;

	//std::cout << "minDepth: " << minDepth << std::endl;
	//std::cout << "maxDepth: " << maxDepth << std::endl;

	//timer.TimeStart();
	float3 zAxis;
	float middleX = 0.0f, middleY = 0.0f, middleZ = 0.0f;
	CalcAxisForPlane2(middleX, middleY, middleZ, zAxis, m_zThresh, m_dVMap,
										minDepth, maxDepth,
										validBox2.x, validBox2.x + validBox2.width, validBox2.y, validBox2.y + validBox2.height,
										m_dPlaneVertexBuf, m_dIndex, true);
	//timer.TimeEnd();
	//std::cout << "3 time: " << timer.TimeGap_in_ms() << std::endl;

	//timer.TimeStart();
	RemoveNonObjectPixels2(middleX, middleY, middleZ,
												 validBox2.x, validBox2.x + validBox2.width, validBox2.y, validBox2.y + validBox2.height,
												 m_dFilteredDepthImg32F,
												 m_dVMap,
												 zAxis, m_zThresh);
	//timer.TimeEnd();
	//std::cout << "4 time: " << timer.TimeGap_in_ms() << std::endl;	

	//timer.TimeStart();
	float3 median;
	m_edgeGen.calcEdge(m_dFilteredDepthImg32F, m_dVMap, box, minDepth, maxDepth, magThreshold, true);
	//timer.TimeEnd();
	//std::cout << "5 time: " << timer.TimeGap_in_ms() << std::endl;
	//timer.TimeStart();
	m_gmsMatcher.computeOnlyOrbFeatures(m_dGrayImg, m_edgeGen.getMaskGpu());
	//timer.TimeEnd();
	//std::cout << "6 time: " << timer.TimeGap_in_ms() << std::endl;

	if (timeStamp == 0) {
		return;
	}

	//timer.TimeStart();
	m_gmsMatcher.getMatches(m_matchesVec, colorImgVec);
	//timer.TimeEnd();
	//std::cout << "7 time: " << timer.TimeGap_in_ms() << std::endl;
	//timer.TimeStart();
	m_edgeGen.getBoxes(m_colorImg, m_candidateBoxVec);
	//timer.TimeEnd();
	//std::cout << "7 time: " << timer.TimeGap_in_ms() << std::endl;

	//timer.TimeStart();
	std::vector<int> cntVec(m_candidateBoxVec.size(), 0);
	std::vector<int> areaVec(m_candidateBoxVec.size(), 0);
	int srcR, srcC, targetR, targetC;
	int featureNumInBox = 0;
	for (int i = 0; i < m_matchesVec.size() / 4; ++i)
	{
		ushort *elem = m_matchesVec.data() + 4 * i;
		srcR = (int)elem[0];
		srcC = (int)elem[1];
		targetR = (int)elem[2];
		targetC = (int)elem[3];

		if (srcR > box.m_top && srcR < box.m_bottom && srcC > box.m_left && srcC < box.m_right)
		{
			++featureNumInBox;
			for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
			{
				Box &candidateBox = m_candidateBoxVec[candiIdx];
				if (targetR > candidateBox.m_top && targetR < candidateBox.m_bottom
						&& targetC > candidateBox.m_left && targetC < candidateBox.m_right)
				{
					++cntVec[candiIdx];
				}
			}
		}
	}
	for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
	{
		Box &candidateBox = m_candidateBoxVec[candiIdx];
		areaVec[candiIdx] = (candidateBox.m_bottom - candidateBox.m_top) * (candidateBox.m_right - candidateBox.m_left);
		//std::cout << areaVec[candiIdx] << std::endl;
	}

	int maxCCNum = 0, maxCCIdx = -1;
	for (int i = 0; i < cntVec.size(); ++i)
	{
		if (cntVec[i] > maxCCNum)
		{
			maxCCNum = cntVec[i];
			maxCCIdx = i;
		}
	}
	if (maxCCIdx == -1) // no matching features, choose the largest
	{
		int maxArea = 0;
		for (int i = 0; i < areaVec.size(); ++i)
		{
			if (areaVec[i] > maxArea)
			{
				maxArea = areaVec[i];
				maxCCIdx = i;
			}
		}
		if (maxArea == 0)
		{
			std::cout << "tracking fail" << std::endl;
			std::exit(0);
		}
	}
	box = Box(Resolution::getInstance().width(), 0, Resolution::getInstance().height(), 0);
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
	{
		if (maxCCIdx == i || (cntVec[i] > 5))// && cntVec[i] > featureNumInBox / 20))
		{
			if (m_candidateBoxVec[i].m_left < box.m_left)
				box.m_left = m_candidateBoxVec[i].m_left;
			if (m_candidateBoxVec[i].m_right > box.m_right)
				box.m_right = m_candidateBoxVec[i].m_right;
			if (m_candidateBoxVec[i].m_top < box.m_top)
				box.m_top = m_candidateBoxVec[i].m_top;
			if (m_candidateBoxVec[i].m_bottom > box.m_bottom)
				box.m_bottom = m_candidateBoxVec[i].m_bottom;
		}
	}

	box.m_left = clamp(box.m_left, 0, Resolution::getInstance().width() - 1);
	box.m_right = clamp(box.m_right, 0, Resolution::getInstance().width() - 1);
	box.m_top = clamp(box.m_top, 0, Resolution::getInstance().height() - 1);
	box.m_bottom = clamp(box.m_bottom, 0, Resolution::getInstance().height() - 1);
	//timer.TimeEnd();
	//std::cout << "8 time: " << timer.TimeGap_in_ms() << std::endl;

#if 0
	m_colorImgVis = m_colorImg.clone();
#if 1
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
	{
		cv::rectangle(m_colorImgVis,
									cv::Rect(m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_top,
									m_candidateBoxVec[i].m_right - m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_bottom - m_candidateBoxVec[i].m_top),
									cv::Scalar(128, 0, 128), 2);
	}
#endif
	cv::rectangle(m_colorImgVis,
								cv::Rect(box.m_left, box.m_top,
								box.m_right - box.m_left, box.m_bottom - box.m_top),
								cv::Scalar(0, 0, 255), 2);
	cv::namedWindow("candidate boxes");
	cv::imshow("candidate boxes", m_colorImgVis);
	cv::waitKey(1);
#endif
}

void xObjectTrackor::trackOpt2(Box &box, int64_t timeStamp, Gravity gravity, float &meanDepth2)
{
	//colorImgVec.push_back(m_colorImg.clone());
	Box prevBox = box;
	float prevBoxArea = (prevBox.m_bottom - prevBox.m_top) * (prevBox.m_right - prevBox.m_left);

	const float magThreshold = GlobalState::getInstance().m_depthThresForEdge / 1000.0f,
		depthThreshold = GlobalState::getInstance().m_maxDepthValue / 1000.0f;

	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();
	mapToGravityCoordinate(gravity, depthThreshold);

	float minDepth = 0.0f, maxDepth = 1.0e24, meanDepth;
	cv::Rect validBox = cv::Rect(box.m_left - 10, box.m_top - 10, box.m_right - box.m_left + 20, box.m_bottom - box.m_top + 20)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
	cv::cuda::GpuMat& dVMapRoi = m_dVMap(validBox);
	CalcMinMeanDepth(minDepth, meanDepth, dVMapRoi, depthThreshold, m_dResultBuf, m_resultBuf, true);
	maxDepth = meanDepth + 1.5 * (meanDepth - minDepth);

	float3 median;
	m_edgeGen.calcEdgeOptWithoutBox(m_dFilteredDepthImg32F, m_dVMap, box, minDepth, maxDepth, magThreshold);
	m_gmsMatcher.computeOnlyOrbFeatures(m_dGrayImg, m_edgeGen.getMaskGpu());

	if (timeStamp == 0)
	{
		return;
	}

	m_gmsMatcher.getMatches(m_matchesVec, colorImgVec);
	m_edgeGen.getBoxes(m_colorImg, m_candidateBoxVec);
#if 0
	cv::Mat boxVis = m_colorImg.clone();
	for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
	{
		Box &candidateBox = m_candidateBoxVec[candiIdx];
		cv::rectangle(boxVis,
									cv::Rect(candidateBox.m_left, candidateBox.m_top,
									candidateBox.m_right - candidateBox.m_left, candidateBox.m_bottom - candidateBox.m_top),
									cv::Scalar(0, 0, 255), 4);
		cv::namedWindow("candidate boxes track2");
		cv::imshow("candidate boxes track2", boxVis);
	}
#endif

	//timer.TimeStart();
	std::vector<int> cntVec(m_candidateBoxVec.size(), 0);
	std::vector<int> areaVec(m_candidateBoxVec.size(), 0);
	int srcR, srcC, targetR, targetC;
	int featureNumInBox = 0;
	for (int i = 0; i < m_matchesVec.size() / 4; ++i)
	{
		ushort *elem = m_matchesVec.data() + 4 * i;
		srcR = (int)elem[0];
		srcC = (int)elem[1];
		targetR = (int)elem[2];
		targetC = (int)elem[3];

		if (srcR > box.m_top && srcR < box.m_bottom && srcC > box.m_left && srcC < box.m_right)
		{
			++featureNumInBox;
			for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
			{
				Box &candidateBox = m_candidateBoxVec[candiIdx];
				if (targetR > candidateBox.m_top && targetR < candidateBox.m_bottom
						&& targetC > candidateBox.m_left && targetC < candidateBox.m_right)
				{
					++cntVec[candiIdx];
				}
			}
		}
	}
	for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
	{
		Box &candidateBox = m_candidateBoxVec[candiIdx];
		areaVec[candiIdx] = (candidateBox.m_bottom - candidateBox.m_top) * (candidateBox.m_right - candidateBox.m_left);
		//std::cout << areaVec[candiIdx] << std::endl;
	}

	int maxCCNum = 0, maxCCIdx = -1;
	for (int i = 0; i < cntVec.size(); ++i)
	{
		if (areaVec[i] < prevBoxArea * 2.0) 
		{
			if (cntVec[i] > maxCCNum)
			{
				maxCCNum = cntVec[i];
				maxCCIdx = i;
			}
		}
	}
	if (maxCCIdx == -1) // no matching features, choose the largest
	{
		int maxArea = 0;
		for (int i = 0; i < areaVec.size(); ++i)
		{
			if (areaVec[i] > maxArea)
			{
				maxArea = areaVec[i];
				maxCCIdx = i;
			}
		}
		if (maxArea == 0)
		{
			std::cout << "tracking fail" << std::endl;
			std::exit(0);
		}
	}
	box = Box(Resolution::getInstance().width(), 0, Resolution::getInstance().height(), 0);
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
	{
		if (areaVec[i] < prevBoxArea * 2.0)
		{
			if (maxCCIdx == i || (cntVec[i] > 5))// && cntVec[i] > featureNumInBox / 20))
			{
				if (m_candidateBoxVec[i].m_left < box.m_left)
					box.m_left = m_candidateBoxVec[i].m_left;
				if (m_candidateBoxVec[i].m_right > box.m_right)
					box.m_right = m_candidateBoxVec[i].m_right;
				if (m_candidateBoxVec[i].m_top < box.m_top)
					box.m_top = m_candidateBoxVec[i].m_top;
				if (m_candidateBoxVec[i].m_bottom > box.m_bottom)
					box.m_bottom = m_candidateBoxVec[i].m_bottom;
			}
		}
	}

	box.m_left = clamp(box.m_left, 0, Resolution::getInstance().width() - 1);
	box.m_right = clamp(box.m_right, 0, Resolution::getInstance().width() - 1);
	box.m_top = clamp(box.m_top, 0, Resolution::getInstance().height() - 1);
	box.m_bottom = clamp(box.m_bottom, 0, Resolution::getInstance().height() - 1);

	float3 zAxis;
	float middleX = 0.0f, middleY = 0.0f, middleZ = 0.0f;
	validBox = cv::Rect(box.m_left - 10, box.m_top - 10, box.m_right - box.m_left + 20, box.m_bottom - box.m_top + 20)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
	CalcAxisForPlane2(middleX, middleY, middleZ, zAxis, m_zThresh, m_dVMap,
										minDepth, maxDepth,
										validBox.x, validBox.x + validBox.width, validBox.y, validBox.y + validBox.height,
										m_dPlaneVertexBuf, m_dIndex, true);
	RemoveNonObjectPixels2(middleX, middleY, middleZ,
												 validBox.x, validBox.x + validBox.width, validBox.y, validBox.y + validBox.height,
												 m_dFilteredDepthImg32F,
												 m_dVMap,
												 zAxis, m_zThresh);

#if 0
	m_colorImgVis = m_colorImg.clone();
#if 1
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
	{
		cv::rectangle(m_colorImgVis,
									cv::Rect(m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_top,
									m_candidateBoxVec[i].m_right - m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_bottom - m_candidateBoxVec[i].m_top),
									cv::Scalar(128, 0, 128), 2);
	}
#endif
	cv::rectangle(m_colorImgVis,
								cv::Rect(box.m_left, box.m_top,
								box.m_right - box.m_left, box.m_bottom - box.m_top),
								cv::Scalar(0, 0, 255), 5);
	cv::namedWindow("boxes");
	cv::imshow("boxes", m_colorImgVis);
	cv::waitKey(1);
#endif
}

void xObjectTrackor::trackOpt3(Box &box, int64_t timeStamp, Gravity gravity, float &meanDepth2)
{
#if 0
	//innoreal::InnoRealTimer timer;	
	const float magThreshold = 5.0f / 1000.0f, depthThreshold = 1.5f;

	//timer.TimeStart();
	mapToGravityCoordinate(gravity, depthThreshold);
	//timer.TimeEnd();
	//std::cout << "1 time: " << timer.TimeGap_in_ms() << std::endl;

	//timer.TimeStart();
	float minDepth = 0.0f, maxDepth = 1.0e24, meanDepth;
	cv::Rect validBox1 = cv::Rect(box.m_left - 5, box.m_top - 5, box.m_right - box.m_left + 10, box.m_bottom - box.m_top + 10)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
#if 0
	cv::Rect validBox2 = cv::Rect(box.m_left - 10, box.m_top - 10, box.m_right - box.m_left + 20, box.m_bottom - box.m_top + 20
	)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
#endif
	cv::Rect validBox2 = validBox1;
	cv::cuda::GpuMat& dVMapRoi = m_dVMap(validBox1);
	CalcMinMeanDepth(minDepth, meanDepth, dVMapRoi, depthThreshold, m_dResultBuf, m_resultBuf, true);
	maxDepth = meanDepth + 1.5 * (meanDepth - minDepth);
	//timer.TimeEnd();
	//std::cout << "2 time: " << timer.TimeGap_in_ms() << std::endl;

	//std::cout << "minDepth: " << minDepth << std::endl;
	//std::cout << "maxDepth: " << maxDepth << std::endl;

	//timer.TimeStart();
	float3 zAxis;
	float middleX = 0.0f, middleY = 0.0f, middleZ = 0.0f;
	CalcAxisForPlane2(middleX, middleY, middleZ, zAxis, m_zThresh, m_dVMap,
										minDepth, maxDepth,
										validBox2.x, validBox2.x + validBox2.width, validBox2.y, validBox2.y + validBox2.height,
										m_dPlaneVertexBuf, m_dIndex, true);
	//timer.TimeEnd();
	//std::cout << "3 time: " << timer.TimeGap_in_ms() << std::endl;

	//timer.TimeStart();
	RemoveNonObjectPixels2(middleX, middleY, middleZ,
												 validBox2.x, validBox2.x + validBox2.width, validBox2.y, validBox2.y + validBox2.height,
												 m_dFilteredDepthImg32F,
												 m_dVMap,
												 zAxis, m_zThresh);
	//timer.TimeEnd();
	//std::cout << "4 time: " << timer.TimeGap_in_ms() << std::endl;	

	//timer.TimeStart();
	float3 median;
	m_edgeGen.calcEdge(m_dFilteredDepthImg32F, m_dVMap, box, minDepth, maxDepth, magThreshold, true);
	//timer.TimeEnd();
	//std::cout << "5 time: " << timer.TimeGap_in_ms() << std::endl;
	//timer.TimeStart();
	m_gmsMatcher.computeOnlyOrbFeatures(m_dGrayImg, m_edgeGen.getMaskGpu());
	//timer.TimeEnd();
	//std::cout << "6 time: " << timer.TimeGap_in_ms() << std::endl;

	if (timeStamp == 0) {
		return;
	}

	//timer.TimeStart();
	m_gmsMatcher.getMatches(m_matchesVec, colorImgVec);
	//timer.TimeEnd();
	//std::cout << "7 time: " << timer.TimeGap_in_ms() << std::endl;
	//timer.TimeStart();
	m_edgeGen.getBoxes(m_colorImg, m_candidateBoxVec);
	//timer.TimeEnd();
	//std::cout << "7 time: " << timer.TimeGap_in_ms() << std::endl;

	//timer.TimeStart();
	std::vector<int> cntVec(m_candidateBoxVec.size(), 0);
	std::vector<int> areaVec(m_candidateBoxVec.size(), 0);
	int srcR, srcC, targetR, targetC;
	int featureNumInBox = 0;
	for (int i = 0; i < m_matchesVec.size() / 4; ++i)
	{
		ushort *elem = m_matchesVec.data() + 4 * i;
		srcR = (int)elem[0];
		srcC = (int)elem[1];
		targetR = (int)elem[2];
		targetC = (int)elem[3];

		if (srcR > box.m_top && srcR < box.m_bottom && srcC > box.m_left && srcC < box.m_right)
		{
			++featureNumInBox;
			for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
			{
				Box &candidateBox = m_candidateBoxVec[candiIdx];
				if (targetR > candidateBox.m_top && targetR < candidateBox.m_bottom
						&& targetC > candidateBox.m_left && targetC < candidateBox.m_right)
				{
					++cntVec[candiIdx];
				}
			}
		}
	}
	for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
	{
		Box &candidateBox = m_candidateBoxVec[candiIdx];
		areaVec[candiIdx] = (candidateBox.m_bottom - candidateBox.m_top) * (candidateBox.m_right - candidateBox.m_left);
		//std::cout << areaVec[candiIdx] << std::endl;
	}

	int maxCCNum = 0, maxCCIdx = -1;
	for (int i = 0; i < cntVec.size(); ++i)
	{
		if (cntVec[i] > maxCCNum)
		{
			maxCCNum = cntVec[i];
			maxCCIdx = i;
		}
	}
	if (maxCCIdx == -1) // no matching features, choose the largest
	{
		int maxArea = 0;
		for (int i = 0; i < areaVec.size(); ++i)
		{
			if (areaVec[i] > maxArea)
			{
				maxArea = areaVec[i];
				maxCCIdx = i;
			}
		}
		if (maxArea == 0)
		{
			std::cout << "tracking fail" << std::endl;
			std::exit(0);
		}
	}
	box = Box(Resolution::getInstance().width(), 0, Resolution::getInstance().height(), 0);
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
	{
		if (maxCCIdx == i || (cntVec[i] > 5))// && cntVec[i] > featureNumInBox / 20))
		{
			if (m_candidateBoxVec[i].m_left < box.m_left)
				box.m_left = m_candidateBoxVec[i].m_left;
			if (m_candidateBoxVec[i].m_right > box.m_right)
				box.m_right = m_candidateBoxVec[i].m_right;
			if (m_candidateBoxVec[i].m_top < box.m_top)
				box.m_top = m_candidateBoxVec[i].m_top;
			if (m_candidateBoxVec[i].m_bottom > box.m_bottom)
				box.m_bottom = m_candidateBoxVec[i].m_bottom;
		}
	}

	box.m_left = clamp(box.m_left, 0, Resolution::getInstance().width() - 1);
	box.m_right = clamp(box.m_right, 0, Resolution::getInstance().width() - 1);
	box.m_top = clamp(box.m_top, 0, Resolution::getInstance().height() - 1);
	box.m_bottom = clamp(box.m_bottom, 0, Resolution::getInstance().height() - 1);
	//timer.TimeEnd();
	//std::cout << "8 time: " << timer.TimeGap_in_ms() << std::endl;

#if 0
	m_colorImgVis = m_colorImg.clone();
#if 1
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
	{
		cv::rectangle(m_colorImgVis,
									cv::Rect(m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_top,
									m_candidateBoxVec[i].m_right - m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_bottom - m_candidateBoxVec[i].m_top),
									cv::Scalar(128, 0, 128), 2);
	}
#endif
	cv::rectangle(m_colorImgVis,
								cv::Rect(box.m_left, box.m_top,
								box.m_right - box.m_left, box.m_bottom - box.m_top),
								cv::Scalar(0, 0, 255), 2);
	cv::namedWindow("candidate boxes");
	cv::imshow("candidate boxes", m_colorImgVis);
	cv::waitKey(1);
#endif
#endif
}

void xObjectTrackor::track(Box &box, int64_t timeStamp, Gravity gravity)
{
	std::cout << "track" << std::endl;
	//colorImgVec.push_back(m_colorImg.clone());
	//std::cout << "timeStamp: " << timeStamp << std::endl;
	//std::cout << "colorImgVec.size(): " << colorImgVec.size() << std::endl;
	//std::cout << "gravity: " << gravity.x << " : " << gravity.y << " : " << gravity.z << std::endl;

	const float magThreshold = 5.0f / 1000.0f, depthThreshold = 1.0f;

	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();
	mapToGravityCoordinate(gravity, depthThreshold);

	float minDepth = 0.0f, maxDepth = 1.0e24, meanDepth;
	cv::Rect validBox = cv::Rect(box.m_left - 10, box.m_top - 10, box.m_right - box.m_left + 20, box.m_bottom - box.m_top + 20)
		& cv::Rect(0, 0, Resolution::getInstance().width(), Resolution::getInstance().height());
	cv::cuda::GpuMat& dVMapRoi = m_dVMap(validBox);
	CalcMinMeanDepth(minDepth, meanDepth, dVMapRoi, depthThreshold, m_dResultBuf, m_resultBuf, true);
	maxDepth = meanDepth + 1.5 * (meanDepth - minDepth);

	//std::cout << "minDepth: " << minDepth << std::endl;
	//std::cout << "maxDepth: " << maxDepth << std::endl;

#if 1
	//timer.TimeStart();
	float3 zAxis;
	float middleX = 0.0f, middleY = 0.0f, middleZ = 0.0f;
	CalcAxisForPlane2(middleX, middleY, middleZ, zAxis, m_zThresh, m_dVMap,
										minDepth, maxDepth,
										validBox.x, validBox.x + validBox.width, validBox.y, validBox.y + validBox.height,
										m_dPlaneVertexBuf, m_dIndex, true);
	//timer.TimeEnd();
	//std::cout << "CalcAxisForPlane2 time: " << timer.TimeGap_in_ms() << std::endl;
	//std::cout << "zAxis: " << zAxis.x << " : " << zAxis.y << " : " << zAxis.z << std::endl;

#if 0
	//if (timeStamp == 20)
	{
		m_dVMap.download(m_vmap);
		savePly(m_vmap, middleX, middleY, middleZ,
						0.0,
						0, 0, -1, make_float3(0.0, 0.0, 0.0), make_float3(0.0, 0.0, 0.0), -zAxis);
		exit(0);
	}
#endif
#endif
	//std::cout << "zAxis: " << zAxis.x << " : " << zAxis.y << " : " << zAxis.z << std::endl;
	//timer.TimeStart();
	RemoveNonObjectPixels2(middleX, middleY, middleZ,
												 validBox.x, validBox.x + validBox.width, validBox.y, validBox.y + validBox.height,
												 m_dFilteredDepthImg32F,
												 m_dVMap,
												 zAxis, m_zThresh);
	//timer.TimeEnd();
	//std::cout << "RemoveNonObjectPixels2 time: " << timer.TimeGap_in_ms() << std::endl;

#if 0
	if (timeStamp == 8)
	{
		m_dVMap.download(m_vmap);
		//float centerX, float centerY, float centerZ
		savePly(m_vmap, middleX, middleY, middleZ,
						0.0,
						zAxis.x, zAxis.y, zAxis.z, make_float3(1.0, 0.0, 0.0), make_float3(0.0, 1.0, 0.0), make_float3(0.0, 0.0, 1.0));
		exit(0);
	}
#endif
#if 0
	cv::Mat filteredDepthImg32F;
	m_dFilteredDepthImg32F.download(filteredDepthImg32F);
	cv::namedWindow("hehe3");
	cv::imshow("hehe3", filteredDepthImg32F * 30);
	cv::waitKey(1);
#endif
#if 0
	float3 median;
	float radiusSquare = 0;
	CalcMedianOnlyZ(median, dVMapRoi, m_dResultBuf, m_resultBuf, minDepth, maxDepth);
	median.z += 100;
#endif
#if 0
	cv::Mat test;
	m_dFilteredDepthImg32F.download(test);
	cv::imshow("test", test);
	cv::waitKey(0);
#endif

	float3 median;
	//std::cout << "median.z: " << median.z << std::endl;
	//timer.TimeStart();
	m_edgeGen.calcEdge(m_dFilteredDepthImg32F, m_dVMap, box, minDepth, maxDepth, magThreshold, true);
	//timer.TimeEnd();
	//std::cout << "calcEdge time: " << timer.TimeGap_in_ms() << std::endl;

	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	//cv::dilate(pruneMat, pruneMat, kernel);

	//timer.TimeStart();
	//std::cout << "median.z: " << median.z << std::endl;
	//m_edgeGen.calcEdge(m_dFilteredDepthImg32F, m_dVMap, median.z, box, true);
	//timer.TimeEnd();
	//std::cout << "time edge: " << timer.TimeGap_in_ms() << std::endl;
	//timer.TimeStart();

	//timer.TimeStart();
	m_gmsMatcher.computeOnlyOrbFeatures(m_dGrayImg, m_edgeGen.getMaskGpu());
	//timer.TimeEnd();
	//std::cout << "computeFeatures time: " << timer.TimeGap_in_ms() << std::endl;
	//timer.TimeEnd();
	//std::cout << "time feature: " << timer.TimeGap_in_ms() << std::endl;
	//std::exit(0);	

	if (timeStamp == 0)
	{
		//timer.TimeStart();
		//fitThe3DArea(box, gravity);
		//fitThe3DArea(box);
		//timer.TimeEnd();
		//std::cout << "time fitThe3DArea: " << timer.TimeGap_in_ms() << std::endl;
#if 0
		cv::Mat filteredDepthImg32F;
		m_dFilteredDepthImg32F.download(filteredDepthImg32F);
		cv::imshow("filteredDepthImg32F", filteredDepthImg32F);
		cv::waitKey(0);
#endif

		//std::exit(0);
		//std::exit(0);
		/*
		m_colorImgVis = m_colorImg.clone();
		cv::rectangle(m_colorImgVis,
			cv::Rect(box.m_left, box.m_top, box.m_right - box.m_left, box.m_bottom - box.m_top),
			cv::Scalar(255, 0, 0), 2);
		cv::imshow("tracking_win", m_colorImgVis);
		*/
		return;
	}

	//timer.TimeStart();	
	m_gmsMatcher.getMatches(m_matchesVec, colorImgVec);
	//timer.TimeEnd();
	//std::cout << "time get matches: " << timer.TimeGap_in_ms() << std::endl;

#if 0
	cv::Mat tmp(m_edgeGen.m_edgeIdx.size(), CV_8UC1);
	for (int r = 0; r < tmp.rows; ++r)
	{
		for (int c = 0; c < tmp.cols; ++c)
		{
			tmp.at<uchar>(r, c) = 0;
			if (m_edgeGen.m_edgeIdx.at<int>(r, c) == -1)
			{
				tmp.at<uchar>(r, c) = 255;
			}
		}
	}
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::dilate(tmp, tmp, kernel);
	cv::imshow("tmp", tmp);
	cv::waitKey(0);
	for (int r = 0; r < tmp.rows; ++r)
	{
		for (int c = 0; c < tmp.cols; ++c)
		{
			m_edgeGen.m_edgeIdx.at<int>(r, c) = 0;
			if (tmp.at<uchar>(r, c) > 0)
			{
				m_edgeGen.m_edgeIdx.at<int>(r, c) = -1;
			}
		}
	}
#endif

	//timer.TimeStart();
	m_edgeGen.getBoxes(m_colorImg, m_candidateBoxVec);
	//timer.TimeEnd();
	//std::cout << "time get boxes: " << timer.TimeGap_in_ms() << std::endl;

#if 0
	cv::Mat ccImgVis = m_colorImg.clone();// cv::Mat::zeros(m_edgeGen.m_edgeIdx.size(), CV_8UC3);
	for (int i = 0; i < m_gmsMatcher.m_keyPoint[m_gmsMatcher.m_prev].size(); ++i)
	{
		if (i % 2 == 0)
		{
			cv::circle(ccImgVis, m_gmsMatcher.m_keyPoint[m_gmsMatcher.m_prev][i].pt, 1, cv::Scalar(0, 255, 255), 1);
		}
	}
	for (int r = 0; r < ccImgVis.rows; r++)
	{
		for (int c = 0; c < ccImgVis.cols; c++)
		{
			int i = m_edgeGen.m_edgeIdx(r, c);
			if (i > 0)
			{
				ccImgVis.at<cv::Vec3b>(r, c)[0] = i <= 0 ? 1 : (123 * i + 128) % 255;
				ccImgVis.at<cv::Vec3b>(r, c)[1] = i <= 0 ? 1 : (79 * (i - 1) + 3) % 255;
				ccImgVis.at<cv::Vec3b>(r, c)[2] = i <= 0 ? 1 : 255;// (1174 * i + 80) % 255;
			}
		}
	}
	cv::imshow("object detect cc", ccImgVis);
	cv::waitKey(0);
#endif

	//timer.TimeStart();
	std::vector<int> cntVec(m_candidateBoxVec.size(), 0);
	std::vector<int> areaVec(m_candidateBoxVec.size(), 0);
	int srcR, srcC, targetR, targetC;
	int featureNumInBox = 0;
	for (int i = 0; i < m_matchesVec.size() / 4; ++i)
	{
		ushort *elem = m_matchesVec.data() + 4 * i;
		srcR = (int)elem[0];
		srcC = (int)elem[1];
		targetR = (int)elem[2];
		targetC = (int)elem[3];

		if (srcR > box.m_top && srcR < box.m_bottom && srcC > box.m_left && srcC < box.m_right)
		{
			++featureNumInBox;
			for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
			{
				Box &candidateBox = m_candidateBoxVec[candiIdx];
				if (targetR > candidateBox.m_top && targetR < candidateBox.m_bottom
						&& targetC > candidateBox.m_left && targetC < candidateBox.m_right)
				{
					++cntVec[candiIdx];
				}
			}
		}
	}
	//std::cout << "area: " << std::endl;
	for (int candiIdx = 0; candiIdx < m_candidateBoxVec.size(); ++candiIdx)
	{
		Box &candidateBox = m_candidateBoxVec[candiIdx];
		areaVec[candiIdx] = (candidateBox.m_bottom - candidateBox.m_top) * (candidateBox.m_right - candidateBox.m_left);
		//std::cout << areaVec[candiIdx] << std::endl;
	}

	int maxCCNum = 0, maxCCIdx = -1;
	for (int i = 0; i < cntVec.size(); ++i)
	{
		if (cntVec[i] > maxCCNum)
		{
			maxCCNum = cntVec[i];
			maxCCIdx = i;
		}
	}
	if (maxCCIdx == -1) // no matching features, choose the largest
	{
		int maxArea = 0;
		for (int i = 0; i < areaVec.size(); ++i)
		{
			if (areaVec[i] > maxArea)
			{
				maxArea = areaVec[i];
				maxCCIdx = i;
			}
		}
		if (maxArea == 0)
		{
			std::cout << "tracking fail" << std::endl;
			std::exit(0);
		}
	}
	box = Box(Resolution::getInstance().width(), 0, Resolution::getInstance().height(), 0);
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
	{
		if (maxCCIdx == i || (cntVec[i] > 5))// && cntVec[i] > featureNumInBox / 20))
		{
			if (m_candidateBoxVec[i].m_left < box.m_left)
				box.m_left = m_candidateBoxVec[i].m_left;
			if (m_candidateBoxVec[i].m_right > box.m_right)
				box.m_right = m_candidateBoxVec[i].m_right;
			if (m_candidateBoxVec[i].m_top < box.m_top)
				box.m_top = m_candidateBoxVec[i].m_top;
			if (m_candidateBoxVec[i].m_bottom > box.m_bottom)
				box.m_bottom = m_candidateBoxVec[i].m_bottom;
		}
	}

	//box.m_top -= 20;
	//box.m_bottom += 20;
	//box.m_left -= 20;
	//box.m_right += 20;

	box.m_left = clamp(box.m_left, 0, Resolution::getInstance().width() - 1);
	box.m_right = clamp(box.m_right, 0, Resolution::getInstance().width() - 1);
	box.m_top = clamp(box.m_top, 0, Resolution::getInstance().height() - 1);
	box.m_bottom = clamp(box.m_bottom, 0, Resolution::getInstance().height() - 1);

	//timer.TimeEnd();
	//std::cout << "tracking time: " << timer.TimeGap_in_ms() << std::endl;

#if 0
	m_colorImgVis = m_colorImg.clone();
#if 1
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
	{
		cv::rectangle(m_colorImgVis,
									cv::Rect(m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_top,
									m_candidateBoxVec[i].m_right - m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_bottom - m_candidateBoxVec[i].m_top),
									cv::Scalar(128, 0, 128), 2);
	}
#endif
	cv::rectangle(m_colorImgVis,
								cv::Rect(box.m_left, box.m_top,
								box.m_right - box.m_left, box.m_bottom - box.m_top),
								cv::Scalar(0, 0, 255), 5);
	cv::namedWindow("candidate boxes");
	cv::imshow("candidate boxes", m_colorImgVis);
	cv::waitKey(0);
#endif

	//timer.TimeEnd();
	//std::cout << "time get box!!!: " << timer.TimeGap_in_ms() << std::endl;

	//fitThe3DArea(box, gravity);
	//fitThe3DArea(box);

#if 0
	cv::Mat filteredDepthImg32F;
	m_dFilteredDepthImg32F.download(filteredDepthImg32F);
	cv::imshow("filteredDepthImg32F", filteredDepthImg32F);
	cv::waitKey(0);
#endif

#ifdef VISIALIZTION	
	for (int r = 0; r < m_edge.rows; r++)
	{
		for (int c = 0; c < m_edge.cols; c++)
		{
			int i = m_edge(r, c);
			if (i > 0)
			{
				m_ccImgVis.at<cv::Vec3b>(r, c)[0] = i <= 0 ? 1 : (123 * i + 128) % 255;
				m_ccImgVis.at<cv::Vec3b>(r, c)[1] = i <= 0 ? 1 : (7 * i + 3) % 255;
				m_ccImgVis.at<cv::Vec3b>(r, c)[2] = i <= 0 ? 1 : (1174 * i + 80) % 255;
			}
		}
	}
#if 0
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::dilate(m_ccImgVis, m_ccImgVis, kernel);
	cv::Mat visMat;
	m_ccImgVis.copyTo(visMat);
	m_colorImg.copyTo(m_ccImgVis);
	for (int r = 0; r < m_edge.rows; r++)
	{
		for (int c = 0; c < m_edge.cols; c++)
		{
			cv::Vec3b color = visMat.at<cv::Vec3b>(r, c);
			if (color[0] != uchar(1))
			{
				m_ccImgVis.at<cv::Vec3b>(r, c) = color;
			}
		}
	}
#endif

	m_ccImgVis.copyTo(m_catImgVis(cv::Rect(0, 0, m_edge.cols, m_edge.rows)));
#endif

#ifdef VISIALIZTION
	for (int r = 0; r < m_edgeNext.rows; r++)
	{
		for (int c = 0; c < m_edgeNext.cols; c++)
		{
			int i = m_edgeNext(r, c);
			if (i > 0)
			{
				m_ccImgVis.at<cv::Vec3b>(r, c)[0] = i <= 0 ? 1 : (1123 * i + 128) % 255;
				m_ccImgVis.at<cv::Vec3b>(r, c)[1] = i <= 0 ? 1 : (7 * i + 3) % 255;
				m_ccImgVis.at<cv::Vec3b>(r, c)[2] = i <= 0 ? 1 : (174 * i + 80) % 255;
			}
		}
	}
#if 0
	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::dilate(m_ccImgVis, m_ccImgVis, kernel);
	//cv::Mat visMat;
	m_ccImgVis.copyTo(visMat);
	m_colorImg.copyTo(m_ccImgVis);
	for (int r = 0; r < m_edge.rows; r++)
	{
		for (int c = 0; c < m_edge.cols; c++)
		{
			cv::Vec3b color = visMat.at<cv::Vec3b>(r, c);
			if (color[0] != uchar(1))
			{
				m_ccImgVis.at<cv::Vec3b>(r, c) = color;
			}
		}
	}
#endif

	m_ccImgVis.copyTo(m_catImgVis(cv::Rect(m_edgeNext.cols, 0, m_edgeNext.cols, m_edgeNext.rows)));
#endif

#ifdef VISIALIZTION
	for (int i = 0; i < m_numMatch; ++i)
	{
		int x1, y1, x2, y2;
		SiftGPU::SiftKeypoint &key1 = m_siftKeyPoints[m_matchBuf[i][0]];
		SiftGPU::SiftKeypoint &key2 = m_siftKeyPointsNext[m_matchBuf[i][1]];
		x1 = key1.x;
		y1 = key1.y;
		cv::circle(m_catImgVis, cv::Point(x1, y1), 5, cv::Scalar(0, 255, 255));
		x2 = key2.x + m_colorImg.cols;
		y2 = key2.y;
		cv::circle(m_catImgVis, cv::Point(x2, y2), 5, cv::Scalar(0, 255, 255));
		cv::line(m_catImgVis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255));
	}
	cv::imshow("matching_sift_points", m_catImgVis);
	//std::vector<int> compression_params;
	//cv::imwrite("D:\\xjm\\document\\ieee_vr\\ieee_vr_templates\\figures\\object_extraction_figures\\frame-000001.correspondence.png", m_catImgVis, compression_params);
	//cv::imshow("depth", m_depthImg * 20);
	//cv::waitKey(0);
	//std::exit(0);
	//cv::waitKey(0);
	//exit(0);	

#if 0
	m_colorImgVis = m_colorImg.clone();
	for (int i = 0; i < m_candidateBoxVec.size(); ++i)
		cv::rectangle(m_colorImgVis,
									cv::Rect(m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_top,
									m_candidateBoxVec[i].m_right - m_candidateBoxVec[i].m_left, m_candidateBoxVec[i].m_bottom - m_candidateBoxVec[i].m_top),
									cv::Scalar(128, 0, 128), 2);
	cv::rectangle(m_colorImgVis,
								cv::Rect(box.m_left, box.m_top,
								box.m_right - box.m_left, box.m_bottom - box.m_top),
								cv::Scalar(255, 0, 0), 2);
	cv::namedWindow("hehe");
	cv::imshow("hehe", m_colorImgVis);
	cv::waitKey(1);
	//std::vector<int> compression_params;
	//compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	//compression_params.push_back(0);
	//cv::imshow("input_color", m_colorImgVis);
	//cv::imwrite("D:\\xjm\\document\\ieee_vr\\xjm_templates\\figures\\frame-000001.next_win.png", m_colorImgVis, compression_params);	
#endif	

	//timer.TimeEnd();
	//std::cout << "time tracking: " << timer.TimeGap_in_ms() << std::endl;
	//cv::waitKey(0);
#endif

#if 0
	cv::imshow("tracking_win", m_colorImgVis);
	cv::waitKey(1);
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);
	std::string depthDir = capture.m_depthPathVec[capture.m_ind - 1];
	std::string colorDir = capture.m_colorPathVec[capture.m_ind - 1];
	//std::string prefixStr = depthDir.substr(0, 79);
	//std::string suffixStr = depthDir.substr(79, depthDir.length() - 79);
	std::string prefixStr = depthDir.substr(0, 41);
	std::string suffixStr = depthDir.substr(47, depthDir.length() - 41);
	depthDir = prefixStr + "\\aftertrack" + suffixStr;
	//prefixStr = colorDir.substr(0, 79);
	//suffixStr = colorDir.substr(79, depthDir.length() - 79);
	prefixStr = colorDir.substr(0, 41);
	suffixStr = colorDir.substr(47, depthDir.length() - 41);
	colorDir = prefixStr + "\\aftertrack" + suffixStr;
	std::cout << depthDir << std::endl;
	std::cout << colorDir << std::endl;

	cv::imwrite(depthDir, m_depthImg, compression_params);
	//cv::imwrite(colorDir, m_colorImgVis, compression_params);
	cv::imwrite(colorDir, m_colorImg, compression_params);
	//cv::imwrite("D:\\xjm\\document\\ieee_vr\\xjm_templates\\figures\\frame-000001.color_object.png", m_colorImgVis, compression_params);
	//std::exit(0);
#endif
}

#if 0
static inline void Rodrigues(float3 *mat, float3 axis, float sintheta, float costheta)
{
	double rx, ry, rz;

	rx = axis.x;
	ry = axis.y;
	rz = axis.z;

	const double I[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

	double c = costheta;
	double s = sintheta;
	double c1 = 1. - c;

	double rrt[] = { rx*rx, rx*ry, rx*rz, rx*ry, ry*ry, ry*rz, rx*rz, ry*rz, rz*rz };
	double _r_x_[] = { 0, -rz, ry, rz, 0, -rx, -ry, rx, 0 };
	double R[9];

	for (int k = 0; k < 9; k++)
	{
		R[k] = c*I[k] + c1*rrt[k] + s*_r_x_[k];
	}

#if 0
	mat[0].x = R[0];
	mat[0].y = R[1];
	mat[0].z = R[2];

	mat[1].x = R[3];
	mat[1].y = R[4];
	mat[1].z = R[5];

	mat[2].x = R[6];
	mat[2].y = R[7];
	mat[2].z = R[8];
#endif
#if 1
	mat[0].x = R[0];
	mat[1].x = R[1];
	mat[2].x = R[2];

	mat[0].y = R[3];
	mat[1].y = R[4];
	mat[2].y = R[5];

	mat[0].z = R[6];
	mat[1].z = R[7];
	mat[2].z = R[8];
#endif
}
#endif

inline void FindMedianAndRadius(float& medianX, float& medianY, float& medianZ, float &radSquare,
																cv::Mat& vmap, Box& box, std::vector<float>& xs, std::vector<float>& ys, std::vector<float>& zs)
{
	int vertexCnt = 0;
	float minZ = 1.0e24f;
	for (int y = box.m_top; y < box.m_bottom; ++y)
	{
		for (int x = box.m_left; x < box.m_right; ++x)
		{
			float3 &vertex = vmap.at<float3>(y, x);
			if (!isnan(vertex.x))
			{
				xs[vertexCnt] = vertex.x;
				ys[vertexCnt] = vertex.y;
				zs[vertexCnt] = vertex.z;
				if (vertex.z < minZ)
				{
					minZ = vertex.z;
				}
				++vertexCnt;
			}
		}
	}
	std::nth_element(xs.begin(), xs.begin() + vertexCnt / 2, xs.begin() + vertexCnt);
	medianX = xs[vertexCnt / 2];
	std::nth_element(ys.begin(), ys.begin() + vertexCnt / 2, ys.begin() + vertexCnt);
	medianY = ys[vertexCnt / 2];
	std::nth_element(zs.begin(), zs.begin() + vertexCnt / 2, zs.begin() + vertexCnt);
	medianZ = zs[vertexCnt / 2];
	medianZ = (medianZ + minZ) / 2.0f;

	float xx, yy, rad;
	radSquare = 0.0f;
	for (int y = box.m_top; y < box.m_bottom; ++y)
	{
		for (int x = box.m_left; x < box.m_right; ++x)
		{
			float3 &vertex = vmap.at<float3>(y, x);
			if (!isnan(vertex.x))
			{
				xx = vertex.x - medianX;
				yy = vertex.y - medianY;
				rad = xx * xx + yy * yy;
				if (rad > radSquare && vertex.z < medianZ)
				{
					radSquare = rad;
				}
			}
		}
	}
}

void xObjectTrackor::fitThe3DArea(Box &box, Gravity &gravity)
{
	int width = Resolution::getInstance().width(), height = Resolution::getInstance().height();

#if 0
	//m_depthDevice.upload(m_depthImg.data, m_depthImg.step[0], m_depthImg.rows, m_depthImg.cols);
	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();
	float3 gravityCam = normalize(make_float3(gravity.x, gravity.y, gravity.z));
	float3 gravityW = { 0.0, 0.0, 1.0 };
	float3 axis = cross(gravityCam, gravityW);
	float theta = asin(norm(axis));
	float sintheta = sin(theta);// norm(axis);
	float costheta = cos(theta);// dot(gravityW, gravityCam);
	float3 RWCam[3];
	Rodrigues(RWCam, normalize(axis), sintheta, costheta);
	//timer.TimeEnd();
#if 0
	std::cout << "time1: " << timer.TimeGap_in_ms() << std::endl;
	std::cout << "theta: " << theta / 3.1415 * 180 << std::endl;
	std::cout << "gravity: " << gravity.x << " : " << gravity.y << " : " << gravity.z << std::endl;
	std::cout << "gravityCam: " << gravityCam.x << " : " << gravityCam.y << " : " << gravityCam.z << std::endl;
	std::cout << "gravityW: " << gravityW.x << " : " << gravityW.y << " : " << gravityW.z << std::endl;
	float3 gravityW2 = RWCam[0] * gravityCam.x + RWCam[1] * gravityCam.y + RWCam[2] * gravityCam.z;
	std::cout << "gravityW2: " << gravityW2.x << " : " << gravityW2.y << " : " << gravityW2.z << std::endl;
#endif

	//createVMap(cameraModel, m_depthDevice, m_vmapDevice, 5.0f, RWCam[0], RWCam[1], RWCam[2]);
	//cv::Mat filteredDepthImg32F;
	//m_dFilteredDepthImg32F.download(filteredDepthImg32F);
	//cv::imshow("filteredDepthImg32F", filteredDepthImg32F);
	//cv::waitKey(0);	
	//timer.TimeStart();
	CreateVMap(m_dVMap, m_dFilteredDepthImg32F,
						 Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(),
						 Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
						 4.0f, RWCam[0], RWCam[1], RWCam[2]);
#endif
	//timer.TimeEnd();
	//std::cout << "time2: " << timer.TimeGap_in_ms() << std::endl;
	//m_dVMap.download(m_vmap);
	//cv::cuda::GpuMat vmapTmp;
	//vmapTmp = m_dVMap.clone();

	//timer.TimeStart();
	//float medianX, medianY, medianZ, radSquare;
	//FindMedianAndRadius(medianX, medianY, medianZ, radSquare,
						//m_vmap, box, m_xs, m_ys, m_zs);
	//timer.TimeEnd();
	//std::cout << "time3: " << timer.TimeGap_in_ms() << std::endl;
#if 0
	std::cout << medianX << std::endl;
	std::cout << medianY << std::endl;
	std::cout << medianZ << std::endl;
	std::cout << radSquare << std::endl;
#endif

	//timer.TimeStart();
	float3 median;
	float radiusSquare = 0;
	cv::cuda::GpuMat& dVMapRoi = m_dVMap(
		cv::Rect(box.m_left, box.m_top, box.m_right - box.m_left, box.m_bottom - box.m_top)
	);
	CalcMedianAndRadius(median, radiusSquare, dVMapRoi, m_dResultBuf, m_resultBuf);

	float3 zAxis;
	CalcAxisForPlane(zAxis, m_zThresh, m_dVMap, median,
									 radiusSquare * 1.0 * 1.0, radiusSquare * 1.3 * 1.3, m_dPlaneVertexBuf, m_dIndex);

	//printf("zThresh: %f %f %f %f\n", zAxis.x, zAxis.y, zAxis.z, zThresh);
#if 0
	cv::Mat pruneMat;
	m_depthImg.convertTo(pruneMat, CV_8UC1, 1.0f / 16.0f);
	cv::Canny(pruneMat, pruneMat, 8, 20, 3);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::dilate(pruneMat, pruneMat, kernel);
	m_pruneMatDevice.upload(pruneMat);
#endif

	RemoveNonObjectPixels(m_dFilteredDepthImg32F,
												m_dVMap,
												median, radiusSquare * 1.2 * 1.2,
												zAxis, m_zThresh);
	//cv::Mat filteredDepthImg32F;
	//m_dFilteredDepthImg32F.download(filteredDepthImg32F);
	//cv::imshow("filteredDepthImg32F", filteredDepthImg32F);
	//cv::waitKey(0);

#if 0
	m_dVMap.download(m_vmap);
	savePly(m_vmap, median.x, median.y, median.z,
					radiusSquare,
					0, 0, 1, zAxis, zAxis, zAxis);

	//std::exit(0);
#endif

#if 0
	timer.TimeEnd();
	std::cout << "time3: " << timer.TimeGap_in_ms() << std::endl;

	savePly(m_vmap, median.x, median.y, median.z,
					radiusSquare,
					0, 0, 1);
#if 0
	savePly(m_vmap, medianX, medianY, medianZ,
					radSquare,
					0, 0, 1);
#endif
	std::exit(0);

#if 1
	timer.TimeStart();
	RemoveVerticesOutOfRange(m_dVMap, medianX, medianY, medianZ,
													 radSquare * 1.1 * 1.1, radSquare * 2.0 * 2.0);
	//std::cout << "radSquare: " << radSquare << std::endl;	
	timer.TimeEnd();
	std::cout << "time4: " << timer.TimeGap_in_ms() << std::endl;
	timer.TimeStart();
	m_dVMap.download(m_vmap);
	timer.TimeEnd();
	std::cout << "time5: " << timer.TimeGap_in_ms() << std::endl;
#endif
#if 1
	std::vector<float3> verticesTmp2;
	int pixelNum = m_vmap.rows * m_vmap.cols;
#if 0
	float3 *vertex = (float3 *)m_vmap.data;
	for (int i = 0; i < pixelNum; ++i)
	{
		if (!isnan(vertex[i].x))
		{
			verticesTmp2.push_back(vertex[i]);
		}
	}
#endif
	for (int r = 0; r < m_vmap.rows; ++r)
	{
		for (int c = 0; c < m_vmap.cols; ++c)
		{
			float3 &vertex = m_vmap.at<float3>(r, c);
			if (!isnan(vertex.x))
			{
				verticesTmp2.push_back(vertex);
			}
		}
	}

	float middleX = 0.0f, middleY = 0.0f, middleZ = 0.0f;
	for (int i = 0; i < verticesTmp2.size(); ++i)
	{
		middleX += verticesTmp2[i].x;
		middleY += verticesTmp2[i].y;
		middleZ += verticesTmp2[i].z;
	}
	middleX /= verticesTmp2.size();
	middleY /= verticesTmp2.size();
	middleZ /= verticesTmp2.size();
	float covMat[9];
	for (int i = 0; i < verticesTmp2.size(); ++i)
	{
		verticesTmp2[i].x -= middleX;
		verticesTmp2[i].y -= middleY;
		verticesTmp2[i].z -= middleZ;
	}
	memset(covMat, 0, sizeof(covMat));
	for (int i = 0; i < verticesTmp2.size(); ++i)
	{
		covMat[0] += verticesTmp2[i].x * verticesTmp2[i].x;
		covMat[3] += verticesTmp2[i].x * verticesTmp2[i].y;
		covMat[6] += verticesTmp2[i].x * verticesTmp2[i].z;

		covMat[1] += verticesTmp2[i].y * verticesTmp2[i].x;
		covMat[4] += verticesTmp2[i].y * verticesTmp2[i].y;
		covMat[7] += verticesTmp2[i].y * verticesTmp2[i].z;

		covMat[2] += verticesTmp2[i].z * verticesTmp2[i].x;
		covMat[5] += verticesTmp2[i].z * verticesTmp2[i].y;
		covMat[8] += verticesTmp2[i].z * verticesTmp2[i].z;
	}
	float3 eigenVectors[3];
	//PCA3x3(eigenVectors, covMat);
	std::cout << eigenVectors[0].x << " : " << eigenVectors[0].y << " : " << eigenVectors[0].z << std::endl;
	std::cout << eigenVectors[1].x << " : " << eigenVectors[1].y << " : " << eigenVectors[1].z << std::endl;
	std::cout << eigenVectors[2].x << " : " << eigenVectors[2].y << " : " << eigenVectors[2].z << std::endl;

#if 1
	gravityW = { 0.0, 0.0, 1.0 };
	axis = cross(eigenVectors[2], gravityW);
	theta = asin(norm(axis));
	sintheta = sin(theta);// norm(axis);
	costheta = cos(theta);// dot(gravityW, gravityCam);
	float3 RRefine[3];
	Rodrigues(RRefine, normalize(axis), sintheta, costheta);
	RotateVertices(vmapTmp, RRefine[0], RRefine[1], RRefine[2]);
#endif
	vmapTmp.download(m_vmap);

	float x, y, z, minZ = 1e24f, maxZ = -1e24f;
	pixelNum = m_vmap.rows * m_vmap.cols;
	float3 *vertex = (float3 *)m_vmap.data;
	for (int i = 0; i < pixelNum; ++i)
	{
		if (!isnan(vertex[i].x))
		{
			z = vertex[i].z;
			if (z < minZ)
			{
				minZ = z;
			}
			if (z > maxZ)
			{
				maxZ = z;
			}
		}
	}
	std::cout << "minZ: " << minZ << std::endl;
	std::cout << "maxZ: " << maxZ << std::endl;
#if 0
	int binNum = (maxZ - minZ) / 0.005;
	std::vector<int> bin(binNum, 0);
	float binSize = (maxZ - minZ) / binNum;
	for (int i = 0; i < pixelNum; ++i)
	{
		if (!isnan(vertex[i].x))
		{
			++bin[(int)((vertex[i].z - minZ) / binSize)];
		}
	}
	for (int i = 0; i < binNum; ++i)
		std::cout << bin[i] << ", ";
	std::cout << std::endl;
	std::exit(0);
#endif
#if 1
	for (int i = 0; i < pixelNum; ++i)
	{
		if (!isnan(vertex[i].x))
		{
			z = vertex[i].z;
			if (z > maxZ - 0.02)
			{
				vertex[i].x = 0;
				vertex[i].y = 0;
				vertex[i].z = 0;
			}
		}
	}
#endif

	//std::cout << dot(eigenVectors[0], eigenVectors[1]) << std::endl;
	//std::cout << dot(eigenVectors[0], eigenVectors[2]) << std::endl;
	//std::cout << dot(eigenVectors[1], eigenVectors[2]) << std::endl;
#endif
	//float3 eigenVectors[3];
#define SAVE_PLY
#ifdef SAVE_PLY
	//savePly(medianX, medianY, medianZ,
	//gravity.x, gravity.y, gravity.z);
	savePly(m_vmap, medianX, medianY, medianZ,
					radSquare,
					0, 0, 1,
					eigenVectors[0], eigenVectors[1], eigenVectors[2]);
	std::exit(0);
#endif

#if 0
	float x, y, z, minZ = 1e24f, maxZ = -1e24f;
	std::vector<float3> vertices1, vertices2;
	int pixelNum = m_vmap.rows * m_vmap.cols;
	float3 *vertex = (float3 *)m_vmap.data;
	for (int i = 0; i < pixelNum; ++i)
	{
		if (!isnan(vertex[i].x))
		{
			vertices1.push_back(vertex[i]);
		}
	}
	for (int i = 0; i < vertices1.size(); ++i)
	{
		z = vertices1[i].z;
		if (z < minZ)
		{
			minZ = z;
		}
		if (z > maxZ)
		{
			maxZ = z;
		}
	}
	std::cout << "minZ: " << minZ << std::endl;
	std::cout << "maxZ: " << maxZ << std::endl;
	int binNum = (maxZ - minZ) / 0.005;
	std::vector<int> bin(binNum, 0);
	float binSize = (maxZ - minZ) / binNum;
	for (int i = 0; i < vertices1.size(); ++i)
	{
		++bin[(int)((vertices1[i].z - minZ) / binSize)];
	}
	for (int i = 0; i < binNum; ++i)
		std::cout << bin[i] << ", ";
	std::cout << std::endl;
	std::exit(0);

	int cnt = 0;
	for (int i = 0; i < verticesTmp1.size() / 3; ++i)
	{
		x = verticesTmp1[3 * i];
		y = verticesTmp1[3 * i + 1];
		z = verticesTmp1[3 * i + 2];
		if (((int)((z - minZ) / binSize)) < binNum - 5)
			continue;
		verticesTmp2.push_back(x);
		verticesTmp2.push_back(y);
		verticesTmp2.push_back(z);
		++cnt;
	}
#endif

#if 0
	float middleX = 0.0f, middleY = 0.0f, middleZ = 0.0f;
	int len = verticesTmp2.size() / 3;
	for (int i = 0; i < len; ++i)
	{
		middleX += verticesTmp2[3 * i];
		middleY += verticesTmp2[3 * i + 1];
		middleZ += verticesTmp2[3 * i + 2];
	}
	middleX /= len;
	middleY /= len;
	middleZ /= len;
	float covMat[9];
	for (int i = 0; i < len; ++i)
	{
		verticesTmp2[3 * i] -= middleX;
		verticesTmp2[3 * i + 1] -= middleY;
		verticesTmp2[3 * i + 2] -= middleZ;
	}
	for (int i = 0; i < len; ++i)
	{
		covMat[0] = verticesTmp2[3 * i] * verticesTmp2[3 * i];
		covMat[1] = verticesTmp2[3 * i] * verticesTmp2[3 * i + 1];
		covMat[2] = verticesTmp2[3 * i] * verticesTmp2[3 * i + 2];
		covMat[3] = verticesTmp2[3 * i + 1] * verticesTmp2[3 * i];
		covMat[4] = verticesTmp2[3 * i + 1] * verticesTmp2[3 * i + 1];
		covMat[5] = verticesTmp2[3 * i + 1] * verticesTmp2[3 * i + 2];
		covMat[6] = verticesTmp2[3 * i + 2] * verticesTmp2[3 * i];
		covMat[7] = verticesTmp2[3 * i + 2] * verticesTmp2[3 * i + 1];
		covMat[8] = verticesTmp2[3 * i + 2] * verticesTmp2[3 * i + 2];
	}
	float3 eigenVectors[3];
	PCA3x3(eigenVectors, covMat);
	std::cout << eigenVectors[0].x << " : " << eigenVectors[0].y << " : " << eigenVectors[0].z << std::endl;
	std::cout << eigenVectors[1].x << " : " << eigenVectors[1].y << " : " << eigenVectors[1].z << std::endl;
	std::cout << eigenVectors[2].x << " : " << eigenVectors[2].y << " : " << eigenVectors[2].z << std::endl;
	Eigen::EigenSolver<Eigen::Matrix3f> es(covMat);
	Eigen::Matrix3d D = es.pseudoEigenvalueMatrix();
	Eigen::Matrix3d V = es.pseudoEigenvectors();
	std::cout << "The pseudo-eigenvalue matrix D is:" << std::endl << D << std::endl;
	std::cout << "The pseudo-eigenvector matrix V is:" << std::endl << V << std::endl;
#endif

	//createVMap(cameraModel, m_depthDevice, m_vmapDevice, 5.0f);
	//m_vmapDevice.download(m_vmapVec.data(), m_depthImg.cols * sizeof(float));

#if 0
	int vertexCnt = 0;
	float medianX, medianY, medianZ;
	for (int y = box.m_top; y < box.m_bottom; ++y)
	{
		for (int x = box.m_left; x < box.m_right; ++x)
		{
			float3 &vertex = m_vmap.at<float3>(y, x);
			if (!isnan(vertex.x))
			{
				m_xs[vertexCnt] = vertex.x;
				m_ys[vertexCnt] = vertex.y;
				m_zs[vertexCnt] = vertex.z;
				++vertexCnt;
			}
		}
	}
	std::nth_element(m_xs.begin(), m_xs.begin() + vertexCnt / 2, m_xs.begin() + vertexCnt);
	medianX = m_xs[vertexCnt / 2];
	std::nth_element(m_ys.begin(), m_ys.begin() + vertexCnt / 2, m_ys.begin() + vertexCnt);
	medianY = m_ys[vertexCnt / 2];
	std::nth_element(m_zs.begin(), m_zs.begin() + vertexCnt / 2, m_zs.begin() + vertexCnt);
	medianZ = m_zs[vertexCnt / 2];



#define SAVE_PLY
#ifdef SAVE_PLY
	//savePly(medianX, medianY, medianZ,
		//gravity.x, gravity.y, gravity.z);
	savePly(medianX, medianY, medianZ,
					0, 0, 1);
	std::exit(0);
#endif
#endif

#if 0
	float xx, yy, zz, ll, maxRadSquare = 0.0f, rad;
	for (int y = box.m_top; y < box.m_bottom; ++y)
	{
		for (int x = box.m_left; x < box.m_right; ++x)
		{
			int n = y * width + x;
			if (!isnan(m_vmapVec[n]) && !isnan(m_nmapVec[n]))
			{
				xx = pVMapX[n] - medianX;
				yy = pVMapY[n] - medianY;
				zz = pVMapZ[n] - medianZ;
				//ll = gravity.x * xx + gravity.y * yy + gravity.z * zz;
				//rad = xx * xx + yy * yy + zz * zz - ll * ll;
				rad = xx * xx + yy * yy;
				if (rad > maxRadSquare)// && zz < 0)
				{
					maxRadSquare = rad;
				}
			}
		}
	}

#if 1
	cv::Mat pruneMat;
	m_depthImg.convertTo(pruneMat, CV_8UC1, 1.0f / 16.0f);
	cv::Canny(pruneMat, pruneMat, 8, 20, 3);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::dilate(pruneMat, pruneMat, kernel);
	m_pruneMatDevice.upload(pruneMat);
#endif

	//m_meanMaxRadSquare = 0.05;// (m_meanMaxRadSquare * m_cntMaxRad + maxRadSquare) / (m_cntMaxRad + 1);
	//std::cout << "m_meanMaxRadSquare: " << maxRadSquare << std::endl;
	//++m_cntMaxRad;

	RemoveNonObjectPixels(m_depthDevice, m_vmapDevice, m_pruneMatDevice,
												medianX, medianY, medianZ, gravity.x, gravity.y, gravity.z, maxRadSquare * 1.5 * 1.5);
	m_depthDevice.download(m_depthImg.data, m_depthImg.cols * sizeof(ushort));
#endif
#endif
}

void xObjectTrackor::fitThe3DArea(Box &box)
{
#if 0
	int width = Resolution::getInstance().width(), height = Resolution::getInstance().height();

	Box extendedBoxLeft, extendedBoxRight;
	extendedBoxLeft.m_top = extendedBoxRight.m_top = (box.m_bottom + box.m_top) / 2;
	extendedBoxLeft.m_bottom = extendedBoxRight.m_bottom = box.m_bottom;;
	extendedBoxLeft.m_left = clamp(box.m_left - (box.m_right - box.m_left) / 4, 0, width);
	extendedBoxLeft.m_right = clamp(box.m_left, 0, width);
	extendedBoxRight.m_left = clamp(box.m_right, 0, width);
	extendedBoxRight.m_right = clamp(box.m_right + (box.m_right - box.m_left) / 4, 0, width);

	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();
	m_depthDevice.upload(m_rawDepthImg.data, m_rawDepthImg.step[0], m_rawDepthImg.rows, m_rawDepthImg.cols);
	CameraModel cameraModel(Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(), Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy());
	createVMap(cameraModel, m_depthDevice, m_vmapDevice, 5.0f);
	createNMap(m_vmapDevice, m_nmapDevice);
	m_vmapDevice.download(m_vmapVec.data(), m_rawDepthImg.cols * sizeof(float));
	m_nmapDevice.download(m_nmapVec.data(), m_rawDepthImg.cols * sizeof(float));
	//timer.TimeEnd();

	int step = m_vmapVec.size() / 3;
	float *pVMapX = m_vmapVec.data(),
		*pVMapY = m_vmapVec.data() + step,
		*pVMapZ = m_vmapVec.data() + 2 * step,
		*pNMapX = m_nmapVec.data(),
		*pNMapY = m_nmapVec.data() + step,
		*pNMapZ = m_nmapVec.data() + 2 * step;
	int normalCnt = 0, vertexCnt = 0;
	float normalX = 0.0, normalY = 0.0, normalZ = 0.0;
	float centerX = 0.0, centerY = 0.0, centerZ = 0.0;
	float medianX, medianY, medianZ;
	for (int y = box.m_top; y < box.m_bottom; ++y)
		//for (int y = 0; y < 480; ++y)
	{
		for (int x = box.m_left; x < box.m_right; ++x)
			//for (int x = 0; x < 640; ++x)
		{
			int n = y * width + x;
			if (!isnan(m_vmapVec[n]) && !isnan(m_nmapVec[n]))
			{
				//centerX += pVMapX[n];
				//centerY += pVMapY[n];
				//centerZ += pVMapZ[n];
				m_xs[vertexCnt] = pVMapX[n];
				m_ys[vertexCnt] = pVMapY[n];
				m_zs[vertexCnt] = pVMapZ[n];
				++vertexCnt;
			}
		}
	}
	std::nth_element(m_xs.begin(), m_xs.begin() + vertexCnt / 2, m_xs.begin() + vertexCnt);
	medianX = m_xs[vertexCnt / 2];
	std::nth_element(m_ys.begin(), m_ys.begin() + vertexCnt / 2, m_ys.begin() + vertexCnt);
	medianY = m_ys[vertexCnt / 2];
	std::nth_element(m_zs.begin(), m_zs.begin() + vertexCnt / 2, m_zs.begin() + vertexCnt);
	medianZ = m_zs[vertexCnt / 2];

	for (int y = extendedBoxLeft.m_top; y < extendedBoxLeft.m_bottom; ++y)
	{
		for (int x = extendedBoxLeft.m_left; x < extendedBoxLeft.m_right; ++x)
		{
			int n = y * width + x;
			if (!isnan(m_vmapVec[n]) && !isnan(m_nmapVec[n]))
			{
				normalX += pNMapX[n];
				normalY += pNMapY[n];
				normalZ += pNMapZ[n];
				++normalCnt;
			}
		}
	}
	for (int y = extendedBoxRight.m_top; y < extendedBoxRight.m_bottom; ++y)
	{
		for (int x = extendedBoxRight.m_left; x < extendedBoxRight.m_right; ++x)
		{
			int n = y * width + x;
			if (!isnan(m_vmapVec[n]) && !isnan(m_nmapVec[n]))
			{
				normalX += pNMapX[n];
				normalY += pNMapY[n];
				normalZ += pNMapZ[n];
				++normalCnt;
			}
		}
	}

	//centerX /= vertexCnt; centerY /= vertexCnt; centerZ /= vertexCnt;
	//medianX = centerX;
	//medianY = centerY;
	//medianZ = centerZ;
	normalX /= normalCnt; normalY /= normalCnt; normalZ /= normalCnt;
	float normalNorm = sqrt(normalX * normalX + normalY * normalY + normalZ * normalZ);
	normalX /= normalNorm; normalY /= normalNorm; normalZ /= normalNorm;
	//std::cout << box.m_top << " : " << box.m_bottom << " : " << box.m_left << " : " << box.m_right << std::endl;
	//std::cout << medianX << " : " << medianY << " : " << medianZ << std::endl;
	//std::cout << normalX << " : " << normalY << " : " << normalZ << std::endl;

#ifdef SAVE_PLY
	savePly(centerX, centerY, centerZ,
					normalX, normalY, normalZ);
#endif

	float xx, yy, zz, ll, maxRadSquare = 0.0f, rad;
	for (int y = box.m_top; y < box.m_bottom; ++y)
	{
		for (int x = box.m_left; x < box.m_right; ++x)
		{
			int n = y * width + x;
			if (!isnan(m_vmapVec[n]) && !isnan(m_nmapVec[n]))
			{
				xx = pVMapX[n] - medianX;
				yy = pVMapY[n] - medianY;
				zz = pVMapZ[n] - medianZ;
				ll = normalX * xx + normalY * yy + normalZ * zz;
				rad = xx * xx + yy * yy + zz * zz - ll * ll;
				if (rad > maxRadSquare && zz < 0)
				{
					maxRadSquare = rad;
				}
			}
		}
	}

#if 1
	cv::Mat pruneMat;
	m_rawDepthImg.convertTo(pruneMat, CV_8UC1, 1.0f / 16.0f);
	cv::Canny(pruneMat, pruneMat, 8, 20, 3);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::dilate(pruneMat, pruneMat, kernel);
	m_pruneMatDevice.upload(pruneMat);
#endif

	m_meanMaxRadSquare = (m_meanMaxRadSquare * m_cntMaxRad + maxRadSquare) / (m_cntMaxRad + 1);
	//std::cout << "m_meanMaxRadSquare: " << maxRadSquare << std::endl;
	++m_cntMaxRad;

	//RemoveNonObjectPixels2(m_depthDevice, m_vmapDevice, m_pruneMatDevice,
		//medianX, medianY, medianZ, normalX, normalY, normalZ, maxRadSquare * 1.5 * 1.5);
	RemoveNonObjectPixels2(m_depthDevice, m_vmapDevice, m_pruneMatDevice,
												 medianX, medianY, medianZ, normalX, normalY, normalZ, m_meanMaxRadSquare * 1.2 * 1.2);
	m_depthDevice.download(m_rawDepthImg.data, m_rawDepthImg.cols * sizeof(ushort));
#endif
}

void xObjectTrackor::savePly(cv::Mat &vmap, float centerX, float centerY, float centerZ,
														 float radSquare,
														 float normalX, float normalY, float normalZ,
														 float3 axis1, float3 axis2, float3 axis3)
{
	std::ofstream fs;
	fs.open("D:\\xjm\\snapshot\\test4_.ply");

	int vertexNum = 0;
	float3 *vertex = (float3 *)vmap.data;
	int pixelNum = vmap.rows * vmap.cols;
	for (int i = 0; i < pixelNum; ++i)
	{
		if (!isnan(vertex[i].x))
		{
			++vertexNum;
		}
	}

	// Write header
	fs << "ply";
	fs << "\nformat " << "ascii" << " 1.0";

	// Vertices
	fs << "\nelement vertex " << vertexNum + 500;
	fs << "\nproperty float x"
		"\nproperty float y"
		"\nproperty float z";

	fs << "\nproperty uchar red"
		"\nproperty uchar green"
		"\nproperty uchar blue";

	fs << "\nproperty float nx"
		"\nproperty float ny"
		"\nproperty float nz";

	fs << "\nend_header\n";

#if 1
	float rad = sqrt(radSquare), theta = 0;
	for (int i = 0; i < 100; ++i)
	{
		fs << centerX + rad * cos(theta) << " " << centerY + rad * sin(theta) << " " << centerZ << " "
			<< (int)0 << " " << (int)250 << " " << (int)0 << " "
			<< 1 << " " << 1 << " " << 1
			<< std::endl;
		theta += (3.1415 * 2 / 100);
	}
	for (int i = 0; i < 100; ++i)
	{
		fs << centerX + normalX * i / 200.0 << " " << centerY + normalY * i / 200.0 << " " << centerZ + normalZ * i / 200.0 << " "
			<< (int)250 << " " << (int)0 << " " << (int)0 << " "
			<< 1 << " " << 1 << " " << 1
			<< std::endl;
	}
#endif
#if 1
	for (int i = 0; i < 100; ++i)
	{
		fs << centerX + axis1.x * i / 200.0 << " "
			<< centerY + axis1.y * i / 200.0 << " "
			<< centerZ + axis1.z * i / 200.0 << " "
			<< (int)0 << " " << (int)255 << " " << (int)0 << " "
			<< 1 << " " << 1 << " " << 1
			<< std::endl;
	}
	for (int i = 0; i < 100; ++i)
	{
		fs << centerX + axis2.x * i / 200.0 << " "
			<< centerY + axis2.y * i / 200.0 << " "
			<< centerZ + axis2.z * i / 200.0 << " "
			<< (int)20 << " " << (int)255 << " " << (int)0 << " "
			<< 1 << " " << 1 << " " << 1
			<< std::endl;
	}
	for (int i = 0; i < 100; ++i)
	{
		fs << centerX + axis3.x * i / 200.0 << " "
			<< centerY + axis3.y * i / 200.0 << " "
			<< centerZ + axis3.z * i / 200.0 << " "
			<< (int)40 << " " << (int)255 << " " << (int)0 << " "
			<< 1 << " " << 1 << " " << 1
			<< std::endl;
	}
#endif

	for (int i = 0; i < pixelNum; ++i)
	{
		if (!isnan(vertex[i].x))
		{
			//z = m_vmapVec[i + step * 2];
			//if (((int)((z - minZ) / binSize)) < binNum - 5)
				//continue;
			fs << vertex[i].x << " " << vertex[i].y << " " << vertex[i].z << " "
				<< (int)240 << " " << (int)240 << " " << (int)240 << " "
				<< 1 << " " << 0 << " " << 0
				<< std::endl;
		}
	}

	// Close file
	fs.close();
	//std::exit(0);
}

void xObjectTrackor::savePly(cv::Mat &vmap, float centerX, float centerY, float centerZ,
														 float radSquare,
														 float normalX, float normalY, float normalZ)
{
	std::ofstream fs;
	fs.open("D:\\xjm\\snapshot\\test4.ply");

	int vertexNum = 0;
	float3 *vertex = (float3 *)vmap.data;
	int pixelNum = vmap.rows * vmap.cols;
	for (int i = 0; i < pixelNum; ++i)
	{
		if (!isnan(vertex[i].x))
		{
			++vertexNum;
		}
	}

	// Write header
	fs << "ply";
	fs << "\nformat " << "ascii" << " 1.0";

	// Vertices
	fs << "\nelement vertex " << vertexNum + 200;// 500;
	fs << "\nproperty float x"
		"\nproperty float y"
		"\nproperty float z";

	fs << "\nproperty uchar red"
		"\nproperty uchar green"
		"\nproperty uchar blue";

	fs << "\nproperty float nx"
		"\nproperty float ny"
		"\nproperty float nz";

	fs << "\nend_header\n";

#if 1
	float rad = sqrt(radSquare), theta = 0;
	for (int i = 0; i < 100; ++i)
	{
		fs << centerX + rad * cos(theta) << " " << centerY + rad * sin(theta) << " " << centerZ << " "
			<< (int)0 << " " << (int)250 << " " << (int)0 << " "
			<< 1 << " " << 1 << " " << 1
			<< std::endl;
		theta += (3.1415 * 2 / 100);
	}
	for (int i = 0; i < 100; ++i)
	{
		fs << centerX + normalX * i / 200.0 << " " << centerY + normalY * i / 200.0 << " " << centerZ + normalZ * i / 200.0 << " "
			<< (int)250 << " " << (int)0 << " " << (int)0 << " "
			<< 1 << " " << 1 << " " << 1
			<< std::endl;
	}
#endif

	for (int i = 0; i < pixelNum; ++i)
	{
		if (!isnan(vertex[i].x))
		{
			//z = m_vmapVec[i + step * 2];
			//if (((int)((z - minZ) / binSize)) < binNum - 5)
			//continue;
			fs << vertex[i].x << " " << vertex[i].y << " " << vertex[i].z << " "
				<< (int)240 << " " << (int)240 << " " << (int)240 << " "
				<< 1 << " " << 0 << " " << 0
				<< std::endl;
		}
	}

	// Close file
	fs.close();
	//std::exit(0);
}

#if 0
while (true)
{
	cv::imshow("CamShift Demo", colorImg);
	cv::waitKey(1);
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);
	std::string depthDir = capture.m_depthPathVec[capture.m_ind - 1];
	std::string colorDir = capture.m_colorPathVec[capture.m_ind - 1];
	std::string prefixStr = depthDir.substr(0, 66);
	std::string suffixStr = depthDir.substr(66, depthDir.length() - 66);
	depthDir = prefixStr + "\\aftertrack\\" + suffixStr;
	prefixStr = colorDir.substr(0, 66);
	suffixStr = colorDir.substr(66, depthDir.length() - 66);
	colorDir = prefixStr + "\\aftertrack\\" + suffixStr;
	std::cout << depthDir << std::endl;
	std::cout << colorDir << std::endl;

	cv::imwrite(depthDir, depthImg, compression_params);
	cv::imwrite(colorDir, colorImg, compression_params);

	capture.next(depthImg, colorImg);

#ifdef GRAY_WORLD_ASSUMPTION
	cv::Mat grayWorldImg = colorImg.clone();
	grayWorldImg = grayWorldImg ^ (1.0 / 2.2);
	cv::namedWindow("hehe");
	cv::imshow("hehe", grayWorldImg);
	double meanRGB[3] = { 0, 0, 0 };
	int numRGB[3] = { 0, 0, 0 };
	for (int y = 0; y < colorImg.rows; ++y)
	{
		for (int x = 0; x < colorImg.cols; ++x)
		{
			cv::Vec3b &rgb = grayWorldImg.at<cv::Vec3b>(y, x);
			if (rgb != cv::Vec3b(0, 0, 0))
			{
				meanRGB[0] += rgb[0];
				meanRGB[1] += rgb[1];
				meanRGB[2] += rgb[2];
				++numRGB[0];
				++numRGB[1];
				++numRGB[2];
			}
		}
	}
	meanRGB[0] /= numRGB[0];
	meanRGB[1] /= numRGB[1];
	meanRGB[2] /= numRGB[2];

	double grayWorldAssum[3] = { 128.0, 128.0, 128.0 };

	for (int y = 0; y < colorImg.rows; ++y)
	{
		for (int x = 0; x < colorImg.cols; ++x)
		{
			cv::Vec3b &rgb = grayWorldImg.at<cv::Vec3b>(y, x);
			if (rgb != cv::Vec3b(0, 0, 0))
			{
				rgb[0] = rgb[0] / meanRGB[0] * grayWorldAssum[0];
				rgb[1] = rgb[1] / meanRGB[1] * grayWorldAssum[1];
				rgb[2] = rgb[2] / meanRGB[2] * grayWorldAssum[2];
			}
		}
	}

	cv::namedWindow("hehe2");
	cv::imshow("hehe2", grayWorldImg);
	cv::waitKey(0);
#endif
}
#endif