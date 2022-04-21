#include "stdafx.h"

#include "GmsGPU/GmsMatcher.h"
#include "GmsGPU/BFMatcher.cuh"
#include "Helpers/xUtils.h"
#include "Helpers/InnorealTimer.hpp"
#include "SiftGPU/xSift.h"

#include <pcl/features/normal_3d.h>

GmsMatcherGPU::GmsMatcherGPU(int maxFeaturePointNum)
	: m_thresFactor(6.0),
	  m_prev(0),
	  m_cur(1)
{
	m_fpfh[0] = pcl::PointCloud<pcl::FPFHSignature33>::Ptr(new pcl::PointCloud<pcl::FPFHSignature33>());
	m_fpfh[1] = pcl::PointCloud<pcl::FPFHSignature33>::Ptr(new pcl::PointCloud<pcl::FPFHSignature33>());

	int width = Resolution::getInstance().width(),
	    height = Resolution::getInstance().height();

	m_maxFeaturePointNum = maxFeaturePointNum;

	m_pOrb = cv::cuda::ORB::create(m_maxFeaturePointNum);
	m_pOrb->setFastThreshold(8);
	m_estFpfh.setNumberOfThreads(4); //4 thread omp

	m_pMatcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
	m_pMatcherOrb = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
	m_pMatcherFPFH = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
	m_pMatcherFPFHCPU = cv::BFMatcher(cv::NORM_L2, true);

	m_matchesAll.reserve(m_maxFeaturePointNum);
	m_vbInliers.reserve(m_maxFeaturePointNum);
	m_keyPoint[0].resize(m_maxFeaturePointNum);
	m_keyPoint[1].resize(m_maxFeaturePointNum);
	m_keyPointOrb[0].resize(m_maxFeaturePointNum);
	m_keyPointOrb[1].resize(m_maxFeaturePointNum);
	m_keyPointFPFH[0].resize(m_maxFeaturePointNum);
	m_keyPointFPFH[1].resize(m_maxFeaturePointNum);

	// Grid initialize
	mGridSizeLeft = cv::Size(20, 20);
	mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

	// Initialize the neihbor of left grid
	mGridNeighborLeft = cv::Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
	InitalizeNiehbors(mGridNeighborLeft, mGridSizeLeft);
}

GmsMatcherGPU::~GmsMatcherGPU()
{
}

void GmsMatcherGPU::next()
{
	std::swap(m_prev, m_cur);
}

void GmsMatcherGPU::computeOnlyOrbFeatures(cv::cuda::GpuMat& dGrayImg, cv::cuda::GpuMat& dMask)
{
	next();
	m_pOrb->setFastThreshold(8);
	m_pOrb->setNLevels(3); // for speed
	m_pOrb->detectAndCompute(dGrayImg, dMask, m_keyPoint[m_cur], m_desc[m_cur]);
	NormalizePoints(m_keyPoint[m_cur], dGrayImg.size(), m_normalizedPoint[m_cur]);
}

void GmsMatcherGPU::computeOrbFeatures(cv::cuda::GpuMat& dGrayImg, cv::cuda::GpuMat& dMask, int idx)
{
	//m_pOrb->setEdgeThreshold(31);
	//m_pOrb->setNLevels(8); // for speed
	cv::Ptr<cv::cuda::ORB> pOrb = cv::cuda::ORB::create();
	pOrb->detectAndCompute(dGrayImg, dMask, m_keyPointOrb[idx], m_descOrb[idx]);
#if 0
	cv::Mat grayImg;
	dGrayImg.download(grayImg);
	cv::drawKeypoints(grayImg, m_keyPointOrb[idx], grayImg);
	cv::imshow("grayImg", grayImg);
	cv::waitKey(0);
	std::cout << "m_descOrb[0].type: " << m_descOrb[idx].type() << std::endl;
	std::cout << "m_descOrb[0].rows: " << m_descOrb[idx].rows << std::endl;
	std::cout << "m_descOrb[0].cols: " << m_descOrb[idx].cols << std::endl;
#endif

#if 0
	cv::Mat testGrayImg;
	dGrayImg.download(testGrayImg);
	for (int i = 0; i < m_keyPoint[m_cur].size(); ++i)
	{
		cv::circle(testGrayImg, m_keyPointOrb[m_cur][i].pt, 2, cv::Scalar(0, 255, 255));
	}
	cv::namedWindow("hehe");
	cv::imshow("hehe", testGrayImg);
	cv::waitKey(0);
#endif
#if 0
	NormalizePoints(m_keyPointOrb[m_cur], dGrayImg.size(), m_normalizedPointOrb[m_cur]);
#endif

#if 0
	for (int i = 0; i < m_keyPoint[m_cur].size(); ++i)
	{
		cv::circle(img, m_keyPoint[m_cur][i].pt, 2, cv::Scalar(0, 255, 255));
	}
	cv::namedWindow("hehe");
	cv::imshow("hehe", img);
	cv::waitKey(0);
#if 0
	char renderedDir[256];
	std::vector<int> pngCompressionParams;
	pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	pngCompressionParams.push_back(0);
	sprintf(renderedDir, "D:\\xjm\\result\\for_demo\\test\\%06d.png", 0);
	cv::imwrite(renderedDir, img, pngCompressionParams);
	cv::waitKey(0);
	std::exit(0);
#endif
#endif
}

void GmsMatcherGPU::computeFPFHFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud,
                                        pcl::PointCloud<pcl::Normal>::Ptr pointCloudNormal,
                                        int* keyPoints, int width, int height, int idx)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree4(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::PointCloud<pcl::Normal>::Ptr pointNormal(new pcl::PointCloud<pcl::Normal>);
	//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> est_normal;
	//est_normal.setInputCloud(pointCloud);
	//est_normal.setSearchMethod(tree4);
	//est_normal.setKSearch(8);
	//est_normal.compute(*pointNormal);

	m_estFpfh.setInputCloud(pointCloud);
	//m_estFpfh.setInputNormals(pointNormal);
	m_estFpfh.setInputNormals(pointCloudNormal);
	m_estFpfh.setSearchMethod(tree4);
	m_estFpfh.setKSearch(8);
	m_estFpfh.compute(*m_fpfh[idx]);
	//m_pointClouds[idx] = pointCloud;
	//m_pointNormals[idx] = pointNormal;

	std::vector<cv::KeyPoint>& keyPoint = m_keyPointFPFH[idx];
	int keyPointNum = pointCloud->size();
	keyPoint.resize(keyPointNum);
	for (size_t i = 0; i < keyPointNum; ++i)
	{
		int n = keyPoints[i];
		keyPoint[i].pt.x = n % width;
		keyPoint[i].pt.y = n / width;
	}

	m_descFPFH[idx].create(m_fpfh[idx]->size(), 33, CV_32FC1);

#if 1
	checkCudaErrors(cudaMemcpy2D(m_descFPFH[idx].data, m_descFPFH[idx].step, m_fpfh[idx]->points.data(),
		sizeof(pcl::FPFHSignature33), sizeof(float) * 33, m_fpfh[idx]->size(), cudaMemcpyHostToDevice));
#endif
#if 0
	cv::Mat descFPFH(m_fpfh[idx]->size(), 33, CV_32FC1);
	for (int i = 0; i < m_fpfh[idx]->size(); ++i)
	{
		for (int j = 0; j < 33; ++j)
		{
			descFPFH.at<float>(i, j) = m_fpfh[idx]->at(i).histogram[j];
		}
	}
	m_descFPFH[idx].upload(descFPFH);
#endif
#if 0
	cv::Mat descFPFH;
	m_descFPFH[idx].download(descFPFH);
	for (int r = 0; r < 10; ++r)
	{ 
		for (int c = 0; c < 33; ++c)
		{
			std::cout << descFPFH.at<float>(r, c) << ", ";
		}
		std::cout << std::endl << "-----" << std::endl;
	}
	std::exit(0);
#endif
#if 0
	NormalizePoints(m_keyPointFPFH[idx], cv::Size(width, height), m_normalizedPointFPFH[idx]);
#endif
}

void GmsMatcherGPU::getMatchesOrb(std::vector<ushort>& matchesVec)
{
	m_keyPoint[0] = m_keyPointOrb[0];
	m_keyPoint[1] = m_keyPointOrb[1];
	m_normalizedPoint[0] = m_normalizedPointOrb[0];
	m_normalizedPoint[1] = m_normalizedPointOrb[1];
	m_desc[0] = m_descOrb[0];
	m_desc[1] = m_descOrb[1];

#if 0
	cv::Mat desc1, desc2;
	m_desc[0].download(desc1);
	m_desc[1].download(desc2);
	for (int i = 0; i < 20; ++i)
	{
		//std::cout << desc1.at<float>(0, i) << ", ";
		std::cout << m_keyPoint[0][i].pt.x << ", ";
	}
	std::cout << std::endl;
	for (int i = 0; i < 20; ++i)
	{
		//std::cout << desc2.at<float>(0, i) << ", ";
		std::cout << m_keyPoint[1][i].pt.x << ", ";
	}
	std::cout << std::endl;
	std::exit(0);
#endif

	m_pMatcherOrb->match(m_desc[m_prev], m_desc[m_cur], m_matchesAll);

	// GMS filter
	mNumberMatches = m_matchesAll.size();
	ConvertMatches(m_matchesAll, mvMatches);

	m_matchNum = GetInlierMask(m_vbInliers, false, false);

	cv::Point2f left_point, right_point;
	for (size_t i = 0; i < m_vbInliers.size(); ++i)
	{
		if (m_vbInliers[i] == true)
		{
#if 1
			left_point = m_keyPoint[m_prev][m_matchesAll[i].queryIdx].pt;
			right_point = m_keyPoint[m_cur][m_matchesAll[i].trainIdx].pt;
			matchesVec.push_back(ushort(left_point.y));
			matchesVec.push_back(ushort(left_point.x));
			matchesVec.push_back(ushort(right_point.y));
			matchesVec.push_back(ushort(right_point.x));
#endif
		}
	}
}

void GmsMatcherGPU::getMultiMatchesOrb(std::vector<cv::DMatch>& matchVec,
																			 std::vector<int>& matchesIdxVec,
                                       std::vector<int>& matchesDistVec,
                                       std::vector<cv::KeyPoint>& src,
                                       std::vector<cv::KeyPoint>& target)
{
	int* dMatchesIdx;
	int* dMatchesDist;	
	checkCudaErrors(cudaMalloc(&dMatchesIdx, m_descOrb[0].rows * 2 * sizeof(int)));
	checkCudaErrors(cudaMalloc(&dMatchesDist, m_descOrb[0].rows * 2 * sizeof(int)));
	OrbBFMatch(dMatchesIdx, dMatchesDist, m_descOrb[0], m_descOrb[1]);

	matchesIdxVec.resize(m_descOrb[0].rows * 2);
	matchesDistVec.resize(m_descOrb[0].rows * 2);
	checkCudaErrors(cudaMemcpy(matchesIdxVec.data(), dMatchesIdx,
		matchesIdxVec.size() * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(matchesDistVec.data(), dMatchesDist,
		matchesDistVec.size() * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(dMatchesIdx));
	checkCudaErrors(cudaFree(dMatchesDist));

	m_pMatcherOrb->match(m_descOrb[0], m_descOrb[1], m_matchesAll);
	matchVec = m_matchesAll;

	src = m_keyPointOrb[0];
	target = m_keyPointOrb[1];
}

void GmsMatcherGPU::getMatchesOrbFPFH(std::vector<ushort>& matchesVec, int width, int height)
{
	NormalizePoints(m_keyPoint[0], cv::Size(width, height), m_normalizedPoint[0]);
	NormalizePoints(m_keyPoint[1], cv::Size(width, height), m_normalizedPoint[1]);

	mNumberMatches = m_matchesAll.size();
	ConvertMatches(m_matchesAll, mvMatches);

	m_matchNum = GetInlierMask(m_vbInliers, false, false);

	cv::Point2f left_point, right_point;
	//for (size_t i = 0; i < m_vbInliers.size(); ++i)
	for (size_t i = 0; i < m_matchesAll.size(); ++i)
	{
		if (m_vbInliers[i] == true)
		{
			left_point = m_keyPoint[0][m_matchesAll[i].queryIdx].pt;
			right_point = m_keyPoint[1][m_matchesAll[i].trainIdx].pt;
			matchesVec.push_back(ushort(left_point.y));
			matchesVec.push_back(ushort(left_point.x));
			matchesVec.push_back(ushort(right_point.y));
			matchesVec.push_back(ushort(right_point.x));
		}
	}
}

void GmsMatcherGPU::getMatchesFPFH(std::vector<ushort>& matchesVec)
{
#if 0
	m_pMatcherFPFH->match(m_descFPFH[0], m_descFPFH[1], m_matchesAll);
#endif
#if 0
	cv::Mat desc0, desc1;
	m_descFPFH[0].download(desc0);
	m_descFPFH[1].download(desc1);
	m_pMatcherFPFHCPU.match(desc0, desc1, m_matchesAll);
#endif

#if 0
	int ptCount = (int)m_matchesAll.size();
	cv::Mat p1(ptCount, 2, CV_32F);
	cv::Mat p2(ptCount, 2, CV_32F);

	// °ÑKeypoint×ª»»ÎªMat
	cv::Point2f pt;
	for (int i = 0; i<ptCount; i++)
	{
		pt = m_keyPointFPFH[0][m_matchesAll[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = m_keyPointFPFH[1][m_matchesAll[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}
#endif

#if 0
	std::vector<uchar> m_RANSACStatus;
	cv::Mat fundamental = cv::findFundamentalMat(p1, p2, m_RANSACStatus, cv::FM_RANSAC);
#endif

#if 0
	// GMS filter
	mNumberMatches = m_matchesAll.size();
	ConvertMatches(m_matchesAll, mvMatches);

	m_matchNum = GetInlierMask(m_vbInliers, false, false);
#endif

#if 0
	cv::Point2f left_point, right_point;
	int size1 = m_keyPointFPFH[0].size(), size2 = m_keyPointFPFH[1].size();
	std::cout << "size1: " << size1 << std::endl;
	std::cout << "size2: " << size2 << std::endl;
	for (size_t i = 0; i < std::min(size1, size2); ++i)
	{
		left_point = m_keyPointFPFH[0][i].pt;
		right_point = m_keyPointFPFH[1][i].pt;
		matchesVec.push_back(ushort(left_point.y));
		matchesVec.push_back(ushort(left_point.x));
		matchesVec.push_back(ushort(right_point.y));
		matchesVec.push_back(ushort(right_point.x));
	}
#endif
#if 1
	cv::Point2f left_point, right_point;
	//for (size_t i = 0; i < m_vbInliers.size(); ++i)
	for (size_t i = 0; i < m_matchesAll.size(); ++i)
	{
#if 1
		//if (m_vbInliers[i] == true)
		//if (m_RANSACStatus[i] != 0)
		//if (i % 100 == 0)
#endif
		{
			left_point = m_keyPointFPFH[0][m_matchesAll[i].queryIdx].pt;
			right_point = m_keyPointFPFH[1][m_matchesAll[i].trainIdx].pt;
			matchesVec.push_back(ushort(left_point.y));
			matchesVec.push_back(ushort(left_point.x));
			matchesVec.push_back(ushort(right_point.y));
			matchesVec.push_back(ushort(right_point.x));
		}
	}
#endif
}

void GmsMatcherGPU::getMultiMatchesFPFH(std::vector<int>& matchesIdxVec,
                                        std::vector<float>& matchesDistVec,
                                        std::vector<cv::KeyPoint>& src,
                                        std::vector<cv::KeyPoint>& target)
{
	int* dMatchesIdx;
	float* dMatchesDist;
	checkCudaErrors(cudaMalloc(&dMatchesIdx, m_descFPFH[0].rows * 2 * sizeof(int)));
	checkCudaErrors(cudaMalloc(&dMatchesDist, m_descFPFH[0].rows * 2 * sizeof(float)));
	NormalizeFPFH(m_descFPFH[0],
	              m_descFPFH[1]);
#if 0
	cv::Mat descFPFH;
	m_descFPFH[0].download(descFPFH);
	for (int i = 0; i < 5; ++i)
	{
		for (int j = 0; j < descFPFH.cols; ++j)
		{
			std::cout << descFPFH.at<float>(i, j) << ", ";
		}
	}
	std::cout << std::endl;
	std::cout << std::endl;
	m_descFPFH[1].download(descFPFH);
	for (int i = 0; i < 5; ++i)
	{
		for (int j = 0; j < descFPFH.cols; ++j)
		{
			std::cout << descFPFH.at<float>(i, j) << ", ";
		}
	}
	std::cout << std::endl;
	std::cout << std::endl;
	std::exit(0);
#endif
	FPFHBFMatch(dMatchesIdx,
	            dMatchesDist,
	            m_descFPFH[0],
	            m_descFPFH[1]);

	matchesIdxVec.resize(m_descFPFH[0].rows * 2);
	matchesDistVec.resize(m_descFPFH[0].rows * 2);
	checkCudaErrors(cudaMemcpy(matchesIdxVec.data(), dMatchesIdx,
		matchesIdxVec.size() * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(matchesDistVec.data(), dMatchesDist,
		matchesDistVec.size() * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(dMatchesIdx));
	checkCudaErrors(cudaFree(dMatchesDist));

	src = m_keyPointFPFH[0];
	target = m_keyPointFPFH[1];
}

void GmsMatcherGPU::calcFPFHForOrbMatches(std::vector<float>& matchesDistVec, 
										  std::vector<int>& coordinateVec,
										  int width,
										  int height)
{
	cv::Mat srcFPFHIdxMat(height, width, CV_32SC1);
	cv::Mat targetFPFHIdxMat(height, width, CV_32SC1);
	for (int i = 0; i < m_keyPointFPFH[0].size(); ++i)
	{
		srcFPFHIdxMat.at<int>(m_keyPointFPFH[0][i].pt.y, m_keyPointFPFH[0][i].pt.x) = i;
	}
	for (int i = 0; i < m_keyPointFPFH[1].size(); ++i)
	{
		targetFPFHIdxMat.at<int>(m_keyPointFPFH[1][i].pt.y, m_keyPointFPFH[1][i].pt.x) = i;
	}
	cv::cuda::GpuMat dSrcFPFHIdxMat, dTargetFPFHIdxMat;
	dSrcFPFHIdxMat.upload(srcFPFHIdxMat);
	dTargetFPFHIdxMat.upload(targetFPFHIdxMat);

	int* dMatchesIdx;
	float* dMatchesDist;
	checkCudaErrors(cudaMalloc(&dMatchesIdx, coordinateVec.size() * sizeof(int)));
	checkCudaErrors(cudaMalloc(&dMatchesDist, coordinateVec.size() / 4 * sizeof(float)));
	NormalizeFPFH(m_descFPFH[0],
				  m_descFPFH[1]);

	cudaMemcpy(dMatchesIdx, coordinateVec.data(), coordinateVec.size() * sizeof(int), cudaMemcpyHostToDevice);
	CalcFPFHForOrbMatches(dMatchesDist,
						  dMatchesIdx,
						  coordinateVec.size() / 4,
						  dSrcFPFHIdxMat,
						  dTargetFPFHIdxMat,
						  m_descFPFH[0],
						  m_descFPFH[1]);
	cudaMemcpy(matchesDistVec.data(), dMatchesDist, coordinateVec.size() / 4 * sizeof(float), cudaMemcpyDeviceToHost);

}

void GmsMatcherGPU::getMatches(std::vector<ushort>& matchesVec, std::vector<cv::Mat>& colorImgVec)
{
	m_pMatcher->match(m_desc[m_prev], m_desc[m_cur], m_matchesAll);

	// GMS filter
	mNumberMatches = m_matchesAll.size();
	ConvertMatches(m_matchesAll, mvMatches);

	m_matchNum = GetInlierMask(m_vbInliers, false, false);

#if 0 // debug
	printf("num_matches: %d\n", m_matchNum);
#endif

	matchesVec.clear();
	cv::Point2f left_point, right_point;
	for (size_t i = 0; i < m_vbInliers.size(); ++i)
	{ 
		if (m_vbInliers[i] == true)
		{
			left_point = m_keyPoint[m_prev][m_matchesAll[i].queryIdx].pt;
			right_point = m_keyPoint[m_cur][m_matchesAll[i].trainIdx].pt;
			matchesVec.push_back(ushort(left_point.y));
			matchesVec.push_back(ushort(left_point.x));
			matchesVec.push_back(ushort(right_point.y));
			matchesVec.push_back(ushort(right_point.x));
		}
	}
	if (matchesVec.size() == 0)
	{
		for (size_t i = 0; i < m_matchesAll.size(); ++i)
		{
			left_point = m_keyPoint[m_prev][m_matchesAll[i].queryIdx].pt;
			right_point = m_keyPoint[m_cur][m_matchesAll[i].trainIdx].pt;
			matchesVec.push_back(ushort(left_point.y));
			matchesVec.push_back(ushort(left_point.x));
			matchesVec.push_back(ushort(right_point.y));
			matchesVec.push_back(ushort(right_point.x));
		}
	}

#if 0 // debug
	std::cout << "matchesVec size: " << matchesVec.size() << std::endl;
	cv::Mat show = DrawMatches(colorImgVec[colorImgVec.size() - 2], 
		colorImgVec[colorImgVec.size() - 1], matchesVec);
	cv::imshow("show", show);
	cv::waitKey(1);
#endif
}

cv::Mat GmsMatcherGPU::DrawMatches(cv::Mat& src1, cv::Mat& src2, std::vector<ushort>& matches)
{
	const int height = std::max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
	src1.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));

	cv::Vec2w temp;
	for (int i = 0; i < matches.size(); i += 4)
	{
		//if (i % 10 == 0)
		{
			cv::Point2f left = cv::Point2f(matches[i + 1], matches[i]);
			cv::Point2f right = cv::Point2f(matches[i + 3] + src1.cols, matches[i + 2]);
			line(output, left, right, cv::Scalar(0, 255, 255));
#if 1
			cv::circle(output, left, 1, cv::Scalar(255, 0, 0), 2);
			cv::circle(output, right, 1, cv::Scalar(0, 255, 0), 2);
#endif
		}
	}
	//std::vector<int> compression_params;
	//compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	//compression_params.push_back(0);
	//cv::imwrite("D:\\xjm\\result\\for_demo\\input_for_vis_tracking\\hehe.png", output, compression_params);
	//std::exit(0);
	return output;
}

cv::Mat GmsMatcherGPU::DrawMatches2(cv::Mat& src1, cv::Mat& src2, std::vector<ushort>& matches1, std::vector<ushort>& matches2)
{
    const int height = std::max(src1.rows, src2.rows);
    const int width = src1.cols + src2.cols;
    cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    src1.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
    src2.copyTo(output(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));

    for (int i = 0; i < matches1.size(); i += 4)
    {
        cv::Point2f left = cv::Point2f(matches1[i + 1], matches1[i]);
        cv::Point2f right = cv::Point2f(matches1[i + 3] + src1.cols, matches1[i + 2]);
        line(output, left, right, cv::Scalar(0, 0, 255));
        cv::circle(output, left, 1, cv::Scalar(255, 0, 0), 2);
        cv::circle(output, right, 1, cv::Scalar(0, 255, 0), 2);
    }
    for (int i = 0; i < matches2.size(); i += 4)
    {
        cv::Point2f left = cv::Point2f(matches2[i + 1], matches2[i]);
        cv::Point2f right = cv::Point2f(matches2[i + 3] + src1.cols, matches2[i + 2]);
        line(output, left, right, cv::Scalar(0, 255, 255));
        cv::circle(output, left, 1, cv::Scalar(255, 0, 0), 2);
        cv::circle(output, right, 1, cv::Scalar(0, 255, 0), 2);
    }
    //std::vector<int> compression_params;
    //compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    //compression_params.push_back(0);
    //cv::imwrite("D:\\xjm\\result\\for_demo\\input_for_vis_tracking\\hehe.png", output, compression_params);
    //std::exit(0);
    return output;
}

// utility
inline cv::Mat DrawInlier(cv::Mat& src1, cv::Mat& src2, std::vector<cv::KeyPoint>& kpt1,
                          std::vector<cv::KeyPoint>& kpt2, std::vector<cv::DMatch>& inlier, int type)
{
	const int height = std::max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
	src1.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));

	if (type == 1)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
			cv::Point2f right = (kpt2[inlier[i].trainIdx].pt + cv::Point2f((float)src1.cols, 0.f));
			line(output, left, right, cv::Scalar(0, 255, 255));
		}
	}
	else if (type == 2)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
			cv::Point2f right = (kpt2[inlier[i].trainIdx].pt + cv::Point2f((float)src1.cols, 0.f));
			line(output, left, right, cv::Scalar(255, 0, 0));
		}

		for (size_t i = 0; i < inlier.size(); i++)
		{
			cv::Point2f left = kpt1[inlier[i].queryIdx].pt;
			cv::Point2f right = (kpt2[inlier[i].trainIdx].pt + cv::Point2f((float)src1.cols, 0.f));
			circle(output, left, 1, cv::Scalar(0, 255, 255), 2);
			circle(output, right, 1, cv::Scalar(0, 255, 0), 2);
		}
	}

	return output;
}

int GmsMatcherGPU::GetGridIndexLeft(const cv::Point2f& pt, int type)
{
	int x = 0, y = 0;

	if (type == 1)
	{
		x = floor(pt.x * mGridSizeLeft.width);
		y = floor(pt.y * mGridSizeLeft.height);
	}

	if (type == 2)
	{
		x = floor(pt.x * mGridSizeLeft.width + 0.5);
		y = floor(pt.y * mGridSizeLeft.height);
	}

	if (type == 3)
	{
		x = floor(pt.x * mGridSizeLeft.width);
		y = floor(pt.y * mGridSizeLeft.height + 0.5);
	}

	if (type == 4)
	{
		x = floor(pt.x * mGridSizeLeft.width + 0.5);
		y = floor(pt.y * mGridSizeLeft.height + 0.5);
	}

	if (x >= mGridSizeLeft.width || y >= mGridSizeLeft.height)
	{
		return -1;
	}

	return x + y * mGridSizeLeft.width;
}

int GmsMatcherGPU::GetGridIndexRight(const cv::Point2f& pt)
{
	int x = floor(pt.x * mGridSizeRight.width);
	int y = floor(pt.y * mGridSizeRight.height);

	return x + y * mGridSizeRight.width;
}

std::vector<int> GmsMatcherGPU::GetNB9(const int idx, const cv::Size& GridSize)
{
	std::vector<int> NB9(9, -1);

	int idx_x = idx % GridSize.width;
	int idx_y = idx / GridSize.width;

	for (int yi = -1; yi <= 1; yi++)
	{
		for (int xi = -1; xi <= 1; xi++)
		{
			int idx_xx = idx_x + xi;
			int idx_yy = idx_y + yi;

			if (idx_xx < 0 || idx_xx >= GridSize.width || idx_yy < 0 || idx_yy >= GridSize.height)
				continue;

			NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize.width;
		}
	}
	return NB9;
}

void GmsMatcherGPU::InitalizeNiehbors(cv::Mat& neighbor, const cv::Size& GridSize)
{
	for (int i = 0; i < neighbor.rows; i++)
	{
		std::vector<int> NB9 = GetNB9(i, GridSize);
		int* data = neighbor.ptr<int>(i);
		memcpy(data, &NB9[0], sizeof(int) * 9);
	}
}

void GmsMatcherGPU::SetScale(int Scale)
{
	// Set Scale
	mGridSizeRight.width = mGridSizeLeft.width * mScaleRatios[Scale];
	mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios[Scale];
	mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;

	// Initialize the neihbor of right grid
	mGridNeighborRight = cv::Mat::zeros(mGridNumberRight, 9, CV_32SC1);
	InitalizeNiehbors(mGridNeighborRight, mGridSizeRight);
}

int GmsMatcherGPU::GetInlierMask(std::vector<bool>& vbInliers, bool WithScale, bool WithRotation)
{
	int max_inlier = 0;

	if (!WithScale && !WithRotation)
	{
		SetScale(0);
		max_inlier = run(1);
		vbInliers = mvbInlierMask;
		return max_inlier;
	}

	if (WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);
			for (int RotationType = 1; RotationType <= 8; RotationType++)
			{
				int num_inlier = run(RotationType);

				if (num_inlier > max_inlier)
				{
					vbInliers = mvbInlierMask;
					max_inlier = num_inlier;
				}
			}
		}
		return max_inlier;
	}

	if (WithRotation && !WithScale)
	{
		for (int RotationType = 1; RotationType <= 8; RotationType++)
		{
			int num_inlier = run(RotationType);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}
		}
		return max_inlier;
	}

	if (!WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);

			int num_inlier = run(1);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}
		}
		return max_inlier;
	}

	return max_inlier;
}

void GmsMatcherGPU::AssignMatchPairs(int GridType)
{
	for (size_t i = 0; i < mNumberMatches; i++)
	{
		cv::Point2f& lp = m_normalizedPoint[m_prev][mvMatches[i].first];
		cv::Point2f& rp = m_normalizedPoint[m_cur][mvMatches[i].second];

		int lgidx = mvMatchPairs[i].first = GetGridIndexLeft(lp, GridType);
		int rgidx = -1;

		if (GridType == 1)
		{
			rgidx = mvMatchPairs[i].second = GetGridIndexRight(rp);
		}
		else
		{
			rgidx = mvMatchPairs[i].second;
		}

		if (lgidx < 0 || rgidx < 0) continue;

		mMotionStatistics.at<int>(lgidx, rgidx)++;
		mNumberPointsInPerCellLeft[lgidx]++;
	}
}

void GmsMatcherGPU::VerifyCellPairs(int RotationType)
{
	const int* CurrentRP = mRotationPatterns[RotationType - 1];

	for (int i = 0; i < mGridNumberLeft; i++)
	{
		if (sum(mMotionStatistics.row(i))[0] == 0)
		{
			mCellPairs[i] = -1;
			continue;
		}

		int max_number = 0;
		for (int j = 0; j < mGridNumberRight; j++)
		{
			int* value = mMotionStatistics.ptr<int>(i);
			if (value[j] > max_number)
			{
				mCellPairs[i] = j;
				max_number = value[j];
			}
		}

		int idx_grid_rt = mCellPairs[i];

		const int* NB9_lt = mGridNeighborLeft.ptr<int>(i);
		const int* NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt);

		int score = 0;
		double thresh = 0;
		int numpair = 0;

		for (size_t j = 0; j < 9; j++)
		{
			int ll = NB9_lt[j];
			int rr = NB9_rt[CurrentRP[j] - 1];
			if (ll == -1 || rr == -1) continue;

			score += mMotionStatistics.at<int>(ll, rr);
			thresh += mNumberPointsInPerCellLeft[ll];
			numpair++;
		}

		thresh = m_thresFactor * sqrt(thresh / numpair);

		if (score < thresh)
			mCellPairs[i] = -2;
	}
}

int GmsMatcherGPU::run(int RotationType)
{
	mvbInlierMask.assign(mNumberMatches, false);

	// Initialize Motion Statisctics
	mMotionStatistics = cv::Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);
	mvMatchPairs.assign(mNumberMatches, std::pair<int, int>(0, 0));

	for (int GridType = 1; GridType <= 4; GridType++)
	{
		// initialize
		mMotionStatistics.setTo(0);
		mCellPairs.assign(mGridNumberLeft, -1);
		mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);

		AssignMatchPairs(GridType);
		VerifyCellPairs(RotationType);

		// Mark inliers
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
			{
				mvbInlierMask[i] = true;
			}
		}
	}
	int num_inlier = cv::sum(mvbInlierMask)[0];
	return num_inlier;
}

void GmsMatcherGPU::ConvertMatches(const std::vector<cv::DMatch>& vDMatches, std::vector<std::pair<int, int>>& vMatches)
{
	vMatches.resize(mNumberMatches);
	for (size_t i = 0; i < mNumberMatches; i++)
	{
		vMatches[i] = std::pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
	}
}

void GmsMatcherGPU::NormalizePoints(const std::vector<cv::KeyPoint>& kp, const cv::Size& size,
                                    std::vector<cv::Point2f>& npts)
{
	const size_t numP = kp.size();
	const int width = size.width;
	const int height = size.height;
	npts.resize(numP);

	for (size_t i = 0; i < numP; i++)
	{
		npts[i].x = kp[i].pt.x / width;
		npts[i].y = kp[i].pt.y / height;
	}
}

void GmsMatcherGPU::SetThreshFactor(double _thresh_factor)
{
	m_thresFactor = _thresh_factor;
}

double GmsMatcherGPU::GetThreshFactor()
{
	return m_thresFactor;
}
