#ifndef __GMS_MATCH_HEADER__
#define __GMS_MATCH_HEADER__

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h> //包含fpfh加速计算的omp(多核并行计算)
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_features.h> //特征的错误对应关系去除
#include <pcl/registration/correspondence_rejection_sample_consensus.h> //随机采样一致性去

class GmsMatcherGPU {
public:
	GmsMatcherGPU(int maxFeaturePointNum = 1000);
	~GmsMatcherGPU();

	void next();

	void computeOnlyOrbFeatures(cv::cuda::GpuMat& dGrayImg, cv::cuda::GpuMat& dMask);

	void computeOrbFeatures(cv::cuda::GpuMat &dGrayImg, cv::cuda::GpuMat &dMask, int idx);

	void computeFPFHFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud,
	                         pcl::PointCloud<pcl::Normal>::Ptr pointCloudNormal,
	                         int* keyPoints, int width, int height, int idx);

	void getMatches(std::vector<ushort> &matchesVec, std::vector<cv::Mat> &colorImgVec);

	void getMatchesOrbFPFH(std::vector<ushort>& matchesVec, int width, int height);

	void getMultiMatchesOrb(std::vector<cv::DMatch>& matchVec,
													std::vector<int>& matchesIdxVec,
	                        std::vector<int>& matchesDistVec,
	                        std::vector<cv::KeyPoint>& src,
	                        std::vector<cv::KeyPoint>& target);
	void getMultiMatchesFPFH(std::vector<int>& matchesIdxVec,
	                         std::vector<float>& matchesDistVec,
	                         std::vector<cv::KeyPoint>& src,
	                         std::vector<cv::KeyPoint>& target);

	void calcFPFHForOrbMatches(std::vector<float>& matchesDistVec,
							   std::vector<int>& matchesIdxVec,
							   int width,
							   int height);

	void getMatchesOrb(std::vector<ushort> &matchesVec);

	void getMatchesFPFH(std::vector<ushort> &matchesVec);
	/*!
	 * Set thresh factor.
	 * @param _thresh_factor Reference paper, initialized to 6.
	 */
	void SetThreshFactor(double _thresh_factor);
	double GetThreshFactor();	

	/*!
	 * Draw matches and input images into a single mat for showing the matches.
	 * @param src1 Original input left image.
	 * @param src2 Origianl input right image.
	 * @param matches Matched matrix which can be get from GetMatchesGPU(src1, src2);
	 * @return Output image contain two input images and matched points, each pair of points is linked with line.
	 */
	cv::Mat DrawMatches(cv::Mat &src1, cv::Mat &src2, std::vector<ushort> &matches);
    cv::Mat DrawMatches2(cv::Mat& src1, cv::Mat& src2, std::vector<ushort>& matches1, std::vector<ushort>& matches2);

public:
	// Get Inlier Mask
	// Return number of inliers
	int GetInlierMask(std::vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false);

	// Normalize Key Points to Range(0 - 1)
	void NormalizePoints(const std::vector<cv::KeyPoint> &kp, const cv::Size &size, std::vector<cv::Point2f> &npts);

	// Convert OpenCV DMatch to Match (pair<int, int>)
	void ConvertMatches(const std::vector<cv::DMatch> &vDMatches, std::vector<std::pair<int, int> > &vMatches);

	int GetGridIndexLeft(const cv::Point2f &pt, int type);

	int GetGridIndexRight(const cv::Point2f &pt);

	// Assign Matches to Cell Pairs
	void AssignMatchPairs(int GridType);

	// Verify Cell Pairs
	void VerifyCellPairs(int RotationType);

	// Get Neighbor 9
	std::vector<int> GetNB9(const int idx, const cv::Size &GridSize);

	//
	void InitalizeNiehbors(cv::Mat &neighbor, const cv::Size &GridSize);

	void SetScale(int Scale);
	// Run
	int run(int RotationType);

	// cv::Matches
	std::vector<std::pair<int, int> > mvMatches;

	// Number of Matches
	size_t mNumberMatches;

	// Grid Size
	cv::Size mGridSizeLeft, mGridSizeRight;
	int mGridNumberLeft;
	int mGridNumberRight;

	// x	  : left grid idx
	// y      :  right grid idx
	// value  : how many matches from idx_left to idx_right
	cv::Mat mMotionStatistics;

	//
	std::vector<int> mNumberPointsInPerCellLeft;

	// Inldex  : grid_idx_left
	// Value   : grid_idx_right
	std::vector<int> mCellPairs;

	// Every Matches has a cell-pair
	// first  : grid_idx_left
	// second : grid_idx_right
	std::vector<std::pair<int, int> > mvMatchPairs;

	// Inlier Mask for output
	std::vector<bool> mvbInlierMask;

	//
	cv::Mat mGridNeighborLeft;
	cv::Mat mGridNeighborRight;

	cv::Ptr<cv::cuda::ORB> m_pOrb;

#if 1
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr m_fpfh[2];
	pcl::PointCloud<pcl::PointXYZ>::Ptr m_pointClouds[2];
	pcl::PointCloud<pcl::Normal>::Ptr m_pointNormals[2];
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> m_estFpfh;
#endif

	cv::Ptr<cv::cuda::DescriptorMatcher> m_pMatcher;
	cv::Ptr<cv::cuda::DescriptorMatcher> m_pMatcherFPFH;
	cv::BFMatcher m_pMatcherFPFHCPU;
	cv::Ptr<cv::cuda::DescriptorMatcher> m_pMatcherOrb;
	int m_matchNum;

	std::vector<cv::KeyPoint> m_keyPoint[2];
	std::vector<cv::Point2f> m_normalizedPoint[2];
	cv::cuda::GpuMat m_desc[2];

	std::vector<cv::KeyPoint> m_keyPointOrb[2];
	std::vector<cv::Point2f> m_normalizedPointOrb[2];
	cv::cuda::GpuMat m_descOrb[2];

	std::vector<cv::KeyPoint> m_keyPointFPFH[2];
	std::vector<cv::Point2f> m_normalizedPointFPFH[2];
	cv::cuda::GpuMat m_descFPFH[2];

	int m_prev, m_cur;

	std::vector<cv::DMatch> m_matchesAll;	
	std::vector<bool> m_vbInliers;
	int m_maxFeaturePointNum;

	double m_thresFactor;
};

// 8 possible rotation and each one is 3 X 3
const int mRotationPatterns[8][9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9,

									 4, 1, 2, 7, 5, 3, 8, 9, 6,

									 7, 4, 1, 8, 5, 2, 9, 6, 3,

									 8, 7, 4, 9, 5, 1, 6, 3, 2,

									 9, 8, 7, 6, 5, 4, 3, 2, 1,

									 6, 9, 8, 3, 5, 7, 2, 1, 4,

									 3, 6, 9, 2, 5, 8, 1, 4, 7,

									 2, 3, 6, 1, 5, 9, 4, 7, 8 };

// 5 level scales
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / std::sqrt(2.0), sqrt(2.0), 2.0 };

#endif