#pragma once

//#define USE_GPU 

#include "Header.h"
#include "gms_matcher.h"

class xGMSManager
{
public:
	xGMSManager()
	{

	}

	void addImg(cv::Mat &keyColorImg)
	{
		keyColorImgVec.push_back(keyColorImg.clone());
	}

	void GmsMatch(std::vector<float> &matchingPointsPosVec, int srcInd, int targetInd) 
	{
		vector<KeyPoint> kp1, kp2;
		Mat d1, d2;
		vector<DMatch> matches_all;

#if 0
		imshow("show", keyColorImgVec[srcInd]);
		waitKey(0);
		imshow("show", keyColorImgVec[targetInd]);
		waitKey(0);
#endif

		cv::Ptr<ORB> orb = ORB::create(500);
		orb->setFastThreshold(0);
		orb->detectAndCompute(keyColorImgVec[srcInd], Mat(), kp1, d1);
		orb->detectAndCompute(keyColorImgVec[targetInd], Mat(), kp2, d2);

#ifdef USE_GPU
		GpuMat gd1(d1), gd2(d2);
		Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
		matcher->match(gd1, gd2, matches_all);
#else
		BFMatcher matcher(NORM_HAMMING);
		matcher.match(d1, d2, matches_all);
#endif

		// GMS filter
		int num_inliers = 0;
		std::vector<bool> vbInliers;
		gms_matcher gms(kp1, keyColorImgVec[srcInd].size(), kp2, keyColorImgVec[targetInd].size(), matches_all);
		num_inliers = gms.GetInlierMask(vbInliers, false, false);

		cout << "Get total " << num_inliers << " matches." << endl;

		// draw matches
		matchingPointsPosVec.clear();
		for (size_t i = 0; i < vbInliers.size(); ++i)
		{
			if (vbInliers[i] == true)
			{
				matchingPointsPosVec.push_back(kp1[matches_all[i].queryIdx].pt.x);
				matchingPointsPosVec.push_back(kp1[matches_all[i].queryIdx].pt.y);
			}
		}
		for (size_t i = 0; i < vbInliers.size(); ++i)
		{
			if (vbInliers[i] == true)
			{
				matchingPointsPosVec.push_back(kp2[matches_all[i].trainIdx].pt.x);
				matchingPointsPosVec.push_back(kp2[matches_all[i].trainIdx].pt.y);
			}
		}	

#if 1
		Mat output(keyColorImgVec[srcInd].rows, keyColorImgVec[srcInd].cols * 2, CV_8UC3, Scalar(0, 0, 0));
		keyColorImgVec[srcInd].copyTo(output(Rect(0, 0, keyColorImgVec[srcInd].cols, keyColorImgVec[srcInd].rows)));
		keyColorImgVec[targetInd].copyTo(output(Rect(keyColorImgVec[srcInd].cols, 0, keyColorImgVec[targetInd].cols, keyColorImgVec[targetInd].rows)));
		for (size_t i = 0; i < vbInliers.size(); ++i)
		{
			if (vbInliers[i] == true)
			{
				Point2f left = kp1[matches_all[i].queryIdx].pt;
				Point2f right = (kp2[matches_all[i].trainIdx].pt + Point2f((float)keyColorImgVec[srcInd].cols, 0.f));
				line(output, left, right, Scalar(0, 255, 255));
			}
		}
		imshow("show", output);
		waitKey(1);
#endif
	}

private:
	std::vector<cv::Mat> keyColorImgVec;
};



