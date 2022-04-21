#pragma once

//#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include "SiftGPU/SiftGPU.h"
#include "SiftGPU/SiftMatch.h"
#include "InnorealTimer.hpp"
//#include "SiftGPU/MatrixConversion.h"
//#include "SiftGPU/SIFTMatchFilter.h"
#include "SiftGPU/SiftCameraParams.h"

class xSiftManager
{
public:
	xSiftManager() : m_ind(0), m_siftDevice(NULL), m_siftMatcherDevice(NULL)
	{
		m_siftDevice = new SiftGPU;
		m_siftDevice->SetParams(COLOR_WIDTH, COLOR_HEIGHT, false, 512, 1, 65535);
		m_siftDevice->InitSiftGPU();
		m_siftMatcherDevice = new SiftMatchGPU(1024);
		m_siftMatcherDevice->InitSiftMatch();

		checkCudaErrors(cudaMalloc(&m_imagePairMatch.d_numMatches, sizeof(int)));
		checkCudaErrors(cudaMalloc(&m_imagePairMatch.d_distances, sizeof(float) * 1024));
		checkCudaErrors(cudaMalloc(&m_imagePairMatch.d_keyPointIndices, sizeof(uint2) * 1024));

		m_numKeyPointsVec.resize(1024);
		m_siftImgDeviceVec.resize(1024, NULL);
		m_keyPointsVec.resize(1024);
		m_keyPointsPosVec.resize(MAX_FRAG_NUM);
		for (int i = 0; i < MAX_FRAG_NUM; ++i)
		{
			m_keyPointsPosVec[i].resize(1024 * 2);
		}
		m_keyPointIndices.resize(1024 * 2);
	}

	~xSiftManager()
	{
		if (m_siftDevice != NULL) delete m_siftDevice;
		if (m_siftMatcherDevice != NULL) delete m_siftMatcherDevice;

		for (int i = 0; i < m_siftImgDeviceVec.size(); ++i)
		{
			if (m_siftImgDeviceVec[i] != NULL)
			{
				delete m_siftImgDeviceVec[i];
			}
		}
	}

	void addImg(float *grayImgArrayDevice, float *depthImgArrayDevice)
	{
		int success = m_siftDevice->RunSIFT(grayImgArrayDevice, depthImgArrayDevice);
		if (!success) throw std::exception("Error running SIFT detection");

		m_siftImageDevice = new SIFTImageGPU;
		checkCudaErrors(cudaMalloc(&m_siftImageDevice->d_keyPoints, sizeof(SIFTKeyPoint) * 1024));
		checkCudaErrors(cudaMalloc(&m_siftImageDevice->d_keyPointDescs, sizeof(SIFTKeyPointDesc) * 1024));

		m_numKeypoints = m_siftDevice->GetKeyPointsAndDescriptorsCUDA(*m_siftImageDevice, depthImgArrayDevice, 1024);
		m_numKeyPointsVec[m_ind] = m_numKeypoints;
		m_siftImgDeviceVec[m_ind] = m_siftImageDevice;

		checkCudaErrors(cudaMemcpy(m_keyPointsVec.data(), m_siftImageDevice->d_keyPoints,
			sizeof(SIFTKeyPoint) * m_numKeypoints, cudaMemcpyDeviceToHost));
		for (int i = 0; i < m_numKeypoints; ++i)
		{
			m_keyPointsPosVec[m_ind][2 * i] = m_keyPointsVec[i].pos.x;
			m_keyPointsPosVec[m_ind][2 * i + 1] = m_keyPointsVec[i].pos.y;
		}

		++m_ind;
	}

	void matchSift(std::vector<float> &matchingPointsPosVec, int srcInd, int targetInd)
	{
		matchingPointsPosVec.clear();

		m_siftMatcherDevice->SetDescriptors(0, m_numKeyPointsVec[srcInd], (unsigned char*)m_siftImgDeviceVec[srcInd]->d_keyPointDescs);
		m_siftMatcherDevice->SetDescriptors(1, m_numKeyPointsVec[targetInd], (unsigned char*)m_siftImgDeviceVec[targetInd]->d_keyPointDescs);

		std::cout << m_numKeyPointsVec[srcInd] << " : " << m_numKeyPointsVec[targetInd] << std::endl;
		m_siftMatcherDevice->GetSiftMatch(1024, m_imagePairMatch,
			make_uint2(0, 0),
			distmax, ratioMax);

		int matchNum;
		checkCudaErrors(cudaMemcpy(&matchNum, &m_imagePairMatch.d_numMatches[0], sizeof(int), cudaMemcpyDeviceToHost));
		std::cout << "matchNum: " << matchNum << std::endl;
		checkCudaErrors(cudaMemcpy(m_keyPointIndices.data(), m_imagePairMatch.d_keyPointIndices, matchNum * sizeof(uint2), cudaMemcpyDeviceToHost));

		uint srcMatchingInd, targetMatchingInd;
		for (int matchInd = 0; matchInd < matchNum; ++matchInd)
		{
			srcMatchingInd = m_keyPointIndices[2 * matchInd];

			matchingPointsPosVec.push_back(m_keyPointsPosVec[srcInd][2 * srcMatchingInd]);
			matchingPointsPosVec.push_back(m_keyPointsPosVec[srcInd][2 * srcMatchingInd + 1]);
		}
		for (int matchInd = 0; matchInd < matchNum; ++matchInd)
		{
			targetMatchingInd = m_keyPointIndices[2 * matchInd + 1];

			matchingPointsPosVec.push_back(m_keyPointsPosVec[targetInd][2 * targetMatchingInd]);
			matchingPointsPosVec.push_back(m_keyPointsPosVec[targetInd][2 * targetMatchingInd + 1]);
		}
	}

public:
	SiftGPU *m_siftDevice;
	SiftMatchGPU *m_siftMatcherDevice;
	SIFTImageGPU *m_siftImageDevice;
	int m_numKeypoints;
	ImagePairMatch m_imagePairMatch;

	std::vector<int> m_numKeyPointsVec;
	std::vector<SIFTImageGPU *> m_siftImgDeviceVec;

	std::vector<SIFTKeyPoint> m_keyPointsVec;
	std::vector<std::vector<float> > m_keyPointsPosVec;
	std::vector<uint> m_keyPointIndices;

	float distmax = 0.7f, ratioMax = 0.7;
	int m_ind;
};



