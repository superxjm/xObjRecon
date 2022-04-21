#pragma once

#include <stdlib.h>
#include <vector>
#include <iostream>
//#define GLEW_STATIC
//#include <GL/glew.h>

#define SIFTGPU_DLL_RUNTIME
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define FREE_MYLIB FreeLibrary
#define GET_MYPROC GetProcAddress

#include "SiftGPU.h"

#if 0

class xSiftGen
{
private:
	HMODULE m_hsiftgpu;
	SiftGPU *m_sift;
	SiftMatchGPU *m_matcher;

public:
	xSiftGen()
	{	
		HMODULE  hsiftgpu = LoadLibraryExA(".\\SiftGPU64.dll", NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
		SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
		SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int) = NULL;
		pCreateNewSiftGPU = (SiftGPU* (*) (int)) GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");
		pCreateNewSiftMatchGPU = (SiftMatchGPU* (*)(int)) GET_MYPROC(hsiftgpu, "CreateNewSiftMatchGPU");
		m_sift = pCreateNewSiftGPU(1);
		m_matcher = pCreateNewSiftMatchGPU(4096);
	}

	~xSiftGen()
	{
		delete m_sift;
		delete m_matcher;	
		FREE_MYLIB(m_hsiftgpu);
	}

	int calcSift(uchar *data, int width, int height, std::vector<SiftGPU::SiftKeypoint> &siftKeyPoints, std::vector<float> &descriptors)
	{
		int num = 0;

		char * argv[] = { "-fo", "-1",  "-v", "0" };
		//char * argv[] = { "-fo", "-1",  "", "0" };

		int argc = sizeof(argv) / sizeof(char*);
		m_sift->ParseParam(argc, argv);

		if (m_sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
		{
			return 0;
		}
	
		m_sift->SetKeypointList(siftKeyPoints.size(), siftKeyPoints.data(), 1);

		if (m_sift->RunSIFT(width, height, data, GL_RGB, GL_UNSIGNED_BYTE))
		{
			num = m_sift->GetFeatureNum();	
			descriptors.resize(128 * num);
			m_sift->GetFeatureVector(siftKeyPoints.data(), descriptors.data());
		}		

		return 0;
	}

	int matchSift(std::vector<float> &descriptors1, std::vector<float> &descriptors2, int (*&matchBuf)[2], int &numMatch)
	{
		int num1 = descriptors1.size() / 128, num2 = descriptors2.size() / 128, num = MAX(num1, num2);

		m_matcher->VerifyContextGL(); //must call once

		m_matcher->SetDescriptors(0, num1, descriptors1.data()); //image 1
		m_matcher->SetDescriptors(1, num2, descriptors2.data()); //image 2

		matchBuf = new int[num][2];
		//use the default thresholds. Check the declaration in SiftGPU.h
		numMatch = m_matcher->GetSiftMatch(num, matchBuf, 0.7, 0.8, 0);

		return 0;
	}
};

#endif

