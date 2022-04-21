#pragma once

#include <cassert>
#include <map>
#include <cuda_runtime_api.h>

#include "Helpers/xUtils.h"

class GPUConfig
{
public:
	static GPUConfig & getInstance()
	{
		static GPUConfig instance;
		return instance;
	}

	int icpStepThreads;
	int icpStepBlocks;

	int rgbStepThreads;
	int rgbStepBlocks;

	int rgbResThreads;
	int rgbResBlocks;

	int so3StepThreads;
	int so3StepBlocks;

private:
	GPUConfig()
		: icpStepThreads(128),
		icpStepBlocks(112),
		rgbStepThreads(128),
		rgbStepBlocks(112),
		rgbResThreads(256),
		rgbResBlocks(336),
		so3StepThreads(160),
		so3StepBlocks(64)
	{
		cudaDeviceProp prop;

		checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

		std::string dev(prop.name);

		icpStepMap["GeForce GTX 780 Ti"] = std::pair<int, int>(128, 112);
		rgbStepMap["GeForce GTX 780 Ti"] = std::pair<int, int>(128, 112);
		rgbResMap["GeForce GTX 780 Ti"] = std::pair<int, int>(256, 336);
		so3StepMap["GeForce GTX 780 Ti"] = std::pair<int, int>(160, 64);

		icpStepMap["GeForce GTX 880M"] = std::pair<int, int>(512, 16);
		rgbStepMap["GeForce GTX 880M"] = std::pair<int, int>(512, 16);
		rgbResMap["GeForce GTX 880M"] = std::pair<int, int>(256, 64);
		so3StepMap["GeForce GTX 880M"] = std::pair<int, int>(384, 16);

		icpStepMap["GeForce GTX 980"] = std::pair<int, int>(512, 32);
		rgbStepMap["GeForce GTX 980"] = std::pair<int, int>(160, 64);
		rgbResMap["GeForce GTX 980"] = std::pair<int, int>(128, 512);
		so3StepMap["GeForce GTX 980"] = std::pair<int, int>(240, 48);

		icpStepMap["GeForce GTX 970"] = std::pair<int, int>(128, 48);
		rgbStepMap["GeForce GTX 970"] = std::pair<int, int>(160, 64);
		rgbResMap["GeForce GTX 970"] = std::pair<int, int>(128, 272);
		so3StepMap["GeForce GTX 970"] = std::pair<int, int>(96, 64);

		icpStepMap["GeForce GTX 675MX"] = std::pair<int, int>(128, 80);
		rgbStepMap["GeForce GTX 675MX"] = std::pair<int, int>(128, 48);
		rgbResMap["GeForce GTX 675MX"] = std::pair<int, int>(128, 80);
		so3StepMap["GeForce GTX 675MX"] = std::pair<int, int>(128, 32);

		icpStepMap["Quadro K620M"] = std::pair<int, int>(32, 48);
		rgbStepMap["Quadro K620M"] = std::pair<int, int>(128, 16);
		rgbResMap["Quadro K620M"] = std::pair<int, int>(448, 48);
		so3StepMap["Quadro K620M"] = std::pair<int, int>(32, 48);

		icpStepMap["GeForce GTX TITAN"] = std::pair<int, int>(128, 96);
		rgbStepMap["GeForce GTX TITAN"] = std::pair<int, int>(112, 96);
		rgbResMap["GeForce GTX TITAN"] = std::pair<int, int>(256, 416);
		so3StepMap["GeForce GTX TITAN"] = std::pair<int, int>(128, 64);

		icpStepMap["GeForce GTX TITAN X"] = std::pair<int, int>(256, 96);
		rgbStepMap["GeForce GTX TITAN X"] = std::pair<int, int>(256, 64);
		rgbResMap["GeForce GTX TITAN X"] = std::pair<int, int>(96, 496);
		so3StepMap["GeForce GTX TITAN X"] = std::pair<int, int>(432, 48);

		icpStepMap["GeForce GTX 980 Ti"] = std::pair<int, int>(320, 64);
		rgbStepMap["GeForce GTX 980 Ti"] = std::pair<int, int>(128, 96);
		rgbResMap["GeForce GTX 980 Ti"] = std::pair<int, int>(224, 384);
		so3StepMap["GeForce GTX 980 Ti"] = std::pair<int, int>(432, 48);

		icpStepMap["GeForce GTX 1060"] = std::pair<int, int>(64, 240);
		rgbStepMap["GeForce GTX 1060"] = std::pair<int, int>(128, 96);
		rgbResMap["GeForce GTX 1060"] = std::pair<int, int>(256, 464);
		so3StepMap["GeForce GTX 1060"] = std::pair<int, int>(256, 48);

		/*
		if(icpStepMap.find(dev) == icpStepMap.end())
		{
		std::stringstream strs;
		strs << "Your GPU \"" << dev << "\" isn't in the ICP Step performance database, please add it";
		std::cout << strs.str() << std::endl;
		}
		else
		*/
		{
			icpStepThreads = icpStepMap["GeForce GTX TITAN"].first;
			icpStepBlocks = icpStepMap["GeForce GTX TITAN"].second;
		}

		/*
		if(rgbStepMap.find(dev) == rgbStepMap.end())
		{
		std::stringstream strs;
		strs << "Your GPU \"" << dev << "\" isn't in the RGB Step performance database, please add it";
		std::cout << strs.str() << std::endl;
		}
		else
		*/
		{
			rgbStepThreads = rgbStepMap["GeForce GTX TITAN"].first;
			rgbStepBlocks = rgbStepMap["GeForce GTX TITAN"].second;
		}

		/*
		if(rgbResMap.find(dev) == rgbResMap.end())
		{
		std::stringstream strs;
		strs << "Your GPU \"" << dev << "\" isn't in the RGB Res performance database, please add it";
		std::cout << strs.str() << std::endl;
		}
		else
		*/
		{
			rgbResThreads = rgbResMap["GeForce GTX TITAN"].first;
			rgbResBlocks = rgbResMap["GeForce GTX TITAN"].second;
		}

		/*
		if(so3StepMap.find(dev) == so3StepMap.end())
		{
		std::stringstream strs;
		strs << "Your GPU \"" << dev << "\" isn't in the SO3 Step performance database, please add it";
		std::cout << strs.str() << std::endl;
		}
		else
		*/
		{
			so3StepThreads = so3StepMap["GeForce GTX TITAN"].first;
			so3StepBlocks = so3StepMap["GeForce GTX TITAN"].second;
		}
	}

	std::map<std::string, std::pair<int, int> > icpStepMap, rgbStepMap, rgbResMap, so3StepMap;
};



