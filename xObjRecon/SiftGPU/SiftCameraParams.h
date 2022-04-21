#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"

#define DEPTH_WIDTH 640
#define DEPTH_HEIGHT 480
#define COLOR_WIDTH 640
#define COLOR_HEIGHT 480

__align__(16)	//has to be aligned to 16 bytes
struct SiftCameraParams {

	unsigned int m_depthWidth;
	unsigned int m_depthHeight;
	unsigned int m_intensityWidth;
	unsigned int m_intensityHeight;

	float4x4 m_siftIntrinsics;
	float4x4 m_siftIntrinsicsInv;

	float4x4 m_downSampIntrinsics;
	float4x4 m_downSampIntrinsicsInv;

	float m_minKeyScale;

	unsigned int dummy0;
	unsigned int dummy1;
	unsigned int dummy2;
};
