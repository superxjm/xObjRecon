#include "SiftCameraParams.h"

#include <cuda_runtime_api.h>
#include <helper_cuda.h>

//__constant__ SiftCameraParams c_siftCameraParams;
__device__ SiftCameraParams *c_siftCameraParams;
//__device__ SiftCameraParams d_siftCameraParams;

extern "C" void updateConstantSiftCameraParams(const SiftCameraParams& params) {
	
	size_t size = sizeof(SiftCameraParams);
	//checkCudaErrors(cudaGetSymbolSize(&size, c_siftCameraParams));
	//checkCudaErrors(cudaMemcpyToSymbol(c_siftCameraParams, &params, size, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc(&c_siftCameraParams, size));
	checkCudaErrors(cudaMemcpy(c_siftCameraParams, &params, size, cudaMemcpyHostToDevice));
	//SiftCameraParams c_siftCameraParams_host;
	//memset(&c_siftCameraParams_host, 0, size);
	//cudaMemcpyFromSymbol(&c_siftCameraParams_host, c_siftCameraParams, size, 0, cudaMemcpyDeviceToHost);
	//std::cout << c_siftCameraParams_host.m_depthWidth << std::endl;
	
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}