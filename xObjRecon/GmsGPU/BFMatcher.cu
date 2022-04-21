#include "GmsGPU/BFMatcher.cuh"

__device__ __forceinline__ int HamminDist(uint64_t* srcOrb,
																					uint64_t* targetOrb)
{
	return __popcll(srcOrb[0] ^ targetOrb[0]) +
		__popcll(srcOrb[1] ^ targetOrb[1]) +
		__popcll(srcOrb[2] ^ targetOrb[2]) +
		__popcll(srcOrb[3] ^ targetOrb[3]);
}

__global__ void OrbBFMatchKernel(int* dMatchesIdx,
                                 int* dMatchesDist,
                                 cv::cuda::PtrStepSz<uchar> src,
                                 cv::cuda::PtrStepSz<uchar> target)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= src.rows)
		return;

	uint64_t* srcOrb = (uint64_t*)src.ptr(idx);
	uint64_t* targetOrb;
	int dist;
	int minDist = 65534, secondMinDist = 65535;
	int minIdx, secondMinIdx;
	for (int i = 0; i < target.rows; ++i)
	{
		targetOrb = (uint64_t*)target.ptr(i);
		dist = HamminDist(srcOrb,
		                  targetOrb);
		if (dist < minDist)
		{
			minDist = dist;
			minIdx = i;
		}
		else
		{
			if (dist < secondMinDist)
			{
				secondMinDist = dist;
				secondMinIdx = i;
			}
		}
	}
#if 1
	dMatchesIdx[2 * idx] = minIdx;
	dMatchesIdx[2 * idx + 1] = secondMinIdx;
	dMatchesDist[2 * idx] = minDist;
	dMatchesDist[2 * idx + 1] = secondMinDist;
#endif
#if 0
	dMatchesIdx[idx] = minIdx;
	dMatchesDist[idx] = minDist;
#endif
}

void OrbBFMatch(int* dMatchesIdx,
                int* dMatchesDist,
                cv::cuda::GpuMat& src,
                cv::cuda::GpuMat& target)
{
	int block = 256;
	int grid = DivUp(src.rows, block);

	OrbBFMatchKernel << <grid, block >> >(dMatchesIdx, dMatchesDist, src, target);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__device__ __forceinline__ float L2Dist(float* srcOrb,
                                        float* targetOrb)
{
	float diff, dist = 0.0f;
	for (int i = 0; i < 33; ++i)
	{
		diff = srcOrb[i] - targetOrb[i];
		dist += diff * diff;
	}

	return sqrt(dist);
}

#if 0
float srcNorm = 0.0f, targetNorm = 0.0f;
for (int i = 0; i < 33; ++i)
{
	srcNorm += srcOrb[i] * srcOrb[i];
	targetNorm += targetOrb[i] * targetOrb[i];
}
srcNorm = sqrt(srcNorm);
targetNorm = sqrt(targetNorm);
#endif

__global__ void FPFHBFMatchKernel(int* dMatchesIdx,
                                  float* dMatchesDist,
                                  cv::cuda::PtrStepSz<float> src,
                                  cv::cuda::PtrStepSz<float> target)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= src.rows)
		return;

	float* srcFPFH = src.ptr(idx);

	float* targetFPFH;
	float dist;
	float minDist = 1.0e24f, secondMinDist = 1.1e24f;
	int minIdx, secondMinIdx;
	for (int i = 0; i < target.rows; ++i)
	{
		targetFPFH = target.ptr(i);
		dist = L2Dist(srcFPFH, targetFPFH);	
		if (dist < minDist)
		{
			minDist = dist;
			minIdx = i;
		}
		else
		{
			if (dist < secondMinDist)
			{
				secondMinDist = dist;
				secondMinIdx = i;
			}
		}
	}
	dMatchesIdx[2 * idx] = minIdx;
	dMatchesIdx[2 * idx + 1] = secondMinIdx;
	dMatchesDist[2 * idx] = minDist;
	dMatchesDist[2 * idx + 1] = secondMinDist;
}

void FPFHBFMatch(int* dMatchesIdx,
                 float* dMatchesDist,
                 cv::cuda::GpuMat& src,
                 cv::cuda::GpuMat& target)
{
	int block = 256;
	int grid = DivUp(src.rows, block);
	
	FPFHBFMatchKernel << <grid, block >> >(dMatchesIdx, dMatchesDist, src, target);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void CalcFPFHForOrbMatchesKernel(float* dMatchesDist,
											int* dMatchesIdx,
											int matehesNum,
											cv::cuda::PtrStepSz<int> srcFPFHIdxMat,
											cv::cuda::PtrStepSz<int> targetFPFHIdxMat,
											cv::cuda::PtrStepSz<float> src,
											cv::cuda::PtrStepSz<float> target)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= matehesNum)
		return;

	int x = dMatchesIdx[4 * idx];
	int y = dMatchesIdx[4 * idx + 1];
	float* srcFPFH = src.ptr(srcFPFHIdxMat.ptr(y)[x]);
	x = dMatchesIdx[4 * idx + 2];
	y = dMatchesIdx[4 * idx + 3];
	float* targetFPFH = target.ptr(targetFPFHIdxMat.ptr(y)[x]);
	
	float dist = L2Dist(srcFPFH, targetFPFH);
		
	dMatchesDist[idx] = dist;
}

void CalcFPFHForOrbMatches(float* dMatchesDist,
						   int* dMatchesIdx,
						   int matehesNum,
						   cv::cuda::GpuMat& srcFPFHIdxMat,
						   cv::cuda::GpuMat& targetFPFHIdxMat,
						   cv::cuda::GpuMat& src,
						   cv::cuda::GpuMat& target)
{
	int block = 256;
	int grid = DivUp(matehesNum, block);

	CalcFPFHForOrbMatchesKernel << <grid, block >> >(dMatchesDist, dMatchesIdx, matehesNum, 
													 srcFPFHIdxMat,
													 targetFPFHIdxMat,
													 src, target);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void NormalizeFPFHKernel(cv::cuda::PtrStepSz<float> fpfh)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= fpfh.rows)
		return;

	float* pFpfh = fpfh.ptr(idx);
	float sum = 0.0f;
	for (int i = 0; i < 33; ++i)
	{
		sum += (pFpfh[i] * pFpfh[i]);
	}
	sum = sqrt(sum);
	for (int i = 0; i < 33; ++i)
	{
		pFpfh[i] /= sum;
	}
}

void NormalizeFPFH(cv::cuda::GpuMat& src,
                   cv::cuda::GpuMat& target)
{
	int block = 256;
	int grid = DivUp(src.rows, block);

	NormalizeFPFHKernel << <grid, block >> >(src);
	NormalizeFPFHKernel << <grid, block >> >(target);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}
