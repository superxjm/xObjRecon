#include "xObjTrack/Cuda/xObjTrackCudaFuncs.cuh"

#include "InnorealTimer.hpp"
#include <device_launch_parameters.h>

__global__ void RemoveNonObjectPixelsKernel2(PtrStepSz<unsigned short> depth, PtrStep<float> vmap,
																						 cv::cuda::PtrStepSz<uchar> pruneMat,
																						 float centerX, float centerY, float centerZ, float normalX, float normalY,
																						 float normalZ, float radSquareThresh)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= depth.cols || y >= depth.rows)
		return;

	int rows = depth.rows;
	float xx, yy, zz, ll;
	if (!isnan(vmap.ptr(y)[x]) && depth.ptr(y)[x] != 0 && pruneMat.ptr(y)[x] != 255)
	{
		xx = vmap.ptr(y)[x] - centerX;
		yy = vmap.ptr(y + rows)[x] - centerY;
		zz = vmap.ptr(y + 2 * rows)[x] - centerZ;
		ll = normalX * xx + normalY * yy + normalZ * zz;
		if (xx * xx + yy * yy + zz * zz - ll * ll > radSquareThresh)
		{
			depth.ptr(y)[x] = 0;
		}
	}
	else
	{
		depth.ptr(y)[x] = 0;
	}
}

void RemoveNonObjectPixels2(DeviceArray2D<unsigned short>& depthDevice, DeviceArray2D<float>& vmapDevice,
														cv::cuda::GpuMat& pruneMatDevice,
														float centerX, float centerY, float centerZ, float normalX, float normalY, float normalZ,
														float radSquareThres)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(depthDevice.cols(), block.x), getGridDim(depthDevice.rows(), block.y));

	RemoveNonObjectPixelsKernel2 << <grid, block >> > (depthDevice, vmapDevice,
																										 pruneMatDevice,
																										 centerX, centerY, centerZ, normalX, normalY, normalZ,
																										 radSquareThres);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
};

__global__ void RemoveNonObjectPixelsKernel(cv::cuda::PtrStepSz<float> dDepth,
																						cv::cuda::PtrStep<float3> dVMap,
																						cv::cuda::PtrStepSz<uchar> dPruneMat,
																						float3 median, float radSquareThesh,
																						float3 zAxis, float zThresh)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dDepth.cols || y >= dDepth.rows)
		return;

	int rows = dDepth.rows;
	float xx, yy, zz, ll;
	float3 vertex = dVMap.ptr(y)[x];
	if (!isnan(vertex.x) && dDepth.ptr(y)[x] != 0 && dPruneMat.ptr(y)[x] != 255)
	{
		xx = vertex.x - median.x;
		yy = vertex.y - median.y;
		zz = vertex.z - median.z;
		ll = zAxis.x * xx + zAxis.y * yy + zAxis.z * zz;
		if (xx * xx + yy * yy + zz * zz - ll * ll > radSquareThesh || radSquareThesh > zThresh)
		{
			dDepth.ptr(y)[x] = 0;
		}
	}
	else
	{
		dDepth.ptr(y)[x] = 0;
	}
}

void RemoveNonObjectPixels(cv::cuda::GpuMat& dDepth,
													 cv::cuda::GpuMat& dVMap,
													 cv::cuda::GpuMat& dPruneMat,
													 float3& median, float radSquareThesh,
													 float3& zAxis, float& zThresh)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(dDepth.cols, block.x);
	grid.y = getGridDim(dDepth.rows, block.y);

	RemoveNonObjectPixelsKernel << <grid, block >> > (dDepth, dVMap, dPruneMat, median, radSquareThesh, zAxis, zThresh);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void RemoveNonObjectPixelsKernel(cv::cuda::PtrStepSz<float> dDepth,
																						cv::cuda::PtrStep<float3> dVMap,
																						float3 median, float radSquareThesh,
																						float3 zAxis, float zThresh)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dDepth.cols || y >= dDepth.rows)
		return;

	float xx, yy, zz, ll;
	float3 vertex = dVMap.ptr(y)[x];
	if (!isnan(vertex.x))
	{
		xx = vertex.x - median.x;
		yy = vertex.y - median.y;
		zz = vertex.z - median.z;
		ll = zAxis.x * xx + zAxis.y * yy + zAxis.z * zz;
		if (xx * xx + yy * yy + zz * zz - ll * ll > radSquareThesh || ll > zThresh)
		{
			dDepth.ptr(y)[x] = 0.0f;
		}
	}
	else
	{
		dDepth.ptr(y)[x] = 0.0f;
	}
}

void RemoveNonObjectPixels(cv::cuda::GpuMat& dDepth,
													 cv::cuda::GpuMat& dVMap,
													 float3& median, float radSquareThesh,
													 float3& zAxis, float& zThresh)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(dDepth.cols, block.x);
	grid.y = getGridDim(dDepth.rows, block.y);
	printf("============\n");
	printf("%f %f %f\n", median.x, median.y, median.z);
	printf("%f %f\n", radSquareThesh, zThresh);
	printf("%f %f %f\n", zAxis.x, zAxis.y, zAxis.z);
	printf("============\n");
	RemoveNonObjectPixelsKernel << <grid, block >> > (dDepth, dVMap, median, radSquareThesh, zAxis, zThresh);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void RemoveNonObjectPixels2Kernel(float middleX, float middleY, float middleZ,
																						 int left, int right, int top, int bottom,
																						 cv::cuda::PtrStepSz<float> dDepth,
																						 cv::cuda::PtrStep<float3> dVMap,
																						 float3 zAxis, float zThresh)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dDepth.cols || y >= dDepth.rows)
		return;

	float xx, yy, zz, ll;
	float3 vertex = dVMap.ptr(y)[x];
	//dDepth.ptr(y)[x] = 0.0f;
	//dDepth.ptr(y)[x] = 0;
	if (!isnan(vertex.x) && x >= left && x <= right && y >= top && y <= bottom)
	{
		xx = vertex.x;
		yy = vertex.y;
		zz = vertex.z;
		ll = zAxis.x * (xx - middleX) + zAxis.y * (yy - middleY) + zAxis.z * (zz - middleZ);
		if (ll > zThresh)
		{
			dDepth.ptr(y)[x] = 0.0f;
			dDepth.ptr(y)[x] = 0;
		}
	}
	else
	{
		dDepth.ptr(y)[x] = 0.0f;
		dDepth.ptr(y)[x] = 0;
	}
}

void RemoveNonObjectPixels2(float middleX, float middleY, float middleZ,
														int left, int right, int top, int bottom,
														cv::cuda::GpuMat& dDepth,
														cv::cuda::GpuMat& dVMap,
														float3& zAxis, float& zThresh)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(dDepth.cols, block.x);
	grid.y = getGridDim(dDepth.rows, block.y);
	RemoveNonObjectPixels2Kernel << <grid, block >> > (middleX, middleY, middleZ,
																										 left, right, top, bottom, dDepth,
																										 dVMap, zAxis, zThresh);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void RemoveVerticesOutOfRangeKernel(cv::cuda::PtrStepSz<float3> dVMap,
																							 float medianX, float medianY, float medianZ,
																							 float lowRadSquare, float hightRadSquare)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dVMap.cols || y >= dVMap.rows)
		return;

	float xx, yy, rad;
	if (!isnan(dVMap.ptr(y)[x].x))
	{
		xx = dVMap.ptr(y)[x].x - medianX;
		yy = dVMap.ptr(y)[x].y - medianY;
		rad = xx * xx + yy * yy;
		if (rad < lowRadSquare || rad > hightRadSquare)
		{
			dVMap.ptr(y)[x].x = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
		}
	}
}

void RemoveVerticesOutOfRange(cv::cuda::GpuMat& dVMap,
															float medianX, float medianY, float medianZ,
															float lowRadSquare, float hightRadSquare)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(dVMap.cols, block.x), getGridDim(dVMap.rows, block.y));

	RemoveVerticesOutOfRangeKernel << <grid, block >> > (dVMap,
																											 medianX, medianY, medianZ,
																											 lowRadSquare, hightRadSquare);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
};

#if 0
__global__ void CalcEdgeIdxKernel(cv::cuda::PtrStepSz<int> dEedgeIdx, cv::cuda::PtrStepSz<uchar> dMask,
																	cv::cuda::PtrStepSz<float> dFiltereddepthImg32, float minDepth, float maxDepth,
																	float magThresholdSquare,
																	int validBoxLeft, int validBoxRight, int validBoxTop, int validBoxBottom)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	dMask.ptr(y)[x] = 0;
	dEedgeIdx.ptr(y)[x] = 0;

	if (x <= validBoxLeft || y <= validBoxTop || x >= validBoxRight || y >= validBoxBottom)
		return;

	float horiDiff = 0.0f, vertiDiff = 0.0f;

	// horizontal [ -1, 0, 1; -2, 0, 2; -1, 0, 1 ] / 8
	horiDiff = dFiltereddepthImg32.ptr(y - 1)[x - 1] * -0.125f
		+ dFiltereddepthImg32.ptr(y - 1)[x] * -0.25f
		+ dFiltereddepthImg32.ptr(y - 1)[x + 1] * -0.125f
		+ dFiltereddepthImg32.ptr(y + 1)[x - 1] * 0.125f
		+ dFiltereddepthImg32.ptr(y + 1)[x] * 0.25f
		+ dFiltereddepthImg32.ptr(y + 1)[x + 1] * 0.125f;
	// vertical [ -1, 0, 1; -2, 0, 2; -1, 0, 1 ] / 8
	vertiDiff = dFiltereddepthImg32.ptr(y - 1)[x - 1] * -0.125f
		+ dFiltereddepthImg32.ptr(y)[x - 1] * -0.25f
		+ dFiltereddepthImg32.ptr(y + 1)[x - 1] * -0.125f
		+ dFiltereddepthImg32.ptr(y - 1)[x + 1] * 0.125f
		+ dFiltereddepthImg32.ptr(y)[x + 1] * 0.25f
		+ dFiltereddepthImg32.ptr(y + 1)[x + 1] * 0.125f;

	float depth = dFiltereddepthImg32.ptr(y)[x];
	if (depth >= minDepth && depth <= maxDepth)
	{
		dMask.ptr(y)[x] = 255;
		if ((vertiDiff * vertiDiff + horiDiff * horiDiff) > magThresholdSquare)
		{
			dEedgeIdx.ptr(y)[x] = -1;
		}
	}
}

void CalcEdgeIdx(cv::cuda::GpuMat& dEdgeIdx, cv::cuda::GpuMat& dMask,
								 cv::cuda::GpuMat& dFiltereddepthImg32F, float minDepth, float maxDepth, float magThresholdSquare,
								 int validBoxLeft, int validBoxRight, int validBoxTop, int validBoxBottom)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(dFiltereddepthImg32F.cols, block.x), getGridDim(dFiltereddepthImg32F.rows, block.y));

	CalcEdgeIdxKernel << <grid, block >> > (dEdgeIdx, dMask,
																					dFiltereddepthImg32F, minDepth, maxDepth, magThresholdSquare,
																					validBoxLeft, validBoxRight, validBoxTop, validBoxBottom);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}
#endif

__global__ void CalcEdgeIdxWithGravityKernel(cv::cuda::PtrStep<int> dEdgeIdx, cv::cuda::PtrStep<uchar> dEdge,
																						 cv::cuda::PtrStep<uchar> dMask,
																						 cv::cuda::PtrStep<float> dFiltereddepthImg32,
																						 cv::cuda::PtrStep<float3> dVMap,
																						 float boxLeft, float boxRight, float boxTop, float boxBottom,
																						 float minDepth, float maxDepth, float magThresholdSquare,
																						 int validBoxLeft, int validBoxRight, int validBoxTop, int validBoxBottom)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	dMask.ptr(y)[x] = 0;
	dEdgeIdx.ptr(y)[x] = 0;
	dEdge.ptr(y)[x] = 0;

	//if (x <= validBoxLeft || y <= validBoxTop || x >= validBoxRight || y >= validBoxBottom)
	//return;
	if (x >= boxLeft && x <= boxRight && y >= boxTop && y <= boxBottom)
	{
		dMask.ptr(y)[x] = 255;
	}

	if (!isnan(dVMap.ptr(y)[x].x) && dFiltereddepthImg32.ptr(y)[x] > 0)
	{
		float depth = -dVMap.ptr(y)[x].y;
		if (depth >= minDepth && depth <= maxDepth)
		{
			// horizontal [ -1, 0, 1; -2, 0, 2; -1, 0, 1 ] / 8
			float horiDiff = dFiltereddepthImg32.ptr(y - 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y - 1)[x] * -0.25f
				+ dFiltereddepthImg32.ptr(y - 1)[x + 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y + 1)[x - 1] * 0.125f
				+ dFiltereddepthImg32.ptr(y + 1)[x] * 0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x + 1] * 0.125f;
			// vertical [ -1, 0, 1; -2, 0, 2; -1, 0, 1 ] / 8
			float vertiDiff = dFiltereddepthImg32.ptr(y - 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y)[x - 1] * -0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y - 1)[x + 1] * 0.125f
				+ dFiltereddepthImg32.ptr(y)[x + 1] * 0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x + 1] * 0.125f;
			if ((vertiDiff * vertiDiff + horiDiff * horiDiff) > magThresholdSquare)
			{
				dEdgeIdx.ptr(y)[x] = -1;
				dEdge.ptr(y)[x] = 255;
			}
		}
	}
}

__global__ void CalcEdgeIdxWithoutGravityKernel(cv::cuda::PtrStep<int> dEdgeIdx, cv::cuda::PtrStep<uchar> dEdge,
																								cv::cuda::PtrStep<uchar> dMask,
																								cv::cuda::PtrStep<float> dFiltereddepthImg32, cv::cuda::PtrStep<float3> dVMap,
																								float boxLeft, float boxRight, float boxTop, float boxBottom,
																								float minDepth, float maxDepth, float magThresholdSquare,
																								int validBoxLeft, int validBoxRight, int validBoxTop, int validBoxBottom)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	dMask.ptr(y)[x] = 0;
	dEdgeIdx.ptr(y)[x] = 0;
	dEdge.ptr(y)[x] = 0;

	//if (x <= validBoxLeft || y <= validBoxTop || x >= validBoxRight || y >= validBoxBottom)
			//return;
	if (x >= boxLeft && x <= boxRight && y >= boxTop && y <= boxBottom)
	{
		dMask.ptr(y)[x] = 255;
	}

	if (!isnan(dVMap.ptr(y)[x].x) && dFiltereddepthImg32.ptr(y)[x] > 0)
	{
		float depth = dVMap.ptr(y)[x].z;
		if (depth >= minDepth && depth <= maxDepth)
		{
			// horizontal [ -1, 0, 1; -2, 0, 2; -1, 0, 1 ] / 8
			float horiDiff = dFiltereddepthImg32.ptr(y - 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y - 1)[x] * -0.25f
				+ dFiltereddepthImg32.ptr(y - 1)[x + 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y + 1)[x - 1] * 0.125f
				+ dFiltereddepthImg32.ptr(y + 1)[x] * 0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x + 1] * 0.125f;
			// vertical [ -1, 0, 1; -2, 0, 2; -1, 0, 1 ] / 8
			float vertiDiff = dFiltereddepthImg32.ptr(y - 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y)[x - 1] * -0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y - 1)[x + 1] * 0.125f
				+ dFiltereddepthImg32.ptr(y)[x + 1] * 0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x + 1] * 0.125f;
			if ((vertiDiff * vertiDiff + horiDiff * horiDiff) > magThresholdSquare)
			{
				dEdgeIdx.ptr(y)[x] = -1;
				dEdge.ptr(y)[x] = 255;
			}
		}
	}
}

void CalcEdgeIdx(cv::cuda::GpuMat& dEdgeIdx, cv::cuda::GpuMat& dEdge, cv::cuda::GpuMat& dMask,
								 cv::cuda::GpuMat& dFiltereddepthImg32F, cv::cuda::GpuMat& dVMap, Box& box,
								 float minDepth, float maxDepth, float magThresholdSquare,
								 int validBoxLeft, int validBoxRight, int validBoxTop, int validBoxBottom, bool withGravity)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(dFiltereddepthImg32F.cols, block.x), getGridDim(dFiltereddepthImg32F.rows, block.y));
	if (withGravity)
	{
		CalcEdgeIdxWithGravityKernel << <grid, block >> > (dEdgeIdx, dEdge, dMask,
																											 dFiltereddepthImg32F, dVMap,
																											 box.m_left, box.m_right, box.m_top, box.m_bottom,
																											 minDepth, maxDepth, magThresholdSquare,
																											 validBoxLeft, validBoxRight, validBoxTop, validBoxBottom);
	}
	else
	{
		CalcEdgeIdxWithoutGravityKernel << <grid, block >> > (dEdgeIdx, dEdge, dMask,
																													dFiltereddepthImg32F, dVMap,
																													box.m_left, box.m_right, box.m_top, box.m_bottom,
																													minDepth, maxDepth, magThresholdSquare,
																													validBoxLeft, validBoxRight, validBoxTop, validBoxBottom);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void CalcEdgeIdxOptWithBoxKernel(cv::cuda::PtrStep<int> dEdgeIdx, cv::cuda::PtrStep<uchar> dEdge,
																						cv::cuda::PtrStep<uchar> dMask,
																						cv::cuda::PtrStep<float> dFiltereddepthImg32, cv::cuda::PtrStep<float3> dVMap,
																						float boxLeft, float boxRight, float boxTop, float boxBottom,
																						float minDepth, float maxDepth, float magThresholdSquare) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	dMask.ptr(y)[x] = 0;
	dEdgeIdx.ptr(y)[x] = 0;
	dEdge.ptr(y)[x] = 0;

	if (x >= boxLeft && x <= boxRight && y >= boxTop && y <= boxBottom) {
		dMask.ptr(y)[x] = 255;
	}
	else
	{
		return;
	}

	if (!isnan(dVMap.ptr(y)[x].x) && dFiltereddepthImg32.ptr(y)[x] > 0) {
		float depth = -dVMap.ptr(y)[x].y;
		if (depth >= minDepth && depth <= maxDepth) {
			// horizontal [ -1, 0, 1; -2, 0, 2; -1, 0, 1 ] / 8
			float horiDiff = dFiltereddepthImg32.ptr(y - 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y - 1)[x] * -0.25f
				+ dFiltereddepthImg32.ptr(y - 1)[x + 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y + 1)[x - 1] * 0.125f
				+ dFiltereddepthImg32.ptr(y + 1)[x] * 0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x + 1] * 0.125f;
			// vertical [ -1, 0, 1; -2, 0, 2; -1, 0, 1 ] / 8
			float vertiDiff = dFiltereddepthImg32.ptr(y - 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y)[x - 1] * -0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y - 1)[x + 1] * 0.125f
				+ dFiltereddepthImg32.ptr(y)[x + 1] * 0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x + 1] * 0.125f;
			if ((vertiDiff * vertiDiff + horiDiff * horiDiff) > magThresholdSquare) {
				dEdgeIdx.ptr(y)[x] = -1;
				dEdge.ptr(y)[x] = 255;
			}
		}
	}
}

void CalcEdgeIdxOptWithBox(cv::cuda::GpuMat& dEdgeIdx, cv::cuda::GpuMat& dEdge, cv::cuda::GpuMat& dMask,
													 cv::cuda::GpuMat& dFiltereddepthImg32F, cv::cuda::GpuMat& dVMap, Box& box,
													 float minDepth, float maxDepth, float magThresholdSquare)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(dFiltereddepthImg32F.cols, block.x), getGridDim(dFiltereddepthImg32F.rows, block.y));
	CalcEdgeIdxOptWithBoxKernel << <grid, block >> > (dEdgeIdx, dEdge, dMask,
																										dFiltereddepthImg32F, dVMap,
																										box.m_left, box.m_right, box.m_top, box.m_bottom,
																										minDepth, maxDepth, magThresholdSquare);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void CalcEdgeIdxOptWithoutBoxKernel(cv::cuda::PtrStep<int> dEdgeIdx, cv::cuda::PtrStep<uchar> dEdge,
																							 cv::cuda::PtrStep<uchar> dMask,
																							 cv::cuda::PtrStep<float> dFiltereddepthImg32, cv::cuda::PtrStep<float3> dVMap,
																							 float minDepth, float maxDepth, float magThresholdSquare,
																							 int validBoxLeft, int validBoxRight, int validBoxTop, int validBoxBottom) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	dMask.ptr(y)[x] = 0;
	dEdgeIdx.ptr(y)[x] = 0;
	dEdge.ptr(y)[x] = 0;

	if (x <= validBoxLeft || y <= validBoxTop || x >= validBoxRight || y >= validBoxBottom)
		return;

	if (!isnan(dVMap.ptr(y)[x].x) && dFiltereddepthImg32.ptr(y)[x] > 0) {
		float depth = -dVMap.ptr(y)[x].y;
		if (depth >= minDepth && depth <= maxDepth) {
			// horizontal [ -1, 0, 1; -2, 0, 2; -1, 0, 1 ] / 8
			float horiDiff = dFiltereddepthImg32.ptr(y - 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y - 1)[x] * -0.25f
				+ dFiltereddepthImg32.ptr(y - 1)[x + 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y + 1)[x - 1] * 0.125f
				+ dFiltereddepthImg32.ptr(y + 1)[x] * 0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x + 1] * 0.125f;
			// vertical [ -1, 0, 1; -2, 0, 2; -1, 0, 1 ] / 8
			float vertiDiff = dFiltereddepthImg32.ptr(y - 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y)[x - 1] * -0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x - 1] * -0.125f
				+ dFiltereddepthImg32.ptr(y - 1)[x + 1] * 0.125f
				+ dFiltereddepthImg32.ptr(y)[x + 1] * 0.25f
				+ dFiltereddepthImg32.ptr(y + 1)[x + 1] * 0.125f;
			if ((vertiDiff * vertiDiff + horiDiff * horiDiff) > magThresholdSquare) {
				dEdgeIdx.ptr(y)[x] = -1;
				dEdge.ptr(y)[x] = 255;
				dMask.ptr(y)[x] = 255;
			}
		}
	}
}

void CalcEdgeIdxOptWithoutBox(cv::cuda::GpuMat& dEdgeIdx, cv::cuda::GpuMat& dEdge, cv::cuda::GpuMat& dMask,
															cv::cuda::GpuMat& dFiltereddepthImg32F, cv::cuda::GpuMat& dVMap,
															float minDepth, float maxDepth, float magThresholdSquare,
															int validBoxLeft, int validBoxRight, int validBoxTop, int validBoxBottom)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(dFiltereddepthImg32F.cols, block.x), getGridDim(dFiltereddepthImg32F.rows, block.y));
	CalcEdgeIdxOptWithoutBoxKernel << <grid, block >> > (dEdgeIdx, dEdge, dMask,
																											 dFiltereddepthImg32F, dVMap,
																											 minDepth, maxDepth, magThresholdSquare,
																											 validBoxLeft, validBoxRight, validBoxTop, validBoxBottom);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void CalcMinSumCntDepthWithGravityKernel(char* dResultBuf, cv::cuda::PtrStepSz<float3> roi,
																										float depthThreshold)
{
	__shared__ float minDepth[128], sumDepth[128];
	__shared__ int validCnt[128];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	minDepth[threadIdx.x] = 1.0e24f;
	sumDepth[threadIdx.x] = 0.0f;
	validCnt[threadIdx.x] = 0;

	if (idx >= roi.cols)
		return;

	int i;
	for (i = 0; i < roi.rows; ++i)
	{
		if (!isnan(roi.ptr(i)[idx].x))
		{
			float depth = -roi.ptr(i)[idx].y;
			if (depth < depthThreshold)
			{
				if (depth < minDepth[threadIdx.x])
				{
					minDepth[threadIdx.x] = depth;
				}
				sumDepth[threadIdx.x] += depth;
				++validCnt[threadIdx.x];
			}
		}
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float minDepthVal = 1.0e23f, sumDepthVal = 0.0f;
		int validCntVal = 0;
		for (i = 0; i < 128; ++i)
		{
			if (minDepth[i] < minDepthVal)
				minDepthVal = minDepth[i];

			sumDepthVal += sumDepth[i];
			validCntVal += validCnt[i];
		}
		float* pMinSumCnt = (float *)dResultBuf;
		pMinSumCnt[blockIdx.x] = minDepthVal;
		pMinSumCnt += gridDim.x;
		pMinSumCnt[blockIdx.x] = sumDepthVal;
		pMinSumCnt += gridDim.x;
		*((int *)pMinSumCnt + blockIdx.x) = validCntVal;
	}
}

__global__ void CalcMinSumCntDepthWithoutGravityKernel(char* dResultBuf, cv::cuda::PtrStepSz<float3> roi,
																											 float depthThreshold)
{
	__shared__ float minDepth[128], sumDepth[128];
	__shared__ int validCnt[128];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	minDepth[threadIdx.x] = 1.0e24f;
	sumDepth[threadIdx.x] = 0.0f;
	validCnt[threadIdx.x] = 0;

	if (idx >= roi.cols)
		return;

	int i;
	for (i = 0; i < roi.rows; ++i)
	{
		if (!isnan(roi.ptr(i)[idx].x))
		{
			float depth = roi.ptr(i)[idx].z;
			if (depth < depthThreshold)
			{
				if (depth < minDepth[threadIdx.x])
				{
					minDepth[threadIdx.x] = depth;
				}
				sumDepth[threadIdx.x] += depth;
				++validCnt[threadIdx.x];
			}
		}
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float minDepthVal = 1.0e23f, sumDepthVal = 0.0f;
		int validCntVal = 0;
		for (i = 0; i < 128; ++i)
		{
			if (minDepth[i] < minDepthVal)
				minDepthVal = minDepth[i];

			sumDepthVal += sumDepth[i];
			validCntVal += validCnt[i];
		}
		float* pMinSumCnt = (float *)dResultBuf;
		pMinSumCnt[blockIdx.x] = minDepthVal;
		pMinSumCnt += gridDim.x;
		pMinSumCnt[blockIdx.x] = sumDepthVal;
		pMinSumCnt += gridDim.x;
		*((int *)pMinSumCnt + blockIdx.x) = validCntVal;
	}
}

bool CalcMinMeanDepth(float& minDepth, float& meanDepth, cv::cuda::GpuMat& dVMapRoi, float depthThreshold,
											char* dResultBuf, char* resultBuf, bool withGravity)
{
	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();
	int block = 128;
	int grid = DivUp(dVMapRoi.cols, block);
	if (withGravity)
	{
		CalcMinSumCntDepthWithGravityKernel << <grid, block >> > (dResultBuf, dVMapRoi, depthThreshold);
	}
	else
	{
		CalcMinSumCntDepthWithoutGravityKernel << <grid, block >> > (dResultBuf, dVMapRoi, depthThreshold);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	checkCudaErrors(cudaMemcpy(
		resultBuf,
		dResultBuf, (sizeof(float) * 2 + sizeof(int)) * grid, cudaMemcpyDeviceToHost));
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());
	//printf("grid: %d\n", grid);

	//timer.TimeStart();
	minDepth = 1.0e22f, meanDepth = 0.0f;
	int cnt = 0;
	float* pMin = (float *)resultBuf;
	float* pSum = pMin + grid;
	int* pCnt = (int *)(pSum + grid);

	for (int i = 0; i < grid; ++i)
	{
		if (pMin[i] < minDepth)
			minDepth = pMin[i];
		meanDepth += pSum[i];
		cnt += pCnt[i];
	}
	meanDepth /= cnt;
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	if (minDepth > 1.0e22f - MYEPS)
	{
		return false;
	}

	return true;
}

#if 0
__global__ void CalcMinSumCntDepthKernel(char* dResultBuf, cv::cuda::PtrStepSz<float> roi)
{
	__shared__ float minDepth[128], sumDepth[128];
	__shared__ int validCnt[128];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	minDepth[threadIdx.x] = 1.0e24f;
	sumDepth[threadIdx.x] = 0.0f;
	validCnt[threadIdx.x] = 0;

	if (idx >= roi.cols)
		return;

	float depth;
	int i;
	for (i = 0; i < roi.rows; ++i)
	{
		depth = roi.ptr(i)[idx];
		if (depth > MYEPS)
		{
			if (depth < minDepth[threadIdx.x])
			{
				minDepth[threadIdx.x] = depth;
			}
			sumDepth[threadIdx.x] += depth;
			++validCnt[threadIdx.x];
		}
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float minDepthVal = 1.0e23f, sumDepthVal = 0.0f;
		int validCntVal = 0;
		for (i = 0; i < 128; ++i)
		{
			if (minDepth[i] < minDepthVal)
				minDepthVal = minDepth[i];

			sumDepthVal += sumDepth[i];
			validCntVal += validCnt[i];
		}
		float* pMinSumCnt = (float *)dResultBuf;
		pMinSumCnt[blockIdx.x] = minDepthVal;
		pMinSumCnt += gridDim.x;
		pMinSumCnt[blockIdx.x] = sumDepthVal;
		pMinSumCnt += gridDim.x;
		*((int *)pMinSumCnt + blockIdx.x) = validCntVal;
	}
}

bool CalcMinMeanDepth(float& minDepth, float& meanDepth, cv::cuda::GpuMat& dDpethRoi,
											char* dResultBuf, char* resultBuf)
{
	innoreal::InnoRealTimer timer;
	timer.TimeStart();

	int block = 128;
	int grid = DivUp(dDpethRoi.cols, block);
	CalcMinSumCntDepthKernel << <grid, block >> > (dResultBuf, dDpethRoi);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	timer.TimeEnd();
	printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	timer.TimeStart();
	checkCudaErrors(cudaMemcpy(
		resultBuf,
		dResultBuf, (sizeof(float) * 2 + sizeof(int)) * grid, cudaMemcpyDeviceToHost));
	timer.TimeEnd();
	printf("time min mean: %lf\n", timer.TimeGap_in_ms());
	printf("grid: %d\n", grid);

	timer.TimeStart();
	minDepth = 1.0e22f, meanDepth = 0.0f;
	int cnt = 0;
	float* pMin = (float *)resultBuf;
	float* pSum = pMin + grid;
	int* pCnt = (int *)(pSum + grid);

	for (int i = 0; i < grid; ++i)
	{
		if (pMin[i] < minDepth)
			minDepth = pMin[i];
		meanDepth += pSum[i];
		cnt += pCnt[i];
	}
	meanDepth /= cnt;
	timer.TimeEnd();
	printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	if (minDepth > 1.0e22f - MYEPS)
	{
		return false;
	}

	return true;
}
#endif

__global__ void CalcMaxSumCntZKernel(char* dResultBuf, cv::cuda::PtrStepSz<float3> dVMapRoi)
{
	__shared__ float maxZ[128], sumZ[128];
	__shared__ int validCnt[128];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	maxZ[threadIdx.x] = -1.0e24f;
	sumZ[threadIdx.x] = 0.0f;
	validCnt[threadIdx.x] = 0;

	if (idx >= dVMapRoi.cols)
		return;

	float depth;
	int i;
	for (i = 0; i < dVMapRoi.rows; ++i)
	{
		float3 vertex = dVMapRoi.ptr(i)[idx];
		if (!isnan(vertex.x))
		{
			if (vertex.z > maxZ[threadIdx.x])
			{
				maxZ[threadIdx.x] = vertex.z;
			}

			sumZ[threadIdx.x] += vertex.z;
			++validCnt[threadIdx.x];
		}
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float maxZVal = -1.0e23f, sumZVal = 0.0f;
		int validCntVal = 0;
		for (i = 0; i < 128; ++i)
		{
			if (maxZ[i] > maxZVal)
				maxZVal = maxZ[i];

			sumZVal += sumZ[i];
			validCntVal += validCnt[i];
		}
		float* pMinSumCnt = (float *)dResultBuf;
		pMinSumCnt[blockIdx.x] = maxZVal;
		pMinSumCnt += gridDim.x;
		pMinSumCnt[blockIdx.x] = sumZVal;
		pMinSumCnt += gridDim.x;
		*((int *)pMinSumCnt + blockIdx.x) = validCntVal;
	}
}

__global__ void CalcSumCntZKernel(char* dResultBuf, cv::cuda::PtrStepSz<float3> dVMapRoi)
{
	__shared__ float sumZ[128];
	__shared__ int validCnt[128];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	sumZ[threadIdx.x] = 0.0f;
	validCnt[threadIdx.x] = 0;

	if (idx >= dVMapRoi.cols)
		return;

	float depth;
	int i;
	for (i = 0; i < dVMapRoi.rows; ++i)
	{
		float3 vertex = dVMapRoi.ptr(i)[idx];
		if (!isnan(vertex.x))
		{
			sumZ[threadIdx.x] += vertex.z;
			++validCnt[threadIdx.x];
		}
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float sumZVal = 0.0f;
		int validCntVal = 0;
		for (i = 0; i < 128; ++i)
		{
			sumZVal += sumZ[i];
			validCntVal += validCnt[i];
		}
		float* pMinSumCnt = (float *)dResultBuf;
		pMinSumCnt[blockIdx.x] = sumZVal;
		pMinSumCnt += gridDim.x;
		*((int *)pMinSumCnt + blockIdx.x) = validCntVal;
	}
}

__global__ void CalcSumCntXYKernel(char* dResultBuf, cv::cuda::PtrStepSz<float3> dVMapRoi, const float medianZ)
{
	__shared__ float sumX[128], sumY[128];
	__shared__ int validCnt[128];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	sumX[threadIdx.x] = 0.0f;
	sumY[threadIdx.x] = 0.0f;
	validCnt[threadIdx.x] = 0;

	if (idx >= dVMapRoi.cols)
		return;

	float depth;
	int i;
	for (i = 0; i < dVMapRoi.rows; ++i)
	{
		float3 vertex = dVMapRoi.ptr(i)[idx];
		if (!isnan(vertex.x) && vertex.z < medianZ)
		{
			sumX[threadIdx.x] += vertex.x;
			sumY[threadIdx.x] += vertex.y;
			++validCnt[threadIdx.x];
		}
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float sumXVal = 0.0f, sumYVal = 0.0f;
		int validCntVal = 0;
		for (i = 0; i < 128; ++i)
		{
			sumXVal += sumX[i];
			sumYVal += sumY[i];
			validCntVal += validCnt[i];
		}
		float* pMinSumCnt = (float *)dResultBuf;
		pMinSumCnt[blockIdx.x] = sumXVal;
		pMinSumCnt += gridDim.x;
		pMinSumCnt[blockIdx.x] = sumYVal;
		pMinSumCnt += gridDim.x;
		*((int *)pMinSumCnt + blockIdx.x) = validCntVal;
	}
}

__global__ void CalcMaxRadiusSquareKernel(char* dResultBuf, cv::cuda::PtrStepSz<float3> dVMapRoi,
																					const float medianX, const float medianY, const float medianZ)
{
	__shared__ float maxRadiusSquare[128];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	maxRadiusSquare[threadIdx.x] = -1.0e24f;

	if (idx >= dVMapRoi.cols)
		return;

	float radSquare, radX, radY;
	int i;
	for (i = 0; i < dVMapRoi.rows; ++i)
	{
		float3 vertex = dVMapRoi.ptr(i)[idx];
		if (!isnan(vertex.x))
		{
			radX = vertex.x - medianX;
			radY = vertex.y - medianY;
			radSquare = radX * radX + radY * radY;
			if (radSquare > maxRadiusSquare[threadIdx.x] && vertex.z < medianZ)
			{
				maxRadiusSquare[threadIdx.x] = radSquare;
			}
		}
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float maxRadiusSquareVal = -1.0e23f;
		for (i = 0; i < 128; ++i)
		{
			if (maxRadiusSquare[i] > maxRadiusSquareVal)
				maxRadiusSquareVal = maxRadiusSquare[i];
		}
		float* pMinSumCnt = (float *)dResultBuf;
		pMinSumCnt[blockIdx.x] = maxRadiusSquareVal;
	}
}

bool CalcMedianZ(float3& median, cv::cuda::GpuMat& dVMapRoi,
								 char* dResultBuf, char* resultBuf)
{
	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();

	int block = 128;
	int grid = DivUp(dVMapRoi.cols, block);

	CalcSumCntZKernel << <grid, block >> > (dResultBuf, dVMapRoi);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	checkCudaErrors(cudaMemcpy(
		resultBuf,
		dResultBuf, (sizeof(float) + sizeof(int)) * grid, cudaMemcpyDeviceToHost));
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	float maxZ = -1.0e22f, meanZ = 0.0f;
	int cnt = 0;
	float* pSum = (float *)resultBuf;
	int* pCnt = (int *)(pSum + grid);

	for (int i = 0; i < grid; ++i)
	{
		meanZ += pSum[i];
		cnt += pCnt[i];
	}
	meanZ /= cnt;
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//medianZ = (maxZ + meanZ) * 0.5f;
	median.z = meanZ;

	if (maxZ < -1.0e22f + MYEPS)
	{
		return false;
	}

	return true;
}

bool CalcMedianXY(float3& median, cv::cuda::GpuMat& dVMapRoi,
									char* dResultBuf, char* resultBuf)
{
	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();

	int block = 128;
	int grid = DivUp(dVMapRoi.cols, block);

	CalcSumCntXYKernel << <grid, block >> > (dResultBuf, dVMapRoi, median.z);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	checkCudaErrors(cudaMemcpy(
		resultBuf,
		dResultBuf, (sizeof(float) * 2 + sizeof(int)) * grid, cudaMemcpyDeviceToHost));
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	float meanX = 0.0f, meanY = 0.0f;
	int cnt = 0;
	float* pSumX = (float *)resultBuf;
	float* pSumY = pSumX + grid;
	int* pCnt = (int *)(pSumY + grid);

	for (int i = 0; i < grid; ++i)
	{
		meanX += pSumX[i];
		meanY += pSumY[i];
		cnt += pCnt[i];
	}
	meanX /= cnt;
	meanY /= cnt;
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());	
	median.x = meanX;
	median.y = meanY;

	if ((meanX < MYEPS * grid && meanX > -MYEPS * grid) || (meanY < MYEPS * grid && meanY > -MYEPS * grid))
	{
		return false;
	}

	return true;
}

bool CalcRadiusSquare(float& radiusSquare, float3& median, cv::cuda::GpuMat& dVMapRoi, char* dResultBuf,
											char* resultBuf)
{
	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();

	int block = 128;
	int grid = DivUp(dVMapRoi.cols, block);

	CalcMaxRadiusSquareKernel << <grid, block >> > (dResultBuf, dVMapRoi, median.x, median.y, median.z);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	checkCudaErrors(cudaMemcpy(
		resultBuf,
		dResultBuf, sizeof(float) * grid, cudaMemcpyDeviceToHost));
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	float maxRadiusSquare = -1.0e22f;
	int cnt = 0;
	float* pMaxRadiusSquare = (float *)resultBuf;

	for (int i = 0; i < grid; ++i)
	{
		if (pMaxRadiusSquare[i] > maxRadiusSquare)
			maxRadiusSquare = pMaxRadiusSquare[i];
	}
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	radiusSquare = maxRadiusSquare;

	if (maxRadiusSquare < -1.0e22f + MYEPS)
	{
		return false;
	}

	return true;
}

void CalcMedianAndRadius(float3& median, float& radiusSquare, cv::cuda::GpuMat& dVMapRoi,
												 char* dResultBuf, char* resultBuf)
{
	bool status1 = CalcMedianZ(median, dVMapRoi, dResultBuf, resultBuf);

	bool status2 = CalcMedianXY(median, dVMapRoi, dResultBuf, resultBuf);

	bool status3 = CalcRadiusSquare(radiusSquare, median, dVMapRoi, dResultBuf, resultBuf);

	if (!status1 || !status2 || !status3)
	{
		std::cout << "tracking error" << std::endl;
		std::exit(0);
	}
}

__global__ void CalcSumCntZOnlyZKernel(char* dResultBuf, cv::cuda::PtrStepSz<float3> dVMapRoi, float minDepth,
																			 float maxDepth)
{
	__shared__ float sumZ[128];
	__shared__ int validCnt[128];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	sumZ[threadIdx.x] = 0.0f;
	validCnt[threadIdx.x] = 0;

	if (idx >= dVMapRoi.cols)
		return;

	float depth;
	int i;
	for (i = 0; i < dVMapRoi.rows; ++i)
	{
		float3 vertex = dVMapRoi.ptr(i)[idx];
		depth = -vertex.y;
		if (!isnan(vertex.x) && depth >= minDepth && depth <= maxDepth)
		{
			sumZ[threadIdx.x] += vertex.z;
			++validCnt[threadIdx.x];
		}
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float sumZVal = 0.0f;
		int validCntVal = 0;
		for (i = 0; i < 128; ++i)
		{
			sumZVal += sumZ[i];
			validCntVal += validCnt[i];
		}
		float* pMinSumCnt = (float *)dResultBuf;
		pMinSumCnt[blockIdx.x] = sumZVal;
		pMinSumCnt += gridDim.x;
		*((int *)pMinSumCnt + blockIdx.x) = validCntVal;
	}
}

bool CalcMedianOnlyZ(float3& median, cv::cuda::GpuMat& dVMapRoi,
										 char* dResultBuf, char* resultBuf, float minDepth, float maxDepth)
{
	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();

	int block = 128;
	int grid = DivUp(dVMapRoi.cols, block);

	CalcSumCntZOnlyZKernel << <grid, block >> > (dResultBuf, dVMapRoi, minDepth, maxDepth);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	checkCudaErrors(cudaMemcpy(
		resultBuf,
		dResultBuf, (sizeof(float) + sizeof(int)) * grid, cudaMemcpyDeviceToHost));
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	float maxZ = -1.0e22f, meanZ = 0.0f;
	int cnt = 0;
	float* pSum = (float *)resultBuf;
	int* pCnt = (int *)(pSum + grid);

	for (int i = 0; i < grid; ++i)
	{
		meanZ += pSum[i];
		cnt += pCnt[i];
	}
	meanZ /= cnt;
	//timer.TimeEnd();
	//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//medianZ = (maxZ + meanZ) * 0.5f;
	median.z = meanZ;

	if (maxZ < -1.0e22f + MYEPS)
	{
		return false;
	}

	return true;
}

__global__ void CalcPlaneVertices2WithGravityKernel(float3* dPlaneVertexBuf, int* dIndex,
																										cv::cuda::PtrStepSz<float3> dVMap,
																										const float minDepth, const float maxDepth,
																										const int left, const int right, const int top, const int bottom)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;

	//int halfWidth = (right - left) / 2, halfHeight = (bottom - top) / 2;
	int halfWidth = 15, halfHeight = halfHeight = (bottom - top) / 2;
	if (((u >= 0 && u >= left - halfWidth && u <= left)
			|| (u >= right && u <= right + halfWidth && u < dVMap.cols))
			&& (v >= bottom - halfHeight && v <= bottom))
	{
		float3 vertex = dVMap.ptr(v)[u];
		if (!isnan(vertex.x)) {
			float depth = -vertex.y;
			if (!isnan(vertex.x))
			{
				if (depth >= minDepth && depth <= maxDepth)
				{
					int oldIndex = atomicAdd(dIndex, 1);
					dPlaneVertexBuf[oldIndex] = vertex;
				}
			}
		}
	}
}

__global__ void CalcPlaneVertices2WithoutGravityKernel(float3* dPlaneVertexBuf, int* dIndex,
																											 cv::cuda::PtrStepSz<float3> dVMap,
																											 const float minDepth, const float maxDepth,
																											 const int left, const int right, const int top, const int bottom)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;

	//int halfWidth = (right - left) / 2, halfHeight = (bottom - top) / 2;
	int halfWidth = 15, halfHeight = (bottom - top) / 2;
	if (((u >= 0 && u >= left - halfWidth && u <= left)
			|| (u >= right && u <= right + halfWidth && u < dVMap.cols))
			&& (v >= bottom - halfHeight && v <= bottom))
	{
		float3 vertex = dVMap.ptr(v)[u];
		if (!isnan(vertex.x)) {
			float depth = vertex.z;
			if (!isnan(vertex.x))
			{
				if (depth >= minDepth && depth <= maxDepth)
				{
					int oldIndex = atomicAdd(dIndex, 1);
					dPlaneVertexBuf[oldIndex] = vertex;
				}
			}
		}
	}
}

bool CalcAxisForPlane2(float& middleX, float& middleY, float& middleZ, float3& zAxis, float& zThresh,
											 cv::cuda::GpuMat& dVMap,
											 float minDepth, float maxDepth,
											 int left, int right, int top, int bottom,
											 float3* dPlaneVertexBuf, int* dIndex, bool withGravity)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(dVMap.cols, block.x);
	grid.y = getGridDim(dVMap.rows, block.y);

	//innoreal::InnoRealTimer timer;

	checkCudaErrors(cudaMemset(dIndex, 0, sizeof(int)));
	if (withGravity)
	{
		CalcPlaneVertices2WithGravityKernel << <grid, block >> > (dPlaneVertexBuf, dIndex, dVMap,
																															minDepth, maxDepth, left, right, top, bottom);
	}
	else
	{
		CalcPlaneVertices2WithoutGravityKernel << <grid, block >> > (dPlaneVertexBuf, dIndex, dVMap,
																																 minDepth, maxDepth, left, right, top, bottom);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	int planeVertexNum = 0;
	checkCudaErrors(cudaMemcpy(
		&planeVertexNum,
		dIndex, sizeof(int), cudaMemcpyDeviceToHost));
	//std::cout << "planeVertexNum: " << planeVertexNum << std::endl;
	std::vector<float3> planeVertexVec(planeVertexNum);
	checkCudaErrors(cudaMemcpy(
		planeVertexVec.data(),
		dPlaneVertexBuf, sizeof(float3) * planeVertexNum, cudaMemcpyDeviceToHost));

#if 0
	std::ofstream fs;
	fs.open("D:\\xjm\\snapshot\\test4_.ply");

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

	//timer.TimeStart();
	middleX = 0.0f, middleY = 0.0f, middleZ = 0.0f;
	float* p = (float*)planeVertexVec.data();
	for (int i = 0; i < planeVertexVec.size(); ++i)
	{
		middleX += *(p++);
		middleY += *(p++);
		middleZ += *(p++);
	}
	middleX /= planeVertexVec.size();
	middleY /= planeVertexVec.size();
	middleZ /= planeVertexVec.size();
	float covMat[9];
	p = (float*)planeVertexVec.data();
	for (int i = 0; i < planeVertexVec.size(); ++i)
	{
		*(p++) -= middleX;
		*(p++) -= middleY;
		*(p++) -= middleZ;
	}
	memset(covMat, 0, sizeof(covMat));
	//std::cout << "planeVertexVec size: " << planeVertexVec.size() << std::endl;
	p = (float*)planeVertexVec.data();
	float x, y, z;
	for (int i = 0; i < planeVertexVec.size(); ++i)
	{
		x = *(p++);
		y = *(p++);
		z = *(p++);

		covMat[0] += x * x;
		covMat[3] += x * y;
		covMat[6] += x * z;

		covMat[1] += y * x;
		covMat[4] += y * y;
		covMat[7] += y * z;

		covMat[2] += z * x;
		covMat[5] += z * y;
		covMat[8] += z * z;
	}
	//timer.TimeEnd();
	//printf("time CalcConvMatForPlaneKernel: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	PCA3x3(zAxis, covMat);
	//std::cout << "zAxis: " << zAxis.x << ", " << zAxis.y << ", " << zAxis.z << std::endl;
	//timer.TimeEnd();
	//printf("time CalcConvMatForPlaneKernel: %lf\n", timer.TimeGap_in_ms());

	if (zAxis.z < 0)
	{
		zAxis *= -1;
	}
#if 1
	if (withGravity && zAxis.z < cos(20.0 / 180.0 * 3.14)) // use the IMU gravity
	{
		std::cout << "zAxis: " << zAxis.x << ", " << zAxis.y << ", " << zAxis.z << std::endl;
		zAxis = make_float3(0, 0, 1);
		std::cout << "use imu gravity" << std::endl;
		//std::exit(0);
	}
#endif

	zThresh = 0; // middleX * zAxis.x + middleY * zAxis.y + middleZ * zAxis.z;
	zThresh -= 0.01;
	//zThresh -= 0.005;
}

__global__ void CalcPlaneVerticesKernel(float3* dPlaneVertexBuf, int* index, cv::cuda::PtrStepSz<float3> dVMap,
																				const float medianX, const float medianY, const float lowRadiusSquareThres,
																				const float highRadiusSquareThres)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u < dVMap.cols && v < dVMap.rows)
	{
		float radSquare, radX, radY;
		float3 vertex = dVMap.ptr(v)[u];
		if (!isnan(vertex.x))
		{
			radX = vertex.x - medianX;
			radY = vertex.y - medianY;
			radSquare = radX * radX + radY * radY;
			if (radSquare >= lowRadiusSquareThres && radSquare <= highRadiusSquareThres)
			{
				int oldIndex = atomicAdd(index, 1);
				dPlaneVertexBuf[oldIndex] = vertex;
			}
		}
	}
}

bool CalcAxisForPlane(float3& zAxis, float& zThresh, cv::cuda::GpuMat& dVMap, float3 median,
											float lowRadiusSquareThres,
											float hightRadiusSquareThres,
											float3* dPlaneVertexBuf, int* dIndex)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(dVMap.cols, block.x);
	grid.y = getGridDim(dVMap.rows, block.y);

	innoreal::InnoRealTimer timer;

	checkCudaErrors(cudaMemset(dIndex, 0, sizeof(int)));
	CalcPlaneVerticesKernel << <grid, block >> > (dPlaneVertexBuf, dIndex, dVMap, median.x, median.y,
																								lowRadiusSquareThres, hightRadiusSquareThres);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	int planeVertexNum = 0;
	checkCudaErrors(cudaMemcpy(
		&planeVertexNum,
		dIndex, sizeof(int), cudaMemcpyDeviceToHost));
	std::vector<float3> planeVertexVec(planeVertexNum);
	checkCudaErrors(cudaMemcpy(
		planeVertexVec.data(),
		dPlaneVertexBuf, sizeof(float3) * planeVertexNum, cudaMemcpyDeviceToHost));

	//timer.TimeStart();
	float middleX = 0.0f, middleY = 0.0f, middleZ = 0.0f;
	float* p = (float*)planeVertexVec.data();
	for (int i = 0; i < planeVertexVec.size(); ++i)
	{
		middleX += *(p++);
		middleY += *(p++);
		middleZ += *(p++);
	}
	middleX /= planeVertexVec.size();
	middleY /= planeVertexVec.size();
	middleZ /= planeVertexVec.size();
	float covMat[9];
	for (int i = 0; i < planeVertexVec.size(); ++i)
	{
		planeVertexVec[i].x -= middleX;
		planeVertexVec[i].y -= middleY;
		planeVertexVec[i].z -= middleZ;
	}
	memset(covMat, 0, sizeof(covMat));
	p = (float*)planeVertexVec.data();
	float x, y, z;
	for (int i = 0; i < planeVertexVec.size(); ++i)
	{
		x = *(p++);
		y = *(p++);
		z = *(p++);

		covMat[0] += x * x;
		covMat[3] += x * y;
		covMat[6] += x * z;

		covMat[1] += y * x;
		covMat[4] += y * y;
		covMat[7] += y * z;

		covMat[2] += z * x;
		covMat[5] += z * y;
		covMat[8] += z * z;
	}
	//timer.TimeEnd();
	//printf("time CalcConvMatForPlaneKernel: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	bool isValid = PCA3x3(zAxis, covMat);
	//timer.TimeEnd();
	//printf("time CalcConvMatForPlaneKernel: %lf\n", timer.TimeGap_in_ms());

	if (zAxis.z < 0)
	{
		zAxis *= -1;
	}
	if (!isValid || zAxis.z < cos(40.0 / 180.0 * 3.14)) // use the IMU gravity
	{
		zAxis = make_float3(0, 0, 1);
	}

	zThresh = (middleX - median.x) * zAxis.x + (middleY - median.y) * zAxis.y + (middleZ - median.z) * zAxis.z;
	zThresh -= 0.01;
	//zThresh += 100;

#if 0
		//timer.TimeEnd();
		//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	checkCudaErrors(cudaMemcpy(
		resultBuf,
		dResultBuf, (sizeof(float) + sizeof(int)) * grid, cudaMemcpyDeviceToHost));
	//timer.TimeEnd();
		//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//timer.TimeStart();
	float maxZ = -1.0e22f, meanZ = 0.0f;
	int cnt = 0;
	float* pSum = (float *)resultBuf;
	int* pCnt = (int *)(pSum + grid);

	for (int i = 0; i < grid; ++i)
	{
		meanZ += pSum[i];
		cnt += pCnt[i];
	}
	meanZ /= cnt;
	//timer.TimeEnd();
		//printf("time min mean: %lf\n", timer.TimeGap_in_ms());

	//medianZ = (maxZ + meanZ) * 0.5f;
	median.z = meanZ;

	if (maxZ < -1.0e22f + MYEPS)
	{
		return false;
	}
#endif

	return true;
}

__global__ void ComputeVmapKernel(const cv::cuda::PtrStepSz<float> depth, cv::cuda::PtrStep<float3> vmap,
																	float fx_inv, float fy_inv, float cx, float cy,
																	float depthCutoff, float3 RX, float3 RY, float3 RZ)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u < depth.cols && v < depth.rows)
	{
		float z = depth.ptr(v)[u];

		if (z > MYEPS && z < depthCutoff)
		{
			float x, y;
			x = z * (u - cx) * fx_inv;
			y = z * (v - cy) * fy_inv;
			vmap.ptr(v)[u] = RX * x + RY * y + RZ * z;
		}
		else
		{
			vmap.ptr(v)[u].x = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
		}
	}
}

void CreateVMap(cv::cuda::GpuMat& vmap, const cv::cuda::GpuMat& depth,
								float fx, float fy, float cx, float cy,
								const float depthCutoff, float3 RX, float3 RY, float3 RZ)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(depth.cols, block.x);
	grid.y = getGridDim(depth.rows, block.y);

	ComputeVmapKernel << <grid, block >> > (depth, vmap, 1.f / fx, 1.f / fy, cx, cy,
																					depthCutoff, RX, RY, RZ);
	checkCudaErrors(cudaGetLastError());
}

__global__ void RotateVerticesKernel(cv::cuda::PtrStepSz<float3> vmap, float3 RX, float3 RY, float3 RZ)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u < vmap.cols && v < vmap.rows)
	{
		float3& vertex = vmap.ptr(v)[u];
		if (!isnan(vertex.x))
		{
			vmap.ptr(v)[u] = RX * vertex.x + RY * vertex.y + RZ * vertex.z;
		}
	}
}

void RotateVertices(cv::cuda::GpuMat& vmap, float3 RX, float3 RY, float3 RZ)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = getGridDim(vmap.cols, block.x);
	grid.y = getGridDim(vmap.rows, block.y);

	RotateVerticesKernel << <grid, block >> > (vmap, RX, RY, RZ);
	checkCudaErrors(cudaGetLastError());
}
