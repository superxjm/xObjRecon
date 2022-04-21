#include "stdafx.h"
#include "RGBDOdometry.h"

#include "helpers/InnorealTimer.hpp"
#include "SiftGPU/xSift.h"

#include <fstream>
#include <pcl/registration/ia_ransac.h>

RGBDOdometry::RGBDOdometry(int width,
                           int height,
                           float cx, float cy, float fx, float fy,
                           float distThresh,
                           float angleThresh)
	: lastICPError(0),
	  lastICPCount(width * height),
	  lastRGBError(0),
	  lastRGBCount(width * height),
	  lastSO3Error(0),
	  lastSO3Count(width * height),
	  //lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
	  //lastb(Eigen::Matrix<double, 6, 1>::Zero()),
	  sobelSize(3),
	  sobelScale(1.0 / pow(2.0, sobelSize)),
	  maxDepthDeltaRGB(0.07),
	  maxDepthRGB(6.0),
	  distThres_(distThresh),
	  angleThres_(angleThresh),
	  width(width),
	  height(height),
	  cx(cx), cy(cy), fx(fx), fy(fy),
	  m_gmsMatcher(4000),
	  m_pointCloud(new pcl::PointCloud<pcl::PointXYZ>),
	  m_pointNormal(new pcl::PointCloud<pcl::Normal>)
{
	sumDataSE3.create(MAX_THREADS);
	outDataSE3.create(1);
	sumResidualRGB.create(MAX_THREADS);

	sumDataSO3.create(MAX_THREADS);
	outDataSO3.create(1);

	sumDataFeatureEstimation.create(MAX_THREADS);
	outDataFeatureEstimation.create(1);

	for (int i = 0; i < NUM_PYRS; i++)
	{
		int2 nextDim = {height >> i, width >> i};
		pyrDims.push_back(nextDim);
	}

	for (int i = 0; i < NUM_PYRS; i++)
	{
		lastDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
		lastImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

		nextDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
		nextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

		lastNextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

		lastdIdx[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
		lastdIdy[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

		pointClouds[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

		corresImg[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
	}

	intr.cx = cx;
	intr.cy = cy;
	intr.fx = fx;
	intr.fy = fy;

	iterations.resize(NUM_PYRS);

	depth_tmp.resize(NUM_PYRS);

	vmaps_g_prev_.resize(NUM_PYRS);
	nmaps_g_prev_.resize(NUM_PYRS);

	vmaps_curr_.resize(NUM_PYRS);
	nmaps_curr_.resize(NUM_PYRS);

	for (int i = 0; i < NUM_PYRS; ++i)
	{
		int pyr_rows = height >> i;
		int pyr_cols = width >> i;

		depth_tmp[i].create(pyr_rows, pyr_cols);

		vmaps_g_prev_[i].create(pyr_rows * 3, pyr_cols);
		nmaps_g_prev_[i].create(pyr_rows * 3, pyr_cols);

		vmaps_curr_[i].create(pyr_rows * 3, pyr_cols);
		nmaps_curr_[i].create(pyr_rows * 3, pyr_cols);
	}
	cudaMalloc((void**)&m_dCompressedVMap, width * height * sizeof(float4));
	cudaMalloc((void**)&m_dCompressedNMap, width * height * sizeof(float4));
	cudaMalloc((void**)&m_dIndexBuf, sizeof(int));
	cudaMalloc((void**)&m_dKeyPoints, width * height * sizeof(int));
	m_keyPoints = (int *)malloc(width * height * sizeof(int));
	cudaMalloc((void**)&m_dMatches, width * height * 4 * sizeof(ushort));
	cudaMalloc((void**)&m_dMatchesOrb, width * height * 4 * sizeof(ushort));
	cudaMalloc((void**)&m_dMatchesFPFH, width * height * 4 * sizeof(ushort));
	m_dGrayImg = cv::cuda::GpuMat(height, width, CV_8UC1);
	m_dDepthImg = cv::cuda::GpuMat(height, width, CV_32FC1);
	m_dMask = cv::cuda::GpuMat(height, width, CV_8UC1);

	vmaps_tmp.create(height * 4 * width);
	nmaps_tmp.create(height * 4 * width);

	minimumGradientMagnitudes.resize(NUM_PYRS);
	minimumGradientMagnitudes[0] = 5;
	minimumGradientMagnitudes[1] = 3;
	minimumGradientMagnitudes[2] = 1;

	m_compressedVMapPrev.reserve(600 * 600);
	m_compressedNMapPrev.reserve(600 * 600);
	m_compressedVMapCurr.reserve(600 * 600);
	m_compressedNMapCurr.reserve(600 * 600);
}

RGBDOdometry::~RGBDOdometry()
{
	cudaFree(m_dCompressedVMap);
	cudaFree(m_dCompressedNMap);
	cudaFree(m_dIndexBuf);
	cudaFree(m_dKeyPoints);
	free(m_keyPoints);
	cudaFree(m_dMatches);
	cudaFree(m_dMatchesOrb);
	cudaFree(m_dMatchesFPFH);
}

void RGBDOdometry::initICP(GPUTexture* filteredDepth, const float depthCutoff)
{
	cudaArray* textPtr;

	cudaGraphicsMapResources(1, &filteredDepth->cudaRes);

	cudaGraphicsSubResourceGetMappedArray(&textPtr, filteredDepth->cudaRes, 0, 0);

	cudaMemcpy2DFromArray(depth_tmp[0].ptr(0), depth_tmp[0].step(), textPtr, 0, 0, depth_tmp[0].colsBytes(),
	                      depth_tmp[0].rows(), cudaMemcpyDeviceToDevice);

	cudaGraphicsUnmapResources(1, &filteredDepth->cudaRes); 

	for (int i = 1; i < NUM_PYRS; ++i)
	{
		pyrDown(depth_tmp[i - 1], depth_tmp[i]);
	}

	for (int i = 0; i < NUM_PYRS; ++i)
	{
		createVMap(intr(i), depth_tmp[i], vmaps_curr_[i], depthCutoff);
		createNMap(vmaps_curr_[i], nmaps_curr_[i]);
	}

	cudaDeviceSynchronize();
}

void RGBDOdometry::initICP(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff)
{
	cudaArray* textPtr;

	cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
	cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
	cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);

	cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
	cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedNormals->cudaRes, 0, 0);
	cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);

	copyMaps(vmaps_tmp, nmaps_tmp, vmaps_curr_[0], nmaps_curr_[0]);

	for (int i = 1; i < NUM_PYRS; ++i)
	{
		resizeVMap(vmaps_curr_[i - 1], vmaps_curr_[i]);
		resizeNMap(vmaps_curr_[i - 1], vmaps_curr_[i]);
	}

	cudaDeviceSynchronize();
}

void RGBDOdometry::initICPModel(GPUTexture* predictedVertices,
                                GPUTexture* predictedNormals,
                                const float depthCutoff,
                                const Eigen::Matrix4f& modelPose)
{
	cudaArray* textPtr;

	cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
	cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
	cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);

	cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
	cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedNormals->cudaRes, 0, 0);
	cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);

	copyMaps(vmaps_tmp, nmaps_tmp, vmaps_g_prev_[0], nmaps_g_prev_[0]);

	for (int i = 1; i < NUM_PYRS; ++i)
	{
		resizeVMap(vmaps_g_prev_[i - 1], vmaps_g_prev_[i]);
		resizeNMap(nmaps_g_prev_[i - 1], nmaps_g_prev_[i]);
	}

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = modelPose.topLeftCorner(3, 3);
	Eigen::Vector3f tcam = modelPose.topRightCorner(3, 1);

	mat33 device_Rcam = Rcam;
	float3 device_tcam = *reinterpret_cast<float3*>(tcam.data());

	for (int i = 0; i < NUM_PYRS; ++i)
	{
		tranformMaps(vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
	}

	cudaDeviceSynchronize();
}

void RGBDOdometry::populateRGBDData(GPUTexture* rgb,
                                    DeviceArray2D<float>* destDepths,
                                    DeviceArray2D<unsigned char>* destImages)
{
	verticesToDepth(vmaps_tmp, destDepths[0], maxDepthRGB);
#if 0
    cv::Mat testDepthImg(m_dDepthImg.rows, m_dDepthImg.cols, CV_16UC1);
    checkCudaErrors(cudaMemcpy2D(testDepthImg.data, testDepthImg.step, depth_tmp[0], depth_tmp[0].step(),
        depth_tmp[0].colsBytes(), depth_tmp[0].rows(), cudaMemcpyDeviceToHost));
    cv::imshow("testDepthImgabc", testDepthImg * 30);
    cv::waitKey(0);
    cv::Mat testDepthImg2(m_dDepthImg.rows, m_dDepthImg.cols, CV_32FC1);
    checkCudaErrors(cudaMemcpy2D(testDepthImg2.data, testDepthImg2.step, destDepths[0], destDepths[0].step(),
        destDepths[0].colsBytes(), destDepths[0].rows(), cudaMemcpyDeviceToHost));
    cv::imshow("testDepthImgdef", testDepthImg2 * 30);
    cv::waitKey(0);
#endif

	for (int i = 0; i + 1 < NUM_PYRS; i++)
	{
		pyrDownGaussF(destDepths[i], destDepths[i + 1]);
	}

	cudaArray* textPtr;

	cudaGraphicsMapResources(1, &rgb->cudaRes);

	cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

	imageBGRToIntensity(textPtr, destImages[0]);

	cudaGraphicsUnmapResources(1, &rgb->cudaRes);

	for (int i = 0; i + 1 < NUM_PYRS; i++)
	{
		pyrDownUcharGauss(destImages[i], destImages[i + 1]);
	}

	cudaDeviceSynchronize();
}

void RGBDOdometry::populateRGBDData2(GPUTexture* rgb,
                                     DeviceArray2D<float>* destDepths,
                                     DeviceArray2D<unsigned char>* destImages)
{
    //verticesToDepth(vmaps_tmp, destDepths[0], maxDepthRGB);
    depthToDepthFloat32(depth_tmp[0], destDepths[0]);
#if 0
    cv::Mat testDepthImg(m_dDepthImg.rows, m_dDepthImg.cols, CV_16UC1);
    checkCudaErrors(cudaMemcpy2D(testDepthImg.data, testDepthImg.step, depth_tmp[0], depth_tmp[0].step(),
        depth_tmp[0].colsBytes(), depth_tmp[0].rows(), cudaMemcpyDeviceToHost));
    cv::imshow("testDepthImgabc", testDepthImg * 30);
    cv::waitKey(0);
    cv::Mat testDepthImg2(m_dDepthImg.rows, m_dDepthImg.cols, CV_32FC1);
    checkCudaErrors(cudaMemcpy2D(testDepthImg2.data, testDepthImg2.step, destDepths[0], destDepths[0].step(),
        destDepths[0].colsBytes(), destDepths[0].rows(), cudaMemcpyDeviceToHost));
    cv::imshow("testDepthImgdef", testDepthImg2 * 30);
    cv::waitKey(0);
#endif

    for (int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownGaussF(destDepths[i], destDepths[i + 1]);
    }

    cudaArray* textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    imageBGRToIntensity(textPtr, destImages[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

    for (int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownUcharGauss(destImages[i], destImages[i + 1]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initRGBModel(GPUTexture* rgb)
{
#if 0
    cudaArray* textPtr;

    cudaGraphicsMapResources(1, &filteredDepthFloat32->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, filteredDepthFloat32->cudaRes, 0, 0);

    cudaMemcpy2DFromArray(lastDepth[0].ptr(0), lastDepth[0].step(), textPtr, 0, 0, lastDepth[0].colsBytes(),
        lastDepth[0].rows(), cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &filteredDepthFloat32->cudaRes);

    for (int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownGaussF(lastDepth[i], lastDepth[i + 1]);
    }

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    imageBGRToIntensity(textPtr, lastImage[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

    for (int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownUcharGauss(lastImage[i], lastImage[i + 1]);
    }

    cudaDeviceSynchronize();
#endif
#if 1
	//NOTE: This depends on vmaps_tmp containing the corresponding depth from initICPModel
	populateRGBDData(rgb, &lastDepth[0], &lastImage[0]);
#endif
}

void RGBDOdometry::initRGB(GPUTexture* filteredDepthFloat32, GPUTexture* rgb)
{
#if 0
    cudaArray* textPtr;

    cudaGraphicsMapResources(1, &filteredDepthFloat32->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, filteredDepthFloat32->cudaRes, 0, 0);

    cudaMemcpy2DFromArray(nextDepth[0].ptr(0), nextDepth[0].step(), textPtr, 0, 0, nextDepth[0].colsBytes(),
        nextDepth[0].rows(), cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &filteredDepthFloat32->cudaRes);

    for (int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownGaussF(nextDepth[i], nextDepth[i + 1]);
    }

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    imageBGRToIntensity(textPtr, nextImage[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

    for (int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownUcharGauss(nextImage[i], nextImage[i + 1]);
    }

    cudaDeviceSynchronize();
#endif
#if 1
	//NOTE: This depends on vmaps_tmp containing the corresponding depth from initICP
	populateRGBDData2(rgb, &nextDepth[0], &nextImage[0]);
#endif
}

void RGBDOdometry::initFirstRGB(GPUTexture* rgb)
{
	cudaArray* textPtr;

	cudaGraphicsMapResources(1, &rgb->cudaRes);

	cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

	imageBGRToIntensity(textPtr, lastNextImage[0]);

	cudaGraphicsUnmapResources(1, &rgb->cudaRes);

	for (int i = 0; i + 1 < NUM_PYRS; i++)
	{
		pyrDownUcharGauss(lastNextImage[i], lastNextImage[i + 1]);
	}
}

class Utility
{
public:
	template <typename Derived>
	static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived>& theta)
	{
		typedef typename Derived::Scalar Scalar_t;

		Eigen::Quaternion<Scalar_t> dq;
		Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
		half_theta /= static_cast<Scalar_t>(2.0);
		dq.w() = static_cast<Scalar_t>(1.0);
		dq.x() = half_theta.x();
		dq.y() = half_theta.y();
		dq.z() = half_theta.z();
		return dq;
	}
};

void RGBDOdometry::calcOrbFPFHFeatures(std::vector<float4>& compressedVMap,
                                       std::vector<float4>& compressedNMap,
                                       DeviceArray2D<float>& dDepthImg,
                                       DeviceArray2D<unsigned char>& dColorImg,
                                       DeviceArray2D<float>& dVMap,
                                       DeviceArray2D<float>& dNMap, int idx, int& vexNum)
{
	checkCudaErrors(cudaMemcpy2D(m_dGrayImg.data, m_dGrayImg.step, dColorImg, dColorImg.step(),
		dColorImg.colsBytes(), dColorImg.rows(), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy2D(m_dDepthImg.data, m_dDepthImg.step, dDepthImg, dDepthImg.step(),
		dDepthImg.colsBytes(), dDepthImg.rows(), cudaMemcpyDeviceToDevice));
	CreateMask(m_dMask, m_dDepthImg); 
	cv::Mat mask;
	m_dMask.download(mask);
	int num = 0;
	for (int u = 0; u < mask.cols; u++) {
		for (int v = 0; v < mask.rows; v++) {
			if (mask.at<ushort>(v, u) != 0)
				num++;
		}
	}
	vexNum = num;
	if (idx == 0) {
		cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		cv::dilate(mask, mask, dilateKernel);
	}
	if (idx == 1) {
		cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13));
		cv::dilate(mask, mask, dilateKernel);
		//cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
		//cv::erode(mask, mask, dilateKernel);
	}
	//cv::imshow("mask", mask);
	//cv::waitKey(0);
	m_dMask.upload(mask);	
	m_gmsMatcher.computeOrbFeatures(m_dGrayImg, m_dMask, idx);	

#if 0

#if 1
	checkCudaErrors(cudaMemset(m_dIndexBuf, 0, sizeof(int)));
	CompressVMapNMap(m_dCompressedVMap, m_dCompressedNMap, m_dKeyPoints, m_dIndexBuf,
	                 dVMap, dNMap);	

	int vertexNum;
	checkCudaErrors(cudaMemcpy(&vertexNum, m_dIndexBuf, sizeof(int), cudaMemcpyDeviceToHost));

	compressedVMap.resize(vertexNum);
	compressedNMap.resize(vertexNum);
	checkCudaErrors(cudaMemcpy(compressedVMap.data(), m_dCompressedVMap,
		vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(compressedNMap.data(), m_dCompressedNMap,
		vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));
#if 0
	std::ofstream fs;
	char dir[256];
	sprintf(dir, "D:\\xjm\\snapshot\\test6_%d.ply", idx);
	fs.open(dir);

	float3 *vertex = (float3 *)compressedVMap.data();
	int pixelNum = vertexNum;

	// Write header
	fs << "ply";
	fs << "\nformat " << "ascii" << " 1.0";

	// Vertices
	fs << "\nelement vertex " << vertexNum;
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

	for (int i = 0; i < pixelNum; ++i)
	{
		if (!isnan(vertex[i].x))
		{
			fs << compressedVMap[i].x << " " << compressedVMap[i].y << " " << compressedVMap[i].z << " "
				<< (int)240 << " " << (int)240 << " " << (int)240 << " "
				<< compressedNMap[i].x << " " << compressedNMap[i].y << " " << compressedNMap[i].z << " "
				<< std::endl;
		}
	}

	// Close file
	fs.close();
	//std::exit(0);
#endif
	
#if 0
	m_pointCloud->resize(vertexNum);
	m_pointNormal->resize(vertexNum);	
#endif
#if 1
	m_pointCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	m_pointNormal = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
	m_pointCloud->resize(vertexNum);
	m_pointNormal->resize(vertexNum);	
#endif

#if 1
	checkCudaErrors(cudaMemcpy(m_pointCloud->points.data(), m_dCompressedVMap, 
		vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(m_pointNormal->points.data(), sizeof(pcl::Normal), 
		m_dCompressedNMap, sizeof(float4), sizeof(float4), vertexNum, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(m_keyPoints, m_dKeyPoints, vertexNum * sizeof(int), cudaMemcpyDeviceToHost));
#endif
#if 0
	std::vector<float4> vec1(vertexNum), vec2(vertexNum);
	checkCudaErrors(cudaMemcpy(vec1.data(), m_dCompressedVMap,
		vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vec2.data(), m_dCompressedNMap,
		vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));
	m_pointCloud->clear();
	m_pointNormal->clear();
	for (int i = 0; i < vertexNum; ++i)
	{
		m_pointCloud->push_back(pcl::PointXYZ(vec1[i].x, vec1[i].y, vec1[i].z));
		m_pointNormal->push_back(pcl::Normal(vec2[i].x, vec2[i].y, vec2[i].z));
	}
#endif

	//std::cout << sizeof(pcl::Normal) << std::endl;
	//std::exit(0);
	//checkCudaErrors(cudaMemcpy(m_pointNormal->points.data(), m_dCompressedNMap, vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));

#if 1
	m_gmsMatcher.computeFPFHFeatures(m_pointCloud, m_pointNormal,
	                                 m_keyPoints, width, height, idx);	
#endif
#endif
#endif
}

void RGBDOdometry::saveForTest(Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RPrev,
                               Eigen::Vector3f& tPrev,
                               Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RCurr,
                               Eigen::Vector3f& tCurr,
                               const char* saveDir)
{
	std::vector<float4> compressedVMapPrev, compressedNMapPrev,
	                    compressedVMapCurr, compressedNMapCurr;

	checkCudaErrors(cudaMemset(m_dIndexBuf, 0, sizeof(int)));
	CompressVMapNMap(m_dCompressedVMap, m_dCompressedNMap, m_dKeyPoints, m_dIndexBuf,
	                 vmaps_g_prev_[0], nmaps_g_prev_[0]);

	int vertexNum;
	checkCudaErrors(cudaMemcpy(&vertexNum, m_dIndexBuf, sizeof(int), cudaMemcpyDeviceToHost));

	compressedVMapPrev.resize(vertexNum);
	compressedNMapPrev.resize(vertexNum);
	checkCudaErrors(cudaMemcpy(compressedVMapPrev.data(), m_dCompressedVMap,
		vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(compressedNMapPrev.data(), m_dCompressedNMap,
		vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemset(m_dIndexBuf, 0, sizeof(int)));
	CompressVMapNMap(m_dCompressedVMap, m_dCompressedNMap, m_dKeyPoints, m_dIndexBuf,
	                 vmaps_curr_[0], nmaps_curr_[0]);

	checkCudaErrors(cudaMemcpy(&vertexNum, m_dIndexBuf, sizeof(int), cudaMemcpyDeviceToHost));

	compressedVMapCurr.resize(vertexNum);
	compressedNMapCurr.resize(vertexNum);
	checkCudaErrors(cudaMemcpy(compressedVMapCurr.data(), m_dCompressedVMap,
		vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(compressedNMapCurr.data(), m_dCompressedNMap,
		vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));

	vertexNum = compressedVMapPrev.size() + compressedVMapCurr.size();

	std::ofstream fs;
	//char dir[256];
	//sprintf(dir, "D:\\xjm\\snapshot\\test6_%d.ply", 3);
	fs.open(saveDir);

	// Write header
	fs << "ply";
	fs << "\nformat " << "ascii" << " 1.0";

	// Vertices
	fs << "\nelement vertex " << vertexNum;
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

	for (int i = 0; i < compressedVMapPrev.size(); ++i)
	{
		fs << compressedVMapPrev[i].x << " " << compressedVMapPrev[i].y << " " << compressedVMapPrev[i].z << " "
			<< (int)0 << " " << (int)240 << " " << (int)0 << " "
			<< compressedNMapPrev[i].x << " " << compressedNMapPrev[i].y << " " << compressedNMapPrev[i].z << " "
			<< std::endl;
	}
	for (int i = 0; i < compressedVMapCurr.size(); ++i)
	{
		//compressedVMapPrevif (!isnan(vertex[i].x))
		{
			Eigen::Vector3f pos(compressedVMapCurr[i].x, compressedVMapCurr[i].y, compressedVMapCurr[i].z);
			Eigen::Vector3f nor(compressedNMapCurr[i].x, compressedNMapCurr[i].y, compressedNMapCurr[i].z);
			pos = RCurr * pos + tCurr;
			nor = RCurr * nor;
			fs << pos.x() << " " << pos.y() << " " << pos.z() << " "
				<< (int)240 << " " << (int)0 << " " << (int)0 << " "
				<< nor.x() << " " << nor.y() << " " << nor.z() << " "
				<< std::endl;
		}
	}

	// Close file
	fs.close();
	//std::exit(0);

#if 0
	vertexNum = compressedVMapCurr.size();

	fs.open(saveDir);

	// Write header
	fs << "ply";
	fs << "\nformat " << "ascii" << " 1.0";

	// Vertices
	fs << "\nelement vertex " << vertexNum;
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

	for (int i = 0; i < vertexNum; ++i)
	{
		//compressedVMapPrevif (!isnan(vertex[i].x))
		{
			Eigen::Vector3f pos(compressedVMapCurr[i].x, compressedVMapCurr[i].y, compressedVMapCurr[i].z);
			Eigen::Vector3f nor(compressedNMapCurr[i].x, compressedNMapCurr[i].y, compressedNMapCurr[i].z);
			pos = RCurr * pos + tCurr;
			nor = RCurr * nor;
			fs << pos.x() << " " << pos.y() << " " << pos.z() << " "
				<< (int)240 << " " << (int)240 << " " << (int)240 << " "
				<< nor.x() << " " << nor.y() << " " << nor.z() << " "
				<< std::endl;
		}
	}

	// Close file
	fs.close();
	//std::exit(0);
#endif
}

void RGBDOdometry::getIncrementalTransformationOrbFPFH(Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RPrev,
                                                       Eigen::Vector3f& tPrev,
                                                       Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RCurr,
                                                       Eigen::Vector3f& tCurr,
																											 int& vexNum,
                                                       std::vector<cv::Mat>* colorImgForVis)
{
	//std::cout << "getIncrementalTransformationOrbFPFH" << std::endl;
	//std::exit(0);
	//innoreal::InnoRealTimer timer;

	//timer.TimeStart();
#if 0
	SiftGPU siftDevice;
	siftDevice.SetParams(Resolution::getInstance().width(), Resolution::getInstance().height(), false, 1024, 0.01f, 100.0f);
	siftDevice.InitSiftGPU();
	SiftMatchGPU siftMatcherDevice(1024);
	siftMatcherDevice.InitSiftMatch();
	ImagePairMatch m_imagePairMatch;
	checkCudaErrors(cudaMalloc(&m_imagePairMatch.d_numMatches, sizeof(int)));
	checkCudaErrors(cudaMalloc(&m_imagePairMatch.d_distances, sizeof(float) * 1024));
	checkCudaErrors(cudaMalloc(&m_imagePairMatch.d_keyPointIndices, sizeof(uint2) * 1024));

	cv::Mat grayImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1);
	cv::Mat colorImg1(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3);
	cv::Mat colorImg2(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3);
	std::vector<uchar> vec;
	int step;
	lastImage[0].download(grayImg.data, Resolution::getInstance().width());
	std::cout << 4 << std::endl;
	for (int r = 0; r < grayImg.rows; ++r)
	{
		for (int c = 0; c < grayImg.cols; ++c)
		{
			colorImg1.at<cv::Vec3b>(r, c)[0] = grayImg.at<uchar>(r, c);
			colorImg1.at<cv::Vec3b>(r, c)[1] = grayImg.at<uchar>(r, c);
			colorImg1.at<cv::Vec3b>(r, c)[2] = grayImg.at<uchar>(r, c);
		}
	}
	nextImage[0].download(grayImg.data, Resolution::getInstance().width());
	for (int r = 0; r < grayImg.rows; ++r)
	{
		for (int c = 0; c < grayImg.cols; ++c)
		{
			colorImg2.at<cv::Vec3b>(r, c)[0] = grayImg.at<uchar>(r, c);
			colorImg2.at<cv::Vec3b>(r, c)[1] = grayImg.at<uchar>(r, c);
			colorImg2.at<cv::Vec3b>(r, c)[2] = grayImg.at<uchar>(r, c);
		}
	}
	cv::imshow("colorImg1", colorImg1);
	cv::imshow("colorImg2", colorImg2);
	cv::waitKey(0);

	std::vector<SIFTKeyPoint> srcKeyPointsVec(1024);
	std::vector<SIFTKeyPoint> targetKeyPointsVec(1024);
#if 0
	{
		cv::Mat grayImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1);
		cv::Mat grayImgFloat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_32FC1);
		cv::Mat depthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1);
		lastImage[0].download(grayImg.data, Resolution::getInstance().width());
		//lastDepth[0].download(depthImg.data, Resolution::getInstance().width());
		grayImg.convertTo(grayImgFloat, CV_32FC1, 1 / 255.0f);
		float *d_GrayImg;
		float *d_DepthImg;
		checkCudaErrors(cudaMalloc(&d_GrayImg, sizeof(float) * grayImgFloat.rows * grayImgFloat.cols));
		checkCudaErrors(cudaMemcpy(d_GrayImg, grayImgFloat.data,
										sizeof(float) * grayImgFloat.rows * grayImgFloat.cols,
										cudaMemcpyHostToDevice));

		int success = siftDevice.RunSIFT(d_GrayImg, lastDepth[0].ptr(0));
		if (!success) throw std::exception("Error running SIFT detection");

		SIFTImageGPU siftImageDevice;
		checkCudaErrors(cudaMalloc(&siftImageDevice.d_keyPoints, sizeof(SIFTKeyPoint) * 1024));
		checkCudaErrors(cudaMalloc(&siftImageDevice.d_keyPointDescs, sizeof(SIFTKeyPointDesc) * 1024));

		int numKeypoints = siftDevice.GetKeyPointsAndDescriptorsCUDA(siftImageDevice, lastDepth[0].ptr(0), 1024);
		checkCudaErrors(cudaMemcpy(srcKeyPointsVec.data(), siftImageDevice.d_keyPoints,
										sizeof(SIFTKeyPoint) * numKeypoints, cudaMemcpyDeviceToHost));
		std::cout << "success: " << success << std::endl;
		std::cout << "numKeypoints: " << numKeypoints << std::endl;
		siftMatcherDevice.SetDescriptors(0, numKeypoints, (unsigned char*)siftImageDevice.d_keyPointDescs);

		for (int i = 0; i < numKeypoints; i++)
		{
			cv::Point2f left = cv::Point2f(srcKeyPointsVec[i].pos.x, srcKeyPointsVec[i].pos.y);
			cv::circle(colorImg1, left, 1, cv::Scalar(255, 0, 0), 2);
		}
	}
#endif
	{
		cv::Mat grayImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1);
		cv::Mat grayImgFloat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_32FC1);
		cv::Mat depthImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1);
		nextImage[0].download(grayImg.data, Resolution::getInstance().width());
		//lastDepth[0].download(depthImg.data, Resolution::getInstance().width());
		cv::imshow("grayImg", grayImg);
		cv::waitKey(0);
		grayImg.convertTo(grayImgFloat, CV_32FC1, 1 / 255.0f);
		float *d_GrayImg;
		float *d_DepthImg;
		checkCudaErrors(cudaMalloc(&d_GrayImg, sizeof(float) * grayImgFloat.rows * grayImgFloat.cols));
		checkCudaErrors(cudaMemcpy(d_GrayImg, grayImgFloat.data,
										sizeof(float) * grayImgFloat.rows * grayImgFloat.cols,
										cudaMemcpyHostToDevice));

		int success = siftDevice.RunSIFT(d_GrayImg, nextDepth[0].ptr(0));
		if (!success) throw std::exception("Error running SIFT detection");

		SIFTImageGPU siftImageDevice;
		checkCudaErrors(cudaMalloc(&siftImageDevice.d_keyPoints, sizeof(SIFTKeyPoint) * 1024));
		checkCudaErrors(cudaMalloc(&siftImageDevice.d_keyPointDescs, sizeof(SIFTKeyPointDesc) * 1024));

		int numKeypoints = siftDevice.GetKeyPointsAndDescriptorsCUDA(siftImageDevice, nextDepth[0].ptr(0), 1024);
		checkCudaErrors(cudaMemcpy(targetKeyPointsVec.data(), siftImageDevice.d_keyPoints,
										sizeof(SIFTKeyPoint) * numKeypoints, cudaMemcpyDeviceToHost));
		std::cout << "success: " << success << std::endl;
		std::cout << "numKeypoints: " << numKeypoints << std::endl;
		siftMatcherDevice.SetDescriptors(1, numKeypoints, (unsigned char*)siftImageDevice.d_keyPointDescs);

		for (int i = 0; i < numKeypoints; i++)
		{
			cv::Point2f left = cv::Point2f(targetKeyPointsVec[i].pos.x, targetKeyPointsVec[i].pos.y);
			cv::circle(colorImg2, left, 1, cv::Scalar(255, 0, 0), 2);
		}
	}
	cv::imshow("colorImg1", colorImg1);
	cv::imshow("colorImg2", colorImg2);

	float distmax = 0.7f, ratioMax = 0.7;
	siftMatcherDevice.GetSiftMatch(1024, m_imagePairMatch,
																 make_uint2(0, 0),
																 distmax, ratioMax);

	int matchNum;
	checkCudaErrors(cudaMemcpy(&matchNum, &m_imagePairMatch.d_numMatches[0], sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << "matchNum: " << matchNum << std::endl;
	std::vector<uint> m_keyPointIndices;
	m_keyPointIndices.resize(1024 * 2);
	checkCudaErrors(cudaMemcpy(m_keyPointIndices.data(), m_imagePairMatch.d_keyPointIndices, matchNum * sizeof(uint2), cudaMemcpyDeviceToHost));

	uint srcMatchingInd, targetMatchingInd;

	std::vector<ushort> testMatchesSift;

	for (int matchInd = 0; matchInd < matchNum; ++matchInd)
	{
		srcMatchingInd = m_keyPointIndices[2 * matchInd];
		targetMatchingInd = m_keyPointIndices[2 * matchInd + 1];
		std::cout << srcMatchingInd << " : " << targetMatchingInd << std::endl;

		cv::Point2f left_point = cv::Point2f(srcKeyPointsVec[srcMatchingInd].pos.x,
																				 srcKeyPointsVec[srcMatchingInd].pos.y);
		cv::Point2f right_point = cv::Point2f(targetKeyPointsVec[targetMatchingInd].pos.x,
																					targetKeyPointsVec[targetMatchingInd].pos.y);
		std::cout << "left point: " << left_point.x << " : " << left_point.y << std::endl;
		std::cout << "right point: " << right_point.x << " : " << right_point.y << std::endl;
		testMatchesSift.push_back(ushort(left_point.y));
		testMatchesSift.push_back(ushort(left_point.x));
		testMatchesSift.push_back(ushort(right_point.y));
		testMatchesSift.push_back(ushort(right_point.x));
	}	
#if 1
	cv::Mat showSift = m_gmsMatcher.DrawMatches(colorImg1, colorImg2, testMatchesSift);
	cv::imshow("showSift", showSift);
	cv::waitKey(0);
#endif
	
	std::exit(0);
#endif
	int vexNum1, vexNum2;
	calcOrbFPFHFeatures(m_compressedVMapPrev,
	                    m_compressedNMapPrev,
	                    lastDepth[0],
	                    lastImage[0],
	                    vmaps_g_prev_[0],
	                    nmaps_g_prev_[0], 0, vexNum1);
	calcOrbFPFHFeatures(m_compressedVMapCurr,
	                    m_compressedNMapCurr,
	                    nextDepth[0],
	                    nextImage[0],
	                    vmaps_curr_[0],
	                    nmaps_curr_[0], 1, vexNum2);
	vexNum = std::min(vexNum1, vexNum2);
	//timer.TimeEnd();
	//std::cout << "calc feature time: " << timer.TimeGap_in_ms() << std::endl;
#if 0
	cv::Mat grayImg(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC1);
	cv::Mat colorImg1(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3);
	cv::Mat colorImg2(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3);
	std::vector<uchar> vec;
	int step;
	lastImage[0].download(grayImg.data, Resolution::getInstance().width());
    std::cout << 4 << std::endl;
	for (int r = 0; r < grayImg.rows; ++r)
	{
		for (int c = 0; c < grayImg.cols; ++c)
		{
			colorImg1.at<cv::Vec3b>(r, c)[0] = grayImg.at<uchar>(r, c);
			colorImg1.at<cv::Vec3b>(r, c)[1] = grayImg.at<uchar>(r, c);
			colorImg1.at<cv::Vec3b>(r, c)[2] = grayImg.at<uchar>(r, c);
		}
	}
	nextImage[0].download(grayImg.data, Resolution::getInstance().width());
	for (int r = 0; r < grayImg.rows; ++r)
	{
		for (int c = 0; c < grayImg.cols; ++c)
		{
			colorImg2.at<cv::Vec3b>(r, c)[0] = grayImg.at<uchar>(r, c);
			colorImg2.at<cv::Vec3b>(r, c)[1] = grayImg.at<uchar>(r, c);
			colorImg2.at<cv::Vec3b>(r, c)[2] = grayImg.at<uchar>(r, c);
		}
	}
	//cv::imshow("colorImg1", colorImg1);
	//cv::imshow("colorImg2", colorImg2);
	//cv::waitKey(1);
#endif

	//timer.TimeStart();
	std::vector<int> matchesIdxOrbVec;
	std::vector<int> matchesDistOrbVec;
	std::vector<cv::KeyPoint> srcKeyPointOrb;
	std::vector<cv::KeyPoint> targetKeyPointOrb;
	std::vector<cv::DMatch> matchVec;
	m_gmsMatcher.getMultiMatchesOrb(matchVec,
																	matchesIdxOrbVec,
	                                matchesDistOrbVec,
	                                srcKeyPointOrb,
	                                targetKeyPointOrb);
#if 0
	std::vector<int> matchesIdxFPFHVec;
	std::vector<float> matchesDistFPFHVec;
	std::vector<cv::KeyPoint> srcKeyPointFPFH;
	std::vector<cv::KeyPoint> targetKeyPointFPFH;
	m_gmsMatcher.getMultiMatchesFPFH(matchesIdxFPFHVec,
																	 matchesDistFPFHVec,
																	 srcKeyPointFPFH,
																	 targetKeyPointFPFH);
#endif

	//timer.TimeEnd();
	//std::cout << "calc match time: " << timer.TimeGap_in_ms() << std::endl;
	//timer.TimeStart();
	std::vector<int> matchesIdxVec;
	std::vector<cv::KeyPoint> srcKeyPoint;
	std::vector<cv::KeyPoint> targetKeyPoint;
#if 1
	for (size_t i = 0; i < matchesIdxOrbVec.size() / 2; ++i)
	{
		//float ratio = matchesDistOrbVec[2 * i] / (float)matchesDistOrbVec[2 * i + 1];
		//std::cout << matchesDistOrbVec[2 * i] << std::endl;
		//if (ratio < 0.9) {
		matchesIdxVec.push_back(i);
		matchesIdxVec.push_back(matchesIdxOrbVec[2 * i]);
		matchesIdxVec.push_back(i);
		matchesIdxVec.push_back(matchesIdxOrbVec[2 * i + 1]);
		//}
	}
	for (size_t i = 0; i < srcKeyPointOrb.size(); ++i)
	{
		srcKeyPoint.push_back(srcKeyPointOrb[i]);
	}
	for (size_t i = 0; i < targetKeyPointOrb.size(); ++i)
	{
		targetKeyPoint.push_back(targetKeyPointOrb[i]);
	}
#else
	for (size_t i = 0; i < matchesIdxFPFHVec.size() / 2; ++i)
	{
		float ratio = matchesDistFPFHVec[2 * i] / (float)matchesDistFPFHVec[2 * i + 1];
		std::cout << ratio << std::endl;
		if (ratio < 0.9) {
			matchesIdxVec.push_back(i);
			matchesIdxVec.push_back(matchesIdxFPFHVec[2 * i]);
		//matchesIdxVec.push_back(i);
		//matchesIdxVec.push_back(matchesIdxFPFHVec[2 * i + 1]);
		}
	}
#endif	

#if 0
	std::cout << 1 << std::endl;
	std::vector<ushort> testMatchesVecFPFH;
	for (size_t i = 0; i < matchesIdxVec.size() / 2; ++i)
	{
		cv::Point2f left_point = srcKeyPointFPFH[matchesIdxVec[2 * i]].pt;
		cv::Point2f right_point = targetKeyPointFPFH[matchesIdxVec[2 * i + 1]].pt;
		testMatchesVecFPFH.push_back(ushort(left_point.y));
		testMatchesVecFPFH.push_back(ushort(left_point.x));
		testMatchesVecFPFH.push_back(ushort(right_point.y));
		testMatchesVecFPFH.push_back(ushort(right_point.x));
#if 0
		left_point = srcKeyPointFPFH[i].pt;
		right_point = targetKeyPointFPFH[matchesIdxFPFHVec[2 * i + 1]].pt;
		testMatchesVecFPFH.push_back(ushort(left_point.y));
		testMatchesVecFPFH.push_back(ushort(left_point.x));
		testMatchesVecFPFH.push_back(ushort(right_point.y));
		testMatchesVecFPFH.push_back(ushort(right_point.x));
#endif
	}
	std::cout << 2 << std::endl;
	cv::Mat showFPFH = m_gmsMatcher.DrawMatches(colorImg1, colorImg2, testMatchesVecFPFH);
	cv::imshow("showFPFH", showFPFH);
	cv::waitKey(0);
	//std::exit(0);
#endif
#if 0
	//for (int nn = 0; nn < 10; ++nn) {
		std::vector<ushort> testMatchesVecOrb;
		for (size_t i = 0; i < matchesIdxVec.size() / 2; ++i)
		{
			//std::cout << fpfhScoreVec[i] << std::endl;
			//if (i % 10 == nn)
			{
				cv::Point2f left_point = srcKeyPointOrb[matchesIdxVec[2 * i]].pt;
				cv::Point2f right_point = targetKeyPointOrb[matchesIdxVec[2 * i + 1]].pt;
				testMatchesVecOrb.push_back(ushort(left_point.y));
				testMatchesVecOrb.push_back(ushort(left_point.x));
				testMatchesVecOrb.push_back(ushort(right_point.y));
				testMatchesVecOrb.push_back(ushort(right_point.x));
			}	
		}
		cv::Mat showOrb = m_gmsMatcher.DrawMatches(colorImg1, colorImg2, testMatchesVecOrb);
		cv::imshow("showOrb", showOrb);
		cv::waitKey(0);
	//}
	//cv::Mat showOrb = m_gmsMatcher.DrawMatches(colorImg1, colorImg2, testMatchesVecOrb);
	//cv::imshow("showOrb", showOrb);
	//cv::waitKey(0);
	//std::exit(0);
#if 0
    cv::Mat showOrbFPFH = m_gmsMatcher.DrawMatches2(colorImgForVis->at(colorImgForVis->size() - 2),
        colorImgForVis->at(colorImgForVis->size() - 1),
        testMatchesVecOrb,
        testMatchesVecFPFH);
    cv::imshow("showOrbFPFH", showOrbFPFH);
    cv::waitKey(0);
#endif
#endif

	int matchesNumOrb = matchesIdxOrbVec.size() / 2;
	int matchesNum = matchesIdxVec.size() / 2;	
	std::vector<int> sampleSet(matchesNum);
	for (int i = 0; i < sampleSet.size(); ++i)
	{
		sampleSet[i] = i;
	}
	srand((unsigned)time(NULL));

#if 0
	std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> RCurrVec;
	std::vector<Eigen::Vector3f> tCurrVec;
	std::vector<float> scoreVec;
#endif
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> bestRCurr;
	Eigen::Vector3f besttCurr;
	float maxScore = 0.0f, minScore = 1.0e24f;
	int firstOrSecond = 0;

	int sampleNum = 5, sampleIdx;
	std::vector<ushort> matchesVec;
	float distThreshForSampleSquare = 3.0 * 3.0;
	//timer.TimeStart();
	int count = 0;
	for (int ii = 0; ii < 2000; ++ii)
	//while (true)
	{
		//timer.Timestart();
		//if (ii > 1000)
		//{
			//distThreshForSampleSquare = 1.0 * 1.0;
		//}
		
		int randIdx, firstOrSecond;
		float diffX, diffY, distForSample, maxDistForSample;
		cv::Point2f leftPoint, rightPoint;
		matchesVec.clear();
		sampleIdx = 0;
		while (sampleIdx < sampleNum)
		{
			//std::cout << matchesNumOrb << std::endl;
			//if (ii < 1000)
			//{
				randIdx = rand() % matchesNumOrb;
			//}
			//else
			//{
				//randIdx = rand() % matchesNum;
			//}
			std::swap(sampleSet[sampleIdx], sampleSet[randIdx]);

			leftPoint = srcKeyPoint[matchesIdxVec[2 * sampleSet[sampleIdx]]].pt;
			rightPoint = targetKeyPoint[matchesIdxVec[2 * sampleSet[sampleIdx] + 1]].pt;
			maxDistForSample = 0.0f;
			for (int preIdx = 0; preIdx < matchesVec.size() / 4; ++preIdx)
			{
				cv::Point2f preLeftPoint = srcKeyPoint[matchesIdxVec[2 * sampleSet[preIdx]]].pt;

				diffX = leftPoint.x - preLeftPoint.x;
				diffY = leftPoint.y - preLeftPoint.y;
				distForSample = diffX * diffX + diffY * diffY;
				if (distForSample > maxDistForSample)
				{
					maxDistForSample = distForSample;
				}
			}
			
			//std::cout << "maxDistForSample: " << maxDistForSample << std::endl;
			if (matchesVec.size() == 0 || maxDistForSample > distThreshForSampleSquare)
			{
				matchesVec.push_back(ushort(leftPoint.y));
				matchesVec.push_back(ushort(leftPoint.x));
				matchesVec.push_back(ushort(rightPoint.y));
				matchesVec.push_back(ushort(rightPoint.x));
				++sampleIdx;
			}
		}

#if 0
		for (size_t i = 0; i < sampleNum; ++i)
		{
			std::cout << sampleSet[i] << ", ";
		}
		std::cout << std::endl;
#endif

#if 0
		cv::Mat show = m_gmsMatcher.DrawMatches(colorImg1, colorImg2, matchesVec);
		cv::imshow("show", show);
		cv::waitKey(0);
#endif
		int matchesNum = matchesVec.size() / 4;
		//std::cout << "matchesNumOrb: " << matchesNumOrb << std::endl;
		checkCudaErrors(cudaMemcpy(m_dMatches, matchesVec.data(),
			matchesNum * sizeof(ushort) * 4,
			cudaMemcpyHostToDevice));
#if 0
		checkCudaErrors(cudaMemcpy(m_dMatchesFPFH, m_matchesVecFPFH.data(),
			matchesNumFPFH * sizeof(ushort) * 4,
			cudaMemcpyHostToDevice));
#endif

#if 1	
#endif

		Eigen::Matrix<float, 3, 3, Eigen::RowMajor> RPrevInv = RPrev.inverse();
		mat33 dRPrevInv = RPrevInv;
		float3 dtPrev = *reinterpret_cast<float3*>(tPrev.data());

		Eigen::Matrix<double, 4, 4, Eigen::RowMajor> relativeRt = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();

		RCurr = RPrev;
		tCurr = tPrev;
		mat33 device_Rcurr;
		float3 device_tcurr;
		float residual[2];
		for (int iter = 0; iter < 3; ++iter)
		{
			Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_p2p;
			Eigen::Matrix<float, 6, 1> b_p2p;

			device_Rcurr = RCurr;
			device_tcurr = *reinterpret_cast<float3*>(tCurr.data());
			featureStep(device_Rcurr,
			            device_tcurr,
			            vmaps_curr_[0],
			            nmaps_curr_[0],
			            dRPrevInv,
			            dtPrev,
			            vmaps_g_prev_[0],
			            nmaps_g_prev_[0],
			            m_dMatches,
			            matchesNum,
			            sumDataSE3,
			            outDataSE3,
			            A_p2p.data(),
			            b_p2p.data(),
			            &residual[0],
			            GPUConfig::getInstance().icpStepThreads,
			            GPUConfig::getInstance().icpStepBlocks);
			float featureError = sqrt(residual[0]) / residual[1];
			float featureCount = residual[1];
			//std::cout << "featureError: " << featureError << " : " << featureCount << std::endl;


			lastA66.setZero();
			lastA66.block<6, 6>(0, 0) = A_p2p.cast<double>();
			lastb61.setZero();
			lastb61.block<6, 1>(0, 0) = b_p2p.cast<double>();

			Eigen::Matrix<double, 6, 1> result;
			result = lastA66.llt().solve(lastb61);

			Eigen::Isometry3f rgbOdom;

			OdometryProvider::computeUpdateSE3(relativeRt, result.block<6, 1>(0, 0), rgbOdom);

			Eigen::Isometry3f currentT;
			currentT.setIdentity();
			currentT.rotate(RPrev);
			currentT.translation() = tPrev;

			currentT = currentT * rgbOdom;

			tCurr = currentT.translation();
			RCurr = currentT.rotation();
		}
		device_Rcurr = RCurr;
		device_tcurr = *reinterpret_cast<float3*>(tCurr.data());

        //Eigen::Vector3f eulerAngles = (RCurr.inverse() * RPrev).eulerAngles(0, 1, 2);
        //if (abs(eulerAngles.x()) > 0.34 || abs(eulerAngles.y()) > 0.34 || abs(eulerAngles.z()) > 0.34)
        //{
            //continue;
        //}

		//std::cout << "intr: " << std::endl;
		//std::cout << intr.fx << " " << intr.fy << std::endl;
		//std::cout << intr.cx << " " << intr.cy << std::endl;
		featureEstimation(device_Rcurr,
		                  device_tcurr,
		                  vmaps_curr_[0],
		                  nmaps_curr_[0],
		                  dRPrevInv,
		                  dtPrev,
		                  intr(0),
		                  vmaps_g_prev_[0],
		                  nmaps_g_prev_[0],
		                  distThres_,
		                  angleThres_,
		                  sumDataFeatureEstimation,
		                  outDataFeatureEstimation,
		                  &residual[0],
		                  GPUConfig::getInstance().icpStepThreads,
		                  GPUConfig::getInstance().icpStepBlocks);
		float featureEstimationError = sqrt(residual[0]) / residual[1];
		float featureEstimationCount = residual[1];
		//std::cout << "featureEstimationError: " << featureEstimationError << " : " << featureEstimationCount << std::endl;
		//timer.TimeEnd();
		//std::cout << "time: " << timer.TimeGap_in_ms() << std::endl;
#if 0
		char saveDir[256];
		sprintf(saveDir, "D:\\xjm\\snapshot\\test6_%f_%f.ply", featureEstimationError, featureEstimationCount);
		saveForTest(RPrev,
		            tPrev,
		            RCurr,
		            tCurr,
		            saveDir);
#endif
		float score = featureEstimationCount;
		if (!isnan(score) && score > maxScore)
		{
			std::cout << "score: " << count << " : " << score << std::endl;
			maxScore = score;
			bestRCurr = RCurr;
			besttCurr = tCurr;
		}
#if 0
		if (maxScore > 2200)
		{
			break;
		}
#endif
#if 0
		RCurrVec.push_back(RCurr);
		tCurrVec.push_back(tCurr);
		scoreVec.push_back(featureEstimationCount);
#endif
		++count;
	}
	//timer.TimeEnd();
	//std::cout << "pause start time: " << timer.TimeGap_in_ms() << std::endl;
	RCurr = bestRCurr;
	tCurr = besttCurr;

#if 0
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> bestRCurr;
	Eigen::Vector3f besttCurr;
	float maxScore = 0.0f, minScore = 1.0e24f;
	for (int i = 0; i < scoreVec.size(); ++i)
	{
		if (!isnan(scoreVec[i]) && scoreVec[i] > maxScore)
		{
			maxScore = scoreVec[i];
			bestRCurr = RCurrVec[i];
			besttCurr = tCurrVec[i];
		}
	}
	std::cout << "score: " << maxScore << std::endl;
#endif
#if 0
	char saveDir[256];
	sprintf(saveDir, "D:\\xjm\\snapshot\\test6_%f_%f.ply", 0.0, 0.0);
	saveForTest(RPrev,
		tPrev,
		RCurr,
		tCurr,
		saveDir);
#endif

	//std::exit(0);

#if 0
	int vertexNum = compressedVMapPrev.size();	

	std::ofstream fs;
	char dir[256];
	sprintf(dir, "D:\\xjm\\snapshot\\test6_%d.ply", 3);
	fs.open(dir);

	// Write header
	fs << "ply";
	fs << "\nformat " << "ascii" << " 1.0";

	// Vertices
	fs << "\nelement vertex " << vertexNum;
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

	for (int i = 0; i < vertexNum; ++i)
	{
		//compressedVMapPrevif (!isnan(vertex[i].x))
		{
			fs << compressedVMapPrev[i].x << " " << compressedVMapPrev[i].y << " " << compressedVMapPrev[i].z << " "
				<< (int)240 << " " << (int)240 << " " << (int)240 << " "
				<< compressedNMapPrev[i].x << " " << compressedNMapPrev[i].y << " " << compressedNMapPrev[i].z << " "
				<< std::endl;
		}
	}

	// Close file
	fs.close();
	//std::exit(0);

	vertexNum = compressedVMapCurr.size();

	sprintf(dir, "D:\\xjm\\snapshot\\test6_%d.ply", 4);
	fs.open(dir);

	// Write header
	fs << "ply";
	fs << "\nformat " << "ascii" << " 1.0";

	// Vertices
	fs << "\nelement vertex " << vertexNum;
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

	for (int i = 0; i < vertexNum; ++i)
	{
		//compressedVMapPrevif (!isnan(vertex[i].x))
		{
			Eigen::Vector3f pos(compressedVMapCurr[i].x, compressedVMapCurr[i].y, compressedVMapCurr[i].z);
			Eigen::Vector3f nor(compressedNMapCurr[i].x, compressedNMapCurr[i].y, compressedNMapCurr[i].z);
			pos = RCurr * pos + tCurr;
			nor = RCurr * nor;
			fs << pos.x() << " " << pos.y() << " " << pos.z() << " "
				<< (int)240 << " " << (int)240 << " " << (int)240 << " "
				<< nor.x() << " " << nor.y() << " " << nor.z() << " "
				<< std::endl;
		}
	}

	// Close file
	fs.close();
	std::exit(0);
#endif
	//std::exit(0);
}

void RGBDOdometry::getIncrementalTransformation(Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RPrev,
                                                Eigen::Vector3f& tPrev,
                                                Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RCurr,
                                                Eigen::Vector3f& tCurr,
                                                float icpWeight,
                                                float rgbWeight,
                                                float imuWeight,
                                                Eigen::Vector3f& velocity,
                                                Eigen::Vector3f& biasAcc,
                                                Eigen::Vector3f& biasGyr,
                                                ImuMeasurements& imuMeasurements,
                                                Gravity& gravityW, bool hasBeenPaused, int& matchesNum)
{
#if 0
    checkCudaErrors(cudaMemcpy2D(m_dDepthImg.data, m_dDepthImg.step, lastDepth[0], lastDepth[0].step(),
        lastDepth[0].colsBytes(), lastDepth[0].rows(), cudaMemcpyDeviceToDevice)); 
    cv::Mat testDepthImg;
    m_dDepthImg.download(testDepthImg);
    cv::imshow("testDepthImg00", testDepthImg * 30);
    cv::waitKey(1);
    checkCudaErrors(cudaMemcpy2D(m_dDepthImg.data, m_dDepthImg.step, nextDepth[0], nextDepth[0].step(),
        nextDepth[0].colsBytes(), nextDepth[0].rows(), cudaMemcpyDeviceToDevice));
    m_dDepthImg.download(testDepthImg);
    cv::imshow("testDepthImg11", testDepthImg * 30);
    cv::waitKey(1);
#endif
	//innoreal::InnoRealTimer timer;

	bool icp = true, rgb = true, imu = false;
#if 1
	iterations[0] = 10;
	iterations[1] = 5;
	iterations[2] = 4;
#endif
#if 0
    iterations[0] = 10 * 2;
    iterations[1] = 5 * 2;
    iterations[2] = 4 * 2;
#endif
#if 0
	if (imuMeasurements.size() == 0)
	{
		//distThres_ = 0.04f,
		//angleThres_ = sin(30.f * 3.14159254f / 180.f);
		iterations[0] = 30;
		imu = false;
		rgb = false;
		imuWeight = 0.0f;
		rgbWeight = 0.0f;
		//iterations[0] = 0;
		//iterations[1] = 0;
		//iterations[2] = 0;
	}
#endif

	Eigen::Vector3f velocityPrev = velocity;
	Eigen::Vector3f biasAccPrev = biasAcc;
	Eigen::Vector3f biasGyrPrev = biasGyr;

	Eigen::Vector3f velocityCurr = velocity;
	Eigen::Vector3f biasAccCurr = biasAcc;
	Eigen::Vector3f biasGyrCurr = biasGyr;

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> RPrevInv = RPrev.inverse();
	const mat33 dRPrevInv = RPrevInv;
	const float3 dtPrev = *reinterpret_cast<float3*>(tPrev.data());

	Eigen::Matrix4f TPrev = Eigen::Matrix4f::Identity(), TCurr = Eigen::Matrix4f::Identity();
	TPrev.block<3, 3>(0, 0) = RPrev;
	TPrev.block<3, 1>(0, 3) = tPrev;
	if (imu)
	{
		Eigen::Vector3f acc_0, gyr_0, acc_1, gyr_1;
		double current_time = -1.0;
		for (int i = 0; i < imuMeasurements.size(); ++i)
		{
			ImuMsg& imuMsg = imuMeasurements[i];
			double t = imuMsg.timeStamp;
			if (current_time < 0)
				current_time = t;
			double dt = (t - current_time);
			current_time = t;

			if (i == 0)
			{
				acc_0 = Eigen::Vector3f({(float)imuMsg.acc.x, (float)imuMsg.acc.y, (float)imuMsg.acc.z});
				gyr_0 = Eigen::Vector3f({(float)imuMsg.gyr.x, (float)imuMsg.gyr.y, (float)imuMsg.gyr.z});
			}
			//std::cout << "acc_0: " << acc_0.x() << " : " << acc_0.y() << " : " << acc_0.z() << std::endl;
			acc_1 = Eigen::Vector3f({(float)imuMsg.acc.x, (float)imuMsg.acc.y, (float)imuMsg.acc.z});
			gyr_1 = Eigen::Vector3f({(float)imuMsg.gyr.x, (float)imuMsg.gyr.y, (float)imuMsg.gyr.z});

			Eigen::Vector3f un_acc_0 = RCurr * (acc_0 - biasAccPrev); // -gravityWCurr;
			Eigen::Vector3f un_gyr = 0.5 * (gyr_0 + gyr_1) - biasGyrPrev;
			RCurr *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();

			Eigen::Vector3f un_acc_1 = RCurr * (acc_1 - biasAccPrev); // -gravityWCurr;
			Eigen::Vector3f un_acc = 0.5 * (un_acc_0 + un_acc_1);
			tCurr += dt * velocityCurr + 0.5 * dt * dt * un_acc;

			velocityCurr += dt * un_acc;

			acc_0 = acc_1;
			gyr_0 = gyr_1;
		}
#if 0
		TCurr.block<3, 3>(0, 0) = RCurr;
		TCurr.block<3, 1>(0, 3) = tCurr;
		std::cout << "relative T prior: " << TPrev.inverse() * TCurr << std::endl;
#endif
	}
	TCurr.block<3, 3>(0, 0) = RCurr;
	TCurr.block<3, 1>(0, 3) = tCurr;
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> relativeRt = (TPrev.inverse() * TCurr).cast<double>();
#if 0
	std::cout << "relative T prior: " << TPrev.inverse() * TCurr << std::endl;
#endif
	if (rgb)
	{
		for (int i = 0; i < NUM_PYRS; i++)
		{
			computeDerivativeImages(lastImage[i], lastdIdx[i], lastdIdy[i]);
		}
	}

	float3 RX, RY, RZ, T;
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();
	for (int i = NUM_PYRS - 1; i >= 0; i--)
	{
		K(0, 0) = intr(i).fx;
		K(1, 1) = intr(i).fy;
		K(0, 2) = intr(i).cx;
		K(1, 2) = intr(i).cy;
		K(2, 2) = 1;
		for (int j = 0; j < iterations[i]; j++)
		{
			//timer.TimeStart();

			if (rgb)
			{
				RX = {(float)relativeRt(0, 0), (float)relativeRt(1, 0), (float)relativeRt(2, 0)};
				RY = {(float)relativeRt(0, 1), (float)relativeRt(1, 1), (float)relativeRt(2, 1)};
				RZ = {(float)relativeRt(0, 2), (float)relativeRt(1, 2), (float)relativeRt(2, 2)};
				T = {(float)relativeRt(0, 3), (float)relativeRt(1, 3), (float)relativeRt(2, 3)};
				projectToPointCloud(nextDepth[i], pointClouds[i], intr, i, RX, RY, RZ, T);
			}

			float sigmaVal;
			if (rgb)
			{
				Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = relativeRt.topLeftCorner(3, 3);
				Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();
				mat33 krkInv;
				memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(), sizeof(mat33));
				Eigen::Vector3d Kt = relativeRt.topRightCorner(3, 1);
				Kt = K * Kt;
				float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};
				int sigma = 0, rgbSize = 0;
				computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0),
				                   lastdIdx[i],
				                   lastdIdy[i],
				                   lastDepth[i],
				                   nextDepth[i],
				                   lastImage[i],
				                   nextImage[i],
				                   corresImg[i],
				                   sumResidualRGB,
				                   maxDepthDeltaRGB,
				                   kt,
				                   krkInv,
				                   sigma,
				                   rgbSize,
				                   GPUConfig::getInstance().rgbResThreads,
				                   GPUConfig::getInstance().rgbResBlocks);
				sigmaVal = std::sqrt((float)sigma / rgbSize == 0 ? 1 : rgbSize);
#if 0
				std::cout << "sigmaVal: " << sigmaVal << std::endl;
				lastRGBError = sqrt(sigma) / rgbSize;
				lastRGBCount = rgbSize;
				std::cout << "lastRGBError: " << lastRGBError << " : " << lastRGBCount << std::endl;
#endif
			}


			Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
			Eigen::Matrix<float, 6, 1> b_icp;
			if (icp)
			{
				mat33 device_Rcurr = RCurr;
				float3 device_tcurr = *reinterpret_cast<float3*>(tCurr.data());
				float residual[2];

				icpStep(device_Rcurr,
				        device_tcurr,
				        vmaps_curr_[i],
				        nmaps_curr_[i],
				        dRPrevInv,
				        dtPrev,
				        intr(i),
				        vmaps_g_prev_[i],
				        nmaps_g_prev_[i],
				        distThres_,
				        angleThres_,
				        sumDataSE3,
				        outDataSE3,
				        A_icp.data(),
				        b_icp.data(),
				        &residual[0],
				        GPUConfig::getInstance().icpStepThreads,
				        GPUConfig::getInstance().icpStepBlocks);
				matchesNum = residual[1];
				if (matchesNum == 0) {
					return;
				}
#if 0
				lastICPError = sqrt(residual[0]) / residual[1];
				lastICPCount = residual[1];
				std::cout << "lastICPError: " << lastICPError << " : " << lastICPCount << std::endl;
#endif
			}

			Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>::Zero();
			Eigen::Matrix<float, 6, 1> b_rgbd = Eigen::Matrix<float, 6, 1>::Zero();
			if (rgb)
			{
				rgbStep(corresImg[i],
				        sigmaVal,
				        pointClouds[i],
				        intr(i).fx,
				        intr(i).fy,
				        lastdIdx[i],
				        lastdIdy[i],
				        sobelScale,
				        sumDataSE3,
				        outDataSE3,
				        A_rgbd.data(),
				        b_rgbd.data(),
				        GPUConfig::getInstance().rgbStepThreads,
				        GPUConfig::getInstance().rgbStepBlocks);
			}

			Eigen::Matrix<double, 15, 15, Eigen::RowMajor> A_imu;
			Eigen::Matrix<double, 15, 1> b_imu;
			if (imu)
			{
				m_integrationBase.imuStep(relativeRt,
				                          RPrevInv,
				                          tPrev,
				                          velocityPrev,
				                          biasAccPrev,
				                          biasGyrPrev,
				                          imuMeasurements,
				                          gravityW,
				                          A_imu,
				                          b_imu);
			}
#if 0
			std::cout << "A_icp:\n" << A_icp << std::endl;
			std::cout << "A_rgbd:\n" << A_rgbd << std::endl;
			std::cout << "A_imu:\n" << A_imu << std::endl;
			std::exit(0);
#endif
			//timer.TimeEnd();
			//std::cout << "time iter: " << timer.TimeGap_in_ms() << std::endl;

			Eigen::Isometry3f rgbOdom;
			if (imu)
			{
				//if (i == 0 && j >= 7)
				if (i == 0 && j >= 100)
				{
					double w1 = icpWeight;
					double w3 = imuWeight;
					lastA1515.setZero();
					lastA1515.block<6, 6>(0, 0) = w1 * w1 * A_icp.cast<double>();
					if (imu)
					{
						lastA1515 += w3 * w3 * A_imu;
					}
					lastb151.setZero();
					lastb151.block<6, 1>(0, 0) = w1 * w1 * b_icp.cast<double>();
					if (imu)
					{
						lastb151 += w3 * w3 * b_imu;
					}
				}
				else
				{
					double w1 = icpWeight;
					double w2 = rgbWeight;
					double w3 = imuWeight;
					lastA1515.setZero();
					lastA1515.block<6, 6>(0, 0) = w1 * w1 * A_icp.cast<double>() + w2 * w2 * A_rgbd.cast<double>();
					if (imu)
					{
						lastA1515 += w3 * w3 * A_imu;
					}
					lastb151.setZero();
					lastb151.block<6, 1>(0, 0) = w1 * w1 * b_icp.cast<double>() + w2 * w2 * b_rgbd.cast<double>();
					if (imu)
					{
						lastb151 += w3 * w3 * b_imu;
					}
				}
				lastA1515 += Eigen::Matrix<double, 15, 15>::Identity() * 0.00001;

				Eigen::Matrix<double, 15, 1> result = lastA1515.ldlt().solve(lastb151);
				//std::cout << "result:\n" << result << std::endl;

				velocityPrev += result.block<3, 1>(6, 0).cast<float>();
				biasAccPrev += result.block<3, 1>(9, 0).cast<float>();
				biasGyrPrev += result.block<3, 1>(12, 0).cast<float>();

				OdometryProvider::computeUpdateSE3(relativeRt, result.block<6, 1>(0, 0), rgbOdom);
			}
			else
			{
#if 0
				if (imuMeasurements.size() == 0)
				{
					float w1 = icpWeight;
					float w2 = rgbWeight;
					std::cout << "A_icp: \n" << w1 * w1 *A_icp << std::endl;
					std::cout << "b_icp: \n" << w1 * w1 * b_icp << std::endl;
					std::cout << "A_rgbd: \n" << w2 * w2 * A_rgbd << std::endl;
					std::cout << "b_rgbd: \n" << w2 * w2 * b_rgbd << std::endl;
				}
#endif
				//if (i == 0 && j >= 100)
				if (i == 0 && j >= 7)
				{
					double w1 = icpWeight;
					lastA66.setZero();
					lastA66.block<6, 6>(0, 0) = w1 * w1 * A_icp.cast<double>();
					lastb61.setZero();
					lastb61.block<6, 1>(0, 0) = w1 * w1 * b_icp.cast<double>();
				}
				else
				{
					double w1 = icpWeight;
					double w2 = rgbWeight;
					//std::cout << "w1: " << w1 << std::endl;
					//std::cout << "w2: " << w2 << std::endl;
					lastA66.setZero();
					lastA66.block<6, 6>(0, 0) = w1 * w1 * A_icp.cast<double>() + w2 * w2 * A_rgbd.cast<double>();
					lastb61.setZero();
					lastb61.block<6, 1>(0, 0) = w1 * w1 * b_icp.cast<double>() + w2 * w2 * b_rgbd.cast<double>();
				}
				Eigen::Matrix<double, 6, 1> result = lastA66.ldlt().solve(lastb61);
#if 0
				if (imuMeasurements.size() == 0)
				{
					std::cout << "result:\n" << result << std::endl;
				}
#endif

				OdometryProvider::computeUpdateSE3(relativeRt, result.block<6, 1>(0, 0), rgbOdom);
			}

			Eigen::Isometry3f currentT;
			currentT.setIdentity();
			currentT.rotate(RPrev);
			currentT.translation() = tPrev;
			currentT = currentT * rgbOdom;

			tCurr = currentT.translation();
			RCurr = currentT.rotation();
		}
	}

#if 0
	std::cout << "velocityPrev:\n" << velocityPrev << std::endl;
	std::cout << "biasAccPrev:\n" << biasAccPrev << std::endl;
	std::cout << "biasGyrPrev:\n" << biasGyrPrev << std::endl;
#endif

#if 0
	if (rgb && (tCurr - tPrev).norm() > 0.3)
	{
		RCurr = RPrev;
		tCurr = tPrev;
	}
#endif

	if (imu)
	{
		Eigen::Matrix<double, 15, 15, Eigen::RowMajor> tmp1;
		Eigen::Matrix<double, 15, 1> tmp2;
		m_integrationBase.imuStep(relativeRt,
		                          RPrevInv,
		                          tPrev,
		                          velocityPrev,
		                          biasAccPrev,
		                          biasGyrPrev,
		                          imuMeasurements,
		                          gravityW,
		                          tmp1,
		                          tmp2);
		velocityCurr = velocityPrev + RPrevInv * m_integrationBase.m_delta_v.cast<float>();
		biasAccCurr = biasAccPrev;
		biasGyrCurr = biasGyrPrev;

		velocity = velocityCurr;
		biasAcc = biasAccCurr;
		biasGyr = biasGyrCurr;
	}
#if 0
	biasAcc = Eigen::Vector3f::Zero();
	biasGyr = Eigen::Vector3f::Zero();
#endif
#if 0
	std::cout << "relative T: " << relativeRt << std::endl;
#endif
}

void RGBDOdometry::getIncrementalTransformation2(Eigen::Vector3f& trans,
                                                 Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& rot,
                                                 const float& icpWeight,
                                                 const float& imuWeight,
                                                 Eigen::Vector3f& velocity,
                                                 Eigen::Vector3f& biasAcc,
                                                 Eigen::Vector3f& biasGyr,
                                                 ImuMeasurements& imuMeasurements,
                                                 Gravity& gravityW)
{
	const bool icp = true;
	const bool rgb = true;
	const bool imu = true;

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> RPrev = rot;
	Eigen::Vector3f tPrev = trans;
	Eigen::Vector3f velocityPrev = velocity;
	Eigen::Vector3f biasAccPrev = biasAcc;
	Eigen::Vector3f biasGyrPrev = biasGyr;

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> RCurr = RPrev;
	Eigen::Vector3f tCurr = tPrev;
	Eigen::Vector3f velocityCurr = velocity;
	Eigen::Vector3f biasAccCurr = biasAcc;
	Eigen::Vector3f biasGyrCurr = biasGyr;

	Eigen::Matrix4f TPrev = Eigen::Matrix4f::Identity(), TCurr = Eigen::Matrix4f::Identity();
	TPrev.block<3, 3>(0, 0) = RPrev;
	TPrev.block<3, 1>(0, 3) = tPrev;
	if (imu)
	{
		Eigen::Vector3f acc_0, gyr_0, acc_1, gyr_1;
		double current_time = -1.0;
		for (int i = 0; i < imuMeasurements.size(); ++i)
		{
			ImuMsg& imuMsg = imuMeasurements[i];
			double t = imuMsg.timeStamp;
			if (current_time < 0)
				current_time = t;
			double dt = (t - current_time);
			current_time = t;

			if (i == 0)
			{
				acc_0 = Eigen::Vector3f({(float)imuMsg.acc.x, (float)imuMsg.acc.y, (float)imuMsg.acc.z});
				gyr_0 = Eigen::Vector3f({(float)imuMsg.gyr.x, (float)imuMsg.gyr.y, (float)imuMsg.gyr.z});
			}
			//std::cout << "acc_0: " << acc_0.x() << " : " << acc_0.y() << " : " << acc_0.z() << std::endl;
			acc_1 = Eigen::Vector3f({(float)imuMsg.acc.x, (float)imuMsg.acc.y, (float)imuMsg.acc.z});
			gyr_1 = Eigen::Vector3f({(float)imuMsg.gyr.x, (float)imuMsg.gyr.y, (float)imuMsg.gyr.z});

			Eigen::Vector3f un_acc_0 = RCurr * (acc_0 - biasAccPrev); // -gravityWCurr;
			Eigen::Vector3f un_gyr = 0.5 * (gyr_0 + gyr_1) - biasGyrPrev;
			RCurr *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();

			Eigen::Vector3f un_acc_1 = RCurr * (acc_1 - biasAccPrev); // -gravityWCurr;
			Eigen::Vector3f un_acc = 0.5 * (un_acc_0 + un_acc_1);
			tCurr += dt * velocityCurr + 0.5 * dt * dt * un_acc;

			velocityCurr += dt * un_acc;

			acc_0 = acc_1;
			gyr_0 = gyr_1;
		}
		//std::cout << "tCurr: " << tCurr.x() << " : " << tCurr.y() << " : " << tCurr.z() << std::endl;

		TCurr.block<3, 3>(0, 0) = RCurr;
		TCurr.block<3, 1>(0, 3) = tCurr;

		std::cout << "relative T prior: " << TPrev.inverse() * TCurr << std::endl;
	}

	if (rgb)
	{
		for (int i = 0; i < NUM_PYRS; i++)
		{
			computeDerivativeImages(lastImage[i], lastdIdx[i], lastdIdy[i]);
		}
	}

	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> resultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

	iterations[0] = 10;
	iterations[1] = 5;
	iterations[2] = 4;

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> RPrevInv = RPrev.inverse();
	mat33 dRPrevInv = RPrevInv;
	float3 dtPrev = *reinterpret_cast<float3*>(tPrev.data());

	TPrev.block<3, 3>(0, 0) = RPrev;
	TPrev.block<3, 1>(0, 3) = tPrev;
	TCurr.block<3, 3>(0, 0) = RCurr;
	TCurr.block<3, 1>(0, 3) = tCurr;
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> relativeRt = (TPrev.inverse() * TCurr).cast<double>();
#if 0
	std::cout << "relative T prior: " << TPrev.inverse() * TCurr << std::endl;
#endif
	//Eigen::Matrix<double, 4, 4, Eigen::RowMajor> relativeRt = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
	float3 RX, RY, RZ, T;

	for (int i = NUM_PYRS - 1; i >= 0; i--)
	{
		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

		K(0, 0) = intr(i).fx;
		K(1, 1) = intr(i).fy;
		K(0, 2) = intr(i).cx;
		K(1, 2) = intr(i).cy;
		K(2, 2) = 1;

		lastRGBError = std::numeric_limits<float>::max();

		for (int j = 0; j < iterations[i]; j++)
		{
			if (rgb)
			{
				RX = {(float)relativeRt(0, 0), (float)relativeRt(1, 0), (float)relativeRt(2, 0)};
				RY = {(float)relativeRt(0, 1), (float)relativeRt(1, 1), (float)relativeRt(2, 1)};
				RZ = {(float)relativeRt(0, 2), (float)relativeRt(1, 2), (float)relativeRt(2, 2)};
				T = {(float)relativeRt(0, 3), (float)relativeRt(1, 3), (float)relativeRt(2, 3)};

				projectToPointCloud(nextDepth[i], pointClouds[i], intr, i, RX, RY, RZ, T);
			}

			//std::cout << "iter: " << j << std::endl;
			Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = relativeRt; // .inverse();

			Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

			Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();
			mat33 krkInv;
			memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(), sizeof(mat33));

			Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
			Kt = K * Kt;
			float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};

			int sigma = 0;
			int rgbSize = 0;

			if (rgb)
			{
				computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0),
				                   lastdIdx[i],
				                   lastdIdy[i],
				                   lastDepth[i],
				                   nextDepth[i],
				                   lastImage[i],
				                   nextImage[i],
				                   corresImg[i],
				                   sumResidualRGB,
				                   maxDepthDeltaRGB,
				                   kt,
				                   krkInv,
				                   sigma,
				                   rgbSize,
				                   GPUConfig::getInstance().rgbResThreads,
				                   GPUConfig::getInstance().rgbResBlocks);
			}

			float sigmaVal = std::sqrt((float)sigma / rgbSize == 0 ? 1 : rgbSize);
			//std::cout << "sigmaVal: " << sigmaVal << std::endl;

			lastRGBError = sqrt(sigma) / rgbSize;
			lastRGBCount = rgbSize;
			std::cout << "lastRGBError: " << lastRGBError << " : " << lastRGBCount << std::endl;

			Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
			Eigen::Matrix<float, 6, 1> b_icp;

			mat33 device_Rcurr = RCurr;
			float3 device_tcurr = *reinterpret_cast<float3*>(tCurr.data());

			DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
			DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];

			DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
			DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

			float residual[2];

			if (icp)
			{
				icpStep(device_Rcurr,
				        device_tcurr,
				        vmap_curr,
				        nmap_curr,
				        dRPrevInv,
				        dtPrev,
				        intr(i),
				        vmap_g_prev,
				        nmap_g_prev,
				        distThres_,
				        angleThres_,
				        sumDataSE3,
				        outDataSE3,
				        A_icp.data(),
				        b_icp.data(),
				        &residual[0],
				        GPUConfig::getInstance().icpStepThreads,
				        GPUConfig::getInstance().icpStepBlocks);
			}

			lastICPError = sqrt(residual[0]) / residual[1];
			lastICPCount = residual[1];
			std::cout << "lastICPError: " << lastICPError << " : " << lastICPCount << std::endl;

			Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
			Eigen::Matrix<float, 6, 1> b_rgbd;

			if (rgb)
			{
				rgbStep(corresImg[i],
				        sigmaVal,
				        pointClouds[i],
				        intr(i).fx,
				        intr(i).fy,
				        lastdIdx[i],
				        lastdIdy[i],
				        sobelScale,
				        sumDataSE3,
				        outDataSE3,
				        A_rgbd.data(),
				        b_rgbd.data(),
				        GPUConfig::getInstance().rgbStepThreads,
				        GPUConfig::getInstance().rgbStepBlocks);
			}

			Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd = A_rgbd.cast<double>();
			Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
			Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();
			Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

			Eigen::Matrix<double, 15, 15, Eigen::RowMajor> A_imu;
			Eigen::Matrix<double, 15, 1> b_imu;
			if (imu)
			{
				m_integrationBase.imuStep(relativeRt,
				                          RPrevInv,
				                          tPrev,
				                          velocityPrev,
				                          biasAccPrev,
				                          biasGyrPrev,
				                          imuMeasurements,
				                          gravityW,
				                          A_imu,
				                          b_imu);
			}

#if 0
			double w = icpWeight;
			lastA.setZero();
			lastA.block<6, 6>(0, 0) = dA_icp;// dA_rgbd + w * w * dA_icp;
			lastA += A_imu;
			lastb.setZero();
			lastb.block<6, 1>(0, 0) = db_icp;// db_rgbd + w * db_icp;
			lastb += b_imu;
			Eigen::Matrix<double, 15, 1> result;

			lastA += Eigen::Matrix<double, 15, 15>::Identity() * 0.001;
			result = lastA.llt().solve(lastb);

			velocityPrev += result.block<3, 1>(6, 0).cast<float>();
			biasAccPrev += result.block<3, 1>(9, 0).cast<float>();
			biasGyrPrev += result.block<3, 1>(12, 0).cast<float>();
			//std::cout << result(0) << " : " << result(1) << " : " << result(2) << std::endl;
#endif
#if 0
			std::cout << "A_icp:\n" << A_icp << std::endl;
			std::cout << "A_rgbd:\n" << A_rgbd << std::endl;
			std::cout << "A_imu:\n" << A_imu << std::endl;
			std::exit(0);
#endif
			if (i == 0 && j >= 7)
			{
				double w2 = 1;
				lastA1515.setZero();
				lastA1515.block<6, 6>(0, 0) = dA_icp;
				lastA1515 += A_imu * w2 * w2;
				lastb151.setZero();
				lastb151.block<6, 1>(0, 0) = db_icp;
				lastb151 += b_imu * w2 * w2;
			}
			else
			{
				double w = icpWeight;
				double w2 = 1;
				lastA1515.setZero();
				lastA1515.block<6, 6>(0, 0) = dA_rgbd + w * w * dA_icp; // dA_icp;// dA_rgbd;// dA_rgbd + w * w * dA_icp;
				lastA1515 += A_imu * w2 * w2;
				lastb151.setZero();
				lastb151.block<6, 1>(0, 0) = db_rgbd + w * w * db_icp; // db_icp;// db_rgbd;// (db_rgbd + w * db_icp);
				lastb151 += b_imu * w2 * w2;
			}
			Eigen::Matrix<double, 15, 1> result;

			lastA1515 += Eigen::Matrix<double, 15, 15>::Identity() * 0.00001;
			result = lastA1515.llt().solve(lastb151);
			//std::cout << "result:\n" << result << std::endl;
			//std::exit(0);

			velocityPrev += result.block<3, 1>(6, 0).cast<float>();
			biasAccPrev += result.block<3, 1>(9, 0).cast<float>();
			biasGyrPrev += result.block<3, 1>(12, 0).cast<float>();

			//std::cout << "result:\n" << result << std::endl;
			//std::exit(0);
#if 0
			if (icp && rgb)
			{
				double w = icpWeight;
				lastA = dA_rgbd + w * w * dA_icp;
				lastb = db_rgbd + w * db_icp;
				result = lastA.ldlt().solve(lastb);
			}
			else if (icp)
			{
				lastA = dA_icp;
				lastb = db_icp;
				result = lastA.ldlt().solve(lastb);
			}
			else if (rgb)
			{
				lastA = dA_rgbd;
				lastb = db_rgbd;
				result = lastA.ldlt().solve(lastb);
			}
			else
			{
				assert(false && "Control shouldn't reach here");
			}
#endif

			Eigen::Isometry3f rgbOdom;

			OdometryProvider::computeUpdateSE3(relativeRt, result.block<6, 1>(0, 0), rgbOdom);

			Eigen::Isometry3f currentT;
			currentT.setIdentity();
			currentT.rotate(RPrev);
			currentT.translation() = tPrev;

			currentT = currentT * rgbOdom; // .inverse();

			//std::cout << "rgbOdom.matrix(): " << rgbOdom.matrix() << std::endl;
			//std::cout << "rgbOdom.inverse().matrix(): " << rgbOdom.inverse().matrix() << std::endl;
			//std::cout << "relativeRt: " << relativeRt << std::endl;

			tCurr = currentT.translation();
			RCurr = currentT.rotation();
		}
	}

	std::cout << "velocityPrev:\n" << velocityPrev << std::endl;
	std::cout << "biasAccPrev:\n" << biasAccPrev << std::endl;
	std::cout << "biasGyrPrev:\n" << biasGyrPrev << std::endl;

#if 1
	if (rgb && (tCurr - tPrev).norm() > 0.3)
	{
		RCurr = RPrev;
		tCurr = tPrev;
	}
#endif

	Eigen::Matrix<double, 15, 15, Eigen::RowMajor> A_imu;
	Eigen::Matrix<double, 15, 1> b_imu;
	m_integrationBase.imuStep(relativeRt,
	                          RPrevInv,
	                          tPrev,
	                          velocityPrev,
	                          biasAccPrev,
	                          biasGyrPrev,
	                          imuMeasurements,
	                          gravityW,
	                          A_imu,
	                          b_imu);
	velocityCurr = velocityPrev + RPrevInv * m_integrationBase.m_delta_v.cast<float>();
	biasAccCurr = biasAccPrev;
	biasGyrCurr = biasGyrPrev;

	trans = tCurr;
	rot = RCurr;

	velocity = velocityCurr;
	biasAcc = biasAccCurr;
	biasGyr = biasGyrCurr;
	//biasAcc = Eigen::Vector3f::Zero();
	//biasGyr = Eigen::Vector3f::Zero();

	TCurr.block<3, 3>(0, 0) = RCurr;
	TCurr.block<3, 1>(0, 3) = tCurr;
	std::cout << "relative T: " << TPrev.inverse() * TCurr << std::endl;
	std::cout << "relative T 2: " << relativeRt << std::endl;

	//outFile.close();
#if 0
	std::cout << "finish" << std::endl;
	std::exit(0);
#endif
}

Eigen::MatrixXd RGBDOdometry::getCovariance()
{
	return lastA66.cast<double>().lu().inverse();
}
