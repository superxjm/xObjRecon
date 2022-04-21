#include "stdafx.h"

#include "xSurfelFusion.h"

#include <utility>
#include <algorithm>
#include <omp.h>

#include "Helpers/xUtils.h"
#include "Helpers/xUtils.cuh"
#include "InnorealTimer.hpp"

xSurfelFusion::xSurfelFusion(int64_t& timeStamp,
                             int& fragIdx,
                             const int timeDelta,
                             const int countThresh,
                             const float errThresh,
                             const float covThresh,
                             const float photoThresh,
                             const float confidence,
                             const float depthCut,
                             const float icpThresh)
	: m_timeStamp(timeStamp),
	  m_fragIdx(fragIdx),
	  frameToModel(Resolution::getInstance().width(),
	               Resolution::getInstance().height(),
	               Intrinsics::getInstance().cx(),
	               Intrinsics::getInstance().cy(),
	               Intrinsics::getInstance().fx(),
	               Intrinsics::getInstance().fy()),
	  currPose(Eigen::Matrix4f::Identity()),
	  timeDelta(timeDelta),
	  icpCountThresh(countThresh),
	  icpErrThresh(errThresh),
	  covThresh(covThresh),
	  maxDepthProcessed(20.0f),
	  icpWeight(icpThresh),
	  confidenceThreshold(confidence),
	  depthCutoff(depthCut)
{
	createTextures();
	createCompute();
	createFeedbackBuffers();	

	m_vertexImg = cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_32FC4);
	
	size_t numBytes;
#if 1
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_vboCudaRes, globalModel.model().first, cudaGraphicsMapFlagsNone));
	cudaGraphicsMapResources(1, &m_vboCudaRes, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_dVboCuda, &numBytes, m_vboCudaRes);
	cudaGraphicsUnmapResources(1, &m_vboCudaRes, 0);
#endif

#if 0
	cudaGraphicsMapResources(1, &textures[GPUTexture::DEPTH_METRIC_FILTERED]->cudaRes, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dFilteredDepth32FCudaMapper, &numBytes, textures[GPUTexture::DEPTH_METRIC_FILTERED]->cudaRes);
	cudaGraphicsUnmapResources(1, &textures[GPUTexture::DEPTH_METRIC_FILTERED]->cudaRes, 0);

	cudaGraphicsMapResources(1, &textures[GPUTexture::RGB]->cudaRes, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dColorCudaMapper, &numBytes, textures[GPUTexture::RGB]->cudaRes);
	cudaGraphicsUnmapResources(1, &textures[GPUTexture::RGB]->cudaRes, 0);

	cudaGraphicsMapResources(1, &indexMap.vertexTex()->cudaRes, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dVertexTexCudaMapper, &numBytes, indexMap.vertexTex()->cudaRes);
	cudaGraphicsUnmapResources(1, &indexMap.vertexTex()->cudaRes, 0);
#endif
}

xSurfelFusion::~xSurfelFusion()
{	
	for (std::map<std::string, GPUTexture*>::iterator it = textures.begin(); it != textures.end(); ++it)
	{
		delete it->second;
	}

	textures.clear();

	for (std::map<std::string, ComputePack*>::iterator it = computePacks.begin(); it != computePacks.end(); ++it)
	{
		delete it->second;
	}

	computePacks.clear();

	for (std::map<std::string, FeedbackBuffer*>::iterator it = feedbackBuffers.begin(); it != feedbackBuffers.end(); ++it)
	{
		delete it->second;
	}

	feedbackBuffers.clear();

	cudaGraphicsUnregisterResource(m_vboCudaRes);
}

void xSurfelFusion::clear()
{
	currPose = Eigen::Matrix4f::Identity();
	checkCudaErrors(cudaGraphicsUnregisterResource(m_vboCudaRes));
	globalModel.clear();
}

void xSurfelFusion::createTextures()
{	
	textures[GPUTexture::RGB] = new GPUTexture(Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		GL_RGBA,
		GL_RGB,
		GL_UNSIGNED_BYTE,
		true,
		true);

	textures[GPUTexture::DEPTH_RAW] = new GPUTexture(Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		GL_LUMINANCE16UI_EXT,
		GL_LUMINANCE_INTEGER_EXT,
		GL_UNSIGNED_SHORT,
		false,
		true);

	textures[GPUTexture::DEPTH_FILTERED] = new GPUTexture(Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		GL_LUMINANCE16UI_EXT,
		GL_LUMINANCE_INTEGER_EXT,
		GL_UNSIGNED_SHORT,
		false,
		true);

	textures[GPUTexture::DEPTH_METRIC] = new GPUTexture(Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		GL_LUMINANCE32F_ARB,
		GL_LUMINANCE,
		GL_FLOAT,
		false,
		true);

	textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		GL_LUMINANCE32F_ARB,
		GL_LUMINANCE,
		GL_FLOAT,
		false,
		true);

	textures[GPUTexture::DEPTH_NORM] = new GPUTexture(Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		GL_LUMINANCE,
		GL_LUMINANCE,
		GL_FLOAT,
		true);
}

void xSurfelFusion::createCompute()
{
	computePacks[ComputePack::NORM] = new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom"),
		textures[GPUTexture::DEPTH_NORM]->texture);

#if 0
	computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_bilateral.frag", "quad.geom"),
		textures[GPUTexture::DEPTH_FILTERED]->texture);
#endif

	computePacks[ComputePack::METRIC] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
		textures[GPUTexture::DEPTH_METRIC]->texture);

	computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
		textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture);
}

void xSurfelFusion::createFeedbackBuffers()
{
	feedbackBuffers[FeedbackBuffer::RAW] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));
	feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));
}

void xSurfelFusion::computeFeedbackBuffers()
{
	feedbackBuffers[FeedbackBuffer::RAW]->compute(textures[GPUTexture::RGB]->texture,
		textures[GPUTexture::DEPTH_METRIC]->texture,
		m_timeStamp,
		maxDepthProcessed);

	feedbackBuffers[FeedbackBuffer::FILTERED]->compute(textures[GPUTexture::RGB]->texture,
		textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture,
		m_timeStamp,
		maxDepthProcessed);	
}

void xSurfelFusion::trackCamera()
{
	//WARNING initICP* must be called before initRGB*
	frameToModel.initICPModel(&fillIn.vertexTexture,
		&fillIn.normalTexture,
		maxDepthProcessed, currPose);
	frameToModel.initRGBModel(&fillIn.imageTexture);

	frameToModel.initICP(textures[GPUTexture::DEPTH_FILTERED], maxDepthProcessed);
	frameToModel.initRGB(textures[GPUTexture::RGB]);

	Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);

	frameToModel.getIncrementalTransformation(trans,
		rot,
		icpWeight);
	xCheckGlDieOnError();
	assert(frameToModel.lastICPError < 1e-04);
	std::cout << "lastICPError: " << frameToModel.lastICPError << std::endl;

	currPose.topRightCorner(3, 1) = trans;
	currPose.topLeftCorner(3, 3) = rot;
}

void xSurfelFusion::dofusion(float weight, int isFrag)
{
	indexMap.predictIndices(currPose, m_timeStamp, globalModel.model(), maxDepthProcessed, timeDelta, m_fragIdx);
	xCheckGlDieOnError();

	globalModel.fuse(currPose,
		m_timeStamp,
		textures[GPUTexture::RGB],
		textures[GPUTexture::DEPTH_METRIC],
		textures[GPUTexture::DEPTH_METRIC_FILTERED],
		indexMap.indexTex(),
		indexMap.vertConfTex(),
		indexMap.colorTimeTex(),
		indexMap.normalRadTex(),
		maxDepthProcessed,
		confidenceThreshold,
		weight,
		m_fragIdx);
	xCheckGlDieOnError();

	indexMap.predictIndices(currPose, m_timeStamp, globalModel.model(), maxDepthProcessed, timeDelta, m_fragIdx);
	xCheckGlDieOnError();

	globalModel.clean(currPose,
		m_timeStamp,
		indexMap.indexTex(),
		indexMap.vertConfTex(),
		indexMap.colorTimeTex(),
		indexMap.normalRadTex(),
		confidenceThreshold,
		timeDelta,
		maxDepthProcessed,
		m_fragIdx,
		isFrag);
	xCheckGlDieOnError();
}

void xSurfelFusion::processFrame(cv::Mat& depthImg,
                                 cv::Mat& colorImg,
                                 cv::cuda::GpuMat& dRenderedDepthImg32F,
                                 float& velocity,
                                 cv::cuda::GpuMat& dRawDepthImg32F,
                                 cv::cuda::GpuMat& dFilteredDepthImg,
                                 cv::cuda::GpuMat& dFilteredDepthImg32F,
                                 cv::cuda::GpuMat& dColorImg,
                                 bool objectDetected)
{
	innoreal::InnoRealTimer timer;
#if 1
	timer.TimeStart();

	textures[GPUTexture::DEPTH_RAW]->texture->Upload(depthImg.data,
		GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
	textures[GPUTexture::RGB]->texture->Upload(colorImg.data,
		GL_RGB, GL_UNSIGNED_BYTE);
	//textures[GPUTexture::DEPTH_RAW]->texture->Download(XXX, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
	//textures[GPUTexture::RGB]->texture->Download(XXX, GL_RGB, GL_UNSIGNED_BYTE);
	xCheckGlDieOnError();

	timer.TimeEnd();
	std::cout << "upload time1: " << timer.TimeGap_in_ms() << std::endl;
#endif
	timer.TimeStart();

#if 0
	cudaGraphicsMapResources(1, &textures[GPUTexture::DEPTH_FILTERED]->cudaRes);
	cudaGraphicsSubResourceGetMappedArray(&m_textPtr, textures[GPUTexture::DEPTH_FILTERED]->cudaRes, 0, 0);
	cudaMemcpy2DToArray(m_textPtr, 0, 0,
		dFilteredDepthImg.data, dFilteredDepthImg.step,
		dFilteredDepthImg.cols * dFilteredDepthImg.elemSize(), dFilteredDepthImg.rows,
		cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &textures[GPUTexture::DEPTH_FILTERED]->cudaRes);
#endif

#if 0
	cudaGraphicsMapResources(1, &textures[GPUTexture::DEPTH_METRIC]->cudaRes);
	cudaGraphicsSubResourceGetMappedArray(&m_textPtr, textures[GPUTexture::DEPTH_METRIC]->cudaRes, 0, 0);
	cudaMemcpy2DToArray(m_textPtr, 0, 0,
	                    dRawDepthImg32F.data, dRawDepthImg32F.step,
	                    dRawDepthImg32F.cols * dRawDepthImg32F.elemSize(), dRawDepthImg32F.rows,
	                    cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &textures[GPUTexture::DEPTH_METRIC]->cudaRes);
#endif

#if 0
	cudaGraphicsMapResources(1, &textures[GPUTexture::DEPTH_METRIC_FILTERED]->cudaRes);
	cudaGraphicsSubResourceGetMappedArray(&m_textPtr, textures[GPUTexture::DEPTH_METRIC_FILTERED]->cudaRes, 0, 0);
	cudaMemcpy2DToArray(m_textPtr, 0, 0,
		dFilteredDepthImg32F.data, dFilteredDepthImg32F.step,
		dFilteredDepthImg32F.cols * dFilteredDepthImg32F.elemSize(), dFilteredDepthImg32F.rows,
		cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &textures[GPUTexture::DEPTH_METRIC_FILTERED]->cudaRes);
#endif

#if 0
	cudaGraphicsMapResources(1, &textures[GPUTexture::RGB]->cudaRes);
	cudaGraphicsSubResourceGetMappedArray(&m_textPtr, textures[GPUTexture::RGB]->cudaRes, 0, 0);
	cudaMemcpy2DToArray(m_textPtr, 0, 0,
		dColorImg.data, dColorImg.step,
		dColorImg.cols * dColorImg.elemSize(), dColorImg.rows,
		cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &textures[GPUTexture::RGB]->cudaRes);
#endif

	timer.TimeEnd();
	std::cout << "upload time2: " << timer.TimeGap_in_ms() << std::endl;

	filterDepth();
	metriciseDepth();
	xCheckGlDieOnError();

	if (m_timeStamp == 1)
	{	
		computeFeedbackBuffers();

		globalModel.initialise(*feedbackBuffers[FeedbackBuffer::RAW], *feedbackBuffers[FeedbackBuffer::FILTERED]);
		frameToModel.initFirstRGB(textures[GPUTexture::RGB]);		
	}
	else
	{	
#if 1
		Eigen::Matrix4f lastPose = currPose;	
			
		trackCamera();	

		//Weight by velocity
		Eigen::Matrix4f diff = currPose.inverse() * lastPose;
		Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
		Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);
		float weight = std::max(diffTrans.norm(), rodrigues2(diffRot).norm());	
		if (objectDetected == false)
		{
			velocity = weight;
		}
		float largest = 0.01, minWeight = 0.5;
		if (weight > largest)
		{
			weight = largest;
		}
		weight = std::max(1.0f - (weight / largest), minWeight);

		if (objectDetected == true)
		{
			dofusion(weight, IsFrag(m_timeStamp));
		}
		else
		{	
			dofusion(weight, 0);
		}
#endif
	}
	
	indexMap.combinedPredict(currPose,
		globalModel.model(),
		maxDepthProcessed,
		confidenceThreshold,
		m_timeStamp,
		timeDelta);

	if (objectDetected == false)
	{
#if 0
		indexMap.vertexTex()->texture->Download(m_vertexImg.data, GL_RGBA, GL_FLOAT);
#pragma omp parallel
		for (int r = 0; r < m_vertexImg.rows; ++r)
		{
			for (int c = 0; c < m_vertexImg.cols; ++c)
			{
				renderedDepthImg.at<ushort>(r, c) = m_vertexImg.at<cv::Vec4f>(r, c)[2] * 1000.0f;
			}
		}
#endif
	}
	//cv::namedWindow("rendered_depth");
	//cv::imshow("rendered_depth", m_vertexImg);
	//cv::waitKey(1);
#if 1
	fillIn.vertex(indexMap.vertexTex(), textures[GPUTexture::DEPTH_FILTERED], false);
	fillIn.normal(indexMap.normalTex(), textures[GPUTexture::DEPTH_FILTERED], false);
	fillIn.image(indexMap.imageTex(), textures[GPUTexture::RGB], false);
#endif
}

void xSurfelFusion::metriciseDepth()
{
#if 1
	std::vector<Uniform> uniforms;

	uniforms.push_back(Uniform("maxD", depthCutoff));

	computePacks[ComputePack::METRIC]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
	computePacks[ComputePack::METRIC_FILTERED]->compute(textures[GPUTexture::DEPTH_FILTERED]->texture, &uniforms);
#endif
}

void xSurfelFusion::filterDepth()
{
#if 1
	std::vector<Uniform> uniforms;

	uniforms.push_back(Uniform("cols", (float)Resolution::getInstance().cols()));
	uniforms.push_back(Uniform("rows", (float)Resolution::getInstance().rows()));
	uniforms.push_back(Uniform("maxD", depthCutoff));

	computePacks[ComputePack::FILTER]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
#endif
}

void xSurfelFusion::normaliseDepth(const float & minVal, const float & maxVal)
{
#if 1
	std::vector<Uniform> uniforms;

	uniforms.push_back(Uniform("maxVal", maxVal * 1000.f));
	uniforms.push_back(Uniform("minVal", minVal * 1000.f));

	computePacks[ComputePack::NORM]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
#endif
}

Eigen::Vector3f xSurfelFusion::rodrigues2(const Eigen::Matrix3f& matrix)
{
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
	Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

	double rx = R(2, 1) - R(1, 2);
	double ry = R(0, 2) - R(2, 0);
	double rz = R(1, 0) - R(0, 1);

	double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
	double c = (R.trace() - 1) * 0.5;
	c = c > 1. ? 1. : c < -1. ? -1. : c;

	double theta = acos(c);

	if (s < 1e-5)
	{
		double t;

		if (c > 0)
			rx = ry = rz = 0;
		else
		{
			t = (R(0, 0) + 1)*0.5;
			rx = sqrt(std::max(t, 0.0));
			t = (R(1, 1) + 1)*0.5;
			ry = sqrt(std::max(t, 0.0)) * (R(0, 1) < 0 ? -1.0 : 1.0);
			t = (R(2, 2) + 1)*0.5;
			rz = sqrt(std::max(t, 0.0)) * (R(0, 2) < 0 ? -1.0 : 1.0);

			if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0))
				rz = -rz;
			theta /= sqrt(rx*rx + ry*ry + rz*rz);
			rx *= theta;
			ry *= theta;
			rz *= theta;
		}
	}
	else
	{
		double vth = 1 / (2 * s);
		vth *= theta;
		rx *= vth; ry *= vth; rz *= vth;
	}
	return Eigen::Vector3d(rx, ry, rz).cast<float>();
}

Eigen::Matrix4f & xSurfelFusion::getCurrPose()
{
	return currPose;
}

GlobalModel & xSurfelFusion::getGlobalModel()
{
	return globalModel;
}
