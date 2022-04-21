#include "stdafx.h"

#include "xSurfelFusion.h"

#include <utility>
#include <algorithm>
#include <omp.h>

#include "Helpers/xUtils.h"
#include "Helpers/xUtils.cuh"
#include "Helpers/InnorealTimer.hpp"

xSurfelFusion::xSurfelFusion(int64_t& timeStamp,
                             int& fragIdx,
                             const float icpWeight,
                             const float rgbWeight,
                             const float imuWeight,
                             const int timeDelta,
                             const int countThresh,
                             const float errThresh,
                             const float covThresh,
                             const float photoThresh,
                             const float confidence,
                             const float depthCut)
	: m_timeStamp(timeStamp),
	  m_fragIdx(fragIdx),
	  m_icpWeight(icpWeight),
	  m_rgbWeight(rgbWeight),
	  m_imuWeight(imuWeight),
	  frameToModel(Resolution::getInstance().width(),
	               Resolution::getInstance().height(),
	               Intrinsics::getInstance().cx(),
	               Intrinsics::getInstance().cy(),
	               Intrinsics::getInstance().fx(),
	               Intrinsics::getInstance().fy()),
	  m_currPose(Eigen::Matrix4f::Identity()),
	  timeDelta(timeDelta),
	  icpCountThresh(countThresh),
	  icpErrThresh(errThresh),
	  covThresh(covThresh),
	  maxDepthProcessed(20.0f),
	  confidenceThreshold(confidence),
	  depthCutoff(depthCut)
{
	createTextures();
	createCompute();
	createFeedbackBuffers();

	m_dVMapFloat4 = cv::cuda::GpuMat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_32FC4);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_vboCudaRes, globalModel.model().first, cudaGraphicsMapFlagsNone));
	size_t numBytes;
	cudaGraphicsMapResources(1, &m_vboCudaRes, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_dVboCuda, &numBytes, m_vboCudaRes);
	cudaGraphicsUnmapResources(1, &m_vboCudaRes, 0);

	m_currVelocity = Eigen::Vector3f::Zero();
	m_currBiasAcc = Eigen::Vector3f::Zero();
	m_currBiasGyr = Eigen::Vector3f::Zero();

	checkCudaErrors(cudaGraphicsMapResources(1, &textures[GPUTexture::DEPTH_RAW]->cudaRes, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_cudaArrayDepthRaw,
		textures[GPUTexture::DEPTH_RAW]->cudaRes,
		0, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &textures[GPUTexture::DEPTH_RAW]->cudaRes, 0));

	checkCudaErrors(cudaGraphicsMapResources(1, &textures[GPUTexture::DEPTH_FILTERED]->cudaRes, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_cudaArrayDepthFiltered,
		textures[GPUTexture::DEPTH_FILTERED]->cudaRes,
		0, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &textures[GPUTexture::DEPTH_FILTERED]->cudaRes, 0));

	checkCudaErrors(cudaGraphicsMapResources(1, &textures[GPUTexture::DEPTH_METRIC]->cudaRes, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_cudaArrayDepthMetric,
		textures[GPUTexture::DEPTH_METRIC]->cudaRes,
		0, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &textures[GPUTexture::DEPTH_METRIC]->cudaRes, 0));

	checkCudaErrors(cudaGraphicsMapResources(1, &textures[GPUTexture::DEPTH_METRIC_FILTERED]->cudaRes, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_cudaArrayDepthMetricFiltered,
		textures[GPUTexture::DEPTH_METRIC_FILTERED]->cudaRes,
		0, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &textures[GPUTexture::DEPTH_METRIC_FILTERED]->cudaRes, 0));

	checkCudaErrors(cudaGraphicsMapResources(1, &textures[GPUTexture::RGB]->cudaRes, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_cudaArrayRGB,
		textures[GPUTexture::RGB]->cudaRes,
		0, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &textures[GPUTexture::RGB]->cudaRes, 0));

	checkCudaErrors(cudaGraphicsMapResources(1, &indexMap.vertexTex()->cudaRes, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_cudaArrayVMapFloat4,
		indexMap.vertexTex()->cudaRes,
		0, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &indexMap.vertexTex()->cudaRes, 0));
}

xSurfelFusion::~xSurfelFusion()
{
	//std::cout << "start clear" << std::endl;
	//globalModel.clear();
	//std::cout << "clear" << std::endl;

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
}

void xSurfelFusion::clear()
{
	m_emptyVBO = true;

	m_currPose = Eigen::Matrix4f::Identity();
	checkCudaErrors(cudaGraphicsUnregisterResource(m_vboCudaRes));
	globalModel.reInitialise();

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_vboCudaRes, globalModel.model().first, cudaGraphicsMapFlagsNone));
	size_t numBytes;
	cudaGraphicsMapResources(1, &m_vboCudaRes, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_dVboCuda, &numBytes, m_vboCudaRes);
	cudaGraphicsUnmapResources(1, &m_vboCudaRes, 0);

	m_currVelocity = Eigen::Vector3f::Zero();
	m_currBiasAcc = Eigen::Vector3f::Zero();
	m_currBiasGyr = Eigen::Vector3f::Zero();
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
	                                                  true,
	                                                  true);
	xCheckGlDieOnError();
}

void xSurfelFusion::createCompute()
{
	xCheckGlDieOnError();
	computePacks[ComputePack::NORM] = new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom"),
	                                                  textures[GPUTexture::DEPTH_NORM]->texture);

	xCheckGlDieOnError();

	computePacks[ComputePack::FILTER] = new ComputePack(
		loadProgramFromFile("empty.vert", "depth_bilateral.frag", "quad.geom"),
		textures[GPUTexture::DEPTH_FILTERED]->texture);

	computePacks[ComputePack::METRIC] = new ComputePack(
		loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
		textures[GPUTexture::DEPTH_METRIC]->texture);

	computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(
		loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
		textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture);
}

void xSurfelFusion::createFeedbackBuffers()
{
	feedbackBuffers[FeedbackBuffer::RAW] = new FeedbackBuffer(
		loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));
	feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(
		loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));
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

void xSurfelFusion::trackCamera(ImuMeasurements& imuMeasurements,
                                Gravity& gravityW)
{
	innoreal::InnoRealTimer pauseRestartTimer;
	if (imuMeasurements.size() == 0)
	{
		pauseRestartTimer.TimeStart();
	}
#if 0
	cudaArray * textPtr;

	cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
	cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
	cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);
#endif
#if 0
	cv::Mat testMat(480, 640, CV_32FC4);
	fillIn.vertexTexture.texture->Download(testMat.data, GL_LUMINANCE, GL_FLOAT);
	std::ofstream testOfstream;
	testOfstream.open("D:\\xjm\\snapshot\\test\\test2.txt");
	for (int r = 0; r < testMat.rows; ++r)
	{
		for (int c = 0; c < testMat.cols; ++c)
		{
			testOfstream << testMat.at<float4>(r, c).x << std::endl;
			testOfstream << testMat.at<float4>(r, c).y << std::endl;
			testOfstream << testMat.at<float4>(r, c).z << std::endl;
			testOfstream << testMat.at<float4>(r, c).w << std::endl;
		}
	}
	testOfstream.close();
	std::exit(0);
#endif
	//innoreal::InnoRealTimer timer;

	//WARNING initICP* must be called before initRGB*
	//timer.TimeStart();
	frameToModel.initICPModel(&fillIn.vertexTexture,
		&fillIn.normalTexture,
		maxDepthProcessed, m_currPose);
	frameToModel.initRGBModel(&fillIn.imageTexture);
	//timer.TimeEnd();
	//std::cout << "initICPModel time:" << timer.TimeGap_in_ms() << std::endl;

	//timer.TimeStart();
	frameToModel.initICP(textures[GPUTexture::DEPTH_FILTERED], maxDepthProcessed);
	frameToModel.initRGB(textures[GPUTexture::DEPTH_METRIC_FILTERED], textures[GPUTexture::RGB]);
	//timer.TimeEnd();
	//std::cout << "initICP time:" << timer.TimeGap_in_ms() << std::endl;

	Eigen::Vector3f transPrev = m_currPose.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotPrev = m_currPose.topLeftCorner(3, 3);
	Eigen::Vector3f transCurr = transPrev;
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotCurr = rotPrev;
#if 0
	std::cout << "Rt0: " << std::endl;
	std::cout << rotPrev << std::endl;
	std::cout << transPrev << std::endl;
	std::cout << rotCurr << std::endl;
	std::cout << transCurr << std::endl;
#endif

#if 1
	float matchRate = 0;
	int matchesNum;
	//if (this->m_timeStamp == 20)
	if (imuMeasurements.size() == 0)
	{
		//innoreal::InnoRealTimer timer;
		//timer.TimeStart();
		while (matchRate < 0.3) {
			transCurr = transPrev;
			rotCurr = rotPrev;
			int vexNum;
			frameToModel.getIncrementalTransformationOrbFPFH(rotPrev,
																											 transPrev,
																											 rotCurr,
																											 transCurr, vexNum);
			//timer.TimeEnd();
			//std::cout << "orb fpfh time: " << timer.TimeGap_in_ms() << std::endl;
			//std::exit(0);

			m_currVelocity = Eigen::Vector3f::Zero();
			m_currBiasAcc = Eigen::Vector3f::Zero();
			m_currBiasGyr = Eigen::Vector3f::Zero();
#if 0
			std::cout << "Rt1: " << std::endl;
			std::cout << rotPrev << std::endl;
			std::cout << transPrev << std::endl;
			std::cout << rotCurr << std::endl;
			std::cout << transCurr << std::endl;
			std::exit(0);
#endif
#if 1
			/*frameToModel.saveForTest(rotPrev,
			transPrev,
			rotCurr,
			transCurr,
			"C:\\xjm\\snapshot\\test6.ply");*/
			//std::exit(0);
			//m_hasBeenPaused = true;
#endif
			frameToModel.getIncrementalTransformation(rotPrev,
																								transPrev,
																								rotCurr,
																								transCurr,
																								m_icpWeight,
																								m_rgbWeight,
																								m_imuWeight,
																								m_currVelocity,
																								m_currBiasAcc,
																								m_currBiasGyr,
																								imuMeasurements,
																								gravityW, m_hasBeenPaused, matchesNum);
			matchRate = 1.0*matchesNum / vexNum;
			//printf("matchesNum: %d\n", matchesNum);
			//printf("vexNum: %d\n", vexNum);
			//printf("matchRate: %f\n", matchRate);
		}

		/*frameToModel.saveForTest(rotPrev,
														 transPrev,
														 rotCurr,
														 transCurr,
														 "C:\\xjm\\snapshot\\test7.ply");*/
#if 0
		frameToModel.saveForTest(rotPrev,
														 transPrev,
														 rotCurr,
														 transCurr);
#endif
		}
#endif

	//std::cout << "icpWeight: " << icpWeight << std::endl;
	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();
#if 1
	//timer.TimeStart();
	else {
		frameToModel.getIncrementalTransformation(rotPrev,
																							transPrev,
																							rotCurr,
																							transCurr,
																							m_icpWeight,
																							m_rgbWeight,
																							m_imuWeight,
																							m_currVelocity,
																							m_currBiasAcc,
																							m_currBiasGyr,
																							imuMeasurements,
																							gravityW, m_hasBeenPaused, matchesNum);
	}
	//timer.TimeEnd();
	//std::cout << "getIncrementalTransformation time:" << timer.TimeGap_in_ms() << std::endl;
	
#if 0
	std::cout << "Rt2: " << std::endl;
	std::cout << rotPrev << std::endl;
	std::cout << transPrev << std::endl;
	std::cout << rotCurr << std::endl;
	std::cout << transCurr << std::endl;
#endif
#endif
#if 0
	std::cout << "imuMeasurements.size(): " << imuMeasurements.size() << std::endl;
	if (imuMeasurements.size() == 0)
	{
		m_hasBeenPaused = true;
#if 0
		frameToModel.saveForTest(rotPrev,
			transPrev,
			rotCurr,
			transCurr);
#endif	
	}
#endif
#if 0
	if (imuMeasurements.size() == 0)
	{
		std::cout << "Rt2: " << std::endl;
		std::cout << rotPrev << std::endl;
		std::cout << transPrev << std::endl;
		std::cout << rotCurr << std::endl;
		std::cout << transCurr << std::endl;

		frameToModel.saveForTest(rotPrev,
			transPrev,
			rotCurr,
			transCurr);
	}
#endif
#if 0
	frameToModel.getIncrementalTransformation2(rotv,
	                                           trans,
	                                           m_icpWeight,
	                                           m_rgbWeight,
	                                           m_currVelocity,
	                                           m_currBiasAcc,
	                                           m_currBiasGyr,
	                                           imuMeasurements,
	                                           gravityW);
#endif
#if 0
	if (m_hasBeenPaused)
	{
		cv::waitKey(0);
	}
#endif
	//timer.TimeEnd();
	//std::cout << "trackCamera time: " << timer.TimeGap_in_ms() << std::endl;
	xCheckGlDieOnError();
	//std::cout << "lastICPError: " << frameToModel.lastICPError << std::endl;
	//assert(frameToModel.lastICPError < 1e-04);

	m_currPose.topRightCorner(3, 1) = transCurr;
	m_currPose.topLeftCorner(3, 3) = rotCurr;

	if (imuMeasurements.size() == 0)
	{
		pauseRestartTimer.TimeEnd();
		std::cout << "pause restart time: " << pauseRestartTimer.TimeGap_in_ms() << std::endl;
		//std::exit(0);
	}
}

void xSurfelFusion::trackCamera(ImuMeasurements& imuMeasurements,
                                Gravity& gravityW,
                                cv::Mat& colorImg)
{
#if 0
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
    cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);
#endif
#if 0
    cv::Mat testMat(480, 640, CV_32FC4);
    fillIn.vertexTexture.texture->Download(testMat.data, GL_LUMINANCE, GL_FLOAT);
    std::ofstream testOfstream;
    testOfstream.open("D:\\xjm\\snapshot\\test\\test2.txt");
    for (int r = 0; r < testMat.rows; ++r)
    {
        for (int c = 0; c < testMat.cols; ++c)
        {
            testOfstream << testMat.at<float4>(r, c).x << std::endl;
            testOfstream << testMat.at<float4>(r, c).y << std::endl;
            testOfstream << testMat.at<float4>(r, c).z << std::endl;
            testOfstream << testMat.at<float4>(r, c).w << std::endl;
        }
    }
    testOfstream.close();
    std::exit(0);
#endif

    //WARNING initICP* must be called before initRGB*
    frameToModel.initICPModel(&fillIn.vertexTexture,
        &fillIn.normalTexture,
        maxDepthProcessed, m_currPose);
    frameToModel.initRGBModel(&fillIn.imageTexture);

    frameToModel.initICP(textures[GPUTexture::DEPTH_FILTERED], maxDepthProcessed);
    frameToModel.initRGB(textures[GPUTexture::DEPTH_FILTERED], textures[GPUTexture::RGB]);

    Eigen::Vector3f transPrev = m_currPose.topRightCorner(3, 1);
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotPrev = m_currPose.topLeftCorner(3, 3);
    Eigen::Vector3f transCurr = transPrev;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotCurr = rotPrev;
#if 0
    std::cout << "Rt0: " << std::endl;
    std::cout << rotPrev << std::endl;
    std::cout << transPrev << std::endl;
    std::cout << rotCurr << std::endl;
    std::cout << transCurr << std::endl;
#endif

#if 1
    //if (this->m_timeStamp == 20)
    m_colorImgForVis.push_back(colorImg.clone());
    if (imuMeasurements.size() == 0)
    {
			  int vexNum;
        std::cout << "getIncrementalTransformationOrbFPFH" << std::endl;
        frameToModel.getIncrementalTransformationOrbFPFH(rotPrev,
                                                         transPrev,
                                                         rotCurr,
                                                         transCurr,
																												 vexNum,
                                                         &m_colorImgForVis);
        m_currVelocity = Eigen::Vector3f::Zero();
        m_currBiasAcc = Eigen::Vector3f::Zero();
        m_currBiasGyr = Eigen::Vector3f::Zero();
#if 0
        std::cout << "Rt1: " << std::endl;
        std::cout << rotPrev << std::endl;
        std::cout << transPrev << std::endl;
        std::cout << rotCurr << std::endl;
        std::cout << transCurr << std::endl;
        std::exit(0);
#endif
#if 0
        frameToModel.saveForTest(rotPrev,
            transPrev,
            rotCurr,
            transCurr);
#endif
    }
#endif

    //std::cout << "icpWeight: " << icpWeight << std::endl;
    //innoreal::InnoRealTimer timer;
    //timer.TimeStart();
#if 1
		int matchesNum = 0;
    frameToModel.getIncrementalTransformation(rotPrev,
                                              transPrev,
                                              rotCurr,
                                              transCurr,
                                              m_icpWeight,
                                              m_rgbWeight,
                                              m_imuWeight,
                                              m_currVelocity,
                                              m_currBiasAcc,
                                              m_currBiasGyr,
                                              imuMeasurements,
                                              gravityW, m_hasBeenPaused, matchesNum);

#if 0
    std::cout << "Rt2: " << std::endl;
    std::cout << rotPrev << std::endl;
    std::cout << transPrev << std::endl;
    std::cout << rotCurr << std::endl;
    std::cout << transCurr << std::endl;
#endif
#endif
#if 0
    std::cout << "imuMeasurements.size(): " << imuMeasurements.size() << std::endl;
    if (imuMeasurements.size() == 0)
    {
        m_hasBeenPaused = true;
#if 0
        frameToModel.saveForTest(rotPrev,
            transPrev,
            rotCurr,
            transCurr);
#endif	
    }
#endif
#if 0
    if (imuMeasurements.size() == 0)
    {
        std::cout << "Rt2: " << std::endl;
        std::cout << rotPrev << std::endl;
        std::cout << transPrev << std::endl;
        std::cout << rotCurr << std::endl;
        std::cout << transCurr << std::endl;

        frameToModel.saveForTest(rotPrev,
            transPrev,
            rotCurr,
            transCurr);
    }
#endif
#if 0
    frameToModel.getIncrementalTransformation2(rotv,
        trans,
        m_icpWeight,
        m_rgbWeight,
        m_currVelocity,
        m_currBiasAcc,
        m_currBiasGyr,
        imuMeasurements,
        gravityW);
#endif
#if 0
    if (imuMeasurements.size() == 0)
    {
        m_hasBeenPaused = true;
    }
    if (m_hasBeenPaused)
    {
        cv::waitKey(0);
    }
#endif
    //timer.TimeEnd();
    //std::cout << "trackCamera time: " << timer.TimeGap_in_ms() << std::endl;
    xCheckGlDieOnError();
    //std::cout << "lastICPError: " << frameToModel.lastICPError << std::endl;
    //assert(frameToModel.lastICPError < 1e-04);

    m_currPose.topRightCorner(3, 1) = transCurr;
    m_currPose.topLeftCorner(3, 3) = rotCurr;
}

void xSurfelFusion::dofusion(float weight, int isFrag)
{
	indexMap.predictIndices(m_currPose, m_timeStamp, globalModel.model(), maxDepthProcessed, timeDelta, m_fragIdx);
	xCheckGlDieOnError();

	globalModel.fuse(m_currPose,
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

	indexMap.predictIndices(m_currPose, m_timeStamp, globalModel.model(), maxDepthProcessed, timeDelta, m_fragIdx);
	xCheckGlDieOnError();

	globalModel.clean(m_currPose,
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

void xSurfelFusion::processFrame2(cv::cuda::GpuMat& dRenderedDepthImg32F,
                                  float& velocity,
                                  cv::cuda::GpuMat& dRawDepthImg,
                                  cv::cuda::GpuMat& dFilteredDepthImg,
                                  cv::cuda::GpuMat& dRawDepthImg32F,
                                  cv::cuda::GpuMat& dFilteredDepthImg32F,
                                  cv::cuda::GpuMat& dColorImgRGBA,
                                  cv::cuda::GpuMat& dColorImgRGB,
                                  ImuMeasurements& imuMeasurements,
                                  Gravity& gravityW,
                                  bool objectDetected)
{
	innoreal::InnoRealTimer timer;
#if 0
    cv::Mat testtest;
    dFilteredDepthImg.download(testtest);
    cv::imshow("testtest", testtest * 60);
    cv::waitKey(1);
#endif

	//timer.TimeStart();
	checkCudaErrors(cudaMemcpy2DToArray(m_cudaArrayDepthRaw, 0, 0,
		dRawDepthImg.data, dRawDepthImg.step,
		dRawDepthImg.cols * sizeof(ushort), dRawDepthImg.rows,
		cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy2DToArray(m_cudaArrayDepthFiltered, 0, 0,
		dFilteredDepthImg.data, dFilteredDepthImg.step,
		dFilteredDepthImg.cols * sizeof(ushort), dFilteredDepthImg.rows,
		cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy2DToArray(m_cudaArrayDepthMetric, 0, 0,
		dRawDepthImg32F.data, dRawDepthImg32F.step,
		dRawDepthImg32F.cols * sizeof(float), dRawDepthImg32F.rows,
		cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy2DToArray(m_cudaArrayDepthMetricFiltered, 0, 0,
		dFilteredDepthImg32F.data, dFilteredDepthImg32F.step,
		dFilteredDepthImg32F.cols * sizeof(float), dFilteredDepthImg32F.rows,
		cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy2DToArray(m_cudaArrayRGB, 0, 0,
		dColorImgRGBA.data, dColorImgRGBA.step,
		dColorImgRGBA.cols * 4, dColorImgRGBA.rows,
		cudaMemcpyDeviceToDevice));	
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	xCheckGlDieOnError();
	//timer.TimeEnd();
	//std::cout << "upload data time:" << timer.TimeGap_in_ms() << std::endl;
	
#if 0
	cv::Mat testDepth(480, 640, CV_16UC1),
	        testColor(480, 640, CV_8UC3),
			testColor2(480, 640, CV_8UC3),
	        testColorRGBA(480, 640, CV_8UC4);
#if 0
	dColorImgRGBA.download(testColorRGBA);
#endif
#if 0
	checkCudaErrors(cudaMemcpy2DFromArray(
		testColor2.data, testColor2.step,
		m_cudaArrayRGB, 0, 0,
		testColor2.cols * 3, testColor2.rows,
		cudaMemcpyDeviceToHost));
	cv::imshow("testColor2", testColor2);
	cv::waitKey(0);
#endif
#if 0
	dRawDepthImg.download(testDepth);
	textures[GPUTexture::DEPTH_RAW]->texture->Upload(testDepth.data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
	testDepth = cv::Mat::zeros(480, 640, CV_16UC1);
	textures[GPUTexture::DEPTH_RAW]->texture->Download(testDepth.data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
	//dRawDepthImg.download(testDepth);
	//dColorImgRGB.download(testColor);
	//textures[GPUTexture::DEPTH_RAW]->texture->Download(testDepth.data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
	//textures[GPUTexture::RGB]->texture->Upload(testColor.data, GL_RGB, GL_UNSIGNED_BYTE);
	//textures[GPUTexture::RGB]->texture->Download(testColor2.data, GL_RGB, GL_UNSIGNED_BYTE);
	cv::imshow("testColor2", testDepth * 30);
	cv::waitKey(0);
#endif
#endif	

	//filterDepth();
	//metriciseDepth();

	if (m_timeStamp == 1)
	{
		computeFeedbackBuffers();
		
		globalModel.initialise(*feedbackBuffers[FeedbackBuffer::RAW], *feedbackBuffers[FeedbackBuffer::FILTERED]);	
		frameToModel.initFirstRGB(textures[GPUTexture::RGB]);
		//std::cout << "lastCount: " << globalModel.lastCount() << std::endl;
		//std::exit(0);
	}
	else
	{
		Eigen::Matrix4f lastPose = m_currPose; 
#if 0
        cv::Mat tmpColorImg;
        dColorImgRGB.download(tmpColorImg);
		std::cout << "curPose:\n" << m_currPose << std::endl;
#endif
		//trackCamera(imuMeasurements, gravityW, tmpColorImg);
		//timer.TimeStart();
		trackCamera(imuMeasurements, gravityW);
		//timer.TimeEnd();
		//std::cout << "track camera time:" << timer.TimeGap_in_ms() << std::endl;

		//Weight by velocity
		Eigen::Matrix4f diff = m_currPose.inverse() * lastPose;
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
		m_emptyVBO = false;
	}

	//timer.TimeStart();
	indexMap.combinedPredict(m_currPose,
	                         globalModel.model(),
	                         maxDepthProcessed,
	                         confidenceThreshold,
	                         m_timeStamp,
	                         timeDelta);
	//timer.TimeEnd();
	//std::cout << "predict time:" << timer.TimeGap_in_ms() << std::endl;

	if (objectDetected == false)
	{
#if 0
		cv::Mat vMapFloat4(480, 640, CV_32FC4);
#endif
		//timer.TimeStart();
		checkCudaErrors(cudaMemcpy2DFromArray(m_dVMapFloat4.data, m_dVMapFloat4.step,
			m_cudaArrayVMapFloat4, 0, 0,
			m_dVMapFloat4.cols * sizeof(float4), m_dVMapFloat4.rows,
			cudaMemcpyDeviceToDevice));
		VMapToDepthMap(dRenderedDepthImg32F, m_dVMapFloat4);
		//timer.TimeEnd();
		//std::cout << "render depth time:" << timer.TimeGap_in_ms() << std::endl;

#if 0
		cv::Mat renderedDepthImg32F;
		dRenderedDepthImg32F.download(renderedDepthImg32F);
		cv::imshow("renderedDepthImg32F2", renderedDepthImg32F * 100);
		cv::waitKey(0);
#endif
	}

#if 1
	//timer.TimeStart();
	fillIn.vertex(indexMap.vertexTex(), textures[GPUTexture::DEPTH_FILTERED], false);
	fillIn.normal(indexMap.normalTex(), textures[GPUTexture::DEPTH_FILTERED], false);
	//fillIn.image(indexMap.imageTex(), textures[GPUTexture::RGB], false);
	fillIn.image(textures[GPUTexture::RGB], textures[GPUTexture::RGB], false);
	//timer.TimeEnd();
	//std::cout << "fillin time:" << timer.TimeGap_in_ms() << std::endl;
#endif
}

void xSurfelFusion::processFrame(cv::Mat& renderedDepthImg,
                                 float& velocity,
                                 cv::Mat& colorImg,
                                 cv::Mat& depthImg,
                                 ImuMeasurements& imuMeasurements,
                                 Gravity& gravityW,
                                 bool objectDetected)
{
	innoreal::InnoRealTimer timer;
	timer.TimeStart();
#if 0
	cv::Mat depthImg1(480, 640, CV_16UC1), depthImg2(480, 640, CV_16UC1);
	cv::Mat depthImg32F1(480, 640, CV_32FC1), depthImg32F2(480, 640, CV_32FC1), depthImg32F3;
	cv::Mat colorImg1(480, 640, CV_8UC3), colorImg2(480, 640, CV_8UC4);
#endif
#if 1
	textures[GPUTexture::DEPTH_RAW]->texture->Upload(depthImg.data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
	textures[GPUTexture::RGB]->texture->Upload(colorImg.data, GL_RGB, GL_UNSIGNED_BYTE);
#if 0
	textures[GPUTexture::DEPTH_RAW]->texture->Download(depthImg1.data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
	textures[GPUTexture::RGB]->texture->Download(colorImg1.data, GL_RGB, GL_UNSIGNED_BYTE);
#endif
	xCheckGlDieOnError();
#endif

	filterDepth();
	metriciseDepth();
	xCheckGlDieOnError();
	timer.TimeEnd();
	std::cout << "time!!!" << timer.TimeGap_in_ms() << std::endl;
	//std::exit(0);

	if (m_timeStamp == 1)
	{
		computeFeedbackBuffers();

		globalModel.initialise(*feedbackBuffers[FeedbackBuffer::RAW], *feedbackBuffers[FeedbackBuffer::FILTERED]);
		std::cout << "lastCount: " << globalModel.lastCount() << std::endl;
		frameToModel.initFirstRGB(textures[GPUTexture::RGB]);
	}
	else
	{
		Eigen::Matrix4f lastPose = m_currPose;

		trackCamera(imuMeasurements, gravityW);
		std::cout << "curPose: " << m_currPose << std::endl;

		//Weight by velocity
		Eigen::Matrix4f diff = m_currPose.inverse() * lastPose;
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
	}

	indexMap.combinedPredict(m_currPose,
	                         globalModel.model(),
	                         maxDepthProcessed,
	                         confidenceThreshold,
	                         m_timeStamp,
	                         timeDelta);

#if 0
	if (objectDetected == false)
	{
		indexMap.vertexTex()->texture->Download(m_vertexImg.data, GL_RGBA, GL_FLOAT);
#pragma omp parallel
		for (int r = 0; r < m_vertexImg.rows; ++r)
		{
			for (int c = 0; c < m_vertexImg.cols; ++c)
			{
				renderedDepthImg.at<ushort>(r, c) = m_vertexImg.at<cv::Vec4f>(r, c)[2] * 1000.0f;
			}
		}
	}
#endif
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
	std::vector<Uniform> uniforms;

	uniforms.push_back(Uniform("maxD", depthCutoff));

	computePacks[ComputePack::METRIC]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
	computePacks[ComputePack::METRIC_FILTERED]->compute(textures[GPUTexture::DEPTH_FILTERED]->texture, &uniforms);
}

void xSurfelFusion::filterDepth()
{
	std::vector<Uniform> uniforms;

	uniforms.push_back(Uniform("cols", (float)Resolution::getInstance().cols()));
	uniforms.push_back(Uniform("rows", (float)Resolution::getInstance().rows()));
	uniforms.push_back(Uniform("maxD", depthCutoff));

	computePacks[ComputePack::FILTER]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
}

void xSurfelFusion::normaliseDepth(const float& minVal, const float& maxVal)
{
	std::vector<Uniform> uniforms;

	uniforms.push_back(Uniform("maxVal", maxVal * 1000.f));
	uniforms.push_back(Uniform("minVal", minVal * 1000.f));

	computePacks[ComputePack::NORM]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
}

Eigen::Vector3f xSurfelFusion::rodrigues2(const Eigen::Matrix3f& matrix)
{
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
	Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

	double rx = R(2, 1) - R(1, 2);
	double ry = R(0, 2) - R(2, 0);
	double rz = R(1, 0) - R(0, 1);

	double s = sqrt((rx * rx + ry * ry + rz * rz) * 0.25);
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
			t = (R(0, 0) + 1) * 0.5;
			rx = sqrt(std::max(t, 0.0));
			t = (R(1, 1) + 1) * 0.5;
			ry = sqrt(std::max(t, 0.0)) * (R(0, 1) < 0 ? -1.0 : 1.0);
			t = (R(2, 2) + 1) * 0.5;
			rz = sqrt(std::max(t, 0.0)) * (R(0, 2) < 0 ? -1.0 : 1.0);

			if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry * rz > 0))
				rz = -rz;
			theta /= sqrt(rx * rx + ry * ry + rz * rz);
			rx *= theta;
			ry *= theta;
			rz *= theta;
		}
	}
	else
	{
		double vth = 1 / (2 * s);
		vth *= theta;
		rx *= vth;
		ry *= vth;
		rz *= vth;
	}
	return Eigen::Vector3d(rx, ry, rz).cast<float>();
}

Eigen::Matrix4f& xSurfelFusion::getCurrPose()
{
	return m_currPose;
}

GlobalModel& xSurfelFusion::getGlobalModel()
{
	return globalModel;
}
