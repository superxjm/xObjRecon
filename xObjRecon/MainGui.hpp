#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <vector>
#include <opencv2/opencv.hpp>

#include "xDeformation/xDeformation.h"
#include "xSurfelFusion/xSurfelFusion.h"
#include "Helpers/xUtils.h"

class MyHandler : public pangolin::Handler3D
{
public:
	MyHandler(bool& show3D, pangolin::OpenGlRenderState& cam_state,
	          pangolin::AxisDirection enforce_up = pangolin::AxisNone, float trans_scale = 0.01f,
	          float zoom_fraction = PANGO_DFLT_HANDLER3D_ZF)
		: Handler3D(cam_state, enforce_up, trans_scale, zoom_fraction),
		  m_show3D(show3D)
	{
	}

#if 0
	void Mouse(pangolin::View& view, pangolin::MouseButton button, int x, int y, bool pressed, int button_state) override
	{
		if (button == pangolin::MouseButtonLeft)
		{
			if (pressed)
			{
				std::cout << x << " : " << y << std::endl;
				m_mouseSelectX = x;
				m_mouseSelectY = y;
			}
		}
	}
	void MouseMotion(pangolin::View& view, int x, int y, int button_state)
	{
		std::cout << x << " : " << y << std::endl;
		if (m_show3D)
		{
			Handler3D::MouseMotion(view, x, y, button_state);
		}
	}
#endif

public:
	int m_mouseSelectX = -1, m_mouseSelectY = -1;
	bool& m_show3D;
};

class xGUI
{
public:
	xGUI(float dispScale) :
		m_dispScale(dispScale)	
	{
		m_renderedModelImg = cv::Mat(dispScale * Resolution::getInstance().height(),
		                             dispScale * Resolution::getInstance().width(), CV_8UC3);
		m_resizedGrayRenderedModelImg = cv::Mat(Resolution::getInstance().height(),
		                             Resolution::getInstance().width(), CV_8UC1);

		// Note: 窗口大小和实际像素大小不同
		pangolin::CreateWindowAndBind("xObjRecon", dispScale * Resolution::getInstance().width(),
		                              dispScale * Resolution::getInstance().height());
		xCheckGlDieOnError();

		pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(150));
		xCheckGlDieOnError();

		m_pauseButton = new pangolin::Var<bool>("ui.pause", false, true);
		m_stepButton = new pangolin::Var<bool>("ui.step", false, false);
		m_saveButton = new pangolin::Var<bool>("ui.save", false, false);
		m_resetButton = new pangolin::Var<bool>("ui.reset", false, false);

		m_debugButton = new pangolin::Var<bool>("ui.debug", false, true);
		//m_srcVertexButton = new pangolin::Var<bool>("ui.Source", true, true);
		m_deformedVertexButton = new pangolin::Var<bool>("ui.deformed", true, true);
		//m_nodeButton = new pangolin::Var<bool>("ui.Node", true, true);
		m_vertexCorrButton = new pangolin::Var<bool>("ui.vertex Corr", true, true);

		//m_srcNormalButton = new pangolin::Var<bool>("ui.Src Normal", true, true);
		m_deformedNormalButton = new pangolin::Var<bool>("ui.deformed normal", true, true);

		m_vertexNumLabel = new pangolin::Var<std::string>("ui.#Points", "0");
		m_nodeNumLabel = new pangolin::Var<std::string>("ui.#Nodes", "0");
		m_fragNumLabel = new pangolin::Var<std::string>("ui.#Frags", "0");

		m_fragIdxA = new pangolin::Var<int>("ui.FragIdxSrc", -1, -1, MAX_FRAG_NUM);
		m_fragIdxB = new pangolin::Var<int>("ui.FragIdxDst", -1, -1, MAX_FRAG_NUM);

		pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(dispScale * Resolution::getInstance().width(),
		                                                         dispScale * Resolution::getInstance().height(),
		                                                         dispScale * Intrinsics::getInstance().fx(),
		                                                         dispScale * Intrinsics::getInstance().fy(),
		                                                         dispScale * Intrinsics::getInstance().cx(),
		                                                         dispScale * Intrinsics::getInstance().cy(),
		                                                         0.1, 1000);
		m_camState = pangolin::OpenGlRenderState(proj, pangolin::ModelViewLookAt(0, 0, 0, 0, 0, 1, pangolin::AxisNegY));
		m_mouseHandler = new MyHandler(m_show3D, m_camState);
		m_view = pangolin::Display("MainView")
		         .SetAspect(Resolution::getInstance().width() / Resolution::getInstance().height())
		         .SetHandler(m_mouseHandler);

		// 3D Mouse handler requires depth testing to be enabled
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		glDepthFunc(GL_LESS);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		pangolin::FinishFrame();
	}

	~xGUI()
	{
		delete m_mouseHandler;

		delete m_pauseButton;
		delete m_stepButton;
		delete m_saveButton;
		delete m_resetButton;

		delete m_debugButton;
		//delete m_srcVertexButton;
		delete m_deformedVertexButton;
		//delete m_nodeButton;
		delete m_vertexCorrButton;

		//delete m_srcNormalButton;
		delete m_deformedNormalButton;

		delete m_vertexNumLabel;
		delete m_nodeNumLabel;
		delete m_fragNumLabel;

		pangolin::DestroyWindow("xObjRecon");
	}

	void active()
	{
		m_view.Activate(m_camState);
	}

	void renderSceneForDebug(xDeformation* pDeform)
	{	
		int fragIdxSrc = m_fragIdxA->Get(), fragIdxDst = m_fragIdxB->Get();
		//std::cout << "fragIdxSrc: " << fragIdxSrc << std::endl;
		//std::cout << "fragIdxDst: " << fragIdxDst << std::endl;
#if 1
		m_corrLineVec.clear();
		m_deformedNormalLineVec.clear();
		std::vector<float4> deformedVertexVecFloat4;
		std::vector<float4> deformedNormalVecFloat4;
		pDeform->getDeformedVertices(deformedVertexVecFloat4);
		pDeform->getDeformedNormals(deformedNormalVecFloat4);
		m_deformedVertexVec.clear();
		m_deformedNormalVec.clear();
		const Eigen::Vector4f invalidStatus(0.0, 0.0, 0.0, 0.0);
		m_deformedVertexVec.resize(deformedVertexVecFloat4.size(), invalidStatus);
		m_deformedNormalVec.resize(deformedNormalVecFloat4.size(), invalidStatus);

		std::vector<int> vertexStrideVec;
		pDeform->getVertexStrideVe(vertexStrideVec);
		int baseIdx, len;
		if (fragIdxSrc != -1 && fragIdxDst != -1)
		{
			baseIdx = vertexStrideVec[fragIdxSrc];
			len = vertexStrideVec[fragIdxSrc + 1] - baseIdx;
			memcpy(m_deformedVertexVec.data() + baseIdx, deformedVertexVecFloat4.data() + baseIdx, len * sizeof(float4));
			memcpy(m_deformedNormalVec.data() + baseIdx, deformedNormalVecFloat4.data() + baseIdx, len * sizeof(float4));
			// The original last elem stands for fragment idx, make it 1.0 for vis
			for (int i = baseIdx; i < baseIdx + len; ++i)
			{
				m_deformedVertexVec[i](3) = 1.0f;
			}
			
			baseIdx = vertexStrideVec[fragIdxDst];
			len = vertexStrideVec[fragIdxDst + 1] - baseIdx;
			memcpy(m_deformedVertexVec.data() + baseIdx, deformedVertexVecFloat4.data() + baseIdx, len * sizeof(float4));
			memcpy(m_deformedNormalVec.data() + baseIdx, deformedNormalVecFloat4.data() + baseIdx, len * sizeof(float4));
			for (int i = baseIdx; i < baseIdx + len; ++i)
			{
				m_deformedVertexVec[i](3) = 1.0f;
			}
		}
		else
		{
			memcpy(m_deformedVertexVec.data(), deformedVertexVecFloat4.data(), deformedVertexVecFloat4.size() * sizeof(float4));
			memcpy(m_deformedNormalVec.data(), deformedNormalVecFloat4.data(), deformedNormalVecFloat4.size() * sizeof(float4));
			for (int i = 0; i < m_deformedVertexVec.size(); ++i)
			{
				m_deformedVertexVec[i](3) = 1.0f;
			}
		}
		for (int i = 0; i < m_deformedVertexVec.size(); ++i)
		{
			if (m_deformedVertexVec[i](3) > MYEPS)
			{
				m_deformedNormalLineVec.push_back(m_deformedVertexVec[i]);
				m_deformedNormalLineVec.push_back(m_deformedVertexVec[i] + m_deformedNormalVec[i] * 1.0 / 100.0);
			}
		}

		std::vector<int> matchingPointIdxVec;
		pDeform->getMatchingPointIndices(matchingPointIdxVec);
		for (int i = 0; i < matchingPointIdxVec.size() / 2; ++i)
		{
			if (m_deformedVertexVec[matchingPointIdxVec[2 * i]](3) > MYEPS 
				&& m_deformedVertexVec[matchingPointIdxVec[2 * i + 1]](3) > MYEPS)
			{
				m_corrLineVec.push_back(m_deformedVertexVec[matchingPointIdxVec[2 * i]]);
				m_corrLineVec.push_back(m_deformedVertexVec[matchingPointIdxVec[2 * i + 1]]);
			}
		}

		glClearColor(0.0, 0.0, 0.0, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		active();
		if (m_deformedNormalButton->Get())
		{	
			glLineWidth(1.0f);
			glColor3f(1.0f, 1.0f, 1.0f);
			pangolin::glDrawLines(m_deformedNormalLineVec);
		}
		if (m_deformedVertexButton->Get())
		{
			glPushAttrib(GL_ALL_ATTRIB_BITS);
			glPointSize(3.0f);
			glColor3f(1.0f, 0.0f, 0.0f);
			pangolin::glDrawVertices(m_deformedVertexVec, GL_POINTS);
			glPopAttrib();
		}
		if (m_vertexCorrButton->Get())
		{
			glColor3f(1.0f, 1.0f, 1.0f);
			glLineWidth(2.0f);
			pangolin::glDrawLines(m_corrLineVec);
		}
#endif
	}

	void renderScene(xSurfelFusion* pFusion,
									 xDeformation* pDeform,
									 int timeStamp,
									 Box& objectBox,
									 std::vector<Box>& boxVec,
									 bool objectDetected,
									 cv::Mat& colorImg,
									 cv::Mat& depthImg)
	{
		if (pFusion->m_emptyVBO)
			return;
#if 1
		pangolin::OpenGlMatrix mv;
		Eigen::Matrix4f currPose = pFusion->getCurrPose();

		Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);
		Eigen::Quaternionf currQuat(currRot);
		Eigen::Vector3f forwardVector(0, 0, 1);
		Eigen::Vector3f upVector(0, -1, 0);
#if 1
		Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
		Eigen::Vector3f up = (currQuat * upVector).normalized();
		Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));
#endif

#if 0
		eye -= forward;
#endif

#if 0
		if (objectDetected == true)
		{
			eye = Eigen::Vector3f(0, 0, 0);
			forward = Eigen::Vector3f(0, 0, 1);
			up = Eigen::Vector3f(0, -1, 0);
			eye -= forward;
			std::cout << "objectDetected" << std::endl;
		}
#endif
		Eigen::Vector3f at = eye + forward;
		Eigen::Vector3f z = (eye - at).normalized(); // Forward
		Eigen::Vector3f x = up.cross(z).normalized(); // Right
		Eigen::Vector3f y = z.cross(x);
		Eigen::Matrix4d m;
		m << x(0), x(1), x(2), -(x.dot(eye)),
			y(0), y(1), y(2), -(y.dot(eye)),
			z(0), z(1), z(2), -(z.dot(eye)),
			0, 0, 0, 1;
		memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));
		m_camState.SetModelViewMatrix(mv);
#endif

#if 0
		if (objectDetected == true)
		{
			m << 0.985093, -0.167752, 0.0380985, 0.0161657,
				-0.0357599, 0.0169406, 0.999217, -0.612957,
				-0.168266, -0.985684, 0.0106893, -1.6841,
				0, 0, 0, 1;
			memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));
			m_camState.SetModelViewMatrix(mv);
		}
		std::cout << "current mv: \n" << m << std::endl;
#endif

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		active();
#if 1
		innoreal::InnoRealTimer timer;
		//timer.TimeStart();
		pFusion->getGlobalModel().renderPointCloud(m_camState.GetProjectionModelViewMatrix(),
												   m_camState.GetModelViewMatrix(),
												   20,
												   true,
												   false,
												   true,
												   false,
												   false,
												   false,
												   timeStamp,
												   2000);
		//timer.TimeEnd();
		//std::cout << "1: " << timer.TimeGap_in_ms() << std::endl;
		//timer.TimeStart();
		glReadPixels(0, 0, m_dispScale * Resolution::getInstance().width(), m_dispScale * Resolution::getInstance().height(),
					 GL_RGB, GL_UNSIGNED_BYTE, m_renderedModelImg.data);
		//pFusion->getGlobalModel().rgbTextureRender.texture->Download(m_renderedModelImg.data, GL_RGB, GL_UNSIGNED_BYTE);
		//timer.TimeEnd();
		//std::cout << "disp size: " << m_dispScale * Resolution::getInstance().width() << " : " <<
			//m_dispScale * Resolution::getInstance().height() << std::endl;
		//std::cout << "read pixel time: " << timer.TimeGap_in_ms() << std::endl;
		//timer.TimeStart();
		cv::resize(m_renderedModelImg, m_resizedGrayRenderedModelImg, cv::Size(Resolution::getInstance().width(), Resolution::getInstance().height()));
		cv::cvtColor(m_resizedGrayRenderedModelImg, m_resizedGrayRenderedModelImg, CV_RGB2GRAY);
		cv::flip(m_resizedGrayRenderedModelImg, m_resizedGrayRenderedModelImg, 0);
		//timer.TimeEnd();
		//std::cout << "3: " << timer.TimeGap_in_ms() << std::endl;
#endif

#if 0
		if (objectDetected == false)
		{
			for (int i = 0; i < boxVec.size(); ++i)
			{
				Box &box = boxVec[i];
				cv::rectangle(colorImg, cv::Rect(box.m_left, box.m_top, box.m_right - box.m_left, box.m_bottom - box.m_top),
							  cv::Scalar(188, 188, 0), 4);
			}
			cv::rectangle(colorImg, cv::Rect(objectBox.m_left, objectBox.m_top, objectBox.m_right - objectBox.m_left, objectBox.m_bottom - objectBox.m_top),
						  cv::Scalar(0, 0, 255), 4);
			//std::cout << "score: " << objectBox.m_score << std::endl;

			int centerX = 648 / 2, centerY = 484 / 2;
			cv::rectangle(colorImg, cv::Rect(centerX - 17, centerY - 13, 35, 27),
						  cv::Scalar(0, 255, 255), 4);
			cv::line(colorImg, cv::Point(centerX - 17, centerY - 13), cv::Point(centerX + 17, centerY + 13),
					 cv::Scalar(0, 255, 255), 4);
			cv::line(colorImg, cv::Point(centerX - 17, centerY + 13), cv::Point(centerX + 17, centerY - 13),
					 cv::Scalar(0, 255, 255), 4);

			cv::imshow("detected color img", colorImg);
			cv::waitKey(1);
#if 0
			std::vector<int> pngCompressionParams;
			pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
			pngCompressionParams.push_back(0);
			char renderedDir[256];
			sprintf(renderedDir, "D:\\xjm\\data_for_video\\first_section\\object_selection\\%06d.png", timeStamp);
			cv::imwrite(renderedDir, colorImg, pngCompressionParams);
#endif
		}

#if 0
		if (objectDetected == false)
		{
			std::vector<int> pngCompressionParams;
			pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
			pngCompressionParams.push_back(0);
			char renderedDir[256];
			sprintf(renderedDir, "D:\\xjm\\data_for_video\\first_section\\object_model_before_selection\\%06d.png", timeStamp);
			cv::imwrite(renderedDir, m_resizedGrayRenderedModelImg, pngCompressionParams);
		}

		if (objectDetected == true)
		{
			std::vector<int> pngCompressionParams;
			pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
			pngCompressionParams.push_back(0);
			char renderedDir[256];
			sprintf(renderedDir, "D:\\xjm\\data_for_video\\first_section\\object_model\\%06d.png", timeStamp);
			cv::imwrite(renderedDir, m_resizedGrayRenderedModelImg, pngCompressionParams);
		}
#endif
	
#if 0
		if (objectDetected == true)
		{
			Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
			K(0, 0) = Intrinsics::getInstance().fx();
			K(1, 1) = Intrinsics::getInstance().fy();
			K(0, 2) = Intrinsics::getInstance().cx();
			K(1, 2) = Intrinsics::getInstance().cy();

			Eigen::Matrix3f Kinv = K.inverse();

			glColor3f(0, 1, 1);
			glLineWidth(3);
			pangolin::glDrawFrustrum(Kinv,
									 Resolution::getInstance().width(),
									 Resolution::getInstance().height(),
									 pFusion->getCurrPose(),
									 0.05f);	
				
			std::vector<Eigen::Matrix4f> updatedKeyPoseVec;
			int fragNum = pDeform->m_fragIdx + 1;
			updatedKeyPoseVec.resize(fragNum);
			checkCudaErrors(cudaMemcpy(updatedKeyPoseVec.data(),
							pDeform->m_dUpdatedKeyPoses, sizeof(Eigen::Matrix4f) * fragNum, cudaMemcpyDeviceToHost));	

			for (int fragIdx = 0; fragIdx < updatedKeyPoseVec.size(); ++fragIdx)
			{
				//updatedKeyPoseVec[fragIdx]
				
				glPushAttrib(GL_ALL_ATTRIB_BITS);
				//glColor3f(1.0f, 0.0f, 0.0f);
				//glLineWidth(2.0f);
				//m_trajectory.push_back(pFusion->getCurrPose().col(3));
				//pangolin::glDrawVertices(m_trajectory, GL_LINE_STRIP);
				glColor3f(1.0f, 0.0f, 0.0f);
				glLineWidth(3);
				pangolin::glDrawFrustrum(Kinv,
										 Resolution::getInstance().width(),
										 Resolution::getInstance().height(),
										 updatedKeyPoseVec[fragIdx],
										 0.05f);

				glPopAttrib();
			}

			glPushAttrib(GL_ALL_ATTRIB_BITS);
			glColor3f(1.0f, 1.0f, 0.0f);
			glLineWidth(2.0f);
			m_trajectory.push_back(pFusion->getCurrPose().col(3));
			pangolin::glDrawVertices(m_trajectory, GL_LINE_STRIP);
			glPopAttrib();

			glReadPixels(0, 0, m_dispScale * Resolution::getInstance().width(), m_dispScale * Resolution::getInstance().height(),
						 GL_RGB, GL_UNSIGNED_BYTE, m_renderedModelImg.data);
			//cv::cvtColor(m_renderedModelImg, m_resizedGrayRenderedModelImg, CV_RGB2GRAY);
			cv::cvtColor(m_renderedModelImg, m_resizedGrayRenderedModelImg, CV_RGB2BGR);
			cv::flip(m_resizedGrayRenderedModelImg, m_resizedGrayRenderedModelImg, 0);

			std::vector<int> pngCompressionParams;
			pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
			pngCompressionParams.push_back(0);
			char renderedDir[256];
			sprintf(renderedDir, "D:\\xjm\\data_for_video\\first_section\\object_with_trajectory\\%06d.png", timeStamp);
			cv::imwrite(renderedDir, m_resizedGrayRenderedModelImg, pngCompressionParams);
		}
#endif
#endif
		
#if 0
		innoreal::InnoRealTimer timer;
		timer.TimeStart();
		glReadPixels(0, 0, m_dispScale * Resolution::getInstance().width(), m_dispScale * Resolution::getInstance().height(), GL_RGB, GL_UNSIGNED_BYTE, m_renderedModelImg.data);
		cv::flip(m_renderedModelImg, m_renderedModelImg, 0);
		timer.TimeEnd();
		std::cout << "read pixel time: " << timer.TimeGap_in_ms() << std::endl;
#endif
#if 0
		//innoreal::InnoRealTimer timer;
		//timer.TimeStart();
		glReadPixels(0, 0, m_dispScale * Resolution::getInstance().width(), m_dispScale * Resolution::getInstance().height(), GL_RGB, GL_UNSIGNED_BYTE, m_renderedModelImg.data);
		//timer.TimeEnd();
		//std::cout << "read pixel time: " << timer.TimeGap_in_ms() << std::endl;
		//cv::resize(m_renderedModelImg, m_resizedRenderedModelImg, cv::Size(Resolution::getInstance().width(), Resolution::getInstance().height()));
		//cv::flip(m_resizedRenderedModelImg, m_resizedRenderedModelImg, 0);
		cv::flip(m_renderedModelImg, m_renderedModelImg, 0);

        char renderedDir[256];
        std::vector<int> pngCompressionParams;
        pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
        pngCompressionParams.push_back(0);
        sprintf(renderedDir, "D:\\xjm\\result\\before_opt\\our_result_without_obj_extract\\%06d.png", timeStamp);
        cv::imwrite(renderedDir, m_renderedModelImg, pngCompressionParams);
#endif
#if 0
		cv::imshow("renderedModelImg", m_resizedRenderedModelImg);
		cv::waitKey(0);
#endif

#if 0
		if (objectDetected == false)
		{
			for (int i = 0; i < boxVec.size(); ++i)
			{
				Box &box = boxVec[i];
				cv::rectangle(colorImg, cv::Rect(box.m_left, box.m_top, box.m_right - box.m_left, box.m_bottom - box.m_top),
					cv::Scalar(188, 188, 0), 4);
			}
			cv::rectangle(colorImg, cv::Rect(objectBox.m_left, objectBox.m_top, objectBox.m_right - objectBox.m_left, objectBox.m_bottom - objectBox.m_top),
				cv::Scalar(0, 0, 255), 4);
            //std::cout << "score: " << objectBox.m_score << std::endl;

            int centerX = 648 / 2, centerY = 484 / 2;
            cv::rectangle(colorImg, cv::Rect(centerX - 17, centerY - 13, 35, 27),
                cv::Scalar(0, 255, 255), 4);
            cv::line(colorImg, cv::Point(centerX - 17, centerY - 13), cv::Point(centerX + 17, centerY + 13),
                cv::Scalar(0, 255, 255), 4);
            cv::line(colorImg, cv::Point(centerX - 17, centerY + 13), cv::Point(centerX + 17, centerY - 13),
                cv::Scalar(0, 255, 255), 4);

			cv::imshow("detected color img", colorImg);
			cv::waitKey(1);
#if 0
            std::vector<int> pngCompressionParams;
            pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
            pngCompressionParams.push_back(0);
            char renderedDir[256];
            sprintf(renderedDir, "D:\\xjm\\result\\detect_process_result\\%04d.png", timeStamp);
            cv::imwrite(renderedDir, colorImg, pngCompressionParams);
#endif
		}
#endif

#if 0
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glColor3f(1.0f, 0.0f, 0.0f);
		glLineWidth(2.0f);
		m_trajectory.push_back(pFusion->getCurrPose().col(3));
		pangolin::glDrawVertices(m_trajectory, GL_LINE_STRIP);
		glPopAttrib();
#endif

#if 0
		glReadPixels(0, 0, 2 * Resolution::getInstance().width(), 2 * Resolution::getInstance().height(), GL_RGB, GL_UNSIGNED_BYTE, renderedImg.data);
		//cv::resize(renderedImg, resizedRenderedImg, cv::Size(Resolution::getInstance().width(), Resolution::getInstance().height()));
		resizedRenderedImg = renderedImg;
		cv::flip(resizedRenderedImg, resizedRenderedImg, 0);
#if 0
		resizedRenderedImg = depthImg.clone();
		resizedRenderedImg = resizedRenderedImg * 40;
#endif
#if 1
		cv::cvtColor(resizedRenderedImg, resizedRenderedImg, CV_BGR2RGB);
		if (objectDetected == false)
		{
			cv::rectangle(resizedRenderedImg, cv::Rect(320 - 17, 240 - 13, 35, 27),
				cv::Scalar(0, 255, 255), 4);
			cv::line(resizedRenderedImg, cv::Point(320 - 17, 240 - 13), cv::Point(320 + 17, 240 + 13),
				cv::Scalar(0, 255, 255), 4);
			cv::line(resizedRenderedImg, cv::Point(320 - 17, 240 + 13), cv::Point(320 + 17, 240 - 13),
				cv::Scalar(0, 255, 255), 4);
		}
#endif
#if 1
		if (objectDetected == false)
		{
			for (int i = 0; i < boxVec.size(); ++i)
			{
				Box &box = boxVec[i];
				cv::rectangle(resizedRenderedImg, cv::Rect(box.m_left, box.m_top, box.m_right - box.m_left, box.m_bottom - box.m_top),
					cv::Scalar(188, 188, 0), 4);
			}
		}
#endif
#if 1
		if (objectDetected == false)
		{
			cv::rectangle(resizedRenderedImg, cv::Rect(objectBox.m_left, objectBox.m_top, objectBox.m_right - objectBox.m_left, objectBox.m_bottom - objectBox.m_top),
				cv::Scalar(0, 0, 255), 4);
		}
#endif
		cv::namedWindow("rendered img");
		cv::imshow("rendered img", resizedRenderedImg);
		cv::waitKey(1);

#if 0
		std::vector<int> pngCompressionParams;
		pngCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
		pngCompressionParams.push_back(0);
		sprintf(renderedDir, "D:\\xjm\\result\\for_demo\\new_new_data\\test3\\%04d.png", totalCntTmp);
		cv::imwrite(renderedDir, resizedRenderedImg, pngCompressionParams);
#endif
#endif

		//pangolin::FinishFrame();
		//glFinish();
	}

	void setPanelInfo(xDeformation* pDeform)
	{
		std::stringstream strs1, strs2, strs3;
		strs1 << pDeform->getSrcVertexNum();
		m_vertexNumLabel->operator=(strs1.str());
		strs2 << pDeform->getSrcNodeNum();
		m_nodeNumLabel->operator=(strs2.str());
		strs3 << pDeform->getFragNum();
		m_fragNumLabel->operator=(strs3.str());
	}

public:
	float m_dispScale;
	bool m_show3D = true;
	pangolin::Var<bool> *m_debugButton, *m_srcVertexButton, *m_deformedVertexButton, *m_nodeButton, *
	                    m_vertexCorrButton;
	pangolin::Var<bool> *m_srcNormalButton, *m_deformedNormalButton;
	pangolin::Var<bool> *m_pauseButton, *m_stepButton, *m_saveButton, *m_resetButton;
	pangolin::Var<bool> *m_withDeformBuffon;
	pangolin::Var<std::string> *m_vertexNumLabel, *m_nodeNumLabel, *m_fragNumLabel;
	pangolin::Var<int> *m_fragIdxA, *m_fragIdxB;

	pangolin::OpenGlRenderState m_camState;
	pangolin::View m_view;
	MyHandler* m_mouseHandler;	

	// For debug 
	std::vector<Eigen::Vector4f> m_deformedVertexVec;
	std::vector<Eigen::Vector4f> m_deformedNormalVec;
	std::vector<Eigen::Vector4f> m_deformedNormalLineVec;
	std::vector<Eigen::Vector4f> m_corrLineVec;

	std::vector<Eigen::Vector4f> m_trajectory;

	cv::Mat m_renderedModelImg;
	cv::Mat m_resizedGrayRenderedModelImg;
};
