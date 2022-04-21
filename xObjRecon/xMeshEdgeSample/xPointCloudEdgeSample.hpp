#pragma once

#include <opencv2/opencv.hpp>
//#include <Eigen/Eigen>

class xPointCloudEdgeSample
{
public:
	xPointCloudEdgeSample(const int renderWidth, const int renderHeight,
		const float renderFX, const float renderFY, const float renderCX, const float renderCY)
		: m_renderWidth(renderWidth), m_renderHeight(renderHeight), 
		m_renderCX(renderCX), m_renderCY(renderCY), m_renderFX(renderFX), m_renderFY(renderFY)
	{
		m_kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
		m_kernel2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(17, 17));
	}

	~xPointCloudEdgeSample()
	{

	}

	void projectVertexId(std::vector<int> &sampledVertexIndVec, cv::Mat_<int> &indImg, float *cameraPoseInv, std::vector<float> &vertices, int sampledNum, int vertexIdBase)
	{
		float posLocal[3];
		int u, v, z;
		indImg = cv::Mat_<int>::ones(m_renderHeight, m_renderWidth) * -1;
		m_zbuffer = cv::Mat_<float>::zeros(m_renderHeight, m_renderWidth);
		//std::cout << m_renderCX << " : " << m_renderCY << std::endl;
		for (int i = 0; i < vertices.size() / 4; ++i)
		{
			float *pos = vertices.data() + 4 * i;	
			posLocal[0] = cameraPoseInv[0] * pos[0] + cameraPoseInv[4] * pos[1] + cameraPoseInv[8] * pos[2] + cameraPoseInv[12];
			posLocal[1] = cameraPoseInv[1] * pos[0] + cameraPoseInv[5] * pos[1] + cameraPoseInv[9] * pos[2] + cameraPoseInv[13];
			posLocal[2] = cameraPoseInv[2] * pos[0] + cameraPoseInv[6] * pos[1] + cameraPoseInv[10] * pos[2] + cameraPoseInv[14];

			u = (int)(posLocal[0] * m_renderFX / posLocal[2] + m_renderCX);
			v = (int)(posLocal[1] * m_renderFY / posLocal[2] + m_renderCY);
			//std::cout << u << " : " << v << std::endl;
			if (u >= 0 && u < m_renderWidth && v >= 0 && v < m_renderHeight)
			{
				z = m_zbuffer(v, u);
				if (z < MYEPS || posLocal[2] < z)
				{
					z = posLocal[2];
					indImg.at<int>(v, u) = i + vertexIdBase;
				}
			}
		}

		if (sampledNum <= 0)
		{
			return;
		}

		m_objectMask = (indImg >= 255);
		cv::dilate(m_objectMask, m_objectMask, m_kernel1);
		cv::Canny(m_objectMask, m_pruneMat, 20, 40, 3);
		cv::dilate(m_pruneMat, m_pruneMat, m_kernel2);
#if 0
		cv::namedWindow("objectMask");
		cv::imshow("objectMask", objectMask);
		cv::namedWindow("pruneMat");
		cv::imshow("pruneMat", pruneMat);
		cv::waitKey(0);
#endif

		int id;
		sampledVertexIndVec.reserve(sampledNum);
		for (int row = 0; row < indImg.rows; ++row)
		{
			for (int col = 0; col < indImg.cols; ++col)
			{
				if (m_pruneMat.at<uchar>(row, col) == 255)
				{
					id = indImg.at<int>(row, col);
					if (id >= 0)
					{
						m_totalVertexIdVec.push_back(id);
					}
				}
			}
		}

		int step = m_totalVertexIdVec.size() / sampledNum;
		int ind = rand() % m_totalVertexIdVec.size();
		sampledVertexIndVec.resize(sampledNum);
		for (int i = 0; i < sampledNum; ++i)
		{
			ind = (ind + step) % m_totalVertexIdVec.size();
			sampledVertexIndVec[i] = m_totalVertexIdVec[ind];
		}
	}

private:
	int m_renderWidth, m_renderHeight;
	int m_renderCX, m_renderCY, m_renderFX, m_renderFY;

	cv::Mat_<float> m_zbuffer;
	cv::Mat m_objectMask, m_pruneMat;
	std::vector<int> m_totalVertexIdVec;

	cv::Mat m_kernel1, m_kernel2;
};
