#pragma once

#include <QtCore/QThread>
#include <opencv2/opencv.hpp>

#include "xSensors/xCapture.hpp"

#define INPUT_DIR "D:\\xjm\\scanner\\bin\\999\\match\\"
#define INPUT_DIR_AFTER_TRACK "D:\\xjm\\scanner\\bin\\999\\match\\"

class xImageImporter : public QThread, public xCapture
{
public:
	explicit xImageImporter(float imgScale) : m_imgScale(imgScale)
	{
		m_depthVec.resize(m_bufLen);
		m_colorVec.resize(m_bufLen);
		m_fullColorVec.resize(m_bufLen);
		m_readInd = m_writeInd = 1;

        this->start();
	}

	virtual ~xImageImporter()
	{

	}

	void run() override
	{
		cv::Mat depthImg, colorImg, resizedColorImg;
		char depthPath[256], colorPath[256];
		while (true)
		{
			sprintf(depthPath, INPUT_DIR_AFTER_TRACK"frame-%06d.depth.png", m_writeInd);
			sprintf(colorPath, INPUT_DIR_AFTER_TRACK"frame-%06d.color.png", m_writeInd);

            std::cout << depthPath << std::endl;
            std::cout << colorPath << std::endl;

			depthImg = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
			colorImg = cv::imread(colorPath, cv::IMREAD_COLOR);

			if (depthImg.rows == 0 || colorImg.rows == 0)
			{
				std::cout << "exit thread" << std::endl;
				quit();
				return;
			}

#if 1
			cv::Rect validBox = cv::Rect(0, 32, 1280, 960);
			resizedColorImg = colorImg(validBox).clone();
			cv::resize(resizedColorImg, resizedColorImg, cv::Size(640, 480));
#endif
#if 0
			resizedColorImg = colorImg;
#endif
#if 0
			cv::Mat colorDepthImg;
			RegisteredImgs(colorDepthImg, resizedColorImg, depthImg);
			cv::imshow("colorDepthImg", colorDepthImg);
			cv::waitKey(0);
#endif	

			while ((m_writeInd % m_bufLen + 1) % m_bufLen == m_readInd % m_bufLen)
			{
			}
			m_depthVec[m_writeInd % m_bufLen] = depthImg.clone();
			m_colorVec[m_writeInd % m_bufLen] = resizedColorImg.clone();

			++m_writeInd;
		}
	}

    bool next(cv::Mat& depthImg, cv::Mat& colorImg,
              ImuMeasurements* imuMeasurements = NULL, Gravity* gravity = NULL,
              int* keyFrameIdxEachFrag = NULL)
    {
        if (imuMeasurements != NULL)
        {
            imuMeasurements->resize(1);
        }
		while (m_writeInd % m_bufLen == m_readInd % m_bufLen)
		{
			if (isFinished())
			{
				std::cout << "thread finished" << std::endl;
				return false;
			}
		}

		m_depthImg = m_depthVec[m_readInd % m_bufLen];
		m_colorImg = m_colorVec[m_readInd % m_bufLen];

		depthImg = m_depthImg;
		colorImg = m_colorImg;

		++m_readInd;

		return true;
	}

private:
	std::vector<cv::Mat> m_depthVec, m_colorVec, m_fullColorVec;
	int m_bufLen = 200;
	volatile int m_readInd;
	volatile int m_writeInd;

	cv::Mat m_depthImg, m_colorImg, m_fullColorImg;
	float m_imgScale;
};

class xImageImporter2 : public QThread, public xCapture
{
public:
    explicit xImageImporter2(float imgScale) : m_imgScale(imgScale)
    {
        m_depthVec.resize(m_bufLen);
        m_colorVec.resize(m_bufLen);
        m_fullColorVec.resize(m_bufLen);
        m_hasBeenPaused.resize(m_bufLen, 0);
        m_readIdx = m_writeIdx = 1;

#if 1
        m_dirs.push_back(std::string("D:\\xjm\\scanner\\bin\\503\\match\\"));
        m_starts.push_back(300);
        m_steps.push_back(-1);
        m_ends.push_back(60);
#endif
        m_dirs.push_back(std::string("D:\\xjm\\scanner\\bin\\510\\match\\"));
        m_starts.push_back(310);
        m_steps.push_back(-1);
        m_ends.push_back(15);
        m_dirs.push_back(std::string("D:\\xjm\\scanner\\bin\\510\\match\\"));
        m_starts.push_back(15);
        m_steps.push_back(1);
        m_ends.push_back(310);
#if 1
        m_dirs.push_back(std::string("D:\\xjm\\scanner\\bin\\505\\match\\"));
        m_starts.push_back(10);
        m_steps.push_back(1);
        m_ends.push_back(490);
        m_dirs.push_back(std::string("D:\\xjm\\scanner\\bin\\505\\match\\"));
        m_starts.push_back(490);
        m_steps.push_back(-1);
        m_ends.push_back(281);
#endif
        m_dirs.push_back(std::string("D:\\xjm\\scanner\\bin\\504\\match\\"));
        m_starts.push_back(20);
        m_steps.push_back(1);
        m_ends.push_back(400);

        this->start();
    }

    virtual ~xImageImporter2()
    {

    }

    void run() override
    {
        cv::Mat depthImg, colorImg;
        char depthPath[256], colorPath[256];
        for (int i = 0; i < m_dirs.size(); ++i)
        {
            for (int idx = m_starts[i]; idx != m_ends[i]; idx += m_steps[i])
            {
                sprintf(depthPath, (m_dirs[i] + "frame-%06d.depth.png").c_str(), idx);
                sprintf(colorPath, (m_dirs[i] + "frame-%06d.color.png").c_str(), idx);

                //std::cout << depthPath << std::endl;
                //std::cout << colorPath << std::endl;

                depthImg = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
                colorImg = cv::imread(colorPath, cv::IMREAD_COLOR);

                if (depthImg.rows == 0 || colorImg.rows == 0)
                {
                    std::cout << "exit thread" << std::endl;
                    quit();
                    return;
                } 

#if 0
                cv::Mat colorDepthImg;
                RegisteredImgs(colorDepthImg, resizedColorImg, depthImg);
                cv::imshow("colorDepthImg", colorDepthImg);
                cv::waitKey(0);
#endif	

                while ((m_writeIdx % m_bufLen + 1) % m_bufLen == m_readIdx % m_bufLen)
                {
                }
                m_depthVec[m_writeIdx % m_bufLen] = depthImg.clone();
                m_colorVec[m_writeIdx % m_bufLen] = colorImg.clone();
                if (idx == m_starts[i])
                {
                    m_hasBeenPaused[m_writeIdx % m_bufLen] = 1;
                }
                else
                {
                    m_hasBeenPaused[m_writeIdx % m_bufLen] = 0;
                }
                ++m_writeIdx;
            }
        }
    }

    bool next(cv::Mat& depthImg, cv::Mat& colorImg,
        ImuMeasurements* imuMeasurements = NULL, Gravity* gravity = NULL,
        int* keyFrameIdxEachFrag = NULL)
    {
        if (imuMeasurements != NULL)
        {
            imuMeasurements->resize(1);
        }
        while (m_writeIdx % m_bufLen == m_readIdx % m_bufLen)
        {
            if (isFinished())
            {
                std::cout << "thread finished" << std::endl;
                return false;
            }
        }

        m_depthImg = m_depthVec[m_readIdx % m_bufLen];
        m_fullColorImg = m_colorVec[m_readIdx % m_bufLen];
        int hasBeenPaused = m_hasBeenPaused[m_readIdx % m_bufLen];

        cv::Rect validBox = cv::Rect(0, 32, 1280, 960);
        cv::resize(m_fullColorImg, m_colorImg, cv::Size(640, 480));
        depthImg = m_depthImg;
        colorImg = m_colorImg.clone();

        ++m_readIdx;

        if (hasBeenPaused == 1)
        {
            std::cout << "has been paused" << std::endl;
            imuMeasurements->clear();
            return true;
            
        }
        return false;
    }

    bool nextForXTION(cv::Mat& depthImg, cv::Mat& colorImg, cv::Mat& fullColorImg,
                      ImuMeasurements* imuMeasurements = NULL, Gravity* gravity = NULL,
                      int* keyFrameIdxEachFrag = NULL)
    {
        if (imuMeasurements != NULL)
        {
            imuMeasurements->resize(1);
        }
        while (m_writeIdx % m_bufLen == m_readIdx % m_bufLen)
        {
            if (isFinished())
            {
                std::cout << "thread finished" << std::endl;
                return false;
            }
        }

        m_depthImg = m_depthVec[m_readIdx % m_bufLen];
        fullColorImg = m_colorVec[m_readIdx % m_bufLen].clone();
        int hasBeenPaused = m_hasBeenPaused[m_readIdx % m_bufLen];

        cv::Rect validBox = cv::Rect(0, 32, 1280, 960);
        cv::resize(fullColorImg(validBox), m_colorImg, cv::Size(640, 480));
        //cv::resize(fullColorImg(validBox), m_colorImg, cv::Size(640, 480));
        depthImg = m_depthImg.clone();
        colorImg = m_colorImg.clone(); 

        ++m_readIdx;

        if (hasBeenPaused == 1)
        {
            std::cout << "has been paused" << std::endl;
            imuMeasurements->clear();
            return true;

        }
        return false;
    }

private:
    std::vector<cv::Mat> m_depthVec, m_colorVec, m_fullColorVec;
    std::vector<int> m_hasBeenPaused;
    int m_bufLen = 500;
    volatile int m_readIdx;
    volatile int m_writeIdx;

    cv::Mat m_depthImg, m_colorImg, m_fullColorImg;
    float m_imgScale;

    std::vector<std::string> m_dirs;
    std::vector<int> m_starts;
    std::vector<int> m_steps;
    std::vector<int> m_ends;
};

