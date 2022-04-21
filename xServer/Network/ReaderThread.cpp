#if 0
#include"ReaderThread.h"
#include <QThread>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <thread>
#include <chrono>
#include <Mmsystem.h>
#include "timer.hpp"

#pragma comment(lib, "winmm.lib") 

extern Color_Buffer color_buffer;

void ReaderThread::run() {
	::timeBeginPeriod(1);
	innoreal::InnoRealTimer readerTimer;
	innoreal::InnoRealTimer sleepTimer;
	cv::namedWindow("Color", CV_WINDOW_AUTOSIZE);
	int delay = 0;
	bool flag = true;
	const size_t sleepTime = 33* 1000000;
	bool didReadData = false;
	while (running_) {
		//std::cout << "Going to here" << std::endl;
		/*while (!color_buffer.readable)
			Sleep(1);*/
		//color_buffer.readable = false;
		mtx.lock();
		//std::cout << "Reader thread lock" << std::endl;
		if (color_buffer.isFull || color_buffer.front != color_buffer.rear % color_buffer.queueSize) {
			readerTimer.TimeEnd();
			/*if (delay > 5)
				std::cout << "Delay" << delay << " " << readerTimer.TimeGap_in_ms() << std::endl;*/
			//delay = delay + readerTimer.TimeGap_in_ms() - 33;
			/*if (readerTimer.TimeGap_in_ms() > 35)
				std::cout << "Time gap in imshow: " << readerTimer.TimeGap_in_ms() << std::endl;*/
			/*if (flag) {
				flag = false;
				delay = 0;
			}
			if (delay > 0) {
				if (sleepTime - delay > 0) {
					sleepTime -= delay;
					delay = 0;
				}
				else {
					delay -= sleepTime;
					sleepTime = 0;
				}
			}
			else if (delay < 0) {
				sleepTime += delay;
				delay = 0;
			}*/
			//std::cout << "delay " << delay << " " << sleepTime << std::endl;
			cv::imshow("Color", color_buffer.color_queue[color_buffer.front % color_buffer.queueSize]);
			cv::waitKey(1);
			color_buffer.front = (color_buffer.front + 1) % color_buffer.queueSize;
			//if (color_buffer.front == color_buffer.rear)
				color_buffer.isFull = false;
			readerTimer.TimeStart();
			sleepTimer.TimeStart();
			// QThread::msleep(sleepTime);
			// Sleep(sleepTime);
			didReadData = true;
		}
		else didReadData = false;
		mtx.unlock();
		std::cout << "Reader thread unlock" << std::endl;
		if (didReadData) {
			std::this_thread::sleep_for(std::chrono::nanoseconds(sleepTime));
			sleepTimer.TimeEnd();
			printf("sleep_ms=%f\n", sleepTimer.TimeGap_in_ms());
		}
		//color_buffer.readable = true;
		
	}
	::timeEndPeriod(1);
}
#endif

