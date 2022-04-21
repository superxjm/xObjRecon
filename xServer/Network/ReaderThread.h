#ifndef READERTHREAD_H
#define READERTHREAD_H
#define COLOR_QUEUE_SIZE 300
#if 0
#include <QThread>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <thread>
#include <mutex>
struct Color_Buffer {
	volatile bool readable = true;
	cv::Mat color_queue[COLOR_QUEUE_SIZE];
	volatile bool isFull = false;
	volatile int front;
	volatile int rear;
	volatile int queueSize = COLOR_QUEUE_SIZE;
};
//extern Color_Buffer color_buffer;
class ReaderThread {
public:
	ReaderThread() {
		running_ = true;
		threadhandle_ = new std::thread(&ReaderThread::run, this);
	}
	~ReaderThread() {
		running_ = false;
		if (threadhandle_->joinable()) {
			threadhandle_->join();
		}
	}
	uint32_t status;
	std::mutex mtx;
	std::condition_variable conv;
	void run();
private:
	bool running_;
	std::thread *threadhandle_;
};
#endif
#endif // !READERTHREAD_H
