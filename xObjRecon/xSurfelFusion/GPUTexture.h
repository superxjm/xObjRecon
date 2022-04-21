#pragma once

#include <pangolin/pangolin.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

class GPUTexture
{
public:
	GPUTexture(const int width,
		const int height,
		const GLenum internalFormat,
		const GLenum format,
		const GLenum dataType,
		const bool draw = false,
		const bool cuda = false);

	virtual ~GPUTexture();

public:
	static const std::string RGB, DEPTH_RAW, DEPTH_FILTERED, DEPTH_METRIC, DEPTH_METRIC_FILTERED, DEPTH_NORM;
	pangolin::GlTexture * texture;
	cudaGraphicsResource * cudaRes;
	const bool draw;

private:	
	const int width;
	const int height;
	const GLenum internalFormat;
	const GLenum format;
	const GLenum dataType;
};


