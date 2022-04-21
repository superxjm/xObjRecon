#include "stdafx.h"
#include "GPUTexture.h"

const std::string GPUTexture::RGB = "RGB";
const std::string GPUTexture::DEPTH_RAW = "DEPTH";
const std::string GPUTexture::DEPTH_FILTERED = "DEPTH_FILTERED";
const std::string GPUTexture::DEPTH_METRIC = "DEPTH_METRIC";
const std::string GPUTexture::DEPTH_METRIC_FILTERED = "DEPTH_METRIC_FILTERED";
const std::string GPUTexture::DEPTH_NORM = "DEPTH_NORM";

GPUTexture::GPUTexture(const int width,
	const int height,
	const GLenum internalFormat,
	const GLenum format,
	const GLenum dataType,
	const bool draw,
	const bool cuda)
	: texture(new pangolin::GlTexture(width, height, internalFormat, draw, 0, format, dataType)),
	draw(draw),
	width(width),
	height(height),
	internalFormat(internalFormat),
	format(format),
	dataType(dataType)
{
	if (cuda)
	{
		//cudaGraphicsGLRegisterImage(&cudaRes, texture->tid, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
		cudaGraphicsGLRegisterImage(&cudaRes, texture->tid, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	}
	else
	{
		cudaRes = 0;
	}
}

GPUTexture::~GPUTexture()
{
	if (texture)
	{
		delete texture;
	}

	if (cudaRes)
	{
		cudaGraphicsUnregisterResource(cudaRes);
	}
}
