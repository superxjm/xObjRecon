#pragma once

#include "xSurfelFusion/Shaders.h"
#include "xSurfelFusion/Uniform.h"
#include "xSurfelFusion/Vertex.h"
#include "GPUTexture.h"
#include "Helpers/xUtils.h"
#include <pangolin/gl/gl.h>
#include <Eigen/LU>

class IndexMap
{
public:
	IndexMap();
	virtual ~IndexMap();

	void predictIndices(const Eigen::Matrix4f & pose,
		const int & time,
		const std::pair<GLuint, GLuint> & model,
		const float depthCutoff,
		const int timeDelta,
		int fragInd);	

	void combinedPredict(const Eigen::Matrix4f & pose,
		const std::pair<GLuint, GLuint> & model,
		const float depthCutoff,
		const float confThreshold,
		const int time,
		const int timeDelta);

	GPUTexture * indexTex()
	{
		return &indexTexture;
	}

	GPUTexture * vertConfTex()
	{
		return &vertConfTexture;
	}

	GPUTexture * colorTimeTex()
	{
		return &colorTimeTexture;
	}

	GPUTexture * normalRadTex()
	{
		return &normalRadTexture;
	}

	GPUTexture * imageTex()
	{
		return &imageTexture;
	}

	GPUTexture * vertexTex()
	{
		return &vertexTexture;
	}

	GPUTexture * normalTex()
	{
		return &normalTexture;
	}

	GPUTexture * timeTex()
	{
		return &timeTexture;
	}

	static const int FACTOR;

private:
	std::shared_ptr<Shader> indexProgram;
	pangolin::GlFramebuffer indexFrameBuffer;
	pangolin::GlRenderBuffer indexRenderBuffer;
	GPUTexture indexTexture;
	GPUTexture vertConfTexture;
	GPUTexture colorTimeTexture;
	GPUTexture normalRadTexture;	

	std::shared_ptr<Shader> combinedProgram;
	pangolin::GlFramebuffer combinedFrameBuffer;
	pangolin::GlRenderBuffer combinedRenderBuffer;
	GPUTexture imageTexture;
	GPUTexture vertexTexture;
	GPUTexture normalTexture;
	GPUTexture timeTexture;
};
