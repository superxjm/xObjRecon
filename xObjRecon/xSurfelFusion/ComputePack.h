#pragma once

#include <pangolin/gl/gl.h>

#include "Uniform.h"
#include "Shaders.h"

class ComputePack
{
public:
	ComputePack(std::shared_ptr<Shader> program,
		pangolin::GlTexture * target);

	virtual ~ComputePack();

	static const std::string NORM, FILTER, METRIC, METRIC_FILTERED;

	void compute(pangolin::GlTexture * input, const std::vector<Uniform> * const uniforms = 0);

private:
	std::shared_ptr<Shader> program;
	pangolin::GlRenderBuffer renderBuffer;
	pangolin::GlTexture * target;
	pangolin::GlFramebuffer frameBuffer;
};
