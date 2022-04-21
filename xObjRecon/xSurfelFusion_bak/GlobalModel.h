#pragma once

#include <pangolin/gl/gl.h>
#include <Eigen/LU>

#include "Helpers/xUtils.h"
#include "xSurfelFusion/Shaders.h"
#include "xSurfelFusion/Uniform.h"
#include "xSurfelFusion/FeedbackBuffer.h"
#include "GPUTexture.h"
#include "IndexMap.h"

class GlobalModel
{
public:
	GlobalModel();
	virtual ~GlobalModel();

	void clear();
	void initialise(const FeedbackBuffer & rawFeedback,
		const FeedbackBuffer & filteredFeedback);

	static const int TEXTURE_DIMENSION;
	static const int MAX_VERTICES;
	static const int NODE_TEXTURE_DIMENSION;
	static const int MAX_NODES;

	void renderPointCloud(pangolin::OpenGlMatrix mvp,
		pangolin::OpenGlMatrix mv,
		const float threshold,
		const bool drawUnstable,
		const bool drawNormals,
		const bool drawColors,
		const bool drawPoints,
		const bool drawWindow,
		const bool drawTimes,
		const int time,
		const int timeDelta);

	const std::pair<GLuint, GLuint> & model();

	void fuse(const Eigen::Matrix4f & pose,
		const int & time,
		GPUTexture * rgb,
		GPUTexture * depthRaw,
		GPUTexture * depthFiltered,
		GPUTexture * indexMap,
		GPUTexture * vertConfMap,
		GPUTexture * colorTimeMap,
		GPUTexture * normRadMap,
		const float depthCutoff,
		const float confThreshold,
		const float weighting,
		int fragInd);

	void clean(const Eigen::Matrix4f & pose,
		const int & time,
		GPUTexture * indexMap,
		GPUTexture * vertConfMap,
		GPUTexture * colorTimeMap,
		GPUTexture * normRadMap,
		const float confThreshold,
		const int timeDelta,
		const float maxDepth,
		int fragInd,
		int isFrag);

	unsigned int lastCount();	

public:
	struct cudaGraphicsResource *vbosCudaResouse[2];

private:
	//First is the vbo, second is the fid
	std::pair<GLuint, GLuint> * vbos;	
	int target, renderSource;

	const int bufferSize;

	GLuint countQuery;
	unsigned int count;

	std::shared_ptr<Shader> initProgram;
	std::shared_ptr<Shader> drawProgram;
	std::shared_ptr<Shader> drawSurfelProgram;

	//For supersample fusing
	std::shared_ptr<Shader> dataProgram;
	std::shared_ptr<Shader> updateProgram;
	std::shared_ptr<Shader> unstableProgram;
	pangolin::GlRenderBuffer renderBuffer;

	//We render updated vertices vec3 + confidences to one texture
	GPUTexture updateMapVertsConfs;

	//We render updated colors vec3 + timestamps to another
	GPUTexture updateMapColorsTime;

	//We render updated normals vec3 + radii to another
	GPUTexture updateMapNormsRadii;	

	GLuint newUnstableVbo, newUnstableFid;

	pangolin::GlFramebuffer frameBuffer;
	GLuint uvo;
	int uvSize;
};
