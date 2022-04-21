#include "stdafx.h"

#include "xSurfelFusion/GlobalModel.h"

const int GlobalModel::TEXTURE_DIMENSION = 3072;
const int GlobalModel::MAX_VERTICES = GlobalModel::TEXTURE_DIMENSION * GlobalModel::TEXTURE_DIMENSION;
const int GlobalModel::NODE_TEXTURE_DIMENSION = 16384;
const int GlobalModel::MAX_NODES = GlobalModel::NODE_TEXTURE_DIMENSION / 16; //16 floats per node

GlobalModel::GlobalModel()
	: target(0),
	renderSource(1),
	bufferSize(MAX_VERTICES * Vertex::SIZE),
	count(0),
	initProgram(loadProgramFromFile("init_unstable.vert")),
	drawProgram(loadProgramFromFile("draw_feedback.vert", "draw_feedback.frag")),
	drawSurfelProgram(loadProgramFromFile("draw_global_surface.vert", "draw_global_surface.frag", "draw_global_surface.geom")),
	dataProgram(loadProgramFromFile("data.vert", "data.frag", "data.geom")),
	updateProgram(loadProgramFromFile("update.vert")),
	unstableProgram(loadProgramGeomFromFile("copy_unstable.vert", "copy_unstable.geom")),
	renderBuffer(TEXTURE_DIMENSION, TEXTURE_DIMENSION),
	updateMapVertsConfs(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
	updateMapColorsTime(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
	updateMapNormsRadii(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT)	
{
	vbos = new std::pair<GLuint, GLuint>[2];

	float * vertices = new float[bufferSize];

	memset(&vertices[0], 0, bufferSize);

	glGenTransformFeedbacks(1, &vbos[0].second);
	glGenBuffers(1, &vbos[0].first);
	glBindBuffer(GL_ARRAY_BUFFER, vbos[0].first);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenTransformFeedbacks(1, &vbos[1].second);
	glGenBuffers(1, &vbos[1].first);
	glBindBuffer(GL_ARRAY_BUFFER, vbos[1].first);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	delete[] vertices;

	vertices = new float[Resolution::getInstance().numPixels() * Vertex::SIZE];
	memset(&vertices[0], 0, Resolution::getInstance().numPixels() * Vertex::SIZE);

	glGenTransformFeedbacks(1, &newUnstableFid);
	glGenBuffers(1, &newUnstableVbo);
	glBindBuffer(GL_ARRAY_BUFFER, newUnstableVbo);
	glBufferData(GL_ARRAY_BUFFER, Resolution::getInstance().numPixels() * Vertex::SIZE, &vertices[0], GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	delete[] vertices;

	std::vector<Eigen::Vector2f> uv;

	for (int i = 0; i < Resolution::getInstance().width(); i++)
	{
		for (int j = 0; j < Resolution::getInstance().height(); j++)
		{
			uv.push_back(Eigen::Vector2f(((float)i / (float)Resolution::getInstance().width()) + 1.0 / (2 * (float)Resolution::getInstance().width()),
				((float)j / (float)Resolution::getInstance().height()) + 1.0 / (2 * (float)Resolution::getInstance().height())));
		}
	}

	uvSize = uv.size();

	glGenBuffers(1, &uvo);
	glBindBuffer(GL_ARRAY_BUFFER, uvo);
	glBufferData(GL_ARRAY_BUFFER, uvSize * sizeof(Eigen::Vector2f), &uv[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	frameBuffer.AttachColour(*updateMapVertsConfs.texture);
	frameBuffer.AttachColour(*updateMapColorsTime.texture);
	frameBuffer.AttachColour(*updateMapNormsRadii.texture);
	frameBuffer.AttachDepth(renderBuffer);

	updateProgram->Bind();

	int locUpdate[3] =
	{
		glGetVaryingLocationNV(updateProgram->programId(), "vPosition0"),
		glGetVaryingLocationNV(updateProgram->programId(), "vColor0"),
		glGetVaryingLocationNV(updateProgram->programId(), "vNormRad0"),
	};

	glTransformFeedbackVaryingsNV(updateProgram->programId(), 3, locUpdate, GL_INTERLEAVED_ATTRIBS);	
	updateProgram->Unbind();

	dataProgram->Bind();

	int dataUpdate[3] =
	{
		glGetVaryingLocationNV(dataProgram->programId(), "vPosition0"),
		glGetVaryingLocationNV(dataProgram->programId(), "vColor0"),
		glGetVaryingLocationNV(dataProgram->programId(), "vNormRad0"),
	};

	glTransformFeedbackVaryingsNV(dataProgram->programId(), 3, dataUpdate, GL_INTERLEAVED_ATTRIBS);

	dataProgram->Unbind();

	unstableProgram->Bind();

	int unstableUpdate[3] =
	{
		glGetVaryingLocationNV(unstableProgram->programId(), "vPosition0"),
		glGetVaryingLocationNV(unstableProgram->programId(), "vColor0"),
		glGetVaryingLocationNV(unstableProgram->programId(), "vNormRad0"),
	};

	glTransformFeedbackVaryingsNV(unstableProgram->programId(), 3, unstableUpdate, GL_INTERLEAVED_ATTRIBS);

	unstableProgram->Unbind();

	initProgram->Bind();

	int locInit[3] =
	{
		glGetVaryingLocationNV(initProgram->programId(), "vPosition0"),
		glGetVaryingLocationNV(initProgram->programId(), "vColor0"),
		glGetVaryingLocationNV(initProgram->programId(), "vNormRad0"),
	};

	glTransformFeedbackVaryingsNV(initProgram->programId(), 3, locInit, GL_INTERLEAVED_ATTRIBS);

	glGenQueries(1, &countQuery);
	//Empty both transform feedbacks
	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[0].second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[0].first);

	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, 0);

	glEndTransformFeedback();

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[1].second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[1].first);

	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, 0);

	glEndTransformFeedback();

	glDisable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	initProgram->Unbind();
}

GlobalModel::~GlobalModel()
{
	glDeleteBuffers(1, &vbos[0].first);
	glDeleteTransformFeedbacks(1, &vbos[0].second);

	glDeleteBuffers(1, &vbos[1].first);
	glDeleteTransformFeedbacks(1, &vbos[1].second);

	glDeleteQueries(1, &countQuery);

	glDeleteBuffers(1, &uvo);

	glDeleteTransformFeedbacks(1, &newUnstableFid);
	glDeleteBuffers(1, &newUnstableVbo);

	delete[] vbos;
}

void GlobalModel::clear()
{
	float * vertices = new float[bufferSize];

	memset(&vertices[0], 0, bufferSize);

	glDeleteBuffers(1, &vbos[0].first);
	glDeleteTransformFeedbacks(1, &vbos[0].second);
	glDeleteBuffers(1, &vbos[1].first);
	glDeleteTransformFeedbacks(1, &vbos[1].second);

	glGenTransformFeedbacks(1, &vbos[0].second);
	glGenBuffers(1, &vbos[0].first);
	glBindBuffer(GL_ARRAY_BUFFER, vbos[0].first);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenTransformFeedbacks(1, &vbos[1].second);
	glGenBuffers(1, &vbos[1].first);
	glBindBuffer(GL_ARRAY_BUFFER, vbos[1].first);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	delete[] vertices;
}

void GlobalModel::initialise(const FeedbackBuffer& rawFeedback,
                const FeedbackBuffer& filteredFeedback)
{
	initProgram->Bind();

	glBindBuffer(GL_ARRAY_BUFFER, rawFeedback.vbo);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

	glBindBuffer(GL_ARRAY_BUFFER, filteredFeedback.vbo);

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[target].second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[target].first);

	glBeginTransformFeedback(GL_POINTS);

	glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

	//It's ok to use either fid because both raw and filtered have the same amount of vertices
	glDrawTransformFeedback(GL_POINTS, rawFeedback.fid);

	glEndTransformFeedback();

	glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

	glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

	glDisable(GL_RASTERIZER_DISCARD);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	initProgram->Unbind();

	glFinish();
}

void GlobalModel::renderPointCloud(pangolin::OpenGlMatrix mvp,
	pangolin::OpenGlMatrix mv,
	const float threshold,
	const bool drawUnstable,
	const bool drawNormals,
	const bool drawColors,
	const bool drawPoints,
	const bool drawWindow,
	const bool drawTimes,
	const int time,
	const int timeDelta)
{
	xCheckGlDieOnError();
	std::shared_ptr<Shader> program = false ? drawProgram : drawSurfelProgram;

	program->Bind();

	program->setUniform(Uniform("MVP", mvp));

	program->setUniform(Uniform("MV", mv));

	program->setUniform(Uniform("threshold", threshold));

	program->setUniform(Uniform("colorType", 0));// (drawNormals ? 1 : drawColors ? 2 : drawTimes ? 3 : 0)));

	program->setUniform(Uniform("unstable", drawUnstable));

	program->setUniform(Uniform("drawWindow", drawWindow));

	program->setUniform(Uniform("time", time));

	program->setUniform(Uniform("timeDelta", timeDelta));

	Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
	//This is for the point shader
	program->setUniform(Uniform("pose", pose));

	glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glDrawTransformFeedback(GL_POINTS, vbos[target].second);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	program->Unbind();
	xCheckGlDieOnError();
}

const std::pair<GLuint, GLuint> & GlobalModel::model()
{
	return vbos[target];
}

void GlobalModel::fuse(const Eigen::Matrix4f & pose,
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
	int fragInd)
{
	//This first part does data association and computes the vertex to merge with, storing
	//in an array that sets which vertices to update by index
	frameBuffer.Bind();

	glPushAttrib(GL_VIEWPORT_BIT);

	glViewport(0, 0, renderBuffer.width, renderBuffer.height);

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	dataProgram->Bind();

	dataProgram->setUniform(Uniform("cSampler", 0));
	dataProgram->setUniform(Uniform("drSampler", 1));
	dataProgram->setUniform(Uniform("drfSampler", 2));
	dataProgram->setUniform(Uniform("indexSampler", 3));
	dataProgram->setUniform(Uniform("vertConfSampler", 4));
	dataProgram->setUniform(Uniform("colorTimeSampler", 5));
	dataProgram->setUniform(Uniform("normRadSampler", 6));
	dataProgram->setUniform(Uniform("time", (float)time));
	dataProgram->setUniform(Uniform("weighting", weighting));
	dataProgram->setUniform(Uniform("fragInd", fragInd));

	dataProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(),
		Intrinsics::getInstance().cy(),
		1.0 / Intrinsics::getInstance().fx(),
		1.0 / Intrinsics::getInstance().fy())));
	dataProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
	dataProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));
	dataProgram->setUniform(Uniform("scale", (float)IndexMap::FACTOR));
	dataProgram->setUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
	dataProgram->setUniform(Uniform("pose", pose));
	dataProgram->setUniform(Uniform("maxDepth", depthCutoff));

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, uvo);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, newUnstableFid);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, newUnstableVbo);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rgb->texture->tid);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depthRaw->texture->tid);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, depthFiltered->texture->tid);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, indexMap->texture->tid);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, vertConfMap->texture->tid);

	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D, colorTimeMap->texture->tid);

	glActiveTexture(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D, normRadMap->texture->tid);

	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, uvSize);

	glEndTransformFeedback();

	frameBuffer.Unbind();

	glBindTexture(GL_TEXTURE_2D, 0);

	glActiveTexture(GL_TEXTURE0);

	glDisableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	dataProgram->Unbind();

	glPopAttrib();

	glFinish();
	
	//Next we update the vertices at the indexes stored in the update textures
	//Using a transform feedback conditional on a texture sample
	updateProgram->Bind();

	updateProgram->setUniform(Uniform("vertSamp", 0));
	updateProgram->setUniform(Uniform("colorSamp", 1));
	updateProgram->setUniform(Uniform("normSamp", 2));
	updateProgram->setUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
	updateProgram->setUniform(Uniform("time", time));
	updateProgram->setUniform(Uniform("fragInd", fragInd));

	glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].first);

	glBeginTransformFeedback(GL_POINTS);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, updateMapVertsConfs.texture->tid);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, updateMapColorsTime.texture->tid);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, updateMapNormsRadii.texture->tid);

	glDrawTransformFeedback(GL_POINTS, vbos[target].second);

	glEndTransformFeedback();

	glDisable(GL_RASTERIZER_DISCARD);

	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	updateProgram->Unbind();

	std::swap(target, renderSource);

	glFinish();
}

void GlobalModel::clean(const Eigen::Matrix4f & pose,
	const int & time,
	GPUTexture * indexMap,
	GPUTexture * vertConfMap,
	GPUTexture * colorTimeMap,
	GPUTexture * normRadMap,
	const float confThreshold,	
	const int timeDelta,
	const float maxDepth,
	int fragInd,
	int isFrag)
{	
	//Next we copy the new unstable vertices from the newUnstableFid transform feedback into the global map
	unstableProgram->Bind();
	unstableProgram->setUniform(Uniform("time", time));
	unstableProgram->setUniform(Uniform("confThreshold", confThreshold));
	unstableProgram->setUniform(Uniform("scale", (float)IndexMap::FACTOR));
	unstableProgram->setUniform(Uniform("indexSampler", 0));
	unstableProgram->setUniform(Uniform("vertConfSampler", 1));
	unstableProgram->setUniform(Uniform("colorTimeSampler", 2));
	unstableProgram->setUniform(Uniform("normRadSampler", 3));	
	unstableProgram->setUniform(Uniform("timeDelta", timeDelta));
	unstableProgram->setUniform(Uniform("maxDepth", maxDepth));
	//std::cout << "fragInd: " << fragInd << std::endl;
	//std::cout << "time: " << time << std::endl;
	unstableProgram->setUniform(Uniform("fragInd", fragInd));
	
	unstableProgram->setUniform(Uniform("isFrag", isFrag));

	Eigen::Matrix4f t_inv = pose.inverse();
	unstableProgram->setUniform(Uniform("t_inv", t_inv));

	unstableProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(),
		Intrinsics::getInstance().cy(),
		Intrinsics::getInstance().fx(),
		Intrinsics::getInstance().fy())));
	unstableProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
	unstableProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

	glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].first);

	glBeginTransformFeedback(GL_POINTS);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, indexMap->texture->tid);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, vertConfMap->texture->tid);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, colorTimeMap->texture->tid);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, normRadMap->texture->tid);	

	glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

	glDrawTransformFeedback(GL_POINTS, vbos[target].second);

	glBindBuffer(GL_ARRAY_BUFFER, newUnstableVbo);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glDrawTransformFeedback(GL_POINTS, newUnstableFid);

	glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

	glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

	glEndTransformFeedback();

	glDisable(GL_RASTERIZER_DISCARD);

	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	unstableProgram->Unbind();

	std::swap(target, renderSource);

	glFinish();	
}

unsigned int GlobalModel::lastCount()
{
	return count;
}



