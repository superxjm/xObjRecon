#include "stdafx.h"

#include "xSurfelFusion/IndexMap.h"

const int IndexMap::FACTOR = 1;

IndexMap::IndexMap()
	: indexProgram(loadProgramFromFile("index_map.vert", "index_map.frag")),
	indexRenderBuffer(Resolution::getInstance().width() * IndexMap::FACTOR, Resolution::getInstance().height() * IndexMap::FACTOR),
	indexTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
		Resolution::getInstance().height() * IndexMap::FACTOR,
		GL_LUMINANCE32UI_EXT,
		GL_LUMINANCE_INTEGER_EXT,
		GL_UNSIGNED_INT),
	vertConfTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
		Resolution::getInstance().height() * IndexMap::FACTOR,
		GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
	colorTimeTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
		Resolution::getInstance().height() * IndexMap::FACTOR,
		GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
	normalRadTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
		Resolution::getInstance().height() * IndexMap::FACTOR,
		GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
	combinedProgram(loadProgramFromFile("splat.vert", "combo_splat.frag")),
	combinedRenderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height()),
	imageTexture(Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		GL_RGBA,
		GL_RGB,
		GL_UNSIGNED_BYTE,
		false,
		true),
	vertexTexture(Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
	normalTexture(Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
	timeTexture(Resolution::getInstance().width(),
		Resolution::getInstance().height(),
		GL_LUMINANCE16UI_EXT,
		GL_LUMINANCE_INTEGER_EXT,
		GL_UNSIGNED_SHORT,
		false,
		true)
{
	printf("test\n");

	indexFrameBuffer.AttachColour(*indexTexture.texture);
	indexFrameBuffer.AttachColour(*vertConfTexture.texture);
	indexFrameBuffer.AttachColour(*colorTimeTexture.texture);
	indexFrameBuffer.AttachColour(*normalRadTexture.texture);
	indexFrameBuffer.AttachDepth(indexRenderBuffer);

	combinedFrameBuffer.AttachColour(*imageTexture.texture);
	combinedFrameBuffer.AttachColour(*vertexTexture.texture);
	combinedFrameBuffer.AttachColour(*normalTexture.texture);
	combinedFrameBuffer.AttachColour(*timeTexture.texture);
	combinedFrameBuffer.AttachDepth(combinedRenderBuffer);
}

IndexMap::~IndexMap()
{
}

void IndexMap::predictIndices(const Eigen::Matrix4f & pose,
	const int & time,
	const std::pair<GLuint, GLuint> & model,
	const float depthCutoff,
	const int timeDelta,
	int fragInd)
{
	indexFrameBuffer.Bind();

	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, indexRenderBuffer.width, indexRenderBuffer.height);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	indexProgram->Bind();

	Eigen::Matrix4f t_inv = pose.inverse();

	Eigen::Vector4f cam(Intrinsics::getInstance().cx() * IndexMap::FACTOR,
		Intrinsics::getInstance().cy() * IndexMap::FACTOR,
		Intrinsics::getInstance().fx() * IndexMap::FACTOR,
		Intrinsics::getInstance().fy() * IndexMap::FACTOR);

	indexProgram->setUniform(Uniform("t_inv", t_inv));
	indexProgram->setUniform(Uniform("cam", cam));
	indexProgram->setUniform(Uniform("maxDepth", depthCutoff));
	indexProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols() * IndexMap::FACTOR));
	indexProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows() * IndexMap::FACTOR));
	indexProgram->setUniform(Uniform("time", time));
	indexProgram->setUniform(Uniform("timeDelta", timeDelta));
	indexProgram->setUniform(Uniform("fragInd", fragInd));

	glBindBuffer(GL_ARRAY_BUFFER, model.first);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glDrawTransformFeedback(GL_POINTS, model.second);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	indexFrameBuffer.Unbind();
	indexProgram->Unbind();

	glPopAttrib();

	glFinish();
}

void IndexMap::combinedPredict(const Eigen::Matrix4f & pose,
	const std::pair<GLuint, GLuint> & model,
	const float depthCutoff,
	const float confThreshold,
	const int time,
	const int timeDelta)
{
	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);

	combinedFrameBuffer.Bind();

	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, combinedRenderBuffer.width, combinedRenderBuffer.height);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	combinedProgram->Bind();

	Eigen::Matrix4f t_inv = pose.inverse();

	Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
		Intrinsics::getInstance().cy(),
		Intrinsics::getInstance().fx(),
		Intrinsics::getInstance().fy());

	combinedProgram->setUniform(Uniform("t_inv", t_inv));
	combinedProgram->setUniform(Uniform("cam", cam));
	combinedProgram->setUniform(Uniform("maxDepth", depthCutoff));
	combinedProgram->setUniform(Uniform("confThreshold", confThreshold));
	combinedProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
	combinedProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));
	combinedProgram->setUniform(Uniform("time", time));
	combinedProgram->setUniform(Uniform("maxTime", time));
	combinedProgram->setUniform(Uniform("timeDelta", timeDelta));

	glBindBuffer(GL_ARRAY_BUFFER, model.first);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glDrawTransformFeedback(GL_POINTS, model.second);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	combinedFrameBuffer.Unbind();
	combinedProgram->Unbind();

	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);

	glPopAttrib();	

	glFinish();
}