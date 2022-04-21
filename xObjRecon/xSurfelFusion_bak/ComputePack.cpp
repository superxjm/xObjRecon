#include "stdafx.h"

#include "xSurfelFusion/ComputePack.h"
#include "Helpers/xUtils.h"

const std::string ComputePack::NORM = "NORM";
const std::string ComputePack::FILTER = "FILTER";
const std::string ComputePack::METRIC = "METRIC";
const std::string ComputePack::METRIC_FILTERED = "METRIC_FILTERED";

ComputePack::ComputePack(std::shared_ptr<Shader> program,
                         pangolin::GlTexture * target)
 : program(program),
   renderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height()),
   target(target)
{
    frameBuffer.AttachColour(*target);
    frameBuffer.AttachDepth(renderBuffer);
}

ComputePack::~ComputePack()
{

}

void ComputePack::compute(pangolin::GlTexture * input, const std::vector<Uniform> * const uniforms)
{
    input->Bind();

    frameBuffer.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, renderBuffer.width, renderBuffer.height);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    program->Bind();

    if(uniforms)
    {
        for(size_t i = 0; i < uniforms->size(); i++)
        {
            program->setUniform(uniforms->at(i));
        }
    }

    glDrawArrays(GL_POINTS, 0, 1);

    frameBuffer.Unbind();

    program->Unbind();

    glPopAttrib();

    glFinish();
}
