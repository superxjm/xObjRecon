#ifndef SHADERS_SHADERS_H_
#define SHADERS_SHADERS_H_

#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glinclude.h>
#include <memory>

#include "xSurfelFusion/Parse.h"
#include "xSurfelFusion/Uniform.h"
#include "xUtils.h"

class Shader : public pangolin::GlSlProgram
{
public:
	Shader()
	{
		//staticGlewInit();
	}

	GLuint programId()
	{
		return prog;
	}

	void setUniform(const Uniform & v)
	{
		GLuint loc = glGetUniformLocation(prog, v.id.c_str());

		switch (v.t)
		{
		case Uniform::INT:
			glUniform1i(loc, v.i);
			break;
		case Uniform::FLOAT:
			glUniform1f(loc, v.f);
			break;
		case Uniform::VEC2:
			glUniform2f(loc, v.v2(0), v.v2(1));
			break;
		case Uniform::VEC3:
			glUniform3f(loc, v.v3(0), v.v3(1), v.v3(2));
			break;
		case Uniform::VEC4:
			glUniform4f(loc, v.v4(0), v.v4(1), v.v4(2), v.v4(3));
			break;
		case Uniform::MAT4:
			glUniformMatrix4fv(loc, 1, false, v.m4.data());
			break;
		default:
			assert(false && "Uniform type not implemented!");
			break;
		}
	}
};

static inline std::shared_ptr<Shader> loadProgramGeomFromFile(const std::string& vertex_shader_file, const std::string& geometry_shader_file)
{
	std::shared_ptr<Shader> program = std::make_shared<Shader>();

	program->AddShaderFromFile(pangolin::GlSlVertexShader, Parse::get().shaderDir() + "/" + vertex_shader_file, {}, { Parse::get().shaderDir() });
	program->AddShaderFromFile(pangolin::GlSlGeometryShader, Parse::get().shaderDir() + "/" + geometry_shader_file, {}, { Parse::get().shaderDir() });
	program->Link();

	xCheckGlDieOnError();

	return program;
}

static inline std::shared_ptr<Shader> loadProgramFromFile(const std::string& vertex_shader_file)
{
	std::shared_ptr<Shader> program = std::make_shared<Shader>();

	program->AddShaderFromFile(pangolin::GlSlVertexShader, Parse::get().shaderDir() + "/" + vertex_shader_file, {}, { Parse::get().shaderDir() });
	program->Link();

	xCheckGlDieOnError();

	return program;
}

static inline std::shared_ptr<Shader> loadProgramFromFile(const std::string& vertex_shader_file, const std::string& fragment_shader_file)
{
	std::shared_ptr<Shader> program = std::make_shared<Shader>();
	xCheckGlDieOnError();
	program->AddShaderFromFile(pangolin::GlSlVertexShader, Parse::get().shaderDir() + "/" + vertex_shader_file, {}, { Parse::get().shaderDir() });
	xCheckGlDieOnError();
	program->AddShaderFromFile(pangolin::GlSlFragmentShader, Parse::get().shaderDir() + "/" + fragment_shader_file, {}, { Parse::get().shaderDir() });
	xCheckGlDieOnError();
	program->Link();
	xCheckGlDieOnError();

	return program;
}

static inline std::shared_ptr<Shader> loadProgramFromFile(const std::string& vertex_shader_file, const std::string& fragment_shader_file, const std::string& geometry_shader_file)
{
	std::shared_ptr<Shader> program = std::make_shared<Shader>();

	program->AddShaderFromFile(pangolin::GlSlVertexShader, Parse::get().shaderDir() + "/" + vertex_shader_file, {}, { Parse::get().shaderDir() });
	program->AddShaderFromFile(pangolin::GlSlGeometryShader, Parse::get().shaderDir() + "/" + geometry_shader_file, {}, { Parse::get().shaderDir() });
	program->AddShaderFromFile(pangolin::GlSlFragmentShader, Parse::get().shaderDir() + "/" + fragment_shader_file, {}, { Parse::get().shaderDir() });

	program->Link();
	xCheckGlDieOnError();

	return program;
}

#endif /* SHADERS_SHADERS_H_ */
