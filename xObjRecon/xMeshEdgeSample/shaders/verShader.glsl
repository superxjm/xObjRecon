#version 430 core

in vec3 in_Position;
flat out vec4 pass_Color;

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

void main(void)
{
    gl_Position = projectionMatrix * modelViewMatrix * vec4(in_Position, 1.0);
	 
	pass_Color = vec4(
	 float(gl_VertexID & 0xFF) / 255.0, 
	 float((gl_VertexID >> 8) & 0xFF) / 255.0, 
	 float((gl_VertexID >> 16) & 0xFF) / 255.0, 
	 float((gl_VertexID >> 24) & 0xFF) / 255.0); 
}

