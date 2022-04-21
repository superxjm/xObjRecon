#version 430 core

layout (location = 0) in vec3 pos; 

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

void main(void)
{
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos.xyz, 1.0);
}