#version 430 core

layout (location = 0) in vec4 posConf; 

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

void main(void)
{
    gl_Position = projectionMatrix * modelViewMatrix * vec4(posConf.xyz, 1.0);
}