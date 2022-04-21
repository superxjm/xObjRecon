#version 430 core

layout (location = 0) in vec4 posConf;
layout (location = 1) in vec4 colorTime;
layout (location = 2) in vec4 normalRad;   

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

out vec3 color;
out vec3 normal;

vec3 decodeColor(float c)
{
    vec3 col;
    col.x = float(int(c) >> 16 & 0xFF) / 255.0f;
    col.y = float(int(c) >> 8 & 0xFF) / 255.0f;
    col.z = float(int(c) & 0xFF) / 255.0f;
    return col;
}

void main(void)
{
    gl_Position = projectionMatrix * modelViewMatrix * vec4(posConf.xyz, 1.0);
	 
	color = decodeColor(colorTime.x);
    normal = normalRad.xyz;
}