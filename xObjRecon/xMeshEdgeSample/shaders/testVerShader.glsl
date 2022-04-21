#version 330 core

layout (location = 0) in vec3 position;

uniform mat4 MV;
uniform mat4 P;

void main()
{
gl_Position = P * MV * vec4(position.x, position.y, position.z, 1.0);
}