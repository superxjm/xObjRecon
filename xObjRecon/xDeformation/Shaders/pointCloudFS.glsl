#version 430 core

in vec4 pos;
in vec3 color;
in vec3 normal;																			

out vec4 FragColor;

void main(void) 
{
	FragColor = vec4(color.xyz, 1.0);
}

