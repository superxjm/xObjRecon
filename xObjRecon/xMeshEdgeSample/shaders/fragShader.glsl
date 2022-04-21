#version 430 core

flat in vec4 pass_Color;

out vec4 out_Color;

void main(void) {
	
	/*
	vec4 color = vec4(
	 float(gl_PrimitiveID & 0xFF) / 255.0, 
	 float((gl_PrimitiveID >> 8) & 0xFF) / 255.0, 
	 float((gl_PrimitiveID >> 16) & 0xFF) / 255.0, 
	 float((gl_PrimitiveID >> 24) & 0xFF) / 255.0);
	
    out_Color = color;
	*/
	
	//out_Color = pass_Color;
	
	out_Color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}