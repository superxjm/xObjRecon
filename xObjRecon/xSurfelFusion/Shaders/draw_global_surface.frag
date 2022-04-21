/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#version 330 core

in vec3 vColor0;
in vec2 texcoord;
in float radius;
flat in int unstablePoint;

uniform mat4 MV;

out vec4 outFragColor;

void main()
{
	mat4 MV_inv = inverse(MV);
	vec4 direction = -MV_inv[2];

    if(dot(texcoord, texcoord) > 1.0)
        discard;
        
    //FragColor = vec4(vColor0, 1.0f);
    
    if(unstablePoint == 1)
	{
		gl_FragDepth = gl_FragCoord.z + radius;
	}
    else
   	{
   		gl_FragDepth = gl_FragCoord.z;
   	}
	
	vec4 v_VaryingLightDir = direction;//vec4(2.0, 2.0, 2.0, 0.0);
	//vec4 v_VaryingLightDir2 = vec4(-2.0, -2.0, 2.0, 0.0);
	vec4 u_AmbientColor = vec4(0.2, 0.2, 0.2, 1.0);
	vec4 u_DiffuseColor = vec4(0.6, 0.6, 0.6, 1.0);
	vec4 u_SpecularColor = vec4(0.5, 0.5, 0.5, 1.0);
	
	vec4 FragColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	float diff = max(0.0f, dot(vec4(vColor0, 1.0f), v_VaryingLightDir));
	FragColor += 1.2 * diff * u_DiffuseColor;
	FragColor += u_AmbientColor;
	//diff = max(0.0f, dot(vec4(vColor0, 1.0f), v_VaryingLightDir2));
	//FragColor += 0.3 * diff * u_DiffuseColor;

	outFragColor = vec4(FragColor.xyz, 1.0);
	//gl_FragDepth = gl_FragCoord.z + u_depthChange;
	//gl_FragColor = vec4(FragColor.xyz, 1.0);
}
