#version 330 core
out vec4 FragColor;
  
in vec2 TexCoord;
in vec3 Normal;

uniform sampler2D texture_basecolor;
uniform sampler2D texture_metallic;
uniform sampler2D texture_roughness;
uniform sampler2D texture_normal;

void main()
{
    // FragColor = texture(texture_basecolor, TexCoord);
    //if (FragColor.w < 3e-1)
    //    discard;
    FragColor = vec4(0.5 * Normal + 0.5,1.0);
}