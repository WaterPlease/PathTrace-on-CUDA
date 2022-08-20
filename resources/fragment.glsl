#version 330 core
out vec4 FragColor;
  
in vec2 TexCoord;

uniform sampler2D texture_basecolor;
uniform sampler2D texture_metallic;
uniform sampler2D texture_roughness;
uniform sampler2D texture_normal;

void main()
{
    FragColor = texture(texture_basecolor, TexCoord);
}