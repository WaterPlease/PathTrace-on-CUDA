#version 330 core
out vec4 FragColor;
  
in vec3 cColor;

void main()
{
    FragColor = vec4(cColor,1.0f);
}