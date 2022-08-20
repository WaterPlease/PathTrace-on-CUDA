#version 330 core
layout(location = 0) in vec3 aPos;

out vec3 oColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    oColor = vec3(1.0f, 0.0f, 0.0f);
}