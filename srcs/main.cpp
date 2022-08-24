#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/common.hpp>

#include <iostream>

#include "shader.h"
#include "texture.h"

#include "renderer.h"

#include "pathtracer.cuh"

const unsigned int SCR_WIDTH =  512;
const unsigned int SCR_HEIGHT = 512;

void GLAPIENTRY
MessageCallback(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam)
{
    fprintf(stdout, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
        (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
        type, severity, message);
}

int main(void)
{

    Renderer& renderer = Renderer::GetInstance();
    renderer.SetScreenSize(glm::uvec2(SCR_WIDTH, SCR_HEIGHT));
    renderer.CreateWindow();


    // During init, enable debug output
    //glEnable(GL_DEBUG_OUTPUT);
    //glDebugMessageCallback(MessageCallback, 0);
    

    
    renderer.Run();
    
    return 0;
}