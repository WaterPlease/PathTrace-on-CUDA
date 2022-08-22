#include "renderer.h"

#include <iostream>

#include "camera.h"
#include "shader.h"
#include "texture.h"
#include "model.h"
#include "bvh.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>

Renderer::Renderer()
{
    /* Initialize the library */
    if (glfwInit() == GLFW_FALSE)
        abort();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    camera = new Camera();
    camera->SetRotation(glm::vec3(0.f,90.f,0.f));
    camera->pos = glm::vec3(0.f, 10.f, 40.0f);
    cameraMoveSpeed = 100.f;
    cameraRotSpeed = 0.1f;

    deltaTime = 0.f;

    bLMBPressed = false;

    renderMode = RENDER_MODE::RM_DEFAULT;
    bShowBVH = false;
}

const glm::uvec2& Renderer::GetScreenSize()
{
    return screenSize;
}

void Renderer::SetScreenSize(const glm::uvec2& size)
{
    screenSize = size;
    camera->aspect = (float)screenSize.x / (float)screenSize.y;
}

void Renderer::CreateWindow()
{

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(screenSize.x, screenSize.y, "Hello World", NULL, NULL);
    if (window == nullptr)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        abort();
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        abort();
    }

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

}

void Renderer::Run()
{
    Shader* shader = new Shader("C:\\Users\\kwonh\\Desktop\\study\\Graphics\\PathTrace_GPGPU\\resources\\vertex.glsl", "C:\\Users\\kwonh\\Desktop\\study\\Graphics\\PathTrace_GPGPU\\resources\\fragment.glsl");
    Shader* shader_cluster = new Shader("C:\\Users\\kwonh\\Desktop\\study\\Graphics\\PathTrace_GPGPU\\resources\\vertex_cluster.glsl", "C:\\Users\\kwonh\\Desktop\\study\\Graphics\\PathTrace_GPGPU\\resources\\fragment_cluster.glsl");

    /* Loop until the user closes the window */

    std::vector<Model*> Models;

    /*
    Model* model_gun = new Model("C:\\Users\\kwonh\\Desktop\\study\\Graphics\\PathTrace_GPGPU\\resources\\models\\weapon_case_final.obj");
    model_gun->translation = glm::vec3(0.f, 0.f, 0.f);
    model_gun->rotation = glm::vec4(0.f);
    model_gun->scale = 1.0f;
    Models.push_back(model_gun);
    */

    Model* sponza = new Model("C:\\Users\\kwonh\\Desktop\\study\\Graphics\\PathTrace_GPGPU\\resources\\models\\sponza\\Sponza.gltf");
    sponza->translation = glm::vec3(0.f, 0.f, 0.f);
    sponza->rotation = glm::vec4(0.f);
    sponza->scale = 1.0f;
    Models.push_back(sponza);

    Model* SciFiHelmet = new Model("C:\\Users\\kwonh\\Desktop\\study\\Graphics\\PathTrace_GPGPU\\resources\\models\\\helmet\\SciFiHelmet.gltf");
    SciFiHelmet->translation = glm::vec3(0.f, 10.f, -30.f);
    SciFiHelmet->rotation = glm::vec4(0.f);
    SciFiHelmet->scale = 10.0f;
    Models.push_back(SciFiHelmet);


    BVH bvh;
    for (auto model : Models)
    {
        bvh.AddModel(model);
    }
    std::cout << "Build BVH" << std::endl;
    bvh.rootCluster = new Cluster();
    std::cout << "insert primitives into root cluster" << std::endl;
    int BVHN = bvh.primitives.size();
    for (int i = 0; i < BVHN; i++)
    {
        bvh.rootCluster->primitives.push_back(i);
    }
    std::cout << "RUN K-Means" << std::endl;
    //bvh.Cluster_K_MEAS(bvh.rootCluster);
    bvh.Cluster_K_MEAS_Recursive(bvh.rootCluster,5);
    std::cout << "Build ClusterMesh" << std::endl;
    bvh.GenClusterMesh(bvh.rootCluster);
    std::cout << "Build BVH tree" << std::endl;
    bvh.rootBVH = bvh.GenBVHTree(bvh.rootCluster);
    std::cout << "Pre process done" << std::endl;

    double currentTime = glfwGetTime();
    double lastTime = currentTime;
    while (!glfwWindowShouldClose(window))
    {
        currentTime = glfwGetTime();
        deltaTime = (float)(currentTime - lastTime);
        lastTime = currentTime;

        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        switch (renderMode)
        {
        case RENDER_MODE::RM_DEFAULT:
            
            shader->use();

            shader->use();
            shader->setMat4("view", camera->GetViewMat());
            shader->setMat4("projection", camera->GetProjMat());

            for (auto model : Models)
            {
                model->Draw(shader);
            }
            break;
        case RENDER_MODE::RM_MESH_CLUSTER:
            shader_cluster->use();
            shader_cluster->setMat4("view", camera->GetViewMat());
            shader_cluster->setMat4("projection", camera->GetProjMat());

            bvh.drawCluster(shader_cluster, bvh.rootCluster, 0);
            break;
        default:
            break;
        }

        if (bShowBVH)
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDisable(GL_CULL_FACE);

            bvh.BoxShader->use();
            bvh.BoxShader->setMat4("view", camera->GetViewMat());
            bvh.BoxShader->setMat4("projection", camera->GetProjMat());
            bvh.drawBVHTree(bvh.rootBVH);

            glEnable(GL_CULL_FACE);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
}

void processInput(GLFWwindow* window)
{
    Renderer& renderer = Renderer::GetInstance();
    
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        renderer.camera->pos += renderer.camera->GetForward() * renderer.cameraMoveSpeed * renderer.deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        renderer.camera->pos -= renderer.camera->GetForward() * renderer.cameraMoveSpeed * renderer.deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        renderer.camera->pos += renderer.camera->GetRight() * renderer.cameraMoveSpeed * renderer.deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        renderer.camera->pos -= renderer.camera->GetRight() * renderer.cameraMoveSpeed * renderer.deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        renderer.camera->pos += glm::vec3(0.0f,1.0f,0.0f) * renderer.cameraMoveSpeed * renderer.deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        renderer.camera->pos -= glm::vec3(0.0f, 1.0f, 0.0f) * renderer.cameraMoveSpeed * renderer.deltaTime;
    }


    if (glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS)
    {
        renderer.renderMode = RENDER_MODE::RM_DEFAULT;
    }
    if (glfwGetKey(window, GLFW_KEY_F2) == GLFW_PRESS)
    {
        renderer.renderMode = RENDER_MODE::RM_MESH_CLUSTER;
    }

    static bool ignoreKeyB = false;
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS && !ignoreKeyB)
    {
        renderer.bShowBVH = !renderer.bShowBVH;
        ignoreKeyB = true;
    }
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_RELEASE)
    {
        ignoreKeyB = false;
    }


    if (renderer.bLMBPressed)
    {
        double x, y;
        glfwGetCursorPos(window, &x, &y);

        glm::vec2 currentMousePos(x, y);

        glm::vec2 deltaRotation = currentMousePos - renderer.prevMousePos;
        renderer.prevMousePos = currentMousePos;

        deltaRotation = deltaRotation * renderer.cameraRotSpeed;
        
        if (glm::length(deltaRotation) > std::numeric_limits<float>::epsilon())
        {
            renderer.camera->AddRotation(glm::vec3(0.f, deltaRotation.y, -deltaRotation.x));
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            Renderer::GetInstance().bLMBPressed = true;

            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            Renderer::GetInstance().prevMousePos = glm::vec2(xpos, ypos);
        }
        else
        {
            Renderer::GetInstance().bLMBPressed = false;
        }
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    Renderer::GetInstance().SetScreenSize(glm::uvec2(width, height));
    glViewport(0, 0, width, height);
}