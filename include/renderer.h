#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "bvh.h"

enum class RENDER_MODE
{
	RM_DEFAULT,
	RM_MESH_CLUSTER,
};

class Renderer
{
public:
	static Renderer& GetInstance()
	{
		static Renderer renderer;

		return renderer;
	}

	GLFWwindow* window;
	class Camera* camera;
	//BVH bvh;
	SAHBVH bvh;

	float deltaTime;
	float cameraMoveSpeed;
	float cameraRotSpeed;

	RENDER_MODE renderMode;

	bool bLMBPressed;
	bool bShowBVH;
	glm::vec2 prevMousePos;

	const glm::uvec2& GetScreenSize();
	void SetScreenSize(const glm::uvec2& size);

	void CreateWindow();
	void Run();
private:
	Renderer();
	Renderer(const Renderer& ref) = delete;
	Renderer& operator=(const Renderer& ref) = delete;
	~Renderer() {};
	glm::uvec2 screenSize;
};



void processInput(GLFWwindow* window);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);