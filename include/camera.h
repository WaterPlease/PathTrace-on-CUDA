#pragma once

#include <glm/glm.hpp>

static const glm::vec3 InitForward(0.0f, 0.0f, -1.0f);
static const glm::vec3 InitUp(0.0f, 1.0f, 0.0f);
static const glm::vec3 InitRight(1.0f, 0.0f, 0.0f);

class Camera
{
public:
	Camera();
	Camera(glm::vec3 _pos);

	glm::vec3 pos;

	unsigned int Screen_W;
	unsigned int Screen_H;

	float fovy;
	float aspect;
	float near;
	float far;

	const glm::vec3 GetForward();
	const glm::vec3 GetUp();
	const glm::vec3 GetRight();

	const Camera& AddRotation(glm::vec3 deltaRotation);
	const Camera& SetRotation(glm::vec3 _rotation);

	const glm::mat4 GetViewMat();
	const glm::mat4 GetProjMat();

private:
	/** rotation = glm::vec3(roll,pitch,yaw) */
	glm::vec3 rotation;

	glm::vec3 forward;
	glm::vec3 up;
};