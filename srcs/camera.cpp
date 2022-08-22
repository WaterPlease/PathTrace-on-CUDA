#include "..\include\camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

Camera::Camera() : forward(InitForward),up(InitUp),rotation(0.f,90.f,0.f), pos(0.f)
{
	fovy = 45.f;
	aspect = 16.f / 9.f;
	near = 1.f;
	far = 5000.f;

	SetRotation(rotation);
}

Camera::Camera(glm::vec3 _pos) : Camera()
{
	pos = _pos;
}

const glm::vec3 Camera::GetForward()
{
	return forward;
}

const glm::vec3 Camera::GetUp()
{
	return up;
}

const glm::vec3 Camera::GetRight()
{
	return glm::normalize(glm::cross(forward, up));
}

const Camera& Camera::AddRotation(glm::vec3 deltaRotation)
{
	return SetRotation(rotation + deltaRotation);
}

const Camera& Camera::SetRotation(glm::vec3 _rotation)
{
	// rotation : roll, pitch, yaw
	_rotation.x = glm::mod(_rotation.x, 360.f);
	_rotation.y = glm::clamp(_rotation.y, 0.f,180.f);
	_rotation.z = glm::mod(_rotation.z, 360.f);

	rotation = _rotation;

	glm::vec3 rotationInRadian = glm::radians(rotation);

	forward = glm::normalize(glm::vec3(
		-sinf(rotationInRadian.y) * sinf(rotationInRadian.z),
		cosf(rotationInRadian.y),
		-sinf(rotationInRadian.y) * cosf(rotationInRadian.z)
	));
	up = glm::normalize(glm::vec3(
		cosf(rotationInRadian.y) * sinf(rotationInRadian.z),
		sinf(rotationInRadian.y),
		cosf(rotationInRadian.y) * cosf(rotationInRadian.z)
	));
	up = glm::normalize(up - glm::dot(forward, up) * glm::normalize(up));

	return (*this);
}

const glm::mat4 Camera::GetViewMat()
{
	glm::mat4 mat = glm::lookAt(
		pos,
		pos + forward,
		up
	);
	return mat;
}

const glm::mat4 Camera::GetProjMat()
{
	glm::mat4 mat = glm::perspective(
		glm::radians(fovy),
		aspect,
		near,
		far
	);
	return mat;
}
