#pragma once

#include <glad/glad.h>
#include <string>

class Texture2D
{
public:
	void LoadImage(const char* imagePath, GLenum textureWarp, GLenum textureFilter, bool bMipmap, GLenum textureMinFilter) noexcept;

	class Image* image;

	inline GLint GetID() noexcept { return ID; }
	inline void SetID(GLuint _ID) noexcept { ID = _ID; }
	inline GLenum GetTarget() noexcept { return target; }
private:
	GLuint ID = 0;
	GLenum target = GL_TEXTURE_2D;
public:
	~Texture2D();
};

class Texture : public Texture2D
{
public:
	std::string type;
	std::string path;
};