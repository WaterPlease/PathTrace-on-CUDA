#include "texture.h"

#include "image.h"

void Texture2D::LoadImage(const char* imagePath, GLenum textureWarp = GL_CLAMP_TO_EDGE, GLenum textureFilter = GL_LINEAR, bool bMipmap = false, GLenum textureMinFilter = GL_LINEAR_MIPMAP_LINEAR) noexcept
{
	image = new Image(imagePath);

	glGenTextures(1, &ID);
	glBindTexture(target, ID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, textureWarp);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, textureWarp);
	if (bMipmap)
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, textureMinFilter);
	}
	else
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, textureFilter);
	}
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, textureFilter);

	if (image->GetData())
	{
		glTexImage2D(target, 0, GL_RGB, image->GetWidth(), image->GetHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, image->GetData());
		if (bMipmap)
		{
			glGenerateMipmap(target);
		}
	}
}

Texture2D::~Texture2D() {
	if (image != nullptr)
	{
		delete image;
	}
}
