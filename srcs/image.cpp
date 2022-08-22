#define _CRT_SECURE_NO_WARNINGS

#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


Image::Image(const char* path)
{
	data = stbi_load(path, &width, &height, &nrChannels, 0);
}

Image::Image(int W, int H, int C) : width(W), height(H), nrChannels(C)
{
	data = (unsigned char*)malloc(sizeof(unsigned char) * W * H * C);// new unsigned char[W * H * C];
}

bool Image::WriteTo(const char* path)
{
	return stbi_write_png(path, width, height, nrChannels, data, width * nrChannels);
}

Image::~Image()
{
	stbi_image_free(data);
}
