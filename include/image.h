#pragma once

class Image
{
public:
	Image(const char* path);

	inline unsigned char* GetData() noexcept { return data; }
	inline int GetWidth() noexcept { return width; }
	inline int GetHeight() noexcept { return height; }
	inline int GetNrChannels() noexcept { return nrChannels; }
private:
	unsigned char* data = nullptr;
	int width;
	int height;
	int nrChannels;
public:
	Image() = delete;
	~Image();
};