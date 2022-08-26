#include "pathtracer.cuh"

#include "image.h"

#include "camera.h"

#include <glm/glm.hpp>

#include "CudaUtil.cuh"
#include "bvh.h"

#include <iostream>
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

// CUDA include
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__constant__ vec3 CameraPos;
__constant__ vec3 CameraForward;
__constant__ vec3 CameraRight;
__constant__ vec3 CameraUp;
__device__ float CameraFovX;
__device__ float CameraFovY;

__constant__ int2 ScreenSize;

__device__ vec3 GetPixelDirection(int px, int py, curandState *s)
{
	vec3 OffsetRight = 2.f*(((float)px + curand_uniform(s)) / (float)(ScreenSize.x - 1) - 0.5f) * tan(CameraFovX * 0.5f) * CameraRight;
	vec3 OffsetUp = -2.f*(((float)py + curand_uniform(s)) / (float)(ScreenSize.y - 1) - 0.5f) * tan(CameraFovY * 0.5f) * CameraUp;
	vec3 direction = Normalize(CameraForward + OffsetRight + OffsetUp);

	return direction;
}
extern __shared__ float array[];
__global__ void StartRender(int W, int H, Color* image, Triangle* triangles, int Nt, CudaBVHNode* BVHTree, int Nb, Triangle* lights, int Nl, int SampleIDX)
{
	/*
	Triangle* blockTriangles = new Triangle[Nt];// (Triangle*)array;
	Triangle* blocklights = new Triangle[Nl];// (Triangle*)&blockTriangles[Nt];
	CudaBVHNode* blockBVHNode = new CudaBVHNode[Nb];// (CudaBVHNode*)&blocklights[Nl];
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < Nt; i++)
		{
			blockTriangles[i].Copy(triangles[i]);
		}
		for (int i = 0; i < Nl; i++)
		{
			blocklights[i].Copy(lights[i]);
		}
		for (int i = 0; i < Nb; i++)
		{
			blockBVHNode[i] = BVHTree[i];
		}
	}
	__syncthreads();
	*/

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	curandState s;
	for (int offset = index; offset < (W*H); offset += (stride))
	{
		curand_init(offset + SampleIDX*W*H, 0, 0, &s);
		int py = offset / W;
		int px = offset % W;
		vec3 direction = GetPixelDirection(px, py, &s);

		Color pixelColor(0.f,0.f,0.f);
		for (int i = 0; i < NUM_SAMPLE; i++)
		{
			pixelColor += GetColor_iter(Ray(CameraPos, direction), triangles, Nt, BVHTree, Nb, lights, Nl, 0, &s);
		}
		image[offset] += pixelColor / (float)(NUM_SAMPLE);
	}
}

__global__ void ClearImage(Color* rawImage, int N)
{
	if (threadIdx.x == 0 && blockIdx.x)
	{
		for (int i = 0; i < N; i++)
			rawImage[i] = Color(0.f, 0.f, 0.f);
	}
}

void exportImage(Image& img, const Color* rawData, const char* path, int H, int W, int SampleCnt)
{
	// write to image
	Pixel* pixels = img.GetData();
	Color pixelColor;
	int cursor = 0;

	for (int i = 0; i < (H * W); i++)
	{
		pixelColor = rawData[i];
		pixelColor /= (float)SampleCnt;
		pixelColor = ACESFilm(pixelColor);
		pixels[cursor] = ConverToUint8(pixelColor.r());
		cursor++;
		pixels[cursor] = ConverToUint8(pixelColor.g());
		cursor++;
		pixels[cursor] = ConverToUint8(pixelColor.b());
		cursor++;
	}

	if (img.WriteTo(path))
	{
		std::cout << "Export Success" << std::endl;
	}
	else
	{
		std::cout << "Export failed" << std::endl;
	}
}

void PathTracer::Render(Camera& camera, BVH* bvh)
{
	int W, H, C;
	std::cout << "Camera : " << camera.Screen_W << " x " << camera.Screen_H << std::endl;
	W = camera.Screen_W;
	H = camera.Screen_H;
	C = 3;
	Image img(W,H,C);
		
	LoadFromBVH(bvh);

	std::cout << "Tree on GPU Size : " << CudaBVH.size()   << std::endl;
	std::cout << "Prim on GPU Size : " << CudaPrims.size() << std::endl;
	std::cout << "Prim on CPU Size : " << bvh->rootBVH->primCnt << std::endl;
	
	std::cout << "Upload world on GPU" << std::endl;

	
	Triangle* triangles;
	int NumTriangle = CudaPrims.size();

	CudaBVHNode* BVHTree;
	int NumTreeNode = CudaBVH.size();

	Triangle* lightSources;
	int NumLightSource = 0;

	checkCudaErrors(cudaMallocManaged(&triangles, sizeof(Triangle) * NumTriangle));
	checkCudaErrors(cudaMallocManaged(&lightSources, sizeof(Triangle) * NumTriangle));
	checkCudaErrors(cudaMallocManaged(&BVHTree, sizeof(CudaBVHNode) * NumTreeNode));

	InitHittables<Triangle> << <1, 1 >> > (triangles, NumTriangle, 0);
	InitHittables<Triangle> << <1, 1 >> > (lightSources, NumTriangle, 0);
	checkCudaErrors(cudaDeviceSynchronize());

	for (int i = 0; i < NumTriangle; i++)
	{
		triangles[i].Copy(CudaPrims[i]);
		if (CudaPrims[i].mat0.emittance.length() > EPS ||
			CudaPrims[i].mat1.emittance.length() > EPS ||
			CudaPrims[i].mat2.emittance.length() > EPS)
		{
			std::cout << "ADD light" << std::endl;
			lightSources[NumLightSource++].Copy(CudaPrims[i]);
		}
	}
	checkCudaErrors(cudaDeviceSynchronize());

	for (int i = 0; i < NumTreeNode; i++)
	{
		BVHTree[i] = CudaBVH[i];
	}
	checkCudaErrors(cudaDeviceSynchronize());


	// copy camera setting
	std::cout << "Upload camera configuration on GPU" << std::endl;
	glm::vec3 pos = camera.pos;
	glm::vec3 forward = camera.GetForward();
	glm::vec3 right = camera.GetRight();
	glm::vec3 up = camera.GetUp();
	float fovy = glm::radians(camera.fovy);
	float fovx = 2.f * atan2(tan(fovy * 0.5f) * camera.aspect, 1.f);
	std::cout << std::endl << std::endl;

	checkCudaErrors(cudaMemcpyToSymbol(CameraPos,     &pos, sizeof(vec3)));
	checkCudaErrors(cudaMemcpyToSymbol(CameraForward, &forward, sizeof(vec3)));
	checkCudaErrors(cudaMemcpyToSymbol(CameraUp, &up, sizeof(vec3)));
	checkCudaErrors(cudaMemcpyToSymbol(CameraRight,   &right, sizeof(vec3)));
	checkCudaErrors(cudaMemcpyToSymbol(CameraFovY, &fovy, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(CameraFovX, &fovx, sizeof(float)));

	int2 screenSize{ W,H };
	checkCudaErrors(cudaMemcpyToSymbol(ScreenSize, &screenSize, sizeof(int2)));

	Color* rawData;
	cudaMallocManaged(&rawData, sizeof(Color) * W * H);
	ClearImage << <1, 1 >> > (rawData, W * H);


	size_t stackSize;
	checkCudaErrors(cudaThreadSetLimit(cudaLimitStackSize, 1024));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaThreadGetLimit(&stackSize, cudaLimitStackSize));
	checkCudaErrors(cudaDeviceSynchronize());
	std::cout << "Stack Size : " << stackSize << std::endl;
	cudaDeviceSynchronize();

	// find optimal grid & block dim
	int gridDim;
	int blockDim;
	cudaOccupancyMaxPotentialBlockSize(&gridDim, &blockDim, StartRender, 0, W*H);
	checkCudaErrors(cudaDeviceSynchronize());

	gridDim = (W * H - blockDim + 1) / (blockDim);

	std::cout << "GridDim x BlockDim : "<<gridDim<<" x " << blockDim << std::endl;

	auto currentMillisecTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

	for (int i = 0; i < NUM_MULTI_SAMPLE; i++)
	{
		//ImageTest<<<numBlock,threadPerBlock>>>(W, H, rawData);
		StartRender << <gridDim, blockDim >> > (W, H, rawData, triangles, NumTriangle, BVHTree, NumTreeNode, lightSources, NumLightSource, i);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		std::cout << "Sample "<< i <<" : Delta time : " << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - currentMillisecTime << " (ms)" << std::endl;

		exportImage(img, rawData, "temp.png", H, W,i+1);
	}
	auto DeltaMillisecTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - currentMillisecTime;
	std::cout << "Delta time : " << DeltaMillisecTime << " (ms)" << std::endl;


	exportImage(img, rawData, "result.png", H, W, NUM_MULTI_SAMPLE);

	checkCudaErrors(cudaFree(rawData));

	cudaFree(triangles);
	cudaFree(BVHTree);
}

