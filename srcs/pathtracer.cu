#include "pathtracer.cuh"

#include "image.h"

#include "camera.h"

#include <glm/glm.hpp>

#include "CudaUtil.cuh"

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
	vec3 OffsetRight =  (((float)px + curand_uniform(s)) / (float)(ScreenSize.x-1) - 0.5f) * tan(CameraFovX * 0.5f) * CameraRight;
	vec3 OffsetUp    = -(((float)py + curand_uniform(s)) / (float)(ScreenSize.y-1) - 0.5f) * tan(CameraFovY * 0.5f) * CameraUp;
	vec3 direction = Normalize(CameraForward + OffsetRight + OffsetUp);

	return direction;
}

__global__ void StartRender(int W, int H, Color* image, Sphere* spheres, Triangle* triangles, int Ns, int Nt, int SampleIDX)
{
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
			pixelColor += GetColor_iter(Ray(CameraPos, direction), spheres, triangles, Ns, Nt, 0, &s);
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

void PathTracer::Render(Camera& camera)
{
	int W, H, C;
	W = 1280;
	H = 720;
	C = 3;
	Image img(W,H,C);

	// load primitives
	Sphere* sphere = new Sphere();
	sphere->center = vec3(0.f, 2.f, 0.0f);
	sphere->rad = 2.f;
	sphere->mat.albedo = Color(0.7f, 0.7f, 0.7f);
	
	// cornell height & width
	float CH = 10.f;
	float CW = 6.f;
	float ALW = 3.0f;
	float offset = -0.01f;

	camera.pos.y = CH * 0.5f;
	
	vec3 V0(-CW, 0, glm::max(CW, camera.pos.z));
	vec3 V1( CW, 0, glm::max(CW, camera.pos.z));
	//vec3 V0(-CW, 0, CW);
	//vec3 V1( CW, 0, CW);
	vec3 V2( CW, 0,-CW);
	vec3 V3(-CW, 0,-CW);

	vec3 V4(-CW,CH, glm::max(CW, camera.pos.z));
	vec3 V5( CW,CH, glm::max(CW, camera.pos.z));
	//vec3 V4(-CW,CH, CW);
	//vec3 V5( CW,CH, CW);
	vec3 V6( CW,CH,-CW);
	vec3 V7(-CW,CH,-CW);

	vec3 V8 (-ALW, CH + offset, ALW);
	vec3 V9 ( ALW, CH + offset, ALW);
	vec3 V10( ALW, CH + offset,-ALW);
	vec3 V11(-ALW, CH + offset,-ALW);

	Material MatGreenWall;
	MatGreenWall.albedo = vec3(0.1f, 1.f, 0.1f);
	MatGreenWall.emittance = vec3(0.f, 0.0f, 0.f);
	Material MatRedWall;
	MatRedWall.albedo = vec3(1.0f, 0.1f, 0.1f);
	MatRedWall.emittance = vec3(0.0f, 0.f, 0.f);
	Material MatWhiteWall;
	MatWhiteWall.albedo = vec3(1.0f, 1.0f, 1.0f);
	MatWhiteWall.emittance = vec3(0.0f, 0.0f, 0.0f);
	Material MatAreaLight;
	MatAreaLight.albedo = vec3(0.3f, 0.3f, 0.3f);
	MatAreaLight.emittance = vec3(5.f, 5.f, 5.f);

	Triangle trs[10+2];

	trs[0].Copy(V3, V7, V4, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatRedWall);
	trs[1].Copy(V4, V0, V3, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatRedWall);

	trs[2].Copy(V5, V6, V2, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatGreenWall);
	trs[3].Copy(V5, V2, V1, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatGreenWall);

	trs[4].Copy(V0, V1, V2, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatWhiteWall);
	trs[5].Copy(V2, V3, V0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatWhiteWall);
	trs[6].Copy(V2, V6, V7, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatWhiteWall);
	trs[7].Copy(V2, V7, V3, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatWhiteWall);
	trs[8].Copy(V4, V7, V6, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatWhiteWall);
	trs[9].Copy(V4, V6, V5, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatWhiteWall);

	trs[10].Copy(V8, V11, V10, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatAreaLight);
	trs[11].Copy(V8, V10, V9 , 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, MatAreaLight);

	std::cout << "Upload world on GPU" << std::endl;

	Sphere* spheres;
	Triangle* triangles;
	checkCudaErrors(cudaMallocManaged(&spheres,     sizeof(Sphere) * 1));
	checkCudaErrors(cudaMallocManaged(&triangles, sizeof(Triangle) * 12));

	InitHittables<Sphere> << <1, 1 >> >   (spheres,1,0);
	InitHittables<Triangle> << <1, 1 >> > (triangles, 12, 0);
	checkCudaErrors(cudaDeviceSynchronize());

	spheres[0].Copy(*sphere);
	for (int i = 0; i < 12; i++)
		triangles[i].Copy(trs[i]);
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
	checkCudaErrors(cudaThreadSetLimit(cudaLimitStackSize, 1024 * 128));
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
		StartRender << <gridDim, blockDim >> > (W, H, rawData, spheres, triangles,1,12, i);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		std::cout << "Sample "<< i <<" : Delta time : " << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - currentMillisecTime << " (ms)" << std::endl;
	}
	auto DeltaMillisecTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - currentMillisecTime;
	std::cout << "Delta time : " << DeltaMillisecTime << " (ms)" << std::endl;


	// write to image
	Pixel* pixels = img.GetData();

	int cursor = 0;

	for (int i = 0; i < (H * W); i++)
	{
		rawData[i] /= (float)NUM_MULTI_SAMPLE;
		rawData[i] = ACESFilm(rawData[i]);
		pixels[cursor] = ConverToUint8(rawData[i].r());
		cursor++;
		pixels[cursor] = ConverToUint8(rawData[i].g());
		cursor++;
		pixels[cursor] = ConverToUint8(rawData[i].b());
		cursor++;
	}

	checkCudaErrors(cudaFree(rawData));
	if (img.WriteTo("test.png"))
	{
		std::cout << "Export Success" << std::endl;
	}
	else
	{
		std::cout << "Export failed" << std::endl;
	}
}