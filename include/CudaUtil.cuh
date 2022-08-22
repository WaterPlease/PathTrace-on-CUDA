#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "CudaVector.cuh"
#include "CudaColor.cuh"
#include "CudaRay.cuh"
#include "CudaPrimitive.cuh"

#define MAX_PATH_DEPTH (1)
#define MAX_BOUNCE (3)
#define NUM_MULTI_SAMPLE (8)
#define NUM_SAMPLE (2048)
#define PI (3.141592f)
#define INV_PI (1.f/3.141592f)

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "'\nmsg : " << cudaGetErrorString(result) << "\n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ float SampleHemisphere(curandState* state, vec3& direction, const vec3& normal)
{
    float prob = 0.5f * INV_PI;

    float x1 = curand_uniform(state);
    float x2 = curand_uniform(state);

    float cosTheta = x1;
    float sinTheta = sqrtf(1.f - x1 * x1);
    float cosPhi = cosf(2.f * PI * x2);
    float sinPhi = sinf(2.f * PI * x2);

    direction.e[0] = cosPhi * sinTheta;
    direction.e[1] = sinPhi * sinTheta;
    direction.e[2] = cosTheta;

    //To world space
    
    if (dot(direction, normal) < 0)
        direction *= -1.f;

    return prob;
}

__device__ void curandInitWithThreadID(curandState* s)
{
    int pid, px, py;
    px = blockDim.x * blockIdx.x + threadIdx.x;
    py = blockDim.y * blockIdx.y + threadIdx.y;
    pid = px + py * blockDim.y * gridDim.y;

    curand_init(pid, pid, 0, s);
}

__device__ bool RayCast(const Ray& ray, Sphere* spheres, Triangle* triangles, int Ns, int Nt, HitResult& hitResult, float t_min = 0.f, float t_max = 999999.f)
{
    HitResult tmpResult;
    bool bHitInWorld = false;
    float closestT = t_max;
    for (int i = 0; i < Ns; i++)
    {
        if (spheres[i].hit(ray, t_min, closestT, tmpResult))
        {
            bHitInWorld = true;
            closestT = tmpResult.t;
            hitResult = tmpResult;
        }
    }
    for (int i = 0; i < Nt; i++)
    {
        if (triangles[i].hit(ray, t_min, closestT, tmpResult))
        {
            bHitInWorld = true;
            closestT = tmpResult.t;
            hitResult = tmpResult;
        }
    }

    return bHitInWorld;
}

__device__ Color GetColor(const Ray& ray, Sphere* spheres, Triangle* triangles, int Ns, int Nt, int Depth, curandState* s)
{
    if (Depth >= MAX_BOUNCE) return Color(0.f, 0.f, 0.f);

    Color color(0.f, 0.f, 0.f);

    HitResult hitResult;
    if (RayCast(ray, spheres, triangles, Ns, Nt, hitResult))
    {
        color = hitResult.mat.emittance;
        //Color irradiance(0.f, 0.f, 0.f);

        //for (int i = 0; i < NUM_SAMPLE; i++)
        //{
            vec3 reflectedRay;
            float prob = SampleHemisphere(s, reflectedRay, hitResult.normal);
            Ray outRay(hitResult.p + hitResult.normal * 0.001, reflectedRay);
            float cosW = dot(hitResult.normal, reflectedRay);
            cosW = (cosW > 0.f) ? cosW : 0.f;
            Color irradiance = INV_PI * hitResult.mat.albedo * GetColor(outRay, spheres, triangles, Ns, Nt, Depth + 1,s) * cosW / prob;
        //}

            color += irradiance;// / (float)NUM_SAMPLE;
    }
    else
    {
        vec3 unit_direction = Normalize(ray.dir);
        float t = 0.5f * (unit_direction.y() + 1.0f);
        color = (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
    }
    return color;
}

__host__ __device__ Color ACESFilm(Color x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}