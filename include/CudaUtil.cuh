#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "CudaVector.cuh"
#include "CudaColor.cuh"
#include "CudaRay.cuh"
#include "CudaPrimitive.cuh"

#define MAX_PATH_DEPTH (1)
#define MAX_BOUNCE (8)
#define RUSSIAN_ROULETTE_BOUNCE (4)
#define PROB_STOP_BOUNCE (0.7f)
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
    const Color Black(0.f, 0.f, 0.f);
    if (Depth >= MAX_BOUNCE) return Color(0.f, 0.f, 0.f);

    Color color = Black;

    HitResult hitResult;
    if (RayCast(ray, spheres, triangles, Ns, Nt, hitResult))
    {
        color = hitResult.mat.emittance;

        vec3 reflectedRay;
        float prob = SampleHemisphere(s, reflectedRay, hitResult.normal);
        Ray outRay(hitResult.p + hitResult.normal * 0.001, reflectedRay);
        float cosW = dot(hitResult.normal, reflectedRay);
        cosW = (cosW > 0.f) ? cosW : 0.f;
        Color irradiance = INV_PI * hitResult.mat.albedo * GetColor(outRay, spheres, triangles, Ns, Nt, Depth + 1,s) * cosW / prob;
        color += irradiance;
    }
    return color;
}
__device__ Color GetColor_iter(const Ray& ray, Sphere* spheres, Triangle* triangles, int Ns, int Nt, int Depth, curandState* s)
{
    const Color Black(0.f, 0.f, 0.f);

    Color color = Black;
    Color reflections[MAX_BOUNCE];
    Color emittances[MAX_BOUNCE];


    HitResult hitResult;
    Ray currentRay = ray;
    for (; Depth < MAX_BOUNCE; Depth++)
    {
        if (RayCast(currentRay, spheres, triangles, Ns, Nt, hitResult))
        {
            emittances[Depth] = hitResult.mat.emittance;

            vec3 reflectedRay;
            float prob = SampleHemisphere(s, reflectedRay, hitResult.normal);
            currentRay = Ray(hitResult.p + hitResult.normal * 0.001, reflectedRay);
            float cosW = dot(hitResult.normal, reflectedRay);
            cosW = (cosW > 0.f) ? cosW : 0.f;

            reflections[Depth] = INV_PI * hitResult.mat.albedo * cosW / prob;

            if (Depth >= RUSSIAN_ROULETTE_BOUNCE)
            {
                float probToStop = MaxFrom(reflections[Depth-1]);
                probToStop = (probToStop < PROB_STOP_BOUNCE)?   probToStop : PROB_STOP_BOUNCE;
                float sample = curand_uniform(s);
                if (sample < probToStop)
                {
                    reflections[Depth] *= (1.f / probToStop);
                }
                else
                {
                    Depth += 1;
                    break;
                }
            }
        }
        else
        {
            reflections[Depth] = Black;
            emittances[Depth] =  Black;
            Depth+=1;
            break;
        }
    }
    Depth -= 1;

    for (; Depth >= 0; --Depth)
    {
        color = emittances[Depth] + color * reflections[Depth];
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