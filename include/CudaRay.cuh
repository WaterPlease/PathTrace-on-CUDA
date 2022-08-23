#ifndef _CUDA_RAY_
#define _CUDA_RAY_
#include "CudaVector.cuh"

class Ray
{
public:
	__host__ __device__ Ray() : org(), dir() {}
	__host__ __device__ Ray(vec3 _org, vec3 _dir) : org(_org), dir(Normalize(_dir)) {}

	__host__ __device__ inline vec3 at(float t) const { return org + t * dir; }

	vec3 org;
	vec3 dir;
};

#endif