#pragma once

#include "CudaRay.cuh"
#include <vector>

#define EPS (0.000001f)

struct Material
{
    Color reflectance;
    Color emittance;
    Color albedo;
};

struct HitResult
{
    Ray ray;

    Material mat;

    vec3 normal;
    vec3 p;

    float t;
    float u;
    float v;

    bool bFrontFace;
    __host__ __device__ inline void SetNormal(const vec3& outward_normal) {
        bFrontFace = dot(ray.dir, outward_normal) < 0;
        normal = bFrontFace ? outward_normal : -outward_normal;
    }
};

class Hittable
{
public:
    //Material mat;
    __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, HitResult& hitResult) const = 0;
};

class World
{
public:
    Hittable** hittables;
};

template <class T>
__global__ void InitHittables(T* hittables, int Cnt, int offset=0)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }

    T* hittableWithVtable = new T;
    for (int i = 0; i < (offset+Cnt); i++)
    {
        memcpy(&hittables[i], hittableWithVtable, sizeof(T));
    }
    delete hittableWithVtable;

    return;
}

class Sphere : public Hittable
{
public:
    __device__ Sphere() = default;
    __device__ Sphere(const Sphere& sphere) : center(sphere.center), rad(sphere.rad) {};
    __device__ Sphere(vec3 center, float rad) : center(center), rad(rad) {};

    __host__ void Copy(const Sphere& sphere) { Copy(sphere.center, sphere.rad, sphere.mat); }
    __host__ void Copy(vec3 _center, float _rad, Material _mat) { center = _center; rad = _rad; mat = _mat; }

    __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, HitResult& hitResult) const override;
    vec3 center;
    float rad;
    Material mat;
};

class Triangle : public Hittable
{
public:
   __device__ Triangle() = default;
   __device__ Triangle(const Triangle& tr) { Triangle(V0, V1, V2); }
   __device__ Triangle(vec3 V0, vec3 V1, vec3 V2) : V0(V0), V1(V1), V2(V2)
   {
        E1 = V1 - V0;
        E2 = V2 - V0;

        normal = Normalize(cross(E1,E2));
    }

    __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, HitResult& hitResult) const override;

    __host__ void Copy(const Triangle& tr) { Copy(tr.V0, tr.V1, tr.V2, tr.u0, tr.u1, tr.u2, tr.v0, tr.v1, tr.v2, tr.mat); }

    __host__ void Copy(vec3 _V0, vec3 _V1, vec3 _V2, float _u0, float _u1, float _u2, float _v0, float _v1, float _v2, Material _mat)
    {
        V0 = _V0;
        V1 = _V1;
        V2 = _V2;

        E1 = V1 - V0;
        E2 = V2 - V0;

        normal = Normalize(cross(E1, E2));

        u0 = _u0;
        u1 = _u1;
        u2 = _u2;

        v0 = _v0;
        v1 = _v1;
        v2 = _v2;

        mat = _mat;
    }

    vec3 V0;
    vec3 V1;
    vec3 V2;

    vec3 normal;

    vec3 E1;
    vec3 E2;

    float u0, v0;
    float u1, v1;
    float u2, v2;

    Material mat;
};

bool Sphere::hit(const Ray& ray, float t_min, float t_max, HitResult& hitResult) const
{
    vec3 oc = ray.org - center;
    auto a = ray.dir.squared_length();
    auto half_b = dot(oc, ray.dir);
    auto c = oc.squared_length() - rad * rad;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }
    
    hitResult.ray = ray;
    hitResult.t = root;
    hitResult.p = ray.at(hitResult.t);
    hitResult.SetNormal((hitResult.p - center) / rad);
    hitResult.mat = mat;

    return true;
}

__host__ __device__ bool Triangle::hit(const Ray& ray, float t_min, float t_max, HitResult& hitResult) const
{
    const vec3& D = ray.dir;
    vec3 T = ray .org-V0;
    vec3 P = cross(D, E2);
    vec3 Q = cross(T, E1);
    float det = dot(P, E1);
    if (det < EPS) return false;
    float invDet = 1.f / det;

    hitResult.ray = ray;
    hitResult.t = dot(Q, E2) * invDet;
    if (hitResult.t < EPS ||
        hitResult.t < t_min ||
        hitResult.t > t_max)
    {
        return false;
    }

    float u = dot(P, T);
    if (u < 0.f || u > det) return false;


    float v = dot(Q, D);
    if (v < 0.f || (v+u) > det) return false;

    u *= invDet;
    v *= invDet;

    hitResult.u = (1.f - v - u) * u0 + v * u1 + u * u2;
    hitResult.v = (1.f - v - u) * v0 + v * v1 + u * v2;

    hitResult.p = ray.at(hitResult.t);
    hitResult.SetNormal(normal);
    hitResult.mat = mat;

    return true;
}
