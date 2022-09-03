#ifndef _CUDA_PRIMITIVE_
#define _CUDA_PRIMITIVE_

#include "CudaRay.cuh"
#include "CudaColor.cuh"

#include <cuda_runtime.h>

#include <vector>

#define EPS (0.0001f)

__host__ void LoadFromBVH(class BVHnode* node);

struct Material
{
    Color emittance;
    Color albedo;
    Color specular;
    float opacity;
    float roughness;
    float metallic;
};

struct HitResult
{
    Ray ray;

    Material mat;

    vec3 normal;
    vec3 tangent;
    vec3 bitangent;
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
    __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, HitResult& hitResult) = 0;
};

template <class T>
__global__ void InitHittables(T* hittables, int Cnt, int offset=0)
{
    /*
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }
    */

    T* hittableWithVtable = new T();
    for (int i = 0; i < (offset+Cnt); i++)
    {
        memcpy(&hittables[i], hittableWithVtable, sizeof(T));
    }
    delete hittableWithVtable;

    return;
}

class alignas(8) Triangle : public Hittable
{
public:
    /*
   __host__ __device__ Triangle() = default;
   __host__ __device__ Triangle(const Triangle& tr) { Triangle(V0, V1, V2); }
   __host__ __device__ Triangle(vec3 V0, vec3 V1, vec3 V2) : V0(V0), V1(V1), V2(V2)
   {
        E1 = V1 - V0;
        E2 = V2 - V0;

        normal = Normalize(cross(E1,E2));
    }
   */

   __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, HitResult& hitResult) override
   {
       const vec3& D = ray.dir;
       vec3 T = ray.org - V0;
       vec3 P = cross(D, E2);
       vec3 Q = cross(T, E1);
       float det = dot(P, E1);
//#define _TEST_CULL_
//#ifdef _TEST_CULL_
       ///*
       if (det < EPS) return false;
       float invDet = 1.f / det;

       hitResult.ray = ray;
       hitResult.t = dot(Q, E2) * invDet;
       if (hitResult.t < t_min ||
           hitResult.t > t_max)
       {
           return false;
       }

       float u = dot(P, T);
       if (u < 0.f || u > det) return false;


       float v = dot(Q, D);
       if (v < 0.f || (v + u) > det) return false;

       u *= invDet;
       v *= invDet;
       //*/
//#else
       /*
       if (det < EPS && det > -EPS) return false;
       float invDet = 1.f / det;

       float u = dot(P, T) * invDet;
       if (u < 0.f || u > 1.f) return false;


       float v = dot(Q, D) * invDet;
       if (v < 0.f || (v + u) > 1.f) return false;

       hitResult.ray = ray;
       hitResult.t = dot(Q, E2) * invDet;
       if (hitResult.t < t_min ||
           hitResult.t > t_max)
       {
           return false;
       }
       */
//#endif
       hitResult.SetNormal(Normalize((1.f - v - u) * N0 + v * N1 + u * N2));
       hitResult.tangent = Normalize((1.f - v - u) * T0 + v * T1 + u * T2);
       hitResult.bitangent = Normalize((1.f - v - u) * B0 + v * B1 + u * B2);

       hitResult.u = (1.f - v - u) * u0 + v * u1 + u * u2;
       hitResult.v = (1.f - v - u) * v0 + v * v1 + u * v2;

       hitResult.p = ray.at(hitResult.t);
       hitResult.mat.albedo = mat0.albedo;
       hitResult.mat.emittance = mat0.emittance;
       hitResult.mat.specular = mat0.specular;
       hitResult.mat.metallic = mat0.metallic;
       hitResult.mat.roughness = mat0.roughness;
       hitResult.mat.opacity = mat0.opacity;

       return true;
   }

    __host__ __device__ void Copy(const Triangle& tr)
    {
        Copy(
            tr.V0, tr.V1, tr.V2,
            tr.T0, tr.T1, tr.T2,
            tr.B0, tr.B1, tr.B2,
            tr.N0, tr.N1, tr.N2,
            tr.mat0, tr.mat1, tr.mat2,
            tr.u0, tr.u1, tr.u2,
            tr.v0, tr.v1, tr.v2 );
    }

    __host__ __device__ void Copy(
        vec3 _V0, vec3 _V1, vec3 _V2,
        vec3 _T0, vec3 _T1, vec3 _T2,
        vec3 _B0, vec3 _B1, vec3 _B2,
        vec3 _N0, vec3 _N1, vec3 _N2,
        Material _mat0, Material _mat1, Material _mat2,
        float _u0, float _u1, float _u2,
        float _v0, float _v1, float _v2)
    {
        V0 = _V0;
        V1 = _V1;
        V2 = _V2;

        T0 = _T0;
        T1 = _T1;
        T2 = _T2;

        B0 = _B0;
        B1 = _B1;
        B2 = _B2;

        N0 = _N0;
        N1 = _N1;
        N2 = _N2;

        mat0 = _mat0;
        mat1 = _mat1;
        mat2 = _mat2;

        E1 = V1 - V0;
        E2 = V2 - V0;

        normal = Normalize(cross(E1, E2));
        
        area = cross(E1, E2).length()*0.5f;

        u0 = _u0;
        u1 = _u1;
        u2 = _u2;

        v0 = _v0;
        v1 = _v1;
        v2 = _v2;

    }

    vec3 V0, T0, B0, N0;
    vec3 V1, T1, B1, N1;
    vec3 V2, T2, B2, N2;

    vec3 normal;

    vec3 E1;
    vec3 E2;

    float u0, v0;
    float u1, v1;
    float u2, v2;

    Material mat0;
    Material mat1;
    Material mat2;

    float area;
};

struct alignas(8) CudaBVHNode
{
    vec3 bMin;
    vec3 bMax;

    int childL;
    int childR;

    int primStart;
    int primEnd;
};

class Sphere
{
public:
    __host__ __device__ Sphere() : center(vec3(0.f,0.f,0.f)), rad(0.f), mat() { }
    __host__ __device__ Sphere(float x, float y, float z, float rad, Material mat)
        : center(vec3(x,y,z)), rad(rad), mat(mat) { }
    __device__ bool hit(const Ray& ray, float t_min, float t_max, HitResult& hitResult) const
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
        hitResult.SetNormal((hitResult.p - center)/rad);

        vec3 v1, v2, v3;
        v2 = vec3(0.f, 0.f, 1.f);
        v2 = vec3(0.f, 1.f, 0.f);
        v3 = vec3(1.f, 0.f, 0.f);

        // Solution for getting tangent and bitangent vector from normal vector
        // by Nathan Reed
        // from : https://computergraphics.stackexchange.com/questions/5498/compute-sphere-tangent-for-normal-mapping
        hitResult.tangent = Normalize(cross(vec3(0.f, 1.f, 0.f), hitResult.normal));
        hitResult.bitangent = cross(hitResult.normal, hitResult.tangent);

        hitResult.u = 0.f;
        hitResult.v = 0.f;

        //hitResult.SetNormal(normal);
        hitResult.mat.albedo = mat.albedo;
        hitResult.mat.emittance = mat.emittance;
        //hitResult.mat.emittance = hitResult.normal * 0.5f + 0.5f;
        hitResult.mat.specular = mat.specular;
        hitResult.mat.metallic = mat.metallic;
        hitResult.mat.roughness = mat.roughness;
        hitResult.mat.opacity = mat.opacity;

        return true;
    }


    __host__ __device__ void Copy(const Sphere& sphere)
    {
        Copy(sphere.center, sphere.rad, sphere.mat);
    }

    __host__ __device__ void Copy(
        vec3 _center, float _rad, Material _mat)
    {
        center = _center;
        rad = _rad;

        mat = _mat;
    }
    vec3 center;
    float rad;

    Material mat;
};

extern std::vector<CudaBVHNode> CudaBVH;
extern std::vector<Triangle>    CudaPrims;
extern std::vector<Sphere>    CudaSpheres;

#include "bvh.h"
#include "mesh.h"

inline vec3 ConvertToCudaVec(const glm::vec3& v)
{
    return vec3(v.x, v.y, v.z);
}

__host__ void LoadFromBVH(BVH* bvh);
#endif