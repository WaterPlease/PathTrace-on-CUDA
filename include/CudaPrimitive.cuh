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
    float metallic;
    float roughness;
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

class Triangle : public Hittable
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
       if (v < 0.f || (v + u) > det) return false;

       u *= invDet;
       v *= invDet;

       hitResult.u = (1.f - v - u) * u0 + v * u1 + u * u2;
       hitResult.v = (1.f - v - u) * v0 + v * v1 + u * v2;

       hitResult.p = ray.at(hitResult.t);
       //hitResult.SetNormal(normal);
       hitResult.SetNormal((1.f - v - u) * N0 + v * N1 + u * N2);
       hitResult.tangent = (1.f - v - u) * T0 + v * T1 + u * T2;
       hitResult.bitangent=(1.f - v - u) * B0 + v * B1 + u * B2;
       hitResult.mat.albedo = mat0.albedo;
       hitResult.mat.emittance = mat0.emittance;

       return true;
   }

    __host__ void Copy(const Triangle& tr)
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

    __host__ void Copy(
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

struct CudaBVHNode
{
    vec3 bMin;
    vec3 bMax;

    int parent;
    int childL;
    int childR;

    int primStart;
    int primEnd;
};
std::vector<CudaBVHNode> CudaBVH;
std::vector<Triangle>    CudaPrims;


#include "bvh.h"
#include "mesh.h"

inline vec3 ConvertToCudaVec(const glm::vec3& v)
{
    return vec3(v.x, v.y, v.z);
}

__host__ void LoadFromBVH(BVH* bvh)
{
    CudaBVH.clear();
    CudaPrims.clear();

    BVHNode* root = bvh->rootBVH;
    std::vector<BVHNode*> stack;
    stack.push_back(root);
    stack.push_back(root);
    stack.push_back((BVHNode*)0);
    stack.push_back((BVHNode*)0);

    int maxDepth = -1;

    while (!stack.empty())
    {
        int depth = (int)stack.back();
        stack.pop_back();
        int parentIdx = (int)stack.back();
        stack.pop_back();
        BVHNode* parent = stack.back();
        stack.pop_back();
        BVHNode* node = stack.back();
        stack.pop_back();

        int idx = CudaBVH.size();

        if (maxDepth < depth)
        {
            maxDepth = depth;
        }

        CudaBVHNode cudaNode;
        cudaNode.bMax = ConvertToCudaVec(node->bMax);
        cudaNode.bMin = ConvertToCudaVec(node->bMin);
        cudaNode.childL = cudaNode.childR = -1;
        cudaNode.primStart = -1;
        cudaNode.primEnd   = -1;

        if (node->ptr_primitives != nullptr)
        {
            std::vector<unsigned int>& primInBB = *(node->ptr_primitives);
            cudaNode.primStart = CudaPrims.size();
            cudaNode.primEnd = CudaPrims.size() + primInBB.size() - 1;
            for (auto primID : primInBB)
            {
                Primitive prim = bvh->primitives[primID];
                Triangle tr;

                tr.V0 = ConvertToCudaVec(prim.v1.Position);
                tr.V1 = ConvertToCudaVec(prim.v2.Position);
                tr.V2 = ConvertToCudaVec(prim.v3.Position);

                tr.E1 = tr.V1 - tr.V0;
                tr.E2 = tr.V2 - tr.V0;

                tr.normal = Normalize(cross(tr.E1, tr.E2));

                tr.T0 = Normalize(ConvertToCudaVec(prim.v1.Tangent));
                tr.T1 = Normalize(ConvertToCudaVec(prim.v2.Tangent));
                tr.T2 = Normalize(ConvertToCudaVec(prim.v3.Tangent));

                tr.B0 = Normalize(ConvertToCudaVec(prim.v1.Bitangent));
                tr.B1 = Normalize(ConvertToCudaVec(prim.v2.Bitangent));
                tr.B2 = Normalize(ConvertToCudaVec(prim.v3.Bitangent));

                tr.N0 = Normalize(ConvertToCudaVec(prim.v1.Normal));
                tr.N1 = Normalize(ConvertToCudaVec(prim.v2.Normal));
                tr.N2 = Normalize(ConvertToCudaVec(prim.v3.Normal));

                tr.u0 = prim.v1.u;
                tr.v0 = prim.v1.v;

                tr.u1 = prim.v2.u;
                tr.v1 = prim.v2.v;

                tr.u2 = prim.v3.u;
                tr.v2 = prim.v3.v;

                tr.mat0.albedo = ConvertToCudaVec(prim.v1.mat.albedo);
                tr.mat0.emittance = ConvertToCudaVec(prim.v1.mat.emittance);
                tr.mat0.metallic = prim.v1.mat.metallic;
                tr.mat0.roughness = prim.v1.mat.roughness;

                tr.mat1.albedo = ConvertToCudaVec(prim.v2.mat.albedo);
                tr.mat1.emittance = ConvertToCudaVec(prim.v2.mat.emittance);
                tr.mat1.metallic = prim.v2.mat.metallic;
                tr.mat1.roughness = prim.v2.mat.roughness;

                tr.mat2.albedo = ConvertToCudaVec(prim.v3.mat.albedo);
                tr.mat2.emittance = ConvertToCudaVec(prim.v3.mat.emittance);
                tr.mat2.metallic = prim.v3.mat.metallic;
                tr.mat2.roughness = prim.v3.mat.roughness;

                CudaPrims.push_back(tr);
            }
        }

        if (idx != parentIdx)
        {
            CudaBVHNode& pNode = CudaBVH[parentIdx];
            if (pNode.childL == -1)
            {
                pNode.childL = idx;
            }
            else if (pNode.childR == -1)
            {
                pNode.childR = idx;
            }
        }

        CudaBVH.push_back(cudaNode);


        if (node->Child[0] != nullptr)
        {
            stack.push_back(node->Child[0]);
            stack.push_back(node);
            stack.push_back((BVHNode*)idx);
            stack.push_back((BVHNode*)(depth+1));
        }
        if (node->Child[1] != nullptr)
        {
            stack.push_back(node->Child[1]);
            stack.push_back(node);
            stack.push_back((BVHNode*)idx);
            stack.push_back((BVHNode*)(depth + 1));
        }
    }

    std::cout << "Maximum depth of tree : " << maxDepth << std::endl;
}

#endif