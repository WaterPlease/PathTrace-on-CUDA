#ifndef _CUDA_UTIL_
#define _CUDA_UTIL_

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "CudaVector.cuh"
#include "CudaColor.cuh"
#include "CudaRay.cuh"
#include "CudaPrimitive.cuh"

#include "Bxdf.cuh"

#define MAX_BOUNCE (8)
#define RUSSIAN_ROULETTE_BOUNCE (3)
#define PROB_STOP_BOUNCE (0.5f)
#define NUM_MULTI_SAMPLE (8)
#define NUM_SAMPLE (1024)
#define PI (3.141592f)
#define INV_PI (1.f/3.141592f)

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#define CMP(x, y) \
	(fabsf(x - y) <= FLT_EPSILON * cudamax(1.0f, cudamax(fabsf(x), fabsf(y))))

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "'\nmsg : " << cudaGetErrorString(result) << "\n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ float SamplePrimitive(curandState* state, vec3& point, const Triangle& prim)
{
    float prob;

    float r1 = sqrtf(curand_uniform(state));
    float r2 = curand_uniform(state);

    point = (1 - r1) * prim.V0 + r1 * (1 - r2) * prim.V1 + r1 * r2 * prim.V2;

    return 1.f / (prim.area);
}

__device__ void curandInitWithThreadID(curandState* s)
{
    int pid, px, py;
    px = blockDim.x * blockIdx.x + threadIdx.x;
    py = blockDim.y * blockIdx.y + threadIdx.y;
    pid = px + py * blockDim.y * gridDim.y;

    curand_init(pid, pid, 0, s);
}

__device__ vec3 inv(const vec3& dir)
{
    return vec3(1.f / dir.x(), 1.f / dir.y(), 1.f / dir.z());
}

__device__ bool intersectionAABB(const Ray& ray,vec3 bMin, vec3 bMax, float tmin, float tmax) 
{
    // Robust BVH Ray Traversal
    // by Thiago Ize, Solid Angle

    vec3 invD = Normalize(inv(ray.dir));
    vec3 bounds[2] = { bMin, bMax };
    int sign[3];
    sign[0] = invD.x() < 0.f;
    sign[1] = invD.y() < 0.f;
    sign[2] = invD.z() < 0.f;

    float txmin = (bounds[sign[0]].x() - ray.org.x()) * invD.x();
    float txmax = (bounds[1-sign[0]].x() - ray.org.x()) * invD.x();
    float tymin = (bounds[sign[1]].y() - ray.org.y()) * invD.y();
    float tymax = (bounds[1-sign[1]].y() - ray.org.y()) * invD.y();
    float tzmin = (bounds[sign[2]].z() - ray.org.z()) * invD.z();
    float tzmax = (bounds[1-sign[2]].z() - ray.org.z()) * invD.z();

    tmin = cudamax(tzmin,cudamax(tymin, cudamax(txmin, tmin)));
    tmax = cudamin(tzmax, cudamin(tymax, cudamin(txmax, tmax)));
    tmax *= 1.00000024f;
    return (tmin) <= (tmax);
}

#define PUSHBACK(val)     (stack[idx++]=val)
#define POPBACK()      (stack[--idx])

__device__ bool RayCast(const Ray& ray, Triangle* triangles, int Nt, CudaBVHNode* BVHTree, int Nb, Sphere* spheres, int Ns, HitResult& hitResult, float t_min = 0.f, float t_max = 999999.f)
{
    HitResult tmpResult;
    bool bHitInWorld = false;
    float closestT = t_max;

    int idx = 0;
    int stack[128];
    PUSHBACK(0);
    while (idx > 0)
    {
        CudaBVHNode node = BVHTree[POPBACK()];

        //if (!intersectionAABB(ray, node.bMin, node.bMax, t_min-EPS, closestT+EPS))
        if (!intersectionAABB(ray, node.bMin, node.bMax, t_min, closestT))
        {
            continue;
        }

        if (node.primStart != -1 && node.primEnd != -1)
        {
            for (int i = node.primStart; i <= node.primEnd; i++)
            {
                if (triangles[i].hit(ray, t_min, closestT, tmpResult))
                {
                    bHitInWorld = true;
                    closestT = tmpResult.t;
                    hitResult = tmpResult;
                }
            }
        }

        if (node.childR > 0)
        {
            PUSHBACK(node.childR);
        }
        if (node.childL > 0)
        {
            PUSHBACK(node.childL);
        }
    }

    //return bHitInWorld;

    for (int i = 0; i < Ns; i++)
    {
        if (spheres[i].hit(ray, t_min, closestT, tmpResult))
        {
            bHitInWorld = true;
            closestT = tmpResult.t;
            hitResult = tmpResult;
        }
    }

    return bHitInWorld;
}

__device__ Color GetLightColor(const vec3& position, const vec3& pointOnLight, Triangle* triangles, int Nt, CudaBVHNode* BVHTree, int Nb, Sphere* spheres, int Ns)
{
    Ray ray(position, pointOnLight - position);
    HitResult hitResult;

    Color color(0.f, 0.f, 0.f);

    if (RayCast(ray, triangles, Nt, BVHTree, Nb, spheres, Ns, hitResult,0.f,(pointOnLight-position).length()+1.0f))
    {
        if ((hitResult.p - pointOnLight).length() < EPS)
        {
            return hitResult.mat.emittance;
        }
    }

    return color;
}

__device__ float GetLightPDF(const Ray& ray, Triangle* lightSources, int Nl)
{
    HitResult hitResult;
    float pdf = 0.f;

    for (int i = 0; i < Nl; i++)
    {
        if (lightSources[i].hit(ray, EPS, 99999.f, hitResult))
        {
            float cosA = (ray.dir.dot(hitResult.normal));
            cosA = (cosA < 0.f)? -cosA:cosA;
            float divisor = (cosA * lightSources->area);
            if (divisor < EPS)
                continue;
            pdf += (ray.org - hitResult.p).squared_length() / divisor;
        }
    }
    return pdf;
}

__device__ float GetHemiSpherePDF(const Ray& ray, vec3 normal)
{
    return dot(ray.dir, normal);
}

__device__ Color GetColor_iter(const Ray& ray, Triangle* triangles, int Nt, CudaBVHNode* BVHTree, int Nb, Triangle* lights, int Nl, Sphere* spheres, int Ns, curandState* s)
{
    const Color Black(0.f, 0.f, 0.f);

    Color emittances = Black;

    HitResult hitResult;
    Ray currentRay = ray;

    Color radiance = Black;
    Color weight(1.f, 1.f, 1.f);
    vec3 wi;
    float pdf;
    float sample_light_pdf;
    float sample_bxdf_pdf;
    Color oldWeight(1.f,1.f,1.f);

    void* a = (void*)nullptr;

    bool bRefracted = false;
    int RefractCnt = 0;

    int Depth = 0;
    for (; Depth < MAX_BOUNCE; Depth++)
    {
        if (RayCast(currentRay, triangles, Nt, BVHTree, Nb, spheres, Ns, hitResult))
        {
            if (hitResult.mat.emittance.squared_length() > EPS)
            {
                radiance += weight * hitResult.mat.emittance;
                //break;
            }

            float selectedPdf;
            float cosW;
            vec3 reflectedRay;
            float prob;

            vec3 ior = reflectivity_to_eta(hitResult.mat.specular);


            // NEE
            int lightIdx = curand(s) % Nl;
            vec3 SampledPoint;
            float pdfLight = SamplePrimitive(s, SampledPoint, lights[lightIdx]) / ((float)Nl);

            Color lightColor = GetLightColor(hitResult.p, SampledPoint, triangles, Nt, BVHTree, Nb, spheres, Ns);

            cosW = dot(hitResult.normal, Normalize(SampledPoint - hitResult.p));
            cosW = (cosW < 0.f) ? 0.f : cosW;

            float cosA = dot(lights[lightIdx].normal, Normalize(hitResult.p - SampledPoint));
            cosA = (cosA < 0.f) ? 0.f : cosA;

            vec3 brdfcos;
            if (hitResult.mat.opacity < (1.f - EPS))
            {
                if (hitResult.mat.roughness < 1e-2f)
                {
                    brdfcos = eval_pure_refractive(hitResult.mat.albedo, ior[0], hitResult, -currentRay.dir, Normalize(SampledPoint - hitResult.p));
                }
                else
                {

                    brdfcos = eval_refractive(hitResult.mat.albedo, ior[0], hitResult.mat.roughness, hitResult, -currentRay.dir, Normalize(SampledPoint - hitResult.p));
                }
            }
            else
            {
                if (hitResult.mat.roughness < 1e-2f)
                {
                    brdfcos = eval_reflective(hitResult.mat.albedo, hitResult.mat.specular, hitResult.mat.roughness, hitResult.mat.metallic, hitResult, -currentRay.dir, Normalize(SampledPoint - hitResult.p));
                }
                else
                {
                    brdfcos = eval_gltfpbr(hitResult.mat.albedo, hitResult.mat.specular, hitResult.mat.roughness, hitResult.mat.metallic, hitResult, -currentRay.dir, Normalize(SampledPoint - hitResult.p));
                }    
            }
            if(!isnan(brdfcos))
                radiance += weight * brdfcos * lightColor * cosA / ((hitResult.p - SampledPoint).squared_length() * (pdfLight));



            // bxdf sampling
            //prob = SampleHemisphere(s, reflectedRay, hitResult.normal, hitResult.tangent, hitResult.bitangent);
            //cosW = dot(hitResult.normal, reflectedRay);
            vec3 w1;
            float w2 = 0.f;
            vec3 currentWeight;

            int OPAQUE;
            if (hitResult.mat.opacity < (1.f-EPS))
            {
                if (hitResult.mat.roughness < 1e-2f)
                {
                    reflectedRay = sample_pure_refractive(hitResult.mat.albedo, ior[0], hitResult, -currentRay.dir, s);
                    w1 = eval_pure_refractive(hitResult.mat.albedo, ior[0], hitResult, -currentRay.dir, reflectedRay);
                    w2 = sample_pure_refractive_pdf(hitResult.mat.albedo, ior[0], hitResult, -currentRay.dir, reflectedRay);
                    w2 = cudamax(w2, 1e-2f);
                    currentWeight = w1 / w2;
                }
                else
                {
                    reflectedRay = sample_refractive(hitResult.mat.albedo, ior[0], hitResult.mat.roughness, hitResult, -currentRay.dir, s);
                    w1 = eval_refractive(hitResult.mat.albedo, ior[0], hitResult.mat.roughness, hitResult, -currentRay.dir, reflectedRay);
                    w2 = sample_refractive_pdf(hitResult.mat.albedo, ior[0], hitResult.mat.roughness, hitResult, -currentRay.dir, reflectedRay);
                    w2 = cudamax(w2, 1e-2f);
                    currentWeight = w1 / w2;
                }

                int transparentProb = 1;
                assert(transparentProb && !isnan(currentWeight));
                assert(!isnan(reflectedRay));

                bRefracted = dot(hitResult.normal, -currentRay.dir) * dot(hitResult.normal, reflectedRay) <= 0.f;
            }
            else
            {
                if (hitResult.mat.roughness < 1e-2f)
                {
                    reflectedRay = sample_reflective(hitResult.mat.albedo, hitResult.normal, -currentRay.dir);
                    w1 = eval_reflective(hitResult.mat.albedo, hitResult.mat.specular, hitResult.mat.roughness, hitResult.mat.metallic, hitResult, -currentRay.dir, reflectedRay);
                    w2 = sample_reflective_pdf(hitResult.mat.albedo, hitResult.normal, -currentRay.dir, reflectedRay);
                    w2 = cudamax(w2, 1e-2f);
                    currentWeight = w1 / w2;
                }
                else
                {
                    reflectedRay = sample_gltfpbr(hitResult.mat.albedo, hitResult.mat.specular, hitResult.mat.roughness, hitResult.mat.metallic, hitResult, -currentRay.dir, s);
                    
                    assert(!isnan(hitResult.tangent));
                    assert(!isnan(hitResult.bitangent));
                    w1 = eval_gltfpbr(hitResult.mat.albedo, hitResult.mat.specular, hitResult.mat.roughness, hitResult.mat.metallic, hitResult, -currentRay.dir, reflectedRay);
                    w2 = sample_gltfpbr_pdf(hitResult.mat.albedo, hitResult.mat.specular, hitResult.mat.roughness, hitResult.mat.metallic, hitResult, -currentRay.dir, reflectedRay);
                    w2 = cudamax(w2, 1e-2f);
                    currentWeight = w1 / w2;
                }


                int opaqueProb = 1;
                assert(opaqueProb && !isnan(currentWeight));
            }
            if (reflectedRay.squared_length() > EPS)
                weight *= currentWeight;
            else
                break;

            assert(!isnan(w1));
            assert(!isnan(w2));
            assert(!isinf(w1));
            assert(!isinf(w2));
            assert(!isinf(1.f / w2));
            assert(!isnan(1.f / w2));
            assert(!isnan(currentWeight));
            //CHECKNAN(weight)

            currentRay.org = hitResult.p + hitResult.normal * (bRefracted? -EPS : EPS);
            currentRay.dir = reflectedRay;
            if (bRefracted)
            {
                Depth--;
                if (RefractCnt++>8)
                {
                    break;
                }
                continue;
            }

            if (Depth >= RUSSIAN_ROULETTE_BOUNCE)
            {
                float sample = curand_uniform(s);
                float rrprob = cudamax(cudamin(MaxFrom(weight), 1.f), PROB_STOP_BOUNCE);
                if (sample < rrprob)
                {
                    weight *= (1.f / rrprob);
                }
                else
                {
                    break;
                }
            }
        }
        else
        {
            radiance += weight * vec3(0.1f,0.1f,0.1f);
            break;
        }
    }
    return radiance;
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

#endif