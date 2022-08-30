#ifndef _CUDA_UTIL_
#define _CUDA_UTIL_

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "CudaVector.cuh"
#include "CudaColor.cuh"
#include "CudaRay.cuh"
#include "CudaPrimitive.cuh"

//#include "Bxdf.cuh"

#define MAX_BOUNCE (6)
#define RUSSIAN_ROULETTE_BOUNCE (3)
#define PROB_STOP_BOUNCE (0.2f)
#define NUM_MULTI_SAMPLE (4)
#define NUM_SAMPLE (16)
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

__device__ float SampleHemisphere(curandState* state, vec3& direction, const vec3& normal, const vec3& tangent, const vec3& bitangent)
{
    float prob;

    //float theta = acosf(sqrtf(curand_uniform(state)) - EPS);
    float phi = 2.f * PI * curand_uniform(state);

    float cosTheta = sqrtf(curand_uniform(state)+EPS)-2.f*EPS;
    float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    prob = cosTheta * INV_PI;

    float x = cosPhi * sinTheta;
    float y = sinPhi * sinTheta;
    float z = cosTheta;

    void* a;

    //To world space
    direction = Normalize(x * tangent + y * bitangent + z * normal);

    return prob;
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

__device__ bool RayCast(const Ray& ray, Triangle* triangles, int Nt, CudaBVHNode* BVHTree, int Nb, HitResult& hitResult, float t_min = 0.f, float t_max = 999999.f)
{
    HitResult tmpResult;
    bool bHitInWorld = false;
    float closestT = t_max;

    int idx = 0;
    int stack[128];
    PUSHBACK(0);
    /*
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
    */
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

    return bHitInWorld;
}

__device__ Color GetLightColor(const vec3& position, const vec3& pointOnLight, Triangle* triangles, int Nt, CudaBVHNode* BVHTree, int Nb)
{
    Ray ray(position, pointOnLight - position);
    HitResult hitResult;

    Color color(0.f, 0.f, 0.f);

    if (RayCast(ray, triangles, Nt, BVHTree, Nb, hitResult,0.f,(pointOnLight-position).length()+1.0f))
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

__device__ Color GetColor_iter(const Ray& ray, Triangle* triangles, int Nt, CudaBVHNode* BVHTree, int Nb, Triangle* lights, int Nl, int Depth, curandState* s)
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
    bool bDirectLight = false;
    Color oldWeight(1.f,1.f,1.f);

    void* a = (void*)nullptr;
    for (; Depth < MAX_BOUNCE; Depth++)
    {
        if (RayCast(currentRay, triangles, Nt, BVHTree, Nb, hitResult))
        {
            
            if (Depth == 0)
            {
                radiance += weight * hitResult.mat.emittance;
                bDirectLight = false;
            }
            //radiance += weight * hitResult.mat.emittance;


            float selectedPdf;
            float cosW;
            vec3 reflectedRay;
            float prob;

            // NEE
            int lightIdx = curand(s) % Nl;
            vec3 SampledPoint;
            float pdfLight = SamplePrimitive(s, SampledPoint, lights[lightIdx]) / ((float)Nl);

            Color lightColor = GetLightColor(hitResult.p, SampledPoint, triangles, Nt, BVHTree, Nb);

            cosW = dot(hitResult.normal, Normalize(SampledPoint - hitResult.p));
            cosW = (cosW < 0.f) ? 0.f : cosW;

            float cosA = dot(lights[lightIdx].normal, Normalize(hitResult.p - SampledPoint));
            cosA = (cosA < 0.f) ? 0.f : cosA;

            radiance += weight * INV_PI * hitResult.mat.albedo * cosW * cosA * lightColor / ((hitResult.p - SampledPoint).squared_length() * (pdfLight));

            // bxdf sampling
            prob = SampleHemisphere(s, reflectedRay, hitResult.normal, hitResult.tangent, hitResult.bitangent);
            cosW = dot(hitResult.normal, reflectedRay);

            weight *= INV_PI * hitResult.mat.albedo * cosW / prob;

            if (Depth >= RUSSIAN_ROULETTE_BOUNCE)
            {
                float sample = curand_uniform(s);
                float rrprob = cudamax(MaxFrom(weight), PROB_STOP_BOUNCE);
                if (sample < rrprob)
                {
                    weight *= (1.f / rrprob);
                }
                else
                {
                    break;
                }
            }

            currentRay.org = hitResult.p + hitResult.normal * 1e-2f;
            currentRay.dir = reflectedRay;
        }
        else
        {
            radiance += weight * Black;
        }
    }
    return radiance;
            
            
            
            /*
            // Direct lighting
            int NUM_LIGHT_SAMPLE_SOURCE = 4;
            if (Nl > 0 && NUM_LIGHT_SAMPLE_SOURCE > 0)
            {
                for (int i = 0; i < NUM_LIGHT_SAMPLE_SOURCE; i++)
                {
                    int lightIdx = curand(s) % Nl;
                    vec3 SampledPoint;
                    float pdfLight = SamplePrimitive(s, SampledPoint, lights[lightIdx]) / ((float)Nl);

                    Color lightColor = GetLightColor(hitResult.p, SampledPoint, triangles, Nt, BVHTree, Nb);

                    float cosW = dot(hitResult.normal, Normalize(SampledPoint - hitResult.p));
                    cosW = (cosW < 0.f) ? 0.f : cosW;

                    float cosA = dot(lights[lightIdx].normal, Normalize(hitResult.p - SampledPoint));
                    cosA = (cosA < 0.f) ? 0.f : cosA;

                    DirectLight[Depth] += INV_PI * hitResult.mat.albedo * cosW * cosW * lightColor / ((hitResult.p - SampledPoint).squared_length() * (pdfLight));
                }
                DirectLight[Depth] /= (float)NUM_LIGHT_SAMPLE_SOURCE;
            }
            if (Depth == 0)
                emittances = hitResult.mat.emittance;
            vec3 reflectedRay;
            // sample specular reflection
            if (curand_uniform(s) < hitResult.mat.specular)
            {
                reflectedRay = hitResult.ray.dir - 2.0f * dot(hitResult.normal, hitResult.ray.dir) * hitResult.normal;
                currentRay = Ray(hitResult.p + hitResult.normal * 0.001, reflectedRay);
                reflections[Depth] = Color(1.0f, 1.0f, 1.0f);
            }
            else // sample diffuse reflection
            {
                float prob = SampleHemisphere(s, reflectedRay, hitResult.normal, hitResult.tangent, hitResult.bitangent);
                currentRay = Ray(hitResult.p + hitResult.normal * 0.001, reflectedRay);
                float cosW = dot(hitResult.normal, reflectedRay);

                reflections[Depth] = INV_PI * hitResult.mat.albedo * cosW / prob;
            }
            reflections[Depth];

            if (Depth >= RUSSIAN_ROULETTE_BOUNCE)
            {
                float probToStop = MaxFrom(reflections[Depth - 1]);
                probToStop = (probToStop < PROB_STOP_BOUNCE) ? probToStop : PROB_STOP_BOUNCE;
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
            // Get light from environment
            reflections[Depth] = Black;
            DirectLight[Depth] = Black;
            Depth += 1;
            break;
        }
    }
    Depth -= 1;
    for (; Depth >= 0; --Depth)
    {
        color = color * reflections[Depth] + DirectLight[Depth];
    }
    return color + emittances;
    */
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