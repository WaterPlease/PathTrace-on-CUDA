#ifndef _BXDF_CUH_
#define _BXDF_CUH_
/*
#include <cuda_runtime.h>
#include "CudaVector.cuh"
#include "CudaPrimitive.cuh"
#include <curand.h>
#include <curand_kernel.h>

#define pif (3.141592f)
#define invPif (1.f/3.141592f)

inline vec3 lerp(const vec3& x, const vec3& y, float alpha)
{
    return x * (1.f - alpha) + y * alpha;
}

inline float mean(const vec3& v)
{
    return (v.x() + v.y() + v.z()) * 0.333333f;
}

__device__ vec3 _SampleHemisphere(curandState* state, vec3& direction, const vec3& normal, const vec3& tangent, const vec3& bitangent)
{
    float prob;

    //float theta = acosf(sqrtf(curand_uniform(state)) - EPS);
    float phi = 2.f * pif * curand_uniform(state);

    float cosTheta = sqrtf(curand_uniform(state) + EPS) - 2.f * EPS;
    float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    //prob = cosTheta * invPif;

    float x = cosPhi * sinTheta;
    float y = sinPhi * sinTheta;
    float z = cosTheta;

    //To world space
    direction = Normalize(x * tangent + y * bitangent + z * normal);

    //return prob;
}


// The code belowe is from https://github.com/xelatihy/yocto-gl/blob/main/libs/yocto/yocto_shading.h
// by Fabio Pellacini.
// Some part of the code may be modified. But lines that contains information about reference is not modified.


// Convert eta to reflectivity
inline vec3 eta_to_reflectivity(const vec3& eta) {
    return ((eta - 1.f) * (eta - 1.f)) / ((eta + 1) * (eta + 1));
}
// Convert reflectivity to  eta.
inline vec3 reflectivity_to_eta(const vec3& reflectivity_) {
    auto reflectivity = clamp(reflectivity_, 0.0f, 0.99f);
    return (1 + sqrtf(reflectivity)) / (1 - sqrtf(reflectivity));
}

inline vec3 fresnel_schlick(
    const vec3& specular, const vec3& normal, const vec3& outgoing) {
    if (specular.squared_length() < EPS) return { 0, 0, 0 };
    auto cosine = dot(normal, outgoing);
    return specular +
        (1 - specular) * pow(clamp(1 - abs(cosine), 0.0f, 1.0f), 5.0f);
}

inline float microfacet_distribution(
    float roughness, const vec3& normal, const vec3& halfway, bool ggx) {
    // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    auto cosine = dot(normal, halfway);
    if (cosine <= 0) return 0;
    auto roughness2 = roughness * roughness;
    auto cosine2 = cosine * cosine;
    if (ggx) {
        return roughness2 / (pif * (cosine2 * roughness2 + 1 - cosine2) *
            (cosine2 * roughness2 + 1 - cosine2));
    }
    else {
        return exp((cosine2 - 1) / (roughness2 * cosine2)) /
            (pif * roughness2 * cosine2 * cosine2);
    }
}

// Evaluate the microfacet shadowing1
inline float microfacet_shadowing1(float roughness, const vec3& normal,
    const vec3& halfway, const vec3& direction, bool ggx) {
    // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-b-brdf-implementation
    auto cosine = dot(normal, direction);
    auto cosineh = dot(halfway, direction);
    if (cosine * cosineh <= 0) return 0;
    auto roughness2 = roughness * roughness;
    auto cosine2 = cosine * cosine;
    if (ggx) {
        return 2 * abs(cosine) /
            (abs(cosine) + sqrt(cosine2 - roughness2 * cosine2 + roughness2));
    }
    else {
        auto ci = abs(cosine) / (roughness * sqrt(1 - cosine2));
        return ci < 1.6f ? (3.535f * ci + 2.181f * ci * ci) /
            (1.0f + 2.276f * ci + 2.577f * ci * ci)
            : 1.0f;
    }
}

// Evaluate microfacet shadowing
inline float microfacet_shadowing(float roughness, const vec3& normal,
    const vec3& halfway, const vec3& outgoing, const vec3& incoming,
    bool ggx) {
    return microfacet_shadowing1(roughness, normal, halfway, outgoing, ggx) *
        microfacet_shadowing1(roughness, normal, halfway, incoming, ggx);
}

// Sample a microfacet distribution.
inline vec3 sample_microfacet(
    float roughness, const HitResult& hitResult, const vec3& rn, bool ggx) {
    auto phi = 2 * pif * rn.x();
    auto theta = 0.0f;
    if (ggx) {
        theta = atan(roughness * sqrt(rn.y() / (1 - rn.y())));
    }
    else {
        auto roughness2 = roughness * roughness;
        theta = atan(sqrt(-roughness2 * log(1 - rn.y())));
    }
    auto local_half_vector = vec3(
        cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
    return local_half_vector.x() * hitResult.tangent + local_half_vector.y() * hitResult.bitangent + local_half_vector.z() * hitResult.normal;
}

// Pdf for microfacet distribution sampling.
inline float sample_microfacet_pdf(
    float roughness, const HitResult& hitResult, const vec3& halfway, bool ggx = true) {
    auto cosine = dot(hitResult.normal, halfway);
    if (cosine < 0) return 0;
    return microfacet_distribution(roughness, hitResult.normal, halfway, ggx) * cosine;
}

inline vec3 eval_gltfpbr(const vec3& color, float ior, float roughness,
    float metallic, const HitResult& hitResult, const vec3& outgoing,
    const vec3& incoming) {
    if (dot(hitResult.normal, incoming) * dot(hitResult.normal, outgoing) <= 0) return { 0, 0, 0 };
    auto reflectivity = lerp(
        eta_to_reflectivity(vec3{ ior, ior, ior }), color, metallic);
    auto up_normal = dot(hitResult.normal, outgoing) <= 0 ? -hitResult.normal : hitResult.normal;
    auto F1 = fresnel_schlick(reflectivity, up_normal, outgoing);
    auto halfway = Normalize(incoming + outgoing);
    auto F = fresnel_schlick(reflectivity, halfway, incoming);
    auto D = microfacet_distribution(roughness, up_normal, halfway);
    auto G = microfacet_shadowing(
        roughness, up_normal, halfway, outgoing, incoming);
    return color * (1 - metallic) * (1 - F1) / pif *
        abs(dot(up_normal, incoming)) +
        F * D * G / (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
        abs(dot(up_normal, incoming));
}

// Sample a specular BRDF lobe.
inline vec3 sample_gltfpbr(const vec3& color, float ior, float roughness,
    float metallic, const HitResult& hitResult, const vec3& outgoing, float rnl,
    const vec3& rn) {
    auto up_normal = dot(hitResult.normal, outgoing) <= 0 ? -hitResult.normal : hitResult.normal;
    auto reflectivity = lerp(
        eta_to_reflectivity(vec3{ ior, ior, ior }), color, metallic);
    if (rnl < mean(fresnel_schlick(reflectivity, up_normal, outgoing))) {
        auto halfway = sample_microfacet(roughness, up_normal, rn);
        auto incoming = reflect(outgoing, halfway);
        if (!same_hemisphere(up_normal, outgoing, incoming)) return { 0, 0, 0 };
        return incoming;
    }
    else {
        return sample_hemisphere_cos(up_normal, rn);
    }
}

// Pdf for specular BRDF lobe sampling.
inline float sample_gltfpbr_pdf(const vec3& color, float ior, float roughness,
    float metallic, const HitResult& hitResult, const vec3& outgoing,
    const vec3& incoming) {
    if (dot(hitResult.normal, incoming) * dot(hitResult.normal, outgoing) <= 0) return 0;
    auto halfway = Normalize(outgoing + incoming);
    auto reflectivity = lerp(
        eta_to_reflectivity(vec3{ ior, ior, ior }), color, metallic);
    auto F = mean(fresnel_schlick(reflectivity, hitResult.normal, outgoing));
    return F * sample_microfacet_pdf(roughness, hitResult, halfway) /
        (4 * abs(dot(outgoing, halfway))) +
        (1 - F) * dot(hitResult.normal, incoming) * invPif;
}
*/
#endif