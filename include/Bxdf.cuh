#ifndef _BXDF_CUH_
#define _BXDF_CUH_

#include <cuda_runtime.h>
#include "CudaVector.cuh"
#include "CudaPrimitive.cuh"
#include <curand.h>
#include <curand_kernel.h>

#define pif (3.141592f)
#define invPif (1.f/3.141592f)

__device__ inline vec3 lerp(const vec3& x, const vec3& y, float alpha)
{
    return x * (1.f - alpha) + y * alpha;
}

__device__ inline float mean(const vec3& v)
{
    return (v.x() + v.y() + v.z()) * 0.333333f;
}

__device__ vec3 SampleHemisphere(curandState* state, const vec3& normal, const vec3& tangent, const vec3& bitangent)
{
    //float theta = acosf(sqrtf(curand_uniform(state)) - EPS);
    float phi = 2.f * pif * curand_uniform(state);

    float cosTheta = sqrtf(curand_uniform(state));
    float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    //prob = cosTheta * invPif;

    float x = cosPhi * sinTheta;
    float y = sinPhi * sinTheta;
    float z = cosTheta;

    //To world space
    return Normalize(x * tangent + y * bitangent + z * normal);
}


// The code belowe is from https://github.com/xelatihy/yocto-gl/blob/main/libs/yocto/yocto_shading.h
// by Fabio Pellacini.
// Some part of the code may be modified. But lines that contains information about reference is not modified.

// Convert eta to reflectivity
__device__ inline vec3 eta_to_reflectivity(const vec3& eta) {
    return ((eta - 1.f) * (eta - 1.f)) / ((eta + 1) * (eta + 1));
}
// Convert reflectivity to  eta.
__device__ inline vec3 reflectivity_to_eta(const vec3& reflectivity_) {
    auto reflectivity = clamp(reflectivity_, 0.0f, 0.99f);
    return (1 + sqrtf(reflectivity)) / (1 - sqrtf(reflectivity));
}

// Compute the fresnel term for dielectrics.
__device__ inline float fresnel_dielectric(
    float eta, const vec3& normal, const vec3& outgoing) {
    // Implementation from
    // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
    auto cosw = abs(dot(normal, outgoing));

    auto sin2 = 1 - cosw * cosw;
    auto eta2 = eta * eta;

    auto cos2t = 1 - sin2 / eta2;
    if (cos2t < 0) return 1;  // tir

    auto t0 = sqrtf(cos2t);
    auto t1 = eta * t0;
    auto t2 = eta * cosw;

    auto rs = (cosw - t1) / (cosw + t1);
    auto rp = (t0 - t2) / (t0 + t2);

    return (rs * rs + rp * rp) / 2;
}

__device__ inline vec3 fresnel_schlick(
    const vec3& specular, const vec3& normal, const vec3& outgoing) {
    if (specular.squared_length() < EPS) return vec3(0, 0, 0);
    auto cosine = dot(normal, outgoing);
    return specular +
        (1.f - specular) * powf(clamp(1 - fabs(cosine), EPS, 0.999f), 5.0f);
}

__device__ inline float microfacet_distribution(
    float roughness, const vec3& normal, const vec3& halfway, bool ggx) {
    // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    auto cosine = dot(normal, halfway);
    if (cosine <= EPS) return 0;
    auto roughness2 = roughness * roughness;
    auto cosine2 = cosine * cosine;
    auto divisor = (cosine2 * roughness2 + 1 - cosine2);
    divisor = cudamax(divisor, 1e-2f);
    if (ggx) {
        return roughness2 / (pif * divisor * divisor);
    }
    else {
        return expf((cosine2 - 1) / (roughness2 * cosine2)) /
            (pif * roughness2 * cosine2 * cosine2);
    }
}

// Evaluate the microfacet shadowing1
__device__ inline float microfacet_shadowing1(float roughness, const vec3& normal,
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
        return 2 * fabs(cosine) /
            (fabs(cosine) + sqrtf(cosine2 - roughness2 * cosine2 + roughness2));
    }
    else {
        auto ci = fabs(cosine) / (roughness * sqrtf(1 - cosine2));
        return ci < 1.6f ? (3.535f * ci + 2.181f * ci * ci) /
            (1.0f + 2.276f * ci + 2.577f * ci * ci)
            : 1.0f;
    }
}

// Evaluate microfacet shadowing
__device__ inline float microfacet_shadowing(float roughness, const vec3& normal,
    const vec3& halfway, const vec3& outgoing, const vec3& incoming,
    bool ggx) {
    return microfacet_shadowing1(roughness, normal, halfway, outgoing, ggx) *
        microfacet_shadowing1(roughness, normal, halfway, incoming, ggx);
}

// Sample a microfacet distribution.
__device__ inline vec3 sample_microfacet(
    float roughness, const HitResult& hitResult, curandState* s, bool entering = true) {
    auto phi = 2 * pif * curand_uniform(s);
    auto ry = curand_uniform(s);
    auto theta = 0.0f;
    theta = atanf(roughness * sqrtf(ry / (1.f - ry)));

    auto local_half_vector = vec3(
        cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
    return local_half_vector.x() * hitResult.tangent + local_half_vector.y() * hitResult.bitangent + local_half_vector.z() * hitResult.normal;
}

// Pdf for microfacet distribution sampling.
__device__ inline float sample_microfacet_pdf(
    float roughness, const HitResult& hitResult, const vec3& halfway, bool ggx = true) {
    auto cosine = dot(hitResult.normal, halfway);
    if (cosine < 0) return 0;
    return microfacet_distribution(roughness, hitResult.normal, halfway, ggx) * cosine;
}

__device__ inline vec3 eval_gltfpbr(const vec3& color, const vec3& _reflectivity, float roughness,
    float metallic, const HitResult& hitResult, const vec3& outgoing,
    const vec3& incoming) {
    if (dot(hitResult.normal, incoming) * dot(hitResult.normal, outgoing) <= 0) return { 0, 0, 0 };
    auto reflectivity = lerp(_reflectivity, color, metallic);
    auto F1 = fresnel_schlick(reflectivity, hitResult.normal, outgoing);
    auto halfway = Normalize(incoming + outgoing);
    auto F = fresnel_schlick(reflectivity, halfway, incoming);
    auto D = microfacet_distribution(roughness, hitResult.normal, halfway, true);
    auto G = microfacet_shadowing(
        roughness, hitResult.normal, halfway, outgoing, incoming, true);
    vec3 k = (1 - metallic) * (1 - F1);
    return color * k * invPif *
        fabs(dot(hitResult.normal, incoming)) +
        F * D * G / (4 * dot(hitResult.normal, outgoing) * dot(hitResult.normal, incoming)) *
        fabs(dot(hitResult.normal, incoming));
}

// Sample a specular BRDF lobe.
__device__ inline vec3 sample_gltfpbr(const vec3& color, const vec3& _reflectivity, float roughness,
    float metallic, const HitResult& hitResult, const vec3& outgoing, curandState* s) {
    auto reflectivity = lerp(_reflectivity, color, metallic);
    if (curand_uniform(s) < mean(fresnel_schlick(reflectivity, hitResult.normal, outgoing))) {
        auto halfway = sample_microfacet(roughness, hitResult, s);
        auto incoming = reflect(outgoing, halfway);
        if (dot(hitResult.normal, incoming) * dot(hitResult.normal, outgoing) < -EPS)
        {
            return { 0, 0, 0 };
        }
        return incoming;
    }
    else {
        return SampleHemisphere(s,hitResult.normal, hitResult.tangent, hitResult.bitangent);
    }
}

// Pdf for specular BRDF lobe sampling.
__device__ inline float sample_gltfpbr_pdf(const vec3& color, const vec3& _reflectivity, float roughness,
    float metallic, const HitResult& hitResult, const vec3& outgoing,
    const vec3& incoming) {
    if (dot(hitResult.normal, incoming) * dot(hitResult.normal, outgoing) <= 0) return 0;
    auto halfway = Normalize(outgoing + incoming);
    auto reflectivity = lerp(_reflectivity, color, metallic);
    auto F = mean(fresnel_schlick(reflectivity, hitResult.normal, outgoing));
    return F * sample_microfacet_pdf(roughness, hitResult, halfway) /
        (4 * fabs(dot(outgoing, halfway))) +
        (1 - F) * dot(hitResult.normal, incoming) * invPif;
}


// Evaluate a delta metal BRDF lobe.
__device__ inline vec3 eval_reflective(const vec3& color, const vec3& _reflectivity, float roughness,
    float metallic, const HitResult& hitResult, const vec3& outgoing,
    const vec3& incoming) {
    if (dot(hitResult.normal, incoming) * dot(hitResult.normal, outgoing) <= 0) return { 0, 0, 0 };
    auto reflectivity = lerp(_reflectivity, color, metallic);
    auto F1 = fresnel_schlick(reflectivity, hitResult.normal, outgoing);
    auto F = fresnel_schlick(reflectivity, hitResult.normal, incoming);
    vec3 k = (1 - metallic) * (1 - F1);
    return color * k * invPif *
        fabs(dot(hitResult.normal, incoming)) +
        F * fabs(dot(hitResult.normal, incoming));
}

// Sample a delta metal BRDF lobe.
__device__ inline vec3 sample_reflective(const vec3& eta,
    const vec3& normal, const vec3& outgoing) {
    return reflect(outgoing, normal);
}

// Pdf for delta metal BRDF lobe sampling.
__device__ inline float sample_reflective_pdf(const vec3& eta,
    const vec3& normal, const vec3& outgoing, const vec3& incoming) {
    return 1;
}

__device__ inline vec3 eval_refractive(const vec3& color, float ior, float roughness,
    const HitResult& hitResult, const vec3& outgoing, const vec3& incoming) {
    auto normal = hitResult.bFrontFace ? hitResult.normal : -hitResult.normal;
    auto entering = dot(normal, outgoing) >= 0;
    auto up_normal = entering ? normal : -normal;
    auto rel_ior = entering ? ior : (1 / ior);
    if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
        auto halfway = Normalize(incoming + outgoing);
        auto F = fresnel_dielectric(rel_ior, halfway, outgoing);
        auto D = microfacet_distribution(roughness, up_normal, halfway,true);
        auto G = microfacet_shadowing(
            roughness, up_normal, halfway, outgoing, incoming, true);
        return color *F * D * G /
            abs(4 * dot(normal, outgoing) * dot(normal, incoming)) *
            abs(dot(normal, incoming));
    }
    else {
        auto halfway = -Normalize(rel_ior * incoming + outgoing) *
            (entering ? 1.0f : -1.0f);
        auto F = fresnel_dielectric(rel_ior, halfway, outgoing);
        auto D = microfacet_distribution(roughness, up_normal, halfway, true);
        auto G = microfacet_shadowing(
            roughness, up_normal, halfway, outgoing, incoming, true);
        // [Walter 2007] equation 21
        return color *
            abs((dot(outgoing, halfway) * dot(incoming, halfway)) /
                (dot(outgoing, normal) * dot(incoming, normal))) *
            (1 - F) * D * G /
            pow(rel_ior * dot(halfway, incoming) + dot(halfway, outgoing),
                2.0f) *
            abs(dot(normal, incoming));
    }
}

// Sample a refraction BRDF lobe.
__device__ inline vec3 sample_refractive(const vec3& color, float ior, float roughness,
    const HitResult& hitResult, const vec3& outgoing, curandState* s) {
    auto normal = hitResult.bFrontFace ? hitResult.normal : -hitResult.normal;
    auto entering = dot(normal, outgoing) >= 0;
    auto up_normal = entering ? normal : -normal;
    auto halfway = sample_microfacet(roughness, hitResult, s);
    // auto halfway = sample_microfacet(roughness, up_normal, outgoing, rn);
    if (curand_uniform(s) < fresnel_dielectric(entering ? ior : (1 / ior), halfway, outgoing)) {
        auto incoming = reflect(outgoing, halfway);
        if (!(dot(normal, outgoing) * dot(normal, incoming) >= 0)) return { 0, 0, 0 };
        return incoming;
    }
    else {
        auto incoming = refract(outgoing, halfway, entering ? (1 / ior) : ior);
        if ((dot(normal, outgoing) * dot(normal, incoming) >= 0)) return { 0, 0, 0 };
        return incoming;
    }
}

// Pdf for refraction BRDF lobe sampling.
__device__ inline float sample_refractive_pdf(const vec3& color, float ior,
    float roughness, const HitResult& hitResult, const vec3& outgoing,
    const vec3& incoming) {
    auto normal = hitResult.bFrontFace ? hitResult.normal : -hitResult.normal;
    auto entering = dot(normal, outgoing) >= 0;
    auto up_normal = entering ? normal : -normal;
    auto rel_ior = entering ? ior : (1 / ior);
    if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
        auto halfway = Normalize(incoming + outgoing);
        return fresnel_dielectric(rel_ior, halfway, outgoing) *
            sample_microfacet_pdf(roughness, hitResult, halfway) /
            //  sample_microfacet_pdf(roughness, up_normal, halfway, outgoing) /
            (4 * abs(dot(outgoing, halfway)));
    }
    else {
        auto halfway = -Normalize(rel_ior * incoming + outgoing) *
            (entering ? 1.0f : -1.0f);
        // [Walter 2007] equation 17
        return (1 - fresnel_dielectric(rel_ior, halfway, outgoing)) *
            sample_microfacet_pdf(roughness, hitResult, halfway) *
            //  sample_microfacet_pdf(roughness, up_normal, halfway, outgoing) /
            abs(dot(halfway, incoming)) /  // here we use incoming as from pbrt
            pow(rel_ior * dot(halfway, incoming) + dot(halfway, outgoing), 2.0f);
    }
}

__device__ inline vec3 eval_pure_refractive(const vec3& color, float ior,
    const HitResult& hitResult, const vec3& outgoing, const vec3& incoming) {
    auto normal = hitResult.bFrontFace ? hitResult.normal : -hitResult.normal;
    auto entering = dot(normal, outgoing) >= 0;
    auto up_normal = entering ? normal : -normal;
    auto rel_ior = entering ? ior : (1 / ior);
    if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
        auto halfway = Normalize(incoming + outgoing);
        auto F = fresnel_dielectric(rel_ior, halfway, outgoing);
        return color * F;
    }
    else {
        auto halfway = -Normalize(rel_ior * incoming + outgoing) *
            (entering ? 1.0f : -1.0f);
        auto F = fresnel_dielectric(rel_ior, halfway, outgoing);
        return color * (1 - F) / (rel_ior * rel_ior);
    }
}

// Sample a refraction BRDF lobe.
__device__ inline vec3 sample_pure_refractive(const vec3& color, float ior,
    const HitResult& hitResult, const vec3& outgoing, curandState* s) {
    auto normal = hitResult.bFrontFace ? hitResult.normal : -hitResult.normal;
    auto entering = dot(normal, outgoing) >= 0;
    auto up_normal = entering ? normal : -normal;
    // auto halfway = sample_microfacet(roughness, up_normal, outgoing, rn);
    if (curand_uniform(s) < fresnel_dielectric(entering ? ior : (1 / ior), up_normal, outgoing)) {
        auto incoming = reflect(outgoing, up_normal);
        return incoming;
    }
    else {
        auto incoming = refract(outgoing, up_normal, entering ? (1 / ior) : ior);
        return incoming;
    }
}

// Pdf for refraction BRDF lobe sampling.
__device__ inline float sample_pure_refractive_pdf(const vec3& color, float ior,
    const HitResult& hitResult, const vec3& outgoing,
    const vec3& incoming) {
    auto normal = hitResult.bFrontFace ? hitResult.normal : -hitResult.normal;
    auto entering = dot(normal, outgoing) >= 0;
    auto up_normal = entering ? normal : -normal;
    auto rel_ior = entering ? ior : (1 / ior);
    if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
        auto halfway = Normalize(incoming + outgoing);
        return fresnel_dielectric(rel_ior, halfway, outgoing);
    }
    else {
        auto halfway = -Normalize(rel_ior * incoming + outgoing) *
            (entering ? 1.0f : -1.0f);
        return (1.f - fresnel_dielectric(rel_ior, halfway, outgoing));
    }
}
#endif