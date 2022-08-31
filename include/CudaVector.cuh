#ifndef _CUDA_VEC_
#define _CUDA_VEC_

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>

#define CHECKNAN(x) if(isnan(x)) { void* a; *(int*)a=1; }
#define CHECKINF(x) if(isinf(x)) { void* a; *(int*)a=1; }
class vec3 {
public:
    __host__ __device__ vec3() { }
    __host__ __device__ vec3(const vec3& v) { e[0] = v.x(); e[1] = v.y(); e[2] = v.z(); }
    __host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3& operator+=(const vec3& v2);
    __host__ __device__ inline vec3& operator-=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const vec3& v2);
    __host__ __device__ inline vec3& operator/=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);

    __host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    __host__ __device__ inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __host__ __device__ inline void make_unit_vector();

    __host__ __device__ inline float dot(const vec3& v2) const;

    float e[3];
};



inline std::istream& operator>>(std::istream& is, vec3& t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator+(const vec3& v, float t) {
    return vec3(t + v.e[0], t + v.e[1], t + v.e[2]);
}

__host__ __device__ inline vec3 operator+(float t, const vec3& v) {
    return vec3(t + v.e[0], t + v.e[1], t + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& v, float t) {
    return vec3(v.e[0] - t, v.e[1] - t, v.e[2] - t);
}

__host__ __device__ inline vec3 operator-(float t, const vec3& v) {
    return vec3(t - v.e[0], t - v.e[1], t - v.e[2]);
}
__host__ __device__ inline float dot(const vec3& v1, const vec3& v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}


__host__ __device__ inline vec3& vec3::operator+=(const vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}
__host__ __device__ inline float MaxFrom(const vec3& v)
{
    float val = (v.x() > v.y()) ? v.x() : v.y();
    val       = (val   > v.z()) ? val   : v.z();
    return val;
}
__host__ __device__ inline float MinFrom(const vec3& v)
{
    float val = (v.x() < v.y()) ? v.x() : v.y();
    val = (val < v.z()) ? val : v.z();
    return val;
}

__host__ __device__ inline vec3 max(const vec3& v1, const vec3& v2)
{
    return vec3(
        (v1.x() > v2.x()) ? v1.x() : v2.x(),
        (v1.y() > v2.y()) ? v1.y() : v2.y(),
        (v1.z() > v2.z()) ? v1.z() : v2.z()
    );
}
__host__ __device__ inline vec3 min(const vec3& v1, const vec3& v2)
{
    return vec3(
        (v1.x() < v2.x()) ? v1.x() : v2.x(),
        (v1.y() < v2.y()) ? v1.y() : v2.y(),
        (v1.z() < v2.z()) ? v1.z() : v2.z()
    );
}

__host__ __device__ inline vec3 max(const vec3& v1, float a)
{
    return vec3(
        (v1.x() > a) ? v1.x() : a,
        (v1.y() > a) ? v1.y() : a,
        (v1.z() > a) ? v1.z() : a
    );
}
__host__ __device__ inline vec3 max(float a, const vec3& v1)
{
    return max(v1, a);
}

__host__ __device__ inline vec3 min(const vec3& v1, float a)
{
    return vec3(
        (v1.x() < a) ? v1.x() : a,
        (v1.y() < a) ? v1.y() : a,
        (v1.z() < a) ? v1.z() : a
    );
}
__host__ __device__ inline vec3 min(float a, const vec3& v1)
{
    return min(v1, a);
}
__host__ __device__ inline vec3 clamp(const vec3& v1, float minVal, float maxVal)
{
    return min(max(v1, minVal),maxVal);
}
__device__ inline float cudamax(const float& v1, const float& v2)
{
    return  (v1 > v2) ? v1 : v2;
}
__device__ inline float cudamin(const float& v1, const float& v2)
{

    return  (v1 < v2)?  v1 : v2;
}

__device__ inline float clamp(const float& v1, float minVal, float maxVal)
{
    return cudamin(cudamax(v1, minVal), maxVal);
}

__device__ inline vec3 fabsf(const vec3& v1)
{

    return  vec3(fabsf(v1.x()), fabsf(v1.y()), fabsf(v1.z()));
}

__device__ inline vec3 sqrtf(const vec3& v1)
{

    return  vec3(sqrtf(v1.x()), sqrtf(v1.y()), sqrtf(v1.z()));
}

__device__ inline vec3 pow(const vec3& base,float expoenent)
{

    return  vec3(powf(base.x(), expoenent), powf(base.y(), expoenent), powf(base.z(), expoenent));
}

__device__ inline vec3 reflect(const vec3& w, const vec3& n) {
    return -w + 2 * dot(n, w) * n;
}

__device__ inline vec3 refract(const vec3& w, const vec3& n, float inv_eta) {
    auto cosine = dot(n, w);
    auto k = 1 + inv_eta * inv_eta * (cosine * cosine - 1);
    if (k < 0) return { 0, 0, 0 };  // tir
    return -w * inv_eta + (inv_eta * cosine - sqrt(k)) * n;
}

__device__ inline bool isnan(const vec3& v)
{
    return isnan(v[0]) || isnan(v[1]) || isnan(v[2]);
}
__device__ inline bool isinf(const vec3& v)
{
    return isinf(v[0]) || isinf(v[1]) || isinf(v[2]);
}
__host__ __device__ inline float vec3::dot(const vec3& v2) const
{
    vec3 multVec = (*this) * v2;
    return multVec[0] + multVec[1] + multVec[2];
}

__host__ __device__ inline vec3 Normalize(const vec3 v)
{
    return v / v.length();
}

__host__ __device__ inline vec3 make_vec3(const float3 _v)
{
    vec3 v;
    v.e[0] = _v.x; v.e[1] = _v.y; v.e[2] = _v.z;
    return v;
}

__host__ __device__ inline vec3 saturate(const vec3& v)
{
    vec3 res;
    res[0] = (v.x() > 0.f) ? ((v.x() < 1.f) ? v.x() : 1.f) : 0.f;
    res[1] = (v.y() > 0.f) ? ((v.y() < 1.f) ? v.y() : 1.f) : 0.f;
    res[2] = (v.z() > 0.f) ? ((v.z() < 1.f) ? v.z() : 1.f) : 0.f;
    return res;
}
#endif