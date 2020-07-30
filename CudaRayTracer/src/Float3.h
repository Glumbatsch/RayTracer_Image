#pragma once
#include <math.h>
#include <cuda_runtime.h>

__host__ __device__ bool compareFloat(const float a, const float b, float epsilon = 0.0001f);

// Unary operators
__host__ __device__ const float3& operator+(const float3& v);
__host__ __device__ float3 operator-(const float3& v);

// Compare uses a compare float function.
__host__ __device__ bool operator==(const float3& a, const float3& b);

// Assigns
__host__ __device__ float3& operator+=(float3& v, const float3& other);
__host__ __device__ float3& operator-=(float3& v, const float3& other);
__host__ __device__ float3& operator*=(float3& v, const float3& other);
__host__ __device__ float3& operator*=(float3& v, const float scalar);
__host__ __device__ float3& operator/=(float3& v, const float scalar);

// Math
__host__ __device__ float Length(const float3& v);
__host__ __device__ float Length2(const float3& v);

__host__ __device__ float3& operator+=(float3& v, const float3& other);

__host__ __device__ float3& operator-=(float3& v, const float3& other);

__host__ __device__ float3& operator*=(float3& v, const float scalar);

__host__ __device__ float3& operator/=(float3& v, const float scalar);

// Normalizes v in place.
__host__ __device__ float3& Normalize(float3& v);

__host__ __device__ float3 operator+(const float3& a, const float3& b);

__host__ __device__ float3 operator-(const float3& a, const float3& b);

__host__ __device__ float3 operator*(const float3& v, const float scale);

__host__ __device__ float3 operator*(const float scale, const float3& v);

__host__ __device__ float3 operator/(const float3& v, const float s);

__host__ __device__ float Dot(const float3& a, const float3& b);

__host__ __device__ float3 Cross(const float3& a, const float3& b);

// Returns a normalized copy of v. Does not alter v.
__host__ __device__ float3 Normalized(const float3& v);

__host__ __device__ float3 Lerp(const float3& a, const float3& b,
	const float t);