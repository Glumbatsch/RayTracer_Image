#pragma once
#include <math.h>
#include <cuda_runtime.h>
#include "Types.h"

__host__ __device__ bool compareFloat(const f32 a, const f32 b, f32 epsilon = 0.0001f);

// Unary operators
__host__ __device__ const float3& operator+(const float3& v);
__host__ __device__ float3 operator-(const float3& v);

// Compare uses a compare f32 function.
__host__ __device__ bool operator==(const float3& a, const float3& b);

// Assigns
__host__ __device__ float3& operator+=(float3& v, const float3& other);
__host__ __device__ float3& operator-=(float3& v, const float3& other);
__host__ __device__ float3& operator*=(float3& v, const float3& other);
__host__ __device__ float3& operator*=(float3& v, const f32 scalar);
__host__ __device__ float3& operator/=(float3& v, const f32 scalar);

// Math
__host__ __device__ f32 Length(const float3& v);
__host__ __device__ f32 Length2(const float3& v);

__host__ __device__ float3& operator+=(float3& v, const float3& other);

__host__ __device__ float3& operator-=(float3& v, const float3& other);

__host__ __device__ float3& operator*=(float3& v, const f32 scalar);

__host__ __device__ float3& operator/=(float3& v, const f32 scalar);

// Normalizes v in place.
__host__ __device__ float3& Normalize(float3& v);

__host__ __device__ float3 operator+(const float3& a, const float3& b);

__host__ __device__ float3 operator-(const float3& a, const float3& b);

__host__ __device__ float3 operator*(const float3& v, const f32 scale);

__host__ __device__ float3 operator*(const f32 scale, const float3& v);

__host__ __device__ float3 operator/(const float3& v, const f32 s);

__host__ __device__ f32 Dot(const float3& a, const float3& b);

__host__ __device__ float3 Cross(const float3& a, const float3& b);

// Returns a normalized copy of v. Does not alter v.
__host__ __device__ float3 Normalized(const float3& v);

__host__ __device__ float3 Lerp(const float3& a, const float3& b,
	const f32 t);