#pragma once
#include <math.h>
#include <cuda_runtime.h>

//////////////////////////////////////////////////////////////////////////
// #VectorMath
//////////////////////////////////////////////////////////////////////////


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

//////////////////////////////////////////////////////////////////////////


__host__ __device__ bool compareFloat(const float a, const float b, float epsilon /*= 0.0001f*/)
{
	return fabsf(a - b) < epsilon;
}

__host__ __device__ float3& operator/=(float3& v, const float scalar)
{
	return v *= (1.0f / scalar);
}

__host__ __device__ float3 operator/(const float3& v, const float s)
{
	return v * (1.0f / s);
}

__host__ __device__ float Dot(const float3& a, const float3& b)
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

__host__ __device__ float3 Cross(const float3& a, const float3& b)
{
	return make_float3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

__host__ __device__ float3 Normalized(const float3& v)
{
	float3 r = v;
	Normalize(r);
	return r;
}

__host__ __device__ float3 Lerp(const float3& a, const float3& b, const float t)
{
	return (1.0f - t) * a + t * b;
}

__host__ __device__ float3& operator*=(float3& v, const float3& other)
{
	v.x *= other.x;
	v.y *= other.y;
	v.z *= other.z;
	return v;
}

__host__ __device__ float3& Normalize(float3& v)
{
	float len = Length(v);
	v.x = v.x / len; 
	v.y = v.y / len;
	v.z = v.z / len;
	return v;
}

__host__ __device__ const float3& operator+(const float3& v)
{
	return v;
}

__host__ __device__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z
	);
}

__host__ __device__ float3 operator-(const float3& v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

__host__ __device__ float3 operator-(const float3& a, const float3& b)
{
	return a + -b;
}

__host__ __device__ float Length(const float3& v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ float Length2(const float3& v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ float3& operator*=(float3& v, const float scalar)
{
	v.x *= scalar;
	v.y *= scalar;
	v.z *= scalar;
	return v;
}

__host__ __device__ float3 operator*(const float3& v, const float scale)
{
	return make_float3(
		v.x * scale,
		v.y * scale,
		v.z * scale
	);
}

__host__ __device__ float3 operator*(const float scale, const float3& v)
{
	return v * scale;
}

__host__ __device__ float3& operator-=(float3& v, const float3& other)
{
	return v += -other;
}

__host__ __device__ float3& operator+=(float3& v, const float3& other)
{
	v.x += other.x;
	v.y += other.y;
	v.z += other.z;
	return v;
}

__host__ __device__ bool operator==(const float3& a, const float3& b)
{
	return compareFloat(a.x, b.x) && compareFloat(a.y, b.y) && compareFloat(a.z, b.z);
}