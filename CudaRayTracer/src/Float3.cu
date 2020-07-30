#include "Float3.h"

__host__ __device__ bool compareFloat(const f32 a, const f32 b, f32 epsilon /*= 0.0001f*/)
{
	return fabsf(a - b) < epsilon;
}

__host__ __device__ float3& operator/=(float3& v, const f32 scalar)
{
	return v *= (1.0f / scalar);
}

__host__ __device__ float3 operator/(const float3& v, const f32 s)
{
	return v * (1.0f / s);
}

__host__ __device__ f32 Dot(const float3& a, const float3& b)
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

__host__ __device__ float3 Lerp(const float3& a, const float3& b, const f32 t)
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
	float3 tmp = v;
	f32 len = Length(tmp);
	//v.x = v.x / len; // #Bug in threads 10, 20 ... v.x /= turns len to 0.0f. => fails if v-component is 0
	//v.y = v.y / len;
	//v.z = v.z / len;
	v.x = v.x / Length(tmp);
	v.y = v.y / Length(tmp);
	v.z = v.z / Length(tmp);
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

__host__ __device__ f32 Length(const float3& v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ f32 Length2(const float3& v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ float3& operator*=(float3& v, const f32 scalar)
{
	v.x *= scalar;
	v.y *= scalar;
	v.z *= scalar;
	return v;
}

__host__ __device__ float3 operator*(const float3& v, const f32 scale)
{
	return make_float3(
		v.x * scale,
		v.y * scale,
		v.z * scale
	);
}

__host__ __device__ float3 operator*(const f32 scale, const float3& v)
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