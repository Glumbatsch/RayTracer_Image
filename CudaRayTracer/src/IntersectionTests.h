#pragma once
#include "Structs.h"
#include "Types.h"
#include "Float3.h"

#include <curand.h>
#include <curand_kernel.h>

__device__ bool IntersectPlane(const Ray& ray, const Plane& plane,
	f32& t)
{
	f32 denom = Dot(plane.normal, ray.direction);
	f32 tolerance = 0.0001f;
	if ((denom < -tolerance) || (denom > tolerance))
	{
		t = (-plane.d - Dot(plane.normal, ray.origin)) / denom;
		if (t >= 0.0f)
			return true;
	}
	return false;
}

__device__ bool IntersectSphere(const Ray& ray,
	const Sphere& sphere, f32& t)
{
	float3 m = ray.origin - sphere.position;
	float b = Dot(m, ray.direction);
	float c = Dot(m, m) - sphere.radius * sphere.radius;

	// Ray's origin outside of the sphere 
	// and ray is pointing away from the sphere.
	if (c > 0.0f && b > 0.0f) return false;

	float discriminant = b * b - c;

	// A negative discriminant means there is no intersection.
	if (discriminant < 0.0f) return false;

	t = -b - sqrtf(discriminant);
	// clamp t to 0
	if (t < 0.0f) t = 0.0f;

	return true;
}

__device__ __forceinline__ float3 GetPointAlongRay(const Ray& ray,
	const f32 t)
{
	return ray.origin + ray.direction * t;
}

__device__ __forceinline__ float3 GetSphereNormal(const Ray& ray,
	const Sphere& sphere, const f32 t)
{
	return Normalized(GetPointAlongRay(ray,t) - sphere.position);
}

__device__ __forceinline__ f32 RandomBilateral(
	curandState* randState)
{
	return -1.0f + 2.0f * curand_uniform(randState);
}

__device__ void Reflect(Ray& ray, f32 t, float3 normal, 
	curandState* randState, f32 roughness)
{
	float3 hitPoint = GetPointAlongRay(ray, t);
	ray.origin = hitPoint;

	float3 pureReflect = Normalized(ray.direction 
		+ Dot(ray.direction, normal) * 2.0f * normal);
	float3 randomDirection = make_float3(
		RandomBilateral(randState),
		RandomBilateral(randState),
		RandomBilateral(randState)
	);
	float3 randomReflect = Normalized(normal+randomDirection);
	ray.direction = Lerp(randomReflect, pureReflect, roughness);
}