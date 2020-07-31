#pragma once
#include "Structs.h"
#include "Float3.h"

#include <curand.h>
#include <curand_kernel.h>

//////////////////////////////////////////////////////////////////////////
// #IntersectionTests (taken from "Real Time Collision Detection" by Christer Ericson)
//////////////////////////////////////////////////////////////////////////

__device__ bool IntersectPlane(const Ray& ray, const Plane& plane,
	float& t)
{
	float denom = Dot(plane.normal, ray.direction);
	float tolerance = 0.0001f;
	if ((denom < -tolerance) || (denom > tolerance))
	{
		t = (-plane.d - Dot(plane.normal, ray.origin)) / denom;
		if (t >= 0.0f)
			return true;
	}
	return false;
}

__device__ bool IntersectSphere(const Ray& ray,
	const Sphere& sphere, float& t)
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
	if (t < 0.0f) return false;

	return true;
}

//////////////////////////////////////////////////////////////////////////
// #IntersectionRelatedFunctions
//////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float3 GetPointAlongRay(const Ray& ray,
	const float t)
{
	return ray.origin + ray.direction * t;
}

__device__ __forceinline__ float3 GetSphereNormal(const Ray& ray,
	const Sphere& sphere, const float t)
{
	return Normalized(GetPointAlongRay(ray, t) - sphere.position);
}

__device__ __forceinline__ float RandomBilateral(
	curandState* randState)
{
	return -1.0f + 2.0f * curand_uniform(randState);
}

__device__ void Reflect(Ray& ray, float t, float3 normal,
	curandState* randState, float roughness)
{
	float3 hitPoint = GetPointAlongRay(ray, t);
	ray.origin = hitPoint;

	float3 pureReflect = Normalized(ray.direction
		- Dot(ray.direction, normal) * 2.0f * normal);
	float3 randomDirection = make_float3(
		RandomBilateral(randState),
		RandomBilateral(randState),
		RandomBilateral(randState)
	);
	float3 randomReflect = Normalized(normal + randomDirection);
	ray.direction = Lerp(pureReflect, randomReflect, roughness);
	Normalize(ray.direction);
	ray.origin += ray.direction * 0.0001f;
}

// Taken from Ray Tracing in a Weekend by Peter Shirley
__device__ bool Refract(const Ray &ray, const float3 normal, 
	float niOverNt, float3& refractedDirection)
{
	float dt = Dot(ray.direction, normal);
	float discr = 1.0f - niOverNt * niOverNt * (1.0f - dt * dt);
	if (discr > 0)
	{
		refractedDirection = niOverNt * (ray.direction - normal * dt) -
			normal * sqrtf(discr);
		return true;
	}
	return false;
}

// Taken from Ray Tracing in a Weekend by Peter Shirley
__device__ float Schlick(float cosine, float refractionIndex)
{
	float r0 = (1.0f - refractionIndex) / (1.0f + refractionIndex);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf((1.0f-cosine),5.0f);
}

// Taken from Ray Tracing in a Weekend by Peter Shirley
__device__ void ReflectOrRefract(Ray& ray, float3 normal,
	curandState* randState, const float refractionIndex)
{
	float3 refract;
	float niOverNt;
	float cosine;
	float reflectProb;
	float3 outNormal;
	if (Dot(ray.direction, normal) > 0.0f)
	{
		outNormal = -normal;
		niOverNt = refractionIndex;
		cosine = refractionIndex * Dot(ray.direction,normal);
	}
	else
	{
		outNormal = normal;
		niOverNt = 1.0f / refractionIndex;
		cosine = -Dot(ray.direction, normal);
	}
	if (Refract(ray, outNormal, niOverNt, refract))
	{
		reflectProb = Schlick(cosine, refractionIndex);
	}
	else
	{
		reflectProb = 1.0f;
	}
	if (curand_uniform(randState) < reflectProb)
	{
		ray.direction = Normalized(refract);
	}
	else
	{
		float3 pureReflect = Normalized(ray.direction
			- Dot(ray.direction, normal) * 2.0f * normal);
		ray.direction = pureReflect;
	}
	ray.origin += ray.direction * 0.0001f;
}