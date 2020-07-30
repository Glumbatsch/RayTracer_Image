#pragma once
#include "Structs.h"
#include "Types.h"
#include "Float3.h"

__device__ bool IntersectPlane(const Ray& ray, const Plane& plane,
	f32& t)
{
	f32 denom = Dot(plane.normal, ray.direction);
	f32 tolerance = 0.0001f;
	if(!((denom < - tolerance) || (denom > tolerance)))
		return false;
	t = (-plane.d - Dot(plane.normal, ray.origin)) / denom;
	return true;
}