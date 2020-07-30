#pragma once
#include "Types.h"
#include "Float3.h"

struct Image {
	u32 width;
	u32 height;
	u32* pixels;
};

struct DeviceImage {
	u32 width;
	u32 height;
	f32 filmWidth;
	f32 filmHeight;
	float3* pixels;
};

struct Camera
{
	float3 position;
	float3 up;
	float3 right;
	float3 forward;
};

struct Material
{
	float3 albedo;
	float3 emitColor;
	f32 roughness;
};

struct Ray 
{
	float3 origin;
	float3 direction;
	float3 color;
	i32 bounces;
};

struct Intersection
{
	float3 normal;
	float t;
	u32 material;
};

struct Plane {
	float3 normal;
	f32 d;
	u32 materialIndex;
};

struct Sphere {
	float3 position;
	f32 radius;
	u32 materialIndex;
};

struct World {
	u32 materialCount;
	Material* materials;

	u32 sphereCount;
	Sphere* spheres;

	u32 planeCount;
	Plane* planes;

	u32 rayCount;
	Ray* rays;
	Intersection* intersections;
};