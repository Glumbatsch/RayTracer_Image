#pragma once
#include "Float3.h"

//////////////////////////////////////////////////////////////////////////
// #Structs  
//////////////////////////////////////////////////////////////////////////

struct Image {
	int width;
	int height;
	int* pixels;
};

struct DeviceImage {
	int width;
	int height;
	float3* pixels;
};

struct Camera
{
	float3 position;
	float3 up;
	float3 right;
	float3 forward;
	float3 film;
};

struct Material
{
	float3 albedo;
	float3 emitColor;
	float roughness;
	float refractionIndex;
};

struct Ray 
{
	float3 origin;
	float3 direction;
	float3 color;
	int bounces;
};

struct Intersection
{
	float3 normal;
	float t;
	int material;
	int id;
};

struct Plane {
	float3 normal;
	float d;
	int materialIndex;
};

struct Sphere {
	float3 position;
	float radius;
	int materialIndex;
};

struct World {
	int materialCount;
	Material* materials;

	int sphereCount;
	Sphere* spheres;

	int planeCount;
	Plane* planes;

	int rayCount;
	Ray* rays;
	Intersection* intersections;
};

struct Config
{
	// [Image]
	int imageWidth;
	int imageHeight;

	// [Camera]
	float filmWidth;
	float filmHeight;
	float filmDistance;
	float cameraDistance;

	// [RT]
	int samplesPerPixel;
	int maxBounces;
	bool bDenoise;

	// [Optimizations]
	bool bUseFastRand;
	bool bSortIntersections;
};