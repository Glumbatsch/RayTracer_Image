#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Structs.h"
#include "IntersectionTests.h"
#include "Util.h"

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <float.h>

#include <curand.h>
#include <curand_kernel.h>

#include <thrust\device_ptr.h>
#include <thrust\sort.h>

#include "OpenImageDenoise\oidn.h"

//////////////////////////////////////////////////////////////////////////
// #Globals 
//////////////////////////////////////////////////////////////////////////
double g_totalSeconds = 0.0;
Config g_cfg;
float3* g_deviceImage;
thrust::device_ptr<Intersection> g_deviceIntersections;
// Device Stuff
World* d_world;
Camera* d_camera;
DeviceImage* d_image;
curandState* d_randStates;

//////////////////////////////////////////////////////////////////////////
// #ForwardDeclarations
//////////////////////////////////////////////////////////////////////////
void Render();
void CudaInit();

__global__ void GeneratePrimaryRaysKernel(DeviceImage* image,
	World* world, Camera* camera, int bounces,
	curandState* randStates);
__global__ void ComputeIntersectionsKernel(DeviceImage* image,
	World* world);
__global__ void InitCurandKernel(int rayCount,
	curandState* randStates, int seed, bool bUseFastRand);
__global__ void ShadeIntersectionsKernel(DeviceImage* image,
	World* world, curandState* randStates, bool bSorted);
__global__ void WriteRayColorToImage(DeviceImage* image,
	World* world, float contributionPerPixel);

//////////////////////////////////////////////////////////////////////////
// #Host
//////////////////////////////////////////////////////////////////////////

int main()
{
	LoadConfig("./rt.ini", g_cfg);

	CudaInit();
	Render();
	cudaCall(cudaDeviceSynchronize());
	printf("Writing image to file...");
	TimeStamp start = std::chrono::high_resolution_clock::now();
	WriteBMP(g_cfg, g_deviceImage);
	TimeStamp end = std::chrono::high_resolution_clock::now();
	double dt = GetElapsedSeconds(start, end);
	g_totalSeconds += dt;
	printf(" done! That took %3.3f seconds.\n", dt);

	printf("\nTotal time: %3.3f seconds.\n\n", g_totalSeconds);

	std::system("start image.bmp");
	return 0;
}

// Init all the device memory and setup the scene.
void CudaInit()
{
	int rayCount = g_cfg.imageWidth * g_cfg.imageHeight;

	Material materials[7] = {};
	materials[0].emitColor = make_float3(0.1f, 0.4f, 0.5f) * 0.0f;
	materials[1].albedo = make_float3(0.5f, 0.5f, 0.5f);
	materials[1].roughness = 0.75f;
	materials[2].emitColor = make_float3(0.7f, 0.5f, 0.3f) * 100.0f;
	materials[2].roughness = 0.75f;
	materials[3].albedo = make_float3(0.9f, 0.1f, 0.1f);
	materials[3].roughness = 1.0f;
	materials[4].albedo = make_float3(0.2f, 0.8f, 0.2f);
	materials[4].roughness = 0.0f;
	materials[5].emitColor = make_float3(0.1f, 1.0f, 0.1f) * 5.0f;
	materials[6].albedo = make_float3(1.0f, 1.0f, 1.0f);
	materials[6].refractionIndex = 1.5f;

	Plane planes[5] = {};
	float3 zero = make_float3(0.0f, 0.0f, 0.0f);
	float3 right = make_float3(1.0f, 0.0f, 0.0f) * 3.0f;
	float3 up = make_float3(0.0f, 0.0f, 1.0f) * -3.0f;
	float3 forward = make_float3(0.0f, -1.0f, 0.0f) * 3.0f;
	// Floor
	planes[0] = CreatePlane(-up, -up - right, -up + forward);
	planes[0].materialIndex = 1;
	// Left wall
	planes[1] = CreatePlane(right, right - up, right + forward);
	planes[1].materialIndex = 1;

	// Right wall
	planes[2] = CreatePlane(-right, -right + up, -right + forward);
	planes[2].materialIndex = 1;

	// Back wall
	planes[3] = CreatePlane(forward, forward + up, forward + right);
	planes[3].materialIndex = 1;

	// Ceiling
	planes[4] = CreatePlane(up, up + right, up + forward);
	planes[4].materialIndex = 1;

	Sphere spheres[7] = {};
	spheres[0].position = make_float3(0.0f, 0.0f, 3.75f);
	spheres[0].radius = 1.0f;
	spheres[0].materialIndex = 2;

	spheres[1].position = make_float3(2.0f, -2.0f, 0.0f);
	spheres[1].radius = 1.0f;
	spheres[1].materialIndex = 3;

	spheres[2].position = make_float3(-2.0f, -1.0f, 2.0f);
	spheres[2].radius = 1.0f;
	spheres[2].materialIndex = 4;

	spheres[3].position = make_float3(0.0f, -1.0f, 0.0f);
	spheres[3].radius = 0.75f;
	spheres[3].materialIndex = 6;

	spheres[4].position = make_float3(0.0f, -1.0f, 0.0f);
	spheres[4].radius = 0.7f;
	spheres[4].materialIndex = 6;

	spheres[5].position = make_float3(0.0f, 2.0f, 0.0f);
	spheres[5].radius = 2.0f;
	spheres[5].materialIndex = 3;

	spheres[6].position = make_float3(0.0f, -50.0f, 0.0f);
	spheres[6].radius = 2.0f;
	spheres[6].materialIndex = 2;


	World w = {};
	w.materialCount = sizeof(materials) / sizeof(materials[0]);
	w.sphereCount = sizeof(spheres) / sizeof(spheres[0]);
	w.planeCount = sizeof(planes) / sizeof(planes[0]);
	w.rayCount = rayCount;

	cudaCall(cudaMalloc(&d_world, sizeof(World)));
	cudaCall(cudaMalloc(&w.materials, sizeof(materials)));
	cudaCall(cudaMalloc(&w.planes, sizeof(planes)));
	cudaCall(cudaMalloc(&w.spheres, sizeof(spheres)));
	cudaCall(cudaMalloc(&w.rays, sizeof(Ray) * rayCount));
	cudaCall(cudaMalloc(&w.intersections,
		sizeof(Intersection) * rayCount));
	g_deviceIntersections = thrust::device_pointer_cast(w.intersections);

	cudaCall(cudaMemcpy(d_world, &w, sizeof(World),
		cudaMemcpyHostToDevice));
	cudaCall(cudaMemcpy(w.materials, &materials, sizeof(materials),
		cudaMemcpyHostToDevice));
	cudaCall(cudaMemcpy(w.planes, &planes, sizeof(planes),
		cudaMemcpyHostToDevice));
	cudaCall(cudaMemcpy(w.spheres, &spheres, sizeof(spheres),
		cudaMemcpyHostToDevice));

	Camera cam = {};
	cam.position = make_float3(0.0f, g_cfg.cameraDistance, 0.0f);
	cam.forward = Normalized(cam.position);
	cam.right = Normalized(Cross(make_float3(0.0f, 0.0f, 1.0f),
		cam.forward));
	cam.up = Normalized(Cross(cam.forward, cam.right));
	cam.film = make_float3(g_cfg.filmWidth, g_cfg.filmHeight,
		g_cfg.filmDistance);
	
	DeviceImage image = {};
	image.width = g_cfg.imageWidth;
	image.height = g_cfg.imageHeight;

	if (image.width > image.height)
	{
		cam.film.y = cam.film.x *
			((float)image.height / (float)image.width);
	}
	else if (image.width < image.height)
	{
		cam.film.x = cam.film.y*
			((float)image.width / (float)image.height);
	}

	cudaCall(cudaMalloc(&d_camera, sizeof(Camera)));
	cudaCall(cudaMemcpy(d_camera, &cam, sizeof(Camera),
		cudaMemcpyHostToDevice));


	cudaCall(cudaMalloc(&image.pixels,
		sizeof(float3) * rayCount));
	g_deviceImage = image.pixels;
	// init image to black
	cudaCall(cudaMemset(g_deviceImage, 0,
		sizeof(float3) * rayCount));


	cudaCall(cudaMalloc(&d_image, sizeof(DeviceImage)));
	cudaCall(cudaMemcpy(d_image, &image, sizeof(DeviceImage),
		cudaMemcpyHostToDevice));

	cudaCall(cudaMalloc(&d_randStates,
		sizeof(curandState) * rayCount));
}

// Renders the scene.
// After completion the back buffer will be filled with the image.
void Render()
{
#ifndef _DEBUG
	// Release
	int threadCount = 1024;
#else
	// Debug
	// In debug mode curand_init() requires an insane stack frame
	// so we need to limit the block size.
	int threadCount = 128; 
#endif 

	// Load config
	int imageWidth = g_cfg.imageWidth;
	int imageHeight = g_cfg.imageHeight;
	int rayCount = imageWidth * imageHeight;
	int maxBounces = g_cfg.maxBounces;
	int samplesPerPixel = g_cfg.samplesPerPixel;
	float contributionPerPixel = 1.0f / (float)samplesPerPixel;
	bool bSortIntersections = g_cfg.bSortIntersections;

	int blockCount = imageWidth * imageHeight / threadCount + 1;


	// Init curandStates:
	printf("Initializing cuRand states... ");
	TimeStamp startTime = std::chrono::high_resolution_clock::now();
	InitCurandKernel << <blockCount, threadCount >> >
		(rayCount, d_randStates, 1234, g_cfg.bUseFastRand);
	cudaCall(cudaDeviceSynchronize());
	TimeStamp endTime = std::chrono::high_resolution_clock::now();
	double dt = GetElapsedSeconds(startTime, endTime);
	g_totalSeconds += dt;
	printf(" done! That took %3.3f seconds.\n", dt);


	// Do the path tracing loop
	printf("Ray casting... ");
	startTime = std::chrono::high_resolution_clock::now();
	for (int s = 0; s < samplesPerPixel; s++)
	{
		GeneratePrimaryRaysKernel << <blockCount, threadCount >> >
			(d_image, d_world, d_camera, maxBounces, d_randStates);

		// iterate over bounces
		for (int i = 0; i < maxBounces; i++)
		{
			ComputeIntersectionsKernel << <blockCount, threadCount >> >
				(d_image, d_world);

			if (bSortIntersections)
			{
				thrust::sort(g_deviceIntersections,
					g_deviceIntersections + rayCount);
			}
			ShadeIntersectionsKernel << <blockCount, threadCount >> >
				(d_image, d_world, d_randStates,bSortIntersections);
		}

		WriteRayColorToImage << <blockCount, threadCount >> >
			(d_image, d_world, contributionPerPixel);
	}

	cudaCall(cudaDeviceSynchronize());
	endTime = std::chrono::high_resolution_clock::now();
	dt = GetElapsedSeconds(startTime, endTime);
	g_totalSeconds += dt;
	printf(" done! That took %3.3f seconds.\n", dt);
}

//////////////////////////////////////////////////////////////////////////
// #Kernels
//////////////////////////////////////////////////////////////////////////

// Initialize the curandStates for later use.
__global__ void InitCurandKernel(int rayCount,
	curandState* randStates, int seed, bool bUseFastRand)
{
	int id = GetIndex();
	if (id >= rayCount) return;


	if (bUseFastRand)
	{
		// #hack 
		// https://forums.developer.nvidia.com/t/curand-initialization-time/19758/3
		// supposedly much faster init, with worse random numbers?
		curand_init((seed << 20) + id, 0, 0, &randStates[id]);
	}
	else {
		curand_init(seed, id, 0, &randStates[id]);
	}
}

// Generate the primary Rays (the ones coming from the camera).
__global__ void GeneratePrimaryRaysKernel(DeviceImage* image,
	World* world, Camera* camera, int bounces,
	curandState* randStates)
{
	int id = GetIndex();
	if (IsOutOfBounds(id, image)) return;

	// Pixel indices
	int pixelX = id % image->width;
	int pixelY = id / image->width;

	// Normalized Pixel indices = pixelX: [0,imageWidth], filmX [-1.0f,1.0f]
	float filmX = -1.0f + 2.0f *
		((float)pixelX / (float)image->width);
	float filmY = -1.0f + 2.0f *
		((float)pixelY / (float)image->height);

	// Half sizes of a pixel
	float halfPixelWidth = 0.5f * (1.0f / (float)image->width);
	float halfPixelHeight = 0.5f * (1.0f / (float)image->height);

	// x,y = width, height; z = distance
	float3 film = camera->film;
	// Jitter the samples by [-1.0f,1.0f] * halfPixel
	filmX += RandomBilateral(&randStates[id]) * halfPixelWidth;
	filmY += RandomBilateral(&randStates[id]) * halfPixelHeight;

	// Ray target location on the camera film
	float3 filmCenter = camera->position - camera->forward * film.z;
	float3 filmP = filmCenter +
		filmX * camera->right * film.x * 0.5f +
		filmY * camera->up * film.y * 0.5f;

	// Construct the actual ray
	Ray r = {};
	r.origin = camera->position;
	r.direction = Normalized(filmP - camera->position);
	r.bounces = bounces;
	r.color = make_float3(1.0f, 1.0f, 1.0f);
	world->rays[id] = r;
}

// Compute the intersections between the current rays and the scene.
__global__ void ComputeIntersectionsKernel(DeviceImage* image,
	World* world)
{
	int id = GetIndex();
	if (IsOutOfBounds(id, image))return;

	Ray ray = world->rays[id];
	if (ray.bounces <= 0) return;

	Intersection closestHit = {};
	closestHit.t = FLT_MAX;

	// Test against all planes
	for (int i = 0; i < world->planeCount; i++)
	{
		float t;
		Plane plane = world->planes[i];
		bool isHit = IntersectPlane(ray, plane, t);
		if (isHit && t < closestHit.t)
		{
			closestHit.t = t;
			closestHit.material = plane.materialIndex;
			closestHit.normal = plane.normal;
		}
	}
	// Test against all spheres
	for (int i = 0; i < world->sphereCount; i++)
	{
		float t;
		Sphere sphere = world->spheres[i];
		bool isHit = IntersectSphere(ray, sphere, t);
		if (isHit && t < closestHit.t)
		{
			closestHit.t = t;
			closestHit.material = sphere.materialIndex;
			closestHit.normal = GetSphereNormal(ray, sphere, t);
		}
	}
	closestHit.id = id;
	world->intersections[id] = closestHit;
}

// Shade the intersections -> i.e. attenuate the ray color and generate 
// the next generation of bounces in place
__global__ void ShadeIntersectionsKernel(DeviceImage* image,
	World* world, curandState* randStates, bool bSorted)
{
	int id = GetIndex();
	if (IsOutOfBounds(id, image)) return;

	Intersection intersection = world->intersections[id];

	int rayId = bSorted ? intersection.id : id;

	Ray r = world->rays[rayId];
	if (r.bounces <= 0) return;

	Material mat = world->materials[intersection.material];
	
	// Decide on how to shade the intersection based on the material
	if (mat.refractionIndex > 0.0f)
	{
		// Refractive Material
		r.color *= mat.albedo; // should be around (1.0f,1.0f,1.0f)
		r.bounces -= 1;
		ReflectOrRefract(r, intersection.normal, &randStates[id],
			mat.refractionIndex);
	}
	else if (Length2(mat.emitColor) > 0.0f)
	{
		// Emissive Material, i.e. light source
		r.color *= mat.emitColor;
		r.bounces = -1;
	}
	else
	{
		// Diffuse Material
		Reflect(r, intersection.t, intersection.normal, &randStates[id],
			mat.roughness);
		r.color *= (mat.albedo + mat.emitColor);
		r.bounces -= 1;
	}
	world->rays[rayId] = r;
}

// Write the paths attenuated color to the back buffer
// The color is weighted by a contribution factor.
__global__ void WriteRayColorToImage(DeviceImage* image,
	World* world, float contributionPerPixel)
{
	int id = GetIndex();
	if (IsOutOfBounds(id, image)) return;

	Ray r = world->rays[id];
	image->pixels[id] += r.color * contributionPerPixel;
}