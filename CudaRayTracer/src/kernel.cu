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

#include "OpenImageDenoise\oidn.h"


double g_totalSeconds = 0.0;
Config g_cfg;
float3* g_deviceImage;
// Device Stuff
World* d_world;
Camera* d_camera;
DeviceImage* d_image;
curandState* d_randStates;

void Raytrace();
void CudaInit();

__global__ void GeneratePrimaryRaysKernel(DeviceImage* image,
	World* world, Camera* camera, int bounces,
	curandState* randStates);
__global__ void ComputeIntersectionsKernel(DeviceImage* image,
	World* world);
__global__ void InitCurandKernel(int rayCount,
	curandState* randStates, int seed, bool bUseFastRand);
__global__ void ShadeIntersectionsKernel(DeviceImage* image,
	World* world, curandState* randStates);
__global__ void WriteRayColorToImage(DeviceImage* image,
	World* world, float contributionPerPixel);

int main()
{
	LoadConfig("./rt.ini",g_cfg);

	CudaInit();
	Raytrace();
	cudaCall(cudaDeviceSynchronize());
	printf("Writing image to file...");
	TimeStamp start = std::chrono::high_resolution_clock::now();
	WriteBMP(g_cfg,g_deviceImage);
	TimeStamp end = std::chrono::high_resolution_clock::now();
	double dt = GetElapsedSeconds(start,end);
	g_totalSeconds += dt;
	printf(" done! That took %3.3f seconds.\n", dt);

	printf("\nTotal time: %3.3f seconds.\n\n", g_totalSeconds);

	Sleep(1000); // #hack sometimes os can't open the image
	std::system("start image.bmp");
	return 0;
}

void CudaInit()
{
	int rayCount = g_cfg.imageWidth * g_cfg.imageHeight;

	Material materials[6] = {};
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

	Sphere spheres[3] = {};
	spheres[0].position = make_float3(0.0f, 0.0f, 3.75f);
	spheres[0].radius = 1.0f;
	spheres[0].materialIndex = 2;

	spheres[1].position = make_float3(2.0f, -2.0f, 0.0f);
	spheres[1].radius = 1.0f;
	spheres[1].materialIndex = 3;

	spheres[2].position = make_float3(-2.0f, -1.0f, 2.0f);
	spheres[2].radius = 1.0f;
	spheres[2].materialIndex = 4;

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


	cudaCall(cudaMemcpy(d_world, &w, sizeof(World),
		cudaMemcpyHostToDevice));
	cudaCall(cudaMemcpy(w.materials, &materials, sizeof(materials),
		cudaMemcpyHostToDevice));
	cudaCall(cudaMemcpy(w.planes, &planes, sizeof(planes),
		cudaMemcpyHostToDevice));
	cudaCall(cudaMemcpy(w.spheres, &spheres, sizeof(spheres),
		cudaMemcpyHostToDevice));

	Camera cam = {};
	cam.position = make_float3(0.0f, -12.5f, 0.0f);
	cam.forward = Normalized(cam.position);
	cam.right = Normalized(Cross(make_float3(0.0f, 0.0f, 1.0f),
		cam.forward));
	cam.up = Normalized(Cross(cam.forward, cam.right));

	cudaCall(cudaMalloc(&d_camera, sizeof(Camera)));
	cudaCall(cudaMemcpy(d_camera, &cam, sizeof(Camera),
		cudaMemcpyHostToDevice));

	DeviceImage image = {};
	image.width = g_cfg.imageWidth;
	image.height = g_cfg.imageHeight;
	image.filmWidth = 1.0f;
	image.filmHeight = 1.0f;

	if (image.width > image.height)
	{
		image.filmHeight = image.filmWidth *
			((float)image.height / (float)image.width);
	}
	else if (image.width < image.height)
	{
		image.filmWidth = image.filmHeight *
			((float)image.width / (float)image.height);
	}

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

__global__ void InitCurandKernel(int rayCount,
	curandState* randStates, int seed,bool bUseFastRand)
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

__global__ void GeneratePrimaryRaysKernel(DeviceImage* image,
	World* world, Camera* camera, int bounces,
	curandState* randStates)
{
	int id = GetIndex();
	if (IsOutOfBounds(id, image)) return;

	float3 filmCenter = camera->position - camera->forward;

	int pixelX = id % image->width;
	int pixelY = id / image->width;

	float filmX = -1.0f + 2.0f *
		((float)pixelX / (float)image->width);
	float filmY = -1.0f + 2.0f *
		((float)pixelY / (float)image->height);

	float halfPixelWidth = 0.5f * (1.0f / (float)image->width);
	float halfPixelHeight = 0.5f * (1.0f / (float)image->height);

	filmX += RandomBilateral(&randStates[id]) * halfPixelWidth;
	filmY += RandomBilateral(&randStates[id]) * halfPixelHeight;

	float3 filmP = filmCenter +
		filmX * camera->right * image->filmWidth * 0.5f +
		filmY * camera->up * image->filmHeight * 0.5f;

	Ray r = {};
	r.origin = camera->position;
	r.direction = Normalized(filmP - camera->position);
	r.bounces = bounces;
	r.color = make_float3(1.0f, 1.0f, 1.0f);
	world->rays[id] = r;
}

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
	world->intersections[id] = closestHit;
}

__global__ void ShadeIntersectionsKernel(DeviceImage* image,
	World* world, curandState* randStates)
{
	int id = GetIndex();
	if (IsOutOfBounds(id, image)) return;

	Ray r = world->rays[id];
	if (r.bounces <= 0) return;

	Intersection intersection = world->intersections[id];
	Material mat = world->materials[intersection.material];

	if (Length2(mat.emitColor) > 0.0f)
	{
		r.color *= mat.emitColor;
		r.bounces = -1;
	}
	else
	{
		Reflect(r, intersection.t, intersection.normal, &randStates[id],
			mat.roughness);
		r.color *= (mat.albedo + mat.emitColor);
		r.bounces -= 1;
	}
	world->rays[id] = r;
}

__global__ void WriteRayColorToImage(DeviceImage* image,
	World* world, float contributionPerPixel)
{
	int id = GetIndex();
	if (IsOutOfBounds(id, image)) return;

	Ray r = world->rays[id];
	image->pixels[id] += r.color * contributionPerPixel;
}

void Raytrace()
{
	// #Note With a higher block size the InitCurandKernel launch will 
	// fail because it requires an insane ~6kb stack frame...
	int threadCount = 512;

	int imageWidth = g_cfg.imageWidth;
	int imageHeight = g_cfg.imageHeight;
	int rayCount = imageWidth * imageHeight;

	int blockCount = imageWidth * imageHeight / threadCount + 1;

	printf("Initializing cuRand states... ");
	// #Time
	TimeStamp startTime = std::chrono::high_resolution_clock::now();
	InitCurandKernel << <blockCount, threadCount >> >
		(rayCount, d_randStates, 1234, g_cfg.bUseFastRand);
	cudaCall(cudaDeviceSynchronize());
	TimeStamp endTime = std::chrono::high_resolution_clock::now();
	double dt = GetElapsedSeconds(startTime,endTime);
	g_totalSeconds += dt;
	printf(" done! That took %3.3f seconds.\n", dt);

	int maxBounces = g_cfg.maxBounces;
	int samplesPerPixel = g_cfg.samplesPerPixel;
	float contributionPerPixel = 1.0f / (float)samplesPerPixel;

	printf("Ray casting... ");
	// #Time
	startTime = std::chrono::high_resolution_clock::now();
	for (int s = 0; s < samplesPerPixel; s++)
	{
		GeneratePrimaryRaysKernel << <blockCount, threadCount >> >
			(d_image, d_world, d_camera, maxBounces, d_randStates);

		for (int i = 0; i < maxBounces; i++)
		{
			ComputeIntersectionsKernel << <blockCount, threadCount >> >
				(d_image, d_world);

			ShadeIntersectionsKernel << <blockCount, threadCount >> >
				(d_image, d_world, d_randStates);
		}

		WriteRayColorToImage << <blockCount, threadCount >> >
			(d_image, d_world, contributionPerPixel);
	}

	cudaCall(cudaDeviceSynchronize());
	endTime = std::chrono::high_resolution_clock::now();
	dt = GetElapsedSeconds(startTime,endTime);
	g_totalSeconds += dt;
	printf(" done! That took %3.3f seconds.\n", dt);
}