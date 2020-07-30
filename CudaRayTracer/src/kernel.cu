#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Structs.h"
#include "IntersectionTests.h"
#include "stb_image_writer.h"

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <float.h>
#include <chrono>

#include <curand.h>
#include <curand_kernel.h>

#include "OpenImageDenoise\oidn.h"

#define DENOISE_FINAL_IMAGE

#define ARRAY_LENGTH(arr) (sizeof(arr)/sizeof((arr)[0]))

#ifndef _DEBUG
// Release
#define cudaCall(x) x
#else
// Debug
#define cudaCall(x) (assert(x == cudaSuccess))
#endif 

#define GetIndex() (threadIdx.x + blockIdx.x * blockDim.x)
#define IsOutOfBounds(id, image) (id >= image->width * image->height)

Config g_cfg;
int g_rayCount;
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
	curandState* randStates, int seed);
__global__ void ShadeIntersectionsKernel(DeviceImage* image,
	World* world, curandState* randStates);
__global__ void WriteRayColorToImage(DeviceImage* image,
	World* world, float contributionPerPixel);

void LoadConfig(const char* fileName, Config& cfg)
{
	cfg.imageWidth = GetPrivateProfileInt("Image", "imageWidth", 
		1024, fileName);
	cfg.imageHeight = GetPrivateProfileInt("Image", "imageHeight",
		720, fileName);

	cfg.samplesPerPixel = GetPrivateProfileInt("RT", 
		"samplesPerPixel", 32, fileName);
	cfg.maxBounces = GetPrivateProfileInt("RT", "maxBounces",
		16, fileName);
	cfg.bDenoise = GetPrivateProfileInt("RT", "bDenoise",
		1, fileName);


	printf("Starting the Path tracer with the following configuration:\n\n");
	printf("\tImage: %dx%d\n", cfg.imageWidth, cfg.imageHeight);
	printf("\tSamples per pixel: %d\n", cfg.samplesPerPixel);
	printf("\tMaximum bounces per path: %d\n",cfg.maxBounces);
	const char en[] = "enabled";
	const char dis[] = "disabled";
	const char* denoiseString = cfg.bDenoise ? en : dis;
	printf("\tDenoising is %s\n", denoiseString);
	printf("\n");
}

// float3 to packed ABGR
int PackColor(float3 color)
{
	int r = (int)(255.9f * color.x);
	int g = (int)(255.9f * color.y);
	int b = (int)(255.9f * color.z);

	int result = 0xFF000000; // alpha
	result += (unsigned char)b << 16;
	result += (unsigned char)g << 8;
	result += (unsigned char)r;
	return result;
}

void WriteBMP()
{
	printf("Writing image to file...");
	auto startTime = std::chrono::high_resolution_clock::now();

	size_t floatImageSize = sizeof(float3) * g_rayCount;
	// OIDN seems not to like pinned memory - difference is just 
	// ~1.5s on 1920x1080 dbg
	float3* fpixels = (float3*)malloc(floatImageSize);
	//cudaCall(cudaMallocHost(&fpixels,floatImageSize));
	int* pixels = (int*)malloc(sizeof(int) * g_rayCount);
	cudaCall(cudaMemcpy(fpixels, g_deviceImage,
		floatImageSize, cudaMemcpyDeviceToHost));

	int imageWidth = g_cfg.imageWidth;
	int imageHeight = g_cfg.imageHeight;

	// #Denoiser
	if (g_cfg.bDenoise)
	{
		OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
		oidnCommitDevice(device);

		OIDNFilter filter = oidnNewFilter(device, "RT");
		oidnSetSharedFilterImage(filter, "color", fpixels,
			OIDN_FORMAT_FLOAT3, imageWidth, imageHeight, 0, 0, 0);
		oidnSetSharedFilterImage(filter, "output", fpixels,
			OIDN_FORMAT_FLOAT3, imageWidth, imageHeight, 0, 0, 0);
		oidnSetFilter1b(filter, "hdr", false);
		oidnCommitFilter(filter);

		oidnExecuteFilter(filter);

		const char* errorMessage;
		if (oidnGetDeviceError(device, &errorMessage) != 
			OIDN_ERROR_NONE)
		{
			printf("Error: %s\n", errorMessage);
		}
		oidnReleaseFilter(filter);
		oidnReleaseDevice(device);
	}

	for (int y = 0; y < imageHeight; y++)
	{
		for (int x = 0; x < imageWidth; x++)
		{
			int id = x + imageWidth * y;
			pixels[id] = PackColor(fpixels[id]);
		}
	}

	// 4 -> ABGR
	stbi_flip_vertically_on_write(true);
	stbi_write_bmp("image.bmp", imageWidth, imageHeight, 4,
		(void*)pixels);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto mili = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	printf(" done! That took %3.3f seconds.\n",(double)mili/1000.0);
}

Plane CreatePlane(const float3& a, const float3& b, const float3& c)
{
	Plane p;
	p.normal = Normalized(Cross(b - a, c - a));
	p.d = Dot(p.normal, a);
	p.materialIndex = 0;
	return p;
}

int main()
{
	LoadConfig("./rt.ini",g_cfg);

	CudaInit();
	Raytrace();
	cudaCall(cudaDeviceSynchronize());
	WriteBMP();
	
	std::system("start image.bmp");
	return 0;
}

void CudaInit()
{
	g_rayCount = g_cfg.imageWidth * g_cfg.imageHeight;

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
	w.rayCount = g_rayCount;

	cudaCall(cudaMalloc(&d_world, sizeof(World)));
	cudaCall(cudaMalloc(&w.materials, sizeof(materials)));
	cudaCall(cudaMalloc(&w.planes, sizeof(planes)));
	cudaCall(cudaMalloc(&w.spheres, sizeof(spheres)));
	cudaCall(cudaMalloc(&w.rays, sizeof(Ray) * g_rayCount));
	cudaCall(cudaMalloc(&w.intersections,
		sizeof(Intersection) * g_rayCount));


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
		sizeof(float3) * g_rayCount));
	g_deviceImage = image.pixels;
	// init image to black
	cudaCall(cudaMemset(g_deviceImage, 0,
		sizeof(float3) * g_rayCount));


	cudaCall(cudaMalloc(&d_image, sizeof(DeviceImage)));
	cudaCall(cudaMemcpy(d_image, &image, sizeof(DeviceImage),
		cudaMemcpyHostToDevice));

	cudaCall(cudaMalloc(&d_randStates,
		sizeof(curandState) * g_rayCount));
}

__global__ void InitCurandKernel(int rayCount,
	curandState* randStates, int seed)
{
	int id = GetIndex();
	if (id >= rayCount) return;

	curand_init(seed, id, 0, &randStates[id]);
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

	int blockCount = imageWidth * imageHeight / threadCount + 1;

	printf("Initializing cuRand states... ");
	// #Time
	auto startTime = std::chrono::high_resolution_clock::now();
	InitCurandKernel << <blockCount, threadCount >> > (g_rayCount, d_randStates, 1234);
	cudaCall(cudaDeviceSynchronize());
	auto endTime = std::chrono::high_resolution_clock::now();
	auto mili = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	printf(" done! That took %3.3f seconds.\n", (double)mili / 1000.0);

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
	mili = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	printf(" done! That took %3.3f seconds.\n", (double)mili / 1000.0);
}
// #Todo remove unnecessary parameters from kernels (e.g. image)