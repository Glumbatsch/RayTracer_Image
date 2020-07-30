#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Structs.h"
#include "IntersectionTests.h"
#include "stb_image_writer.h"

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <float.h>

#include <curand.h>
#include <curand_kernel.h>

#include "OpenImageDenoise\oidn.h"

#define DENOISE_FINAL_IMAGE

#ifndef _DEBUG
// Release
#define cudaCall(x) x
#else
// Debug
#define cudaCall(x) assert(x == cudaSuccess)
#endif 

#define GetIndex() threadIdx.x + blockIdx.x * blockDim.x
#define IsOutOfBounds(id, image) id >= image->width * image->height

__global__ void DebugRaysKernel(World* world)
{
	int a = 1;
	int b = 2;
	int c = a++ + b;
}


__global__ void PrintMaterialsKernel(World* world)
{
	int count = world->materialCount;
	for (int i = 0; i < count; i++)
	{
		Material m = world->materials[i];
		printf("Material %d: \n", i);
		printf("\t Albedo: %1.3f %1.3f %1.3f\n", m.albedo.x, m.albedo.y, m.albedo.z);
		printf("\t EmitColor: %1.3f %1.3f %1.3f\n", m.emitColor.x, m.emitColor.y, m.emitColor.z);
		printf("\t Roughness: %1.3f\n\n", m.roughness);
	}
}

int g_rayCount;
float3* g_deviceImage;
// Device Stuff
World* d_world;
Camera* d_camera;
DeviceImage* d_image;
curandState* d_randStates;

void Raytrace(int imageWidth, int imageHeight);
void CudaInit(int imageWidth, int imageHeight);

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

void WriteBMP(int imageHeight, int imageWidth)
{
	size_t floatImageSize = sizeof(float3) * g_rayCount;
	// OIDN seems not to like pinned memory - difference is just 
	// ~1.5s on 1920x1080 dbg
	float3* fpixels = (float3*)malloc(floatImageSize);
	//cudaCall(cudaMallocHost(&fpixels,floatImageSize));
	int* pixels = (int*)malloc(sizeof(int) * g_rayCount);
	cudaCall(cudaMemcpy(fpixels, g_deviceImage,
		floatImageSize, cudaMemcpyDeviceToHost));

#ifdef DENOISE_FINAL_IMAGE 
	// #Denoiser
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
	if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
	{
		printf("Error: %s\n", errorMessage);
	}
	oidnReleaseFilter(filter);
	oidnReleaseDevice(device);
#endif
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
}

int main()
{
	int imageWidth = 1920;
	int imageHeight = 1080;

	CudaInit(imageWidth, imageHeight);
	Raytrace(imageWidth, imageHeight);
	cudaCall(cudaDeviceSynchronize());
	printf("Writing image to file...");
	WriteBMP(imageHeight, imageWidth);
	printf(" done!\n");
	std::system("start image.bmp");
	return 0;
}

void CudaInit(int imageWidth, int imageHeight)
{
	g_rayCount = imageWidth * imageHeight;

	Material materials[3] = {};
	materials[0].emitColor = make_float3(0.1f, 0.4f, 0.5f);
	materials[1].albedo = make_float3(0.5f, 0.5f, 0.5f);
	materials[1].roughness = 0.75f;
	materials[2].albedo = make_float3(0.7f, 0.5f, 0.3f);
	materials[2].roughness = 0.75f;

	Plane planes[1] = {};

	// x-y plane in the origin
	planes[0].normal = make_float3(0.0f, 0.0f, 1.0f);
	planes[0].d = 0;
	planes[0].materialIndex = 1;

	Sphere spheres[2] = {};
	spheres[0].position = make_float3(0.0f, 0.0f, 0.0f);
	spheres[0].radius = 0.75f;
	spheres[0].materialIndex = 2;

	spheres[1].position = make_float3(3.0f, -2.0f, 0.0f);
	spheres[1].radius = 0.75f;
	spheres[1].materialIndex = 2;


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
	//PrintMaterialsKernel << <1, 1 >> > (d_world);
	cudaCall(cudaMemcpy(w.planes, &planes, sizeof(planes),
		cudaMemcpyHostToDevice));
	cudaCall(cudaMemcpy(w.spheres, &spheres, sizeof(spheres),
		cudaMemcpyHostToDevice));

	Camera cam = {};
	cam.position = make_float3(0.0f, -10.0f, 1.0f);
	cam.forward = Normalized(cam.position);
	cam.right = Normalized(Cross(make_float3(0.0f, 0.0f, 1.0f),
		cam.forward));
	cam.up = Normalized(Cross(cam.forward, cam.right));

	cudaCall(cudaMalloc(&d_camera, sizeof(Camera)));
	cudaCall(cudaMemcpy(d_camera, &cam, sizeof(Camera),
		cudaMemcpyHostToDevice));

	DeviceImage image = {};
	image.width = imageWidth;
	image.height = imageHeight;
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
	r.color *= (mat.albedo + mat.emitColor);

	Reflect(r, intersection.t, intersection.normal, &randStates[id],
		mat.roughness);
	if (Length2(mat.emitColor) > 0.0f)
	{
		r.bounces = -1;
	}
	r.bounces -= 1;
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

void Raytrace(int imageWidth, int imageHeight)
{
	// #Note With a higher block size the InitCurandKernel launch will fail because it requires an insane ~6kb stack frame...
	int threadCount = 128;
	int blockCount = imageWidth * imageHeight / threadCount + 1;
	printf("Initializing cuRand states... ");
	InitCurandKernel << <blockCount, threadCount >> > (g_rayCount, d_randStates, 1234);
	cudaCall(cudaDeviceSynchronize());
	printf("done! \n");

	int maxBounces = 8;
	int samplesPerPixel = 16;
	float contributionPerPixel = 1.0f / (float)samplesPerPixel;
	printf("Ray casting... ");

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
	printf("done!\n");
}
// #Todo remove unnecessary parameters from kernels (e.g. image)