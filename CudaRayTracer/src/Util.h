#pragma once
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdio.h>
#include "OpenImageDenoise\oidn.h"
#include "stb_image_writer.h"
#include "Structs.h"
#include <chrono>

typedef std::chrono::steady_clock::time_point TimeStamp;

#ifndef _DEBUG
// Release
#define cudaCall(x) x
#else
// Debug
#define cudaCall(x) (assert(x == cudaSuccess))
#endif 

#define GetIndex() (threadIdx.x + blockIdx.x * blockDim.x)
#define IsOutOfBounds(id, image) (id >= image->width * image->height)

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
	printf("\tMaximum bounces per path: %d\n", cfg.maxBounces);
	const char en[] = "enabled";
	const char dis[] = "disabled";
	const char* denoiseString = cfg.bDenoise ? en : dis;
	printf("\tDenoising is %s.\n", denoiseString);
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

void WriteBMP(Config cfg, float3* deviceImage)
{
	

	int rayCount = cfg.imageWidth * cfg.imageHeight;
	size_t floatImageSize = sizeof(float3) * rayCount;
	
	// OIDN does not like pinned memory
	float3* fpixels = (float3*)malloc(floatImageSize);
	//cudaCall(cudaMallocHost(&fpixels,floatImageSize));
	int* pixels = (int*)malloc(sizeof(int) * rayCount);
	cudaCall(cudaMemcpy(fpixels, deviceImage,
		floatImageSize, cudaMemcpyDeviceToHost));

	int imageWidth = cfg.imageWidth;
	int imageHeight = cfg.imageHeight;

	// #Denoiser
	if (cfg.bDenoise)
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
}

Plane CreatePlane(const float3& a, const float3& b, const float3& c)
{
	Plane p;
	p.normal = Normalized(Cross(b - a, c - a));
	p.d = Dot(p.normal, a);
	p.materialIndex = 0;
	return p;
}

double GetElapsedSeconds(TimeStamp& start,TimeStamp& end)
{
	__int64 mili = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	return (double)mili / 1000.0;
}