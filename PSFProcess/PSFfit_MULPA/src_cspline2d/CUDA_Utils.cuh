#include <stdio.h>
#include <cuda_runtime.h>
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH


void cudasafe(cudaError_t err, const char* str, int lineNumber)
{
	if (err != cudaSuccess)
	{
		//reset all cuda devices
		int deviceCount = 0;
		int ii = 0;
		cudasafe(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", __LINE__); //query number of GPUs
		for (ii = 0; ii < deviceCount; ii++) {
			cudaSetDevice(ii);
			cudaDeviceReset();
		}
		printf("%s failed with error code %i at line %d\n", str, err, lineNumber);
		exit(1); 
	}
}


void CUDAERROR(const char *instr, int lineNumber)
{
	cudaError_t errornum;
	const char *str;
	if (errornum = cudaGetLastError()) {
		//reset all cuda devices
		int deviceCount = 0;
		int ii = 0;
		cudasafe(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", __LINE__); //query number of GPUs
		for (ii = 0; ii < deviceCount; ii++) {
			cudaSetDevice(ii);
			cudaDeviceReset();
		}
		str = cudaGetErrorString(errornum);
		cudaDeviceReset();
		printf("cudaGetLastError code=%i: %s in %s\n", errornum, str, instr);
		exit(1); // might not stop matlab
	}
}


void cudaavailable(int silent) 
{
	int driverVersion = 0, runtimeVersion = 0, deviceCount = 0;
    // driver version
    if (cudaSuccess == cudaDriverGetVersion(&driverVersion)) 
        if (silent == 0)
			printf("CUDA driver version: %d\n", driverVersion);
    else  
        printf("Could not query CUDA driver version\n");

    // runtime version
    if (cudaSuccess == cudaRuntimeGetVersion(&runtimeVersion))
		if (silent == 0)
			printf("CUDA rt version: %d\n", runtimeVersion);
    else 
		printf("Could not query CUDA runtime version\n");
	
    // device count    
	if (cudaSuccess == cudaGetDeviceCount(&deviceCount))
		if (silent == 0)
			printf("CUDA devices detected: %d\n", deviceCount);
    else
        printf("Could not query CUDA device count\n");
        
	if (deviceCount < 1)
		printf("No CUDA capable devices were detected");
}

#endif