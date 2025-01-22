#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "CUDA_Utils.cuh"

__global__ void myadd(int NFits, float* arr_a, float* arr_b, float* arr_c)
{
    unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    float *dum = nullptr;

    if (Idx < NFits) {
        dum = (arr_b) ? arr_b + Idx : nullptr;
        if (dum)
            arr_c[Idx] = arr_a[Idx] + *dum;
        else
            arr_c[Idx] = arr_a[Idx] + 100.0f;
    }
    __syncthreads();
    return;
}



static inline int intceil(int n, int m) { return (n - 1) / m + 1; }
static inline dim3 dimGrid1(int n, int blocksz) { dim3 gridsz(intceil(n, blocksz), 1, 1); return gridsz; }
static inline dim3 dimBlock1(int n) { dim3 blocksz(n, 1, 1); return blocksz; }


int main()
{
    int NFits = 10;
    dim3 grids = dimGrid1(NFits, 64);
    dim3 blcks = dimBlock1(64);

    float *h_arr_a = (float*)malloc(NFits * sizeof(float));
    float *h_arr_b = (float*)malloc(NFits * sizeof(float));
    float *h_arr_c = (float*)malloc(NFits * sizeof(float));
    float *h_arr_d = (float*)malloc(NFits * sizeof(float));
    for (unsigned int i = 0; i < NFits; i++) {
        h_arr_a[i] = (float)i;
        h_arr_b[i] = 10.0f - (float)i;
    }
    memset(h_arr_c, 0, NFits * sizeof(float));
    memset(h_arr_d, 0, NFits * sizeof(float));

    float *d_arr_a = nullptr, *d_arr_b = nullptr, *d_arr_c = nullptr, *d_arr_d = nullptr;
    cudasafe(cudaMalloc((void**)&d_arr_a, NFits * sizeof(float)), "malloc d_arr_a", __LINE__);
    cudasafe(cudaMalloc((void**)&d_arr_b, NFits * sizeof(float)), "malloc d_arr_b", __LINE__);
    cudasafe(cudaMalloc((void**)&d_arr_c, NFits * sizeof(float)), "malloc d_arr_c", __LINE__);
    cudasafe(cudaMalloc((void**)&d_arr_d, NFits * sizeof(float)), "malloc d_arr_d", __LINE__);
    
    cudasafe(cudaMemcpy(d_arr_a, h_arr_a, NFits * sizeof(float), cudaMemcpyHostToDevice), "transfer h_arr_a to device", __LINE__);
    cudasafe(cudaMemcpy(d_arr_b, h_arr_b, NFits * sizeof(float), cudaMemcpyHostToDevice), "transfer h_arr_b to device", __LINE__);
    myadd<<<grids, blcks>>>(NFits, d_arr_a, d_arr_b, d_arr_c);
    myadd<<<grids, blcks>>>(NFits, d_arr_a, nullptr, d_arr_d);
    cudasafe(cudaMemcpy(h_arr_c, d_arr_c, NFits * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_arr_c to host", __LINE__);
    cudasafe(cudaMemcpy(h_arr_d, d_arr_d, NFits * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_arr_d to host", __LINE__);

    printf("arr_c: ");
    for (unsigned int i = 0; i < NFits - 1; i++)
        printf("%.1f, ", h_arr_c[i]);
    printf("%.1f\n", h_arr_c[NFits - 1]);

    printf("arr_d: ");
    for (unsigned int i = 0; i < NFits - 1; i++)
        printf("%.1f, ", h_arr_d[i]);
    printf("%.1f\n", h_arr_d[NFits - 1]);

    cudasafe(cudaFree(d_arr_a), "free d_arr_a on device", __LINE__);
    cudasafe(cudaFree(d_arr_b), "free d_arr_b on device", __LINE__);
    cudasafe(cudaFree(d_arr_c), "free d_arr_c on device", __LINE__);
    cudasafe(cudaFree(d_arr_d), "free d_arr_d on device", __LINE__);

    free(h_arr_a);
    free(h_arr_b);
    free(h_arr_c);
    free(h_arr_d);
    return 0;
}