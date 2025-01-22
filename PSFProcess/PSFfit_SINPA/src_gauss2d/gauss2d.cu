#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "DLL_Macros.h"
#include "CUDA_Utils.cuh"

#include "definitions.h"
#include "kernels_1ch.cuh"
#include "kernels_2ch.cuh"

/*! @param[in]	NFits: 				int, number of PSF squares
    @param[in]	boxsz: 				int, size of the PSF square
	@param[in]	warpdeg:			int, the degree of the polynomial function that warps 2d locations from one channel to the other
    @param[in]	h_data: 			(NFits, nchannels, boxsz, boxsz) flattened, float, multi-channel PSF data
	@param[in]	h_var: 				(NFits, nchannels, boxsz, boxsz) flattened, float, multi-channel pixel-dependent variance of camera readout noise
	@param[in]	h_lu:				(NFits, nchannels, 2) flattened, int, [lc, uc] the left and upper corner of the PSF square in each channel (image coordinates)
    @param[in]	h_PSFsigmax: 		(nchannels) float, sigma (x-axis) of the Gaussian PSF
	@param[in]	h_PSFsigmay: 		(nchannels) float, sigma (y-axis) of the Gaussian PSF
	@param[in]	h_coeff_R2T:		(nchannels - 1, 2, warpdeg * warpdeg) flattened, float, the [coeffx_B2A, coeffy_B2A] the coefficients of the polynomial warping 
									from each of the target channel (i-th channel, i > 0) to the reference channel (0-th channel)
	@param[in]	opt:				int, (0 or 1) optimization method. 0 for MLE and 1 for LSQ
    @param[in]	MaxIters:			int, number of maximum iterations
    @param[out]	h_xvec:				(NFits, vnum) float, see definitions.h
	@param[out]	h_CRLB:			    (NFits, vnum) float, CRLB variance corresponding to parameters in g_xvec
	@param[out]	h_Loss:				(NFits) float, Loss value for the optimization of a PSF fitting.
    @param		d_push_flag: 		(NFits) int, flag for whether iteration should proceed.
	@param		d_maxJump: 			(NFits, vnum) flattened, float, maxJump control vector for each of the parameter for xvec.
	@param		d_grad: 			(NFits, vnum) flattened, float, gradient vector of w.r.t the xvec.
	@param		d_Hessian: 			(NFits, vnum, vnum) flattened, float, Hessian matrix w.r.t the xvec.
	@param		d_pvec: 			(NFits, vnum) flattened, float, increment of xvec for a single iteration.
	@param		d_lambda: 			(NFits) float, lambda value for Levenberg-Marquardt optimization.	
*/



static inline int intceil(int n, int m) { return (n - 1) / m + 1; }
static inline dim3 dimGrid1(int n, int blocksz) { dim3 gridsz(intceil(n, blocksz), 1, 1); return gridsz; }
static inline dim3 dimBlock1(int n) { dim3 blocksz(n, 1, 1); return blocksz; }



CDLL_EXPORT void gauss2d_1ch(int NFits, int boxsz, float* h_data, float* h_var, float* h_PSFsigmax, float* h_PSFsigmay, 
    float* h_xvec, float* h_CRLB, float* h_Loss, int opt, int MaxIters)
{
    const int vnum = NDIM + 2;
    
    float *d_data = nullptr, *d_var = nullptr, *d_PSFsigmax = nullptr, *d_PSFsigmay = nullptr;
    float *d_xvec = nullptr, *d_Loss = nullptr, *d_CRLB = nullptr;

    float *d_maxJump = nullptr, *d_grad = nullptr, *d_Hessian = nullptr, *d_pvec = nullptr, *d_lambda = nullptr;
    char* d_push_flag = nullptr;
    
    dim3 grids = dimGrid1(NFits, BLCKSZ);
    dim3 blcks = dimBlock1(BLCKSZ);

    // malloc space on device
    cudasafe(cudaMalloc((void**)&d_data, NFits * boxsz * boxsz * sizeof(float)), "malloc d_data", __LINE__);
    cudasafe(cudaMalloc((void**)&d_PSFsigmax, sizeof(float)), "malloc d_PSFsigmax", __LINE__);
    cudasafe(cudaMalloc((void**)&d_PSFsigmay, sizeof(float)), "malloc d_PSFsigmay", __LINE__);

    cudasafe(cudaMalloc((void**)&d_xvec, NFits * vnum * sizeof(float)), "malloc d_xvec", __LINE__);
    cudasafe(cudaMalloc((void**)&d_CRLB, NFits * vnum * sizeof(float)), "malloc d_CRLB", __LINE__);
    cudasafe(cudaMalloc((void**)&d_Loss, NFits * sizeof(float)), "malloc d_Loss", __LINE__);
    
    cudasafe(cudaMalloc((void**)&d_maxJump, NFits * vnum * sizeof(float)), "malloc d_maxJump", __LINE__);
    cudasafe(cudaMalloc((void**)&d_grad, NFits * vnum * sizeof(float)), "malloc d_grad", __LINE__);
    cudasafe(cudaMalloc((void**)&d_Hessian, NFits * vnum * vnum * sizeof(float)), "malloc d_Hessian", __LINE__);
    cudasafe(cudaMalloc((void**)&d_pvec, NFits * vnum * sizeof(float)), "malloc d_pvec", __LINE__);
    cudasafe(cudaMalloc((void**)&d_lambda, NFits * sizeof(float)), "malloc d_lambda", __LINE__);
    cudasafe(cudaMalloc((void**)&d_push_flag, NFits * sizeof(char)), "malloc d_push_flag", __LINE__);

    // transfer data from host to device
    cudasafe(cudaMemcpy(d_data, h_data, NFits * boxsz * boxsz * sizeof(float), cudaMemcpyHostToDevice), "transfer h_data to device", __LINE__);
    cudasafe(cudaMemcpy(d_PSFsigmax, h_PSFsigmax, sizeof(float), cudaMemcpyHostToDevice), "transfer h_PSFsigmax to device", __LINE__);
    cudasafe(cudaMemcpy(d_PSFsigmay, h_PSFsigmay, sizeof(float), cudaMemcpyHostToDevice), "transfer h_PSFsigmay to device", __LINE__);
    if (h_var) {
        cudasafe(cudaMalloc((void**)&d_var, NFits * boxsz * boxsz * sizeof(float)), "malloc d_var", __LINE__);
        cudasafe(cudaMemcpy(d_var, h_var, NFits * boxsz * boxsz * sizeof(float), cudaMemcpyHostToDevice), "transfer h_var to device", __LINE__);
    }

    Init_1ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay,
        d_xvec, d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_push_flag, opt);

    for (int iter = 0; iter < MaxIters; iter++)
        LMupdate_1ch<<<grids, blcks>>>(NFits, d_push_flag, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay,
            d_maxJump, d_lambda, d_xvec, d_grad, d_Hessian, d_pvec, d_Loss, opt);
    
    getCRLB_1ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, 
        d_xvec, d_Loss, d_Hessian, d_CRLB, opt);   
    
    cudasafe(cudaMemcpy(h_xvec, d_xvec, NFits * vnum * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_xvec to host", __LINE__);
    cudasafe(cudaMemcpy(h_CRLB, d_CRLB, NFits * vnum * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_CRLB to host", __LINE__);
    cudasafe(cudaMemcpy(h_Loss, d_Loss, NFits * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_Loss to host", __LINE__);
    
    cudasafe(cudaFree(d_data), "free d_data on device", __LINE__);
    cudasafe(cudaFree(d_PSFsigmax), "free d_PSFsigmax on device", __LINE__);
    cudasafe(cudaFree(d_PSFsigmay), "free d_PSFsigmay on device", __LINE__);
    cudasafe(cudaFree(d_xvec), "free d_xvec on device", __LINE__);
    cudasafe(cudaFree(d_CRLB), "free d_CRLB on device", __LINE__);
    cudasafe(cudaFree(d_Loss), "free d_Loss on device", __LINE__);
    
    cudasafe(cudaFree(d_maxJump), "free d_maxJump on device", __LINE__);
    cudasafe(cudaFree(d_grad), "free d_grad on device", __LINE__);
    cudasafe(cudaFree(d_Hessian), "free d_Hessian on device", __LINE__);
    cudasafe(cudaFree(d_pvec), "free d_pvec on device", __LINE__);
    cudasafe(cudaFree(d_lambda), "free d_lambda on device", __LINE__);
    cudasafe(cudaFree(d_push_flag), "free d_push_flag on device", __LINE__);
    if (h_var)
        cudasafe(cudaFree(d_var), "free d_var on device", __LINE__);
    return;
}  



CDLL_EXPORT void gauss2d_2ch(int NFits, int boxsz, float* h_data, float* h_var, float* h_PSFsigmax, float* h_PSFsigmay, 
    int warpdeg, int* h_lu, float* h_coeff_R2T,
    float* h_xvec, float* h_CRLB, float* h_Loss, int opt, int MaxIters)
{
    const int nchannels = 2;
    const int vnum = NDIM + 4;
    
    int *d_lu = nullptr;
    float *d_data = nullptr, *d_var = nullptr, *d_PSFsigmax = nullptr, *d_PSFsigmay = nullptr, *d_coeff_R2T = nullptr;
    float *d_xvec = nullptr, *d_Loss = nullptr, *d_CRLB = nullptr;

    float *d_maxJump = nullptr, *d_grad = nullptr, *d_Hessian = nullptr, *d_pvec = nullptr, *d_lambda = nullptr;
    char* d_push_flag = nullptr;
    
    dim3 grids = dimGrid1(NFits, BLCKSZ);
    dim3 blcks = dimBlock1(BLCKSZ);

    // malloc space on device
    cudasafe(cudaMalloc((void**)&d_data, NFits * nchannels * boxsz * boxsz * sizeof(float)), "malloc d_data", __LINE__);
    cudasafe(cudaMalloc((void**)&d_lu, NFits * nchannels * 2 * sizeof(int)), "malloc d_lu", __LINE__);
    cudasafe(cudaMalloc((void**)&d_PSFsigmax, nchannels * sizeof(float)), "malloc d_PSFsigmax", __LINE__);
    cudasafe(cudaMalloc((void**)&d_PSFsigmay, nchannels * sizeof(float)), "malloc d_PSFsigmay", __LINE__);
    cudasafe(cudaMalloc((void**)&d_coeff_R2T, (nchannels - 1) * 2 * warpdeg * warpdeg * sizeof(float)), "malloc d_coeff_R2T", __LINE__);

    cudasafe(cudaMalloc((void**)&d_xvec, NFits * vnum * sizeof(float)), "malloc d_xvec", __LINE__);
    cudasafe(cudaMalloc((void**)&d_CRLB, NFits * vnum * sizeof(float)), "malloc d_CRLB", __LINE__);
    cudasafe(cudaMalloc((void**)&d_Loss, NFits * sizeof(float)), "malloc d_Loss", __LINE__);
    
    cudasafe(cudaMalloc((void**)&d_maxJump, NFits * vnum * sizeof(float)), "malloc d_maxJump", __LINE__);
    cudasafe(cudaMalloc((void**)&d_grad, NFits * vnum * sizeof(float)), "malloc d_grad", __LINE__);
    cudasafe(cudaMalloc((void**)&d_Hessian, NFits * vnum * vnum * sizeof(float)), "malloc d_Hessian", __LINE__);
    cudasafe(cudaMalloc((void**)&d_pvec, NFits * vnum * sizeof(float)), "malloc d_pvec", __LINE__);
    cudasafe(cudaMalloc((void**)&d_lambda, NFits * sizeof(float)), "malloc d_lambda", __LINE__);
    cudasafe(cudaMalloc((void**)&d_push_flag, NFits * sizeof(char)), "malloc d_push_flag", __LINE__);

    // transfer data from host to device
    cudasafe(cudaMemcpy(d_data, h_data, NFits * nchannels * boxsz * boxsz * sizeof(float), cudaMemcpyHostToDevice), "transfer h_data to device", __LINE__);
    cudasafe(cudaMemcpy(d_lu, h_lu, NFits * nchannels * 2 * sizeof(int), cudaMemcpyHostToDevice), "transfer h_lu to device", __LINE__);
    cudasafe(cudaMemcpy(d_PSFsigmax, h_PSFsigmax, nchannels * sizeof(float), cudaMemcpyHostToDevice), "transfer h_PSFsigmax to device", __LINE__);
    cudasafe(cudaMemcpy(d_PSFsigmay, h_PSFsigmay, nchannels * sizeof(float), cudaMemcpyHostToDevice), "transfer h_PSFsigmay to device", __LINE__);
    cudasafe(cudaMemcpy(d_coeff_R2T, h_coeff_R2T, (nchannels - 1) * 2 * warpdeg * warpdeg * sizeof(float), cudaMemcpyHostToDevice), "transfer h_coeff_R2T to device", __LINE__);
    if (h_var) {
        cudasafe(cudaMalloc((void**)&d_var, NFits * nchannels * boxsz * boxsz * sizeof(float)), "malloc d_var", __LINE__);
        cudasafe(cudaMemcpy(d_var, h_var, NFits * nchannels * boxsz * boxsz * sizeof(float), cudaMemcpyHostToDevice), "transfer h_var to device", __LINE__);
    }

    Init_2ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T,
        d_xvec, d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_push_flag, opt);
    
    for (int iter = 0; iter < MaxIters; iter++)
        LMupdate_2ch<<<grids, blcks>>>(NFits, d_push_flag, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T,
            d_maxJump, d_lambda, d_xvec, d_grad, d_Hessian, d_pvec, d_Loss, opt);
    
    getCRLB_2ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T,
        d_xvec, d_Loss, d_Hessian, d_CRLB, opt);
    
    cudasafe(cudaMemcpy(h_xvec, d_xvec, NFits * vnum * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_xvec to host", __LINE__);
    cudasafe(cudaMemcpy(h_CRLB, d_CRLB, NFits * vnum * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_CRLB to host", __LINE__);
    cudasafe(cudaMemcpy(h_Loss, d_Loss, NFits * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_Loss to host", __LINE__);
    
    cudasafe(cudaFree(d_data), "free d_data on device", __LINE__);
    cudasafe(cudaFree(d_lu), "free d_lu on device", __LINE__);
    cudasafe(cudaFree(d_PSFsigmax), "free d_PSFsigmax on device", __LINE__);
    cudasafe(cudaFree(d_PSFsigmay), "free d_PSFsigmay on device", __LINE__);
    cudasafe(cudaFree(d_coeff_R2T), "free d_coeff_R2T on device", __LINE__);

    cudasafe(cudaFree(d_xvec), "free d_xvec on device", __LINE__);
    cudasafe(cudaFree(d_CRLB), "free d_CRLB on device", __LINE__);
    cudasafe(cudaFree(d_Loss), "free d_Loss on device", __LINE__);
    
    cudasafe(cudaFree(d_maxJump), "free d_maxJump on device", __LINE__);
    cudasafe(cudaFree(d_grad), "free d_grad on device", __LINE__);
    cudasafe(cudaFree(d_Hessian), "free d_Hessian on device", __LINE__);
    cudasafe(cudaFree(d_pvec), "free d_pvec on device", __LINE__);
    cudasafe(cudaFree(d_lambda), "free d_lambda on device", __LINE__);
    cudasafe(cudaFree(d_push_flag), "free d_push_flag on device", __LINE__);
    if (h_var)
        cudasafe(cudaFree(d_var), "free d_var on device", __LINE__);
    return;
}