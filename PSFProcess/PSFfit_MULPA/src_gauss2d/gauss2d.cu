#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "DLL_Macros.h"
#include "CUDA_Utils.cuh"

#include "definitions.h"
#include "kernels_1.cuh"
#include "kernels_2.cuh"
#include "kernels_3.cuh"
#include "kernels_4.cuh"
#include "kernels_5.cuh"
#include "kernels_6.cuh"
#include "kernels_7.cuh"


/*! @param[in]	NFits: 				int, number of PSF squares
    @param[in]	nchannels:			int, the number of channels
	@param[in]	boxsz: 				int, size of the PSF square
	@param[in]	h_data: 			(NFits, nchannels, boxsz, boxsz) flattened, float, multi-channel PSF data
	@param[in]	h_var: 				(NFits, nchannels, boxsz, boxsz) flattened, float, multi-channel pixel-dependent variance of camera readout noise
    @param[in]	PSFsigmax: 		    (nchannels) float, sigma (x-axis) of the Gaussian PSF in each channel
	@param[in]	PSFsigmay: 		    (nchannels) float, sigma (y-axis) of the Gaussian PSF in each channel
    @param[in]	warpdeg:			int, the degree of the polynomial function that warps 2d locations from one channel to the other
    @param[in]	h_lu:				(NFits, nchannels, 2) flattened, int, [lc, uc] the left and upper corner of the PSF square in each channel (image coordinates)
    @param[in]	h_coeff_R2T:		((nchannels - 1) * 2 * warpdeg * warpdeg) float, the [coeffx_B2A, coeffy_B2A] coefficients of the polynomial warping from refernce channel to each of the target channel	
	@param[in]	MaxIters:			int, number of maximum iterations
    @param[out]	h_nnum:		        (NFits) int, number of emitters for the optimization of a PSF fitting.
    @param[out]	h_Loss:		        (NFits) float, Loss value for the optimization of a PSF fitting.
    @param[out]	h_xvec: 		    (NFits, NMAX * (2 + nchannels) + nchannels) float, see definitions.h
	@param[out]	h_CRLB:			    (NFits, NMAX * (2 + nchannels) + nchannels) float, CRLB variance corresponding to parameters in g_xvec
    @param		d_iter_push: 		(NFits,) char, flag for whether iteration should proceed.
	@param		d_maxJump: 			(NFits, VMAX) flattened, float, maxJump control vector for each of the parameter for xvec.
	@param		d_grad: 			(NFits, VMAX) flattened, float, gradient vector of w.r.t the xvec.
	@param		d_Hessian: 			(NFits, VMAX, VMAX) flattened, float, Hessian matrix w.r.t the xvec.
	@param		d_pvec: 			(NFits, VMAX) flattened, float, increment of xvec for a single iteration.
	@param		d_lambda: 			(NFits,) float, lambda value for Levenberg-Marquardt optimization.	
*/



static inline int intceil(int n, int m) { return (n - 1) / m + 1; }
static inline dim3 dimGrid1(int n, int blocksz) { dim3 gridsz(intceil(n, blocksz), 1, 1); return gridsz; }
static inline dim3 dimBlock1(int n) { dim3 blocksz(n, 1, 1); return blocksz; }


CDLL_EXPORT void gauss2d_1ch(int NFits, int nPSF, int boxsz, float* h_data, float* h_var, float* h_PSFsigmax, float* h_PSFsigmay,
    float* h_xvec, float* h_CRLB, float* h_Loss, int opt, int MaxIters)
{
    const int nchannels = 1, vnum0 = NDIM + nchannels;
    const int nnum = (nPSF > NMAX) ? NMAX : nPSF;
    const int vnum = nnum * vnum0 + nchannels;
    
    float *d_data = nullptr, *d_var = nullptr, *d_PSFsigmax = nullptr, *d_PSFsigmay = nullptr;
    
    float *d_maxJump = nullptr, *d_grad = nullptr, *d_Hessian = nullptr, *d_pvec = nullptr, *d_lambda = nullptr;
    char *d_iter_push = nullptr;

    float *d_xvec = nullptr, *d_CRLB = nullptr, *d_Loss = nullptr;
    
    dim3 grids = dimGrid1(NFits, BLCKSZ);
    dim3 blcks = dimBlock1(BLCKSZ);

    // malloc space on device
    cudasafe(cudaMalloc((void**)&d_data, NFits * nchannels * boxsz * boxsz * sizeof(float)), "malloc d_data", __LINE__);
    cudasafe(cudaMalloc((void**)&d_PSFsigmax, nchannels * sizeof(float)), "malloc d_PSFsigmax", __LINE__);
    cudasafe(cudaMalloc((void**)&d_PSFsigmay, nchannels * sizeof(float)), "malloc d_PSFsigmay", __LINE__);
    
    cudasafe(cudaMalloc((void**)&d_maxJump, NFits * vnum * sizeof(float)), "malloc d_maxJump", __LINE__);
    cudasafe(cudaMalloc((void**)&d_grad, NFits * vnum * sizeof(float)), "malloc d_grad", __LINE__);
    cudasafe(cudaMalloc((void**)&d_Hessian, NFits * vnum * vnum * sizeof(float)), "malloc d_Hessian", __LINE__);
    cudasafe(cudaMalloc((void**)&d_pvec, NFits * vnum * sizeof(float)), "malloc d_pvec", __LINE__);
    cudasafe(cudaMalloc((void**)&d_lambda, NFits * sizeof(float)), "malloc d_lambda", __LINE__);
    cudasafe(cudaMalloc((void**)&d_iter_push, NFits * sizeof(char)), "malloc d_iter_push", __LINE__);

    cudasafe(cudaMalloc((void**)&d_Loss, NFits * sizeof(float)), "malloc d_Loss", __LINE__);
    cudasafe(cudaMalloc((void**)&d_xvec, NFits * vnum * sizeof(float)), "malloc d_xvec", __LINE__);
    cudasafe(cudaMalloc((void**)&d_CRLB, NFits * vnum * sizeof(float)), "malloc d_CRLB", __LINE__);

    // transfer data from host to device
    cudasafe(cudaMemcpy(d_data, h_data, NFits * nchannels * boxsz * boxsz * sizeof(float), cudaMemcpyHostToDevice), "transfer h_data to device", __LINE__);
    cudasafe(cudaMemcpy(d_PSFsigmax, h_PSFsigmax, nchannels * sizeof(float), cudaMemcpyHostToDevice), "transfer h_PSFsigmax to device", __LINE__);
    cudasafe(cudaMemcpy(d_PSFsigmay, h_PSFsigmay, nchannels * sizeof(float), cudaMemcpyHostToDevice), "transfer h_PSFsigmay to device", __LINE__);
    cudasafe(cudaMemcpy(d_xvec, h_xvec, NFits * vnum * sizeof(float), cudaMemcpyHostToDevice), "transfer h_xvec to device", __LINE__);
    if (h_var) {
        cudasafe(cudaMalloc((void**)&d_var, NFits * nchannels * boxsz * boxsz * sizeof(float)), "malloc d_var", __LINE__);
        cudasafe(cudaMemcpy(d_var, h_var, NFits * nchannels * boxsz * boxsz * sizeof(float), cudaMemcpyHostToDevice), "transfer h_var to device", __LINE__);
    }

    if (nnum == 1) {   
        PreInit<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_xvec);
        Init_1_1ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_1_1ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_1_1ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, d_CRLB);
    }
    else if (nnum == 2) {   
        PushnPull_2<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_2_1ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_2_1ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_2_1ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec,d_CRLB);
    }
    else if (nnum == 3) {   
        PushnPull_3<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_3_1ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_3_1ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_3_1ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, d_CRLB);
    }
    else if (nnum == 4) {   
        PushnPull_4<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_4_1ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_4_1ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_4_1ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, d_CRLB);
    }
    else if (nnum == 5) {   
        PushnPull_5<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_5_1ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_5_1ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_5_1ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, d_CRLB);
    }
    else if (nnum == 6) {   
        PushnPull_6<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_6_1ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_6_1ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_6_1ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, d_CRLB);
    }
    else {   
        PushnPull_7<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_7_1ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_7_1ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_7_1ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, d_xvec, d_CRLB);
    }
    
    
    cudasafe(cudaMemcpy(h_Loss, d_Loss, NFits * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_Loss to host", __LINE__);
    cudasafe(cudaMemcpy(h_xvec, d_xvec, NFits * vnum * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_xvec to host", __LINE__);
    cudasafe(cudaMemcpy(h_CRLB, d_CRLB, NFits * vnum * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_CRLB to host", __LINE__);
    
    cudasafe(cudaFree(d_data), "free d_data on device", __LINE__);
    cudasafe(cudaFree(d_PSFsigmax), "free d_PSFsigmax on device", __LINE__);
    cudasafe(cudaFree(d_PSFsigmay), "free d_PSFsigmay on device", __LINE__);
    
    cudasafe(cudaFree(d_Loss), "free d_Loss on device", __LINE__);
    cudasafe(cudaFree(d_xvec), "free d_xvec on device", __LINE__);
    cudasafe(cudaFree(d_CRLB), "free d_CRLB on device", __LINE__);
    
    cudasafe(cudaFree(d_maxJump), "free d_maxJump on device", __LINE__);
    cudasafe(cudaFree(d_grad), "free d_grad on device", __LINE__);
    cudasafe(cudaFree(d_Hessian), "free d_Hessian on device", __LINE__);
    cudasafe(cudaFree(d_pvec), "free d_pvec on device", __LINE__);
    cudasafe(cudaFree(d_lambda), "free d_lambda on device", __LINE__);
    cudasafe(cudaFree(d_iter_push), "free d_iter_push on device", __LINE__);
    if (h_var)
        cudasafe(cudaFree(d_var), "free d_var on device", __LINE__);    
    return;
}  



CDLL_EXPORT void gauss2d_2ch(int NFits, int nPSF, int boxsz, float* h_data, float* h_var, float* h_PSFsigmax, float* h_PSFsigmay, 
    int warpdeg, int* h_lu, float* h_coeff_R2T, 
    float* h_xvec, float* h_CRLB, float* h_Loss, int opt, int MaxIters)
{    
    const int nchannels = 2, vnum0 = NDIM + nchannels;
    const int nnum = (nPSF > NMAX) ? NMAX : nPSF;
    const int vnum = nnum * vnum0 + nchannels;
    
    float *d_data = nullptr, *d_var = nullptr, *d_PSFsigmax = nullptr, *d_PSFsigmay = nullptr, *d_coeff_R2T = nullptr;
    int *d_lu = nullptr;
    
    float *d_maxJump = nullptr, *d_grad = nullptr, *d_Hessian = nullptr, *d_pvec = nullptr, *d_lambda = nullptr;
    char *d_iter_push = nullptr;

    float *d_xvec = nullptr, *d_CRLB = nullptr, *d_Loss = nullptr;

    dim3 grids = dimGrid1(NFits, BLCKSZ);
    dim3 blcks = dimBlock1(BLCKSZ);

    // malloc space on device
    cudasafe(cudaMalloc((void**)&d_data, NFits * nchannels * boxsz * boxsz * sizeof(float)), "malloc d_data", __LINE__);
    cudasafe(cudaMalloc((void**)&d_PSFsigmax, nchannels * sizeof(float)), "malloc d_PSFsigmax", __LINE__);
    cudasafe(cudaMalloc((void**)&d_PSFsigmay, nchannels * sizeof(float)), "malloc d_PSFsigmay", __LINE__);
    cudasafe(cudaMalloc((void**)&d_lu, NFits * nchannels * 2 * sizeof(int)), "malloc d_lu", __LINE__);
    cudasafe(cudaMalloc((void**)&d_coeff_R2T, (nchannels - 1) * 2 * warpdeg * warpdeg * sizeof(float)), "malloc d_coeff_R2T", __LINE__);

    cudasafe(cudaMalloc((void**)&d_Loss, NFits * sizeof(float)), "malloc d_Loss", __LINE__);
    cudasafe(cudaMalloc((void**)&d_xvec, NFits * vnum * sizeof(float)), "malloc d_xvec", __LINE__);
    cudasafe(cudaMalloc((void**)&d_CRLB, NFits * vnum * sizeof(float)), "malloc d_CRLB", __LINE__);

    cudasafe(cudaMalloc((void**)&d_maxJump, NFits * vnum * sizeof(float)), "malloc d_maxJump", __LINE__);
    cudasafe(cudaMalloc((void**)&d_grad, NFits * vnum * sizeof(float)), "malloc d_grad", __LINE__);
    cudasafe(cudaMalloc((void**)&d_Hessian, NFits * vnum * vnum * sizeof(float)), "malloc d_Hessian", __LINE__);
    cudasafe(cudaMalloc((void**)&d_pvec, NFits * vnum * sizeof(float)), "malloc d_pvec", __LINE__);
    cudasafe(cudaMalloc((void**)&d_lambda, NFits * sizeof(float)), "malloc d_lambda", __LINE__);
    cudasafe(cudaMalloc((void**)&d_iter_push, NFits * sizeof(char)), "malloc d_iter_push", __LINE__);

    // transfer data from host to device
    cudasafe(cudaMemcpy(d_data, h_data, NFits * nchannels * boxsz * boxsz * sizeof(float), cudaMemcpyHostToDevice), "transfer h_data to device", __LINE__);
    cudasafe(cudaMemcpy(d_PSFsigmax, h_PSFsigmax, nchannels * sizeof(float), cudaMemcpyHostToDevice), "transfer h_PSFsigmax to device", __LINE__);
    cudasafe(cudaMemcpy(d_PSFsigmay, h_PSFsigmay, nchannels * sizeof(float), cudaMemcpyHostToDevice), "transfer h_PSFsigmay to device", __LINE__);
    cudasafe(cudaMemcpy(d_lu, h_lu, NFits * nchannels * 2 * sizeof(int), cudaMemcpyHostToDevice), "transfer h_lu to device", __LINE__);
    cudasafe(cudaMemcpy(d_coeff_R2T, h_coeff_R2T, (nchannels - 1) * 2 * warpdeg * warpdeg * sizeof(float), cudaMemcpyHostToDevice), "transfer h_coeff_R2T to device", __LINE__);
    cudasafe(cudaMemcpy(d_xvec, h_xvec, NFits * vnum * sizeof(float), cudaMemcpyHostToDevice), "transfer h_xvec to device", __LINE__);
    if (h_var) {
        cudasafe(cudaMalloc((void**)&d_var, NFits * nchannels * boxsz * boxsz * sizeof(float)), "malloc d_var", __LINE__);
        cudasafe(cudaMemcpy(d_var, h_var, NFits * nchannels * boxsz * boxsz * sizeof(float), cudaMemcpyHostToDevice), "transfer h_var to device", __LINE__);
    }

    if (nnum == 1) {   
        PreInit<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_xvec);
        Init_1_2ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_1_2ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_1_2ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, d_CRLB);
    }
    else if (nnum == 2) {   
        PushnPull_2<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_2_2ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_2_2ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_2_2ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, d_CRLB);
    }
    else if (nnum == 3) {   
        PushnPull_3<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_3_2ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_3_2ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_3_2ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, d_CRLB);
    }
    else if (nnum == 4) {   
        PushnPull_4<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_4_2ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_4_2ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_4_2ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, d_CRLB);
    }
    else if (nnum == 5) {   
        PushnPull_5<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_5_2ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_5_2ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_5_2ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, d_CRLB);
    }
    else if (nnum == 6) {   
        PushnPull_6<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_6_2ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_6_2ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_6_2ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, d_CRLB);
    }
    else if (nnum == 7) {   
        PushnPull_7<<<grids, blcks>>>(NFits, nchannels, boxsz, d_data, d_PSFsigmax, d_PSFsigmay, d_xvec);
        Init_7_2ch<<<grids, blcks>>>(NFits, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, 
            d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_iter_push);
        for (unsigned int iter = 0; iter < MaxIters; iter++)
            LMupdate_7_2ch<<<grids, blcks>>>(NFits, d_iter_push, boxsz, d_data, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T,
                d_xvec, d_Loss, d_maxJump, d_lambda, d_grad, d_Hessian, d_pvec);
        getCRLB_7_2ch<<<grids, blcks>>>(NFits, boxsz, d_var, d_PSFsigmax, d_PSFsigmay, warpdeg, d_lu, d_coeff_R2T, d_xvec, d_CRLB);
    }
    
    cudasafe(cudaMemcpy(h_Loss, d_Loss, NFits * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_Loss to host", __LINE__);
    cudasafe(cudaMemcpy(h_xvec, d_xvec, NFits * vnum * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_xvec to host", __LINE__);
    cudasafe(cudaMemcpy(h_CRLB, d_CRLB, NFits * vnum * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_CRLB to host", __LINE__);
    
    cudasafe(cudaFree(d_data), "free d_data on device", __LINE__);
    cudasafe(cudaFree(d_PSFsigmax), "free d_PSFsigmax on device", __LINE__);
    cudasafe(cudaFree(d_PSFsigmay), "free d_PSFsigmay on device", __LINE__);
    cudasafe(cudaFree(d_lu), "free d_lu on device", __LINE__);
    cudasafe(cudaFree(d_coeff_R2T), "free d_coeff_R2T on device", __LINE__);
    
    cudasafe(cudaFree(d_Loss), "free d_Loss on device", __LINE__);
    cudasafe(cudaFree(d_xvec), "free d_xvec on device", __LINE__);
    cudasafe(cudaFree(d_CRLB), "free d_CRLB on device", __LINE__);
    
    cudasafe(cudaFree(d_maxJump), "free d_maxJump on device", __LINE__);
    cudasafe(cudaFree(d_grad), "free d_grad on device", __LINE__);
    cudasafe(cudaFree(d_Hessian), "free d_Hessian on device", __LINE__);
    cudasafe(cudaFree(d_pvec), "free d_pvec on device", __LINE__);
    cudasafe(cudaFree(d_lambda), "free d_lambda on device", __LINE__);
    cudasafe(cudaFree(d_iter_push), "free d_iter_push on device", __LINE__);
    if (h_var)
        cudasafe(cudaFree(d_var), "free d_var on device", __LINE__);
    return;
}  