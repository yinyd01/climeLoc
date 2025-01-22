#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "DLL_Macros.h"
#include "CUDA_Utils.cuh"

#include "definitions.h"
#include "kernels.cuh"


/*! @param[in]	NFits: 				int, number of PSF squares
    @param[in]	boxsz: 				int, size of the PSF square
	@param[in]	h_data: 			(NFits, boxsz, boxsz) flattened, float, PSF data
    @param[in]	h_var: 			    (NFits, boxsz, boxsz) flattened, float, cam var, NULL for EMCCD
	@param[in]	PSFsigmax: 		    float, sigma (x-axis) of the Gaussian PSF
	@param[in]	PSFsigmay: 		    float, sigma (y-axis) of the Gaussian PSF
    @param[in]	fixs:				int, (0 or 1) 1 for PSFsigmax and PSfsigmay are fixed at a given values during fitting, 0 for free fit of PSFsigmax and PSFsigmay
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



CDLL_EXPORT void gauss2d(int NFits, int boxsz, float* h_data, float* h_var, float PSFsigmax, float PSFsigmay, 
    float* h_xvec, float* h_CRLB, float* h_Loss, int fixs, int opt, int MaxIters)
{
    const int vnum = (fixs == 1) ? NDIM + 2 : NDIM + 4;
    
    float *d_data = nullptr, *d_var = nullptr, *d_xvec = nullptr, *d_Loss = nullptr, *d_CRLB = nullptr;
    float *d_maxJump = nullptr, *d_grad = nullptr, *d_Hessian = nullptr, *d_pvec = nullptr, *d_lambda = nullptr;
    char* d_push_flag = nullptr;
    
    dim3 grids = dimGrid1(NFits, BLCKSZ);
    dim3 blcks = dimBlock1(BLCKSZ);

    // malloc space on device
    cudasafe(cudaMalloc((void**)&d_data, NFits * boxsz * boxsz * sizeof(float)), "malloc d_data", __LINE__);
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
    if (h_var) {
        cudasafe(cudaMalloc((void**)&d_var, NFits * boxsz * boxsz * sizeof(float)), "malloc d_var", __LINE__);
        cudasafe(cudaMemcpy(d_var, h_var, NFits * boxsz * boxsz * sizeof(float), cudaMemcpyHostToDevice), "transfer h_var to device", __LINE__);
    }

    Init<<<grids, blcks>>>(fixs, NFits, boxsz, d_data, d_var, PSFsigmax, PSFsigmay,
        d_xvec, d_maxJump, d_Loss, d_grad, d_Hessian, d_pvec, d_lambda, d_push_flag, opt);

    for (unsigned int iter = 0; iter < MaxIters; iter++)
        LMupdate<<<grids, blcks>>>(fixs, NFits, d_push_flag, boxsz, d_data, d_var, PSFsigmax, PSFsigmay,
            d_maxJump, d_lambda, d_xvec, d_grad, d_Hessian, d_pvec, d_Loss, opt);
    
    getCRLB<<<grids, blcks>>>(fixs, NFits, boxsz, d_var, PSFsigmax, PSFsigmay, 
        d_xvec, d_Loss, d_Hessian, d_CRLB, opt);   
    
    cudasafe(cudaMemcpy(h_xvec, d_xvec, NFits * vnum * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_xvec to host", __LINE__);
    cudasafe(cudaMemcpy(h_CRLB, d_CRLB, NFits * vnum * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_CRLB to host", __LINE__);
    cudasafe(cudaMemcpy(h_Loss, d_Loss, NFits * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_Loss to host", __LINE__);
    
    cudasafe(cudaFree(d_data), "free d_data on device", __LINE__);
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