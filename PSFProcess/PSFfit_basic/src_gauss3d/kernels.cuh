#pragma once
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "matrix.cuh"
#include "functionals.cuh"

#ifndef KERNELS_CUH
#define KERNELS_CUH


/*! @brief	kernel functions for PSFfit_SINPA
	@param[in]	NFits: 				int, the number of PSF squares
	@param[in]	boxsz: 				int, the size of the PSF square
	@param[in]	zrange:				int, the range (z-axis) of the axial calibration of the astigmatic PSF
	@param[in]	g_data: 			(NFits * nchannels * boxsz * boxsz) float, PSF data
	@param[in]	g_var: 				(NFits * nchannels * boxsz * boxsz) float, pixel-dependent variance, NULL for pixel-independent variance
	@param[in]	g_astigs:			(nchannels * 9) float, [PSFsigmax0, PSFsigmay0, shiftx, shifty, dof, Ax, Bx, Ay, By] the astigmatic parameters for each channel
	@param[in]	opt:				int, optimization method. 0 for MLE and 1 for LSQ
	@param[out]	g_xvec: 			(NFits * vnum) float, see definitions.h
	@param[out]	g_CRLB:				(NFits * vnum) float, the CRLB corresponding to each of the variables in xvec, respectively
	@param[out]	g_Loss:				(NFits) float, Loss value for the optimization of a PSF fitting.
	@param		g_push_flag: 		(NFits) char, flag for whether iteration should proceed.
	@param		g_maxJump: 			(NFits * vnum) float, maxJump control vector for each of the parameter for xvec.
	@param		g_grad: 			(NFits * vnum) float, gradient vector of w.r.t the xvec.
	@param		g_Hessian: 			(NFits * vnum * vnum) float, Hessian matrix w.r.t the xvec.
	@param		g_pvec: 			(NFits * vnum) float, increment of xvec for a single iteration.
	@param		g_lambda: 			(NFits) float, lambda value for Levenberg-Marquardt optimization.	
*/



/*! @brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
			Initialize the g_xvec, g_maxJump	
			Initialize the g_Loss, g_grad, g_Hessian
			Initialize the g_lambda at INIT_LAMBDA for Levenberg-Marquardt optimization.
			Initialize the g_pvec at INIT_ERR
			Initialize the g_push_flag at 1
*/
__global__ void Init(int NFits, int boxsz, int zrange, float* g_data, float* g_var, float* astigs,
	float* g_xvec, float* g_maxJump, float* g_Loss, float* g_grad, float* g_Hessian, float* g_pvec, float* g_lambda, char* g_push_flag, int opt)
{
	const int vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *varim = nullptr;
	
	float Loss = 0.0f, xvec[VNUM] = {0.0f}, maxJump[VNUM] = {0.0f}, grad[VNUM] = {0.0f}, Hessian[VNUM * VNUM] = {0.0f};
	
	if (Idx < NFits) {
		// load data from the global memory
		dataim = g_data + Idx * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * boxsz * boxsz : nullptr;
		
		init_xvec(boxsz, zrange, dataim, xvec);
		init_maxJump(boxsz, zrange, xvec, maxJump);
		get_lgH(boxsz, dataim, varim, astigs, xvec, &Loss, grad, Hessian, opt);

		memcpy(g_xvec + Idx * vnum, xvec, vnum * sizeof(float));
		memcpy(g_maxJump + Idx * vnum, maxJump, vnum * sizeof(float));
		g_Loss[Idx] = Loss;
		memcpy(g_grad + Idx * vnum, grad, vnum * sizeof(float));
		memcpy(g_Hessian + Idx * vnum * vnum, Hessian, vnum * vnum * sizeof(float));
		
		// initialize the lambda and pvec.
		g_lambda[Idx] = INIT_LAMBDA;
		for (unsigned int i = 0; i < vnum; i++)
			g_pvec[Idx * vnum + i] = INIT_ERR;

		// initialize the push_flag
		g_push_flag[Idx] = 1;
	}
	__syncthreads();
}



/*!	@brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
			Single iteration of a Levenberg-Marguardt update of the xvec.
*/
__global__ void LMupdate(int NFits, char* g_push_flag, int boxsz, int zrange, float* g_data, float* g_var, float* astigs, 
	float* g_maxJump, float* g_lambda, float* g_xvec, float* g_grad, float* g_Hessian, float* g_pvec, float* g_Loss, int opt)
{
	const int vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *varim = nullptr;

	char push_flag = 0, decomp_flag = 0;
	
	float Loss = 0.0f, lambda = 0.0f, mu = 1.0f;
	float xvec[VNUM] = {0.0f}, grad[VNUM] = {0.0f}, Hessian[VNUM * VNUM] = {0.0f}, pvec[VNUM] = {0.0f}, maxJump[VNUM] = {0.0f}, L[VNUM * VNUM] = {0.0f}; 

	if (Idx < NFits)
		push_flag = g_push_flag[Idx];
	__syncthreads();

	if (push_flag) {
		// LOAD DATA FROM THE GLOBAL MEMORY
		dataim = g_data + Idx * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * boxsz * boxsz : nullptr;
		lambda = g_lambda[Idx];
		mu = 1.0f + lambda;
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
		memcpy(grad, g_grad + Idx * vnum, vnum * sizeof(float));
		memcpy(maxJump, g_maxJump + Idx * vnum, vnum * sizeof(float));
		memcpy(Hessian, g_Hessian + Idx * vnum * vnum, vnum * vnum * sizeof(float));
		
		// CHOLESKY DECOMPOSITION OF THE HEISSIAN
		decomp_flag = Cholesky_decomp(Hessian, vnum, L);
		
		// DECOMPOSIBLE
		if ( decomp_flag == 1 ) {
			// calculate the pvec H * pvec = grad
			Cholesky_solve(L, grad, vnum, pvec);
			
			// update and refine xvec and pvec
			for (unsigned int i = 0; i < vnum; i++) {
				if (pvec[i] / g_pvec[Idx * vnum + i] < -0.5f)
					maxJump[i] *= 0.5f; 
				pvec[i] = pvec[i] / (1.0f + fabs(pvec[i] / maxJump[i]));
				xvec[i] = xvec[i] - pvec[i];
			}
			boxConstrain(boxsz, zrange, xvec);

			// calculate the loss, grad, and Hessian
			get_lgH(boxsz, dataim, varim, astigs, xvec, &Loss, grad, Hessian, opt);

			// evaluate the Loss
			if (Loss >= ACCEPTANCE * g_Loss[Idx]) {
				mu = max((1 + lambda * SCALE_UP) / (1 + lambda), 1.3f);
				g_lambda[Idx] = SCALE_UP * lambda;
				for (unsigned int i = 0; i < vnum; i++)
					g_Hessian[Idx * vnum * vnum + i * vnum + i] = g_Hessian[Idx * vnum * vnum + i * vnum + i] * mu;
			}
			else {
				if (Loss < g_Loss[Idx]) {
					mu = 1 + SCALE_DOWN * lambda;
					g_lambda[Idx] = SCALE_DOWN * lambda;
				}
				memcpy(g_xvec + Idx * vnum, xvec, vnum * sizeof(float));
				memcpy(g_pvec + Idx * vnum, pvec, vnum * sizeof(float));
				memcpy(g_maxJump + Idx * vnum, maxJump, vnum * sizeof(float));
				memcpy(g_grad + Idx * vnum, grad, vnum * sizeof(float));
				memcpy(g_Hessian + Idx * vnum * vnum, Hessian, vnum * vnum * sizeof(float));
				g_push_flag[Idx] = (fabs(Loss - g_Loss[Idx]) >= (fabs(Loss) * OPTTOL));
				g_Loss[Idx] = Loss;
			}
		}

		// NOT DECOMPOSIBLE
		else {
			mu = max((1 + lambda * SCALE_UP) / (1 + lambda), 1.3f);
			g_lambda[Idx] = SCALE_UP * lambda;
			for (unsigned int i = 0; i < vnum; i++)
				g_Hessian[Idx * vnum * vnum + i * vnum + i] = g_Hessian[Idx * vnum * vnum + i * vnum + i] * mu;
		}
	}
	__syncthreads();
}



/*!	@brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
			calculate the CRLB	
*/
__global__ void getCRLB(int NFits, int boxsz, float* g_var, float* astigs, 
	float* g_xvec, float* g_Loss, float* g_Hessian, float* g_CRLB, int opt)
{
	const int vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *varim = nullptr;

	char cholesky_info = 0;

	float xvec[VNUM] = {0.0f}, FIM[VNUM * VNUM] = {0.0f}, L[VNUM * VNUM] = {0.0f}; 

	if (Idx < NFits) {
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
	
		// construct the Fisher Information Matrix (FIM)
		if (opt == 0) {
			varim = (g_var) ? g_var + Idx * boxsz * boxsz : nullptr;
			get_FIM(boxsz, varim, astigs, xvec, FIM);
		}
		else
			memcpy(FIM, g_Hessian + Idx * vnum * vnum, vnum * vnum * sizeof(float));
			
		cholesky_info = Cholesky_decomp(FIM, vnum, L);
		if (cholesky_info == 1) {
			Cholesky_invert(L, vnum, FIM);
			for (unsigned int i = 0; i < vnum; i++)
				if (opt == 0)
					g_CRLB[Idx * vnum + i] = FIM[i * vnum + i];
				else
					g_CRLB[Idx * vnum + i] = g_Loss[Idx] / (float)boxsz / (float)boxsz * FIM[i * vnum + i];
		}
	}	
	__syncthreads();	
}

#endif