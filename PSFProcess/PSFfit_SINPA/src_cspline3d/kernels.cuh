#pragma once
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matrix.cuh"
#include "functionals.cuh"

#ifndef KERNELS_CUH
#define KERNELS_CUH


/*! @brief	kernel functions for dualview PSFfit_SINPA
	@param		nchannels:			const int, 2, the number of channels 
	@param		vnum:				const int, number of optimizatin parameters
	@param[in]	NFits: 				int, the number of nchannel-PSF
	@param[in]	boxsz: 				int, the size of the PSF square
	@param[in]	splineszx: 			int, the size of the splines
	@param[in]	splineszz:			int, the size of the splines cube (axis-axis)
	@param[in]	g_data: 			(NFits * nchannels * boxsz * boxsz) float, PSF data
	@param[in]	g_var: 				(NFits * nchannels * boxsz * boxsz) float, pixel-dependent variance, NULL for pixel-independent variance
	@param[in]	g_coeff_PSF: 		(nchannels * splinesz * splinesz * 16) float, spline coefficient for the nchannel 2d PSF
	@param[in]	warpdeg:			int, the degree of the polynomial function that warps 2d locations from one channel to the other
	@param[in]	g_lu:				(NFits * nchannels * 2) int, [lc, uc] the left and upper corner of the PSF square in each channel (image coordinates)
	@param[in]	g_coeff_R2T:		((nchannels - 1) * 2 * warpdeg * warpdeg) float, the [coeffx_B2A, coeffy_B2A] the coefficients of the polynomial warping 
									from each of the target channel (i-th channel, i > 0) to the reference channel (0-th channel)	
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
__global__ void Init_1ch(int NFits, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF,
	float* g_xvec, float* g_maxJump, float* g_Loss, float* g_grad, float* g_Hessian, float* g_pvec, float* g_lambda, char* g_push_flag, int opt)
{
	const int vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;

	float Loss = 0.0f, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};
	float *dataim = nullptr, *varim = nullptr;
	float *xvec = nullptr, *maxJump = nullptr;

	if (Idx < NFits) {
		// load data from the global memory
		dataim = g_data + Idx * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * boxsz * boxsz : nullptr;
		xvec = g_xvec + Idx * vnum;
		maxJump = g_maxJump + Idx * vnum;
		init_xvec_1ch(boxsz, splineszz, dataim, xvec);
		init_maxJump_1ch(boxsz, splineszz, xvec, maxJump);

		// initialize the loss, grad, and hessian
		get_lgH_1ch(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, xvec, &Loss, grad, Hessian, opt);

		g_Loss[Idx] = Loss;
		for (unsigned int i = 0; i < vnum; i++)
			g_grad[Idx * vnum + i] = grad[i];
		for (unsigned int i = 0; i < vnum * vnum; i++)
			g_Hessian[Idx * vnum * vnum + i] = Hessian[i];

		// initialize the lambda and pvec
		g_lambda[Idx] = INIT_LAMBDA;
		for (unsigned int i = 0; i < vnum; i++)
			g_pvec[Idx * vnum + i] = INIT_ERR;
		
		// initialize the push_flag.
		g_push_flag[Idx] = 1;
	}
	__syncthreads();
}



/*! @brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
			Initialize the g_xvec, g_maxJump	
			Initialize the g_Loss, g_grad, g_Hessian
			Initialize the g_lambda at INIT_LAMBDA for Levenberg-Marquardt optimization.
			Initialize the g_pvec at INIT_ERR
			Initialize the g_push_flag at 1
*/
__global__ void Init_2ch(int NFits, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF,
	int warpdeg, int* g_lu, float* coeff_R2T, 
	float* g_xvec, float* g_maxJump, float* g_Loss, float* g_grad, float* g_Hessian, float* g_pvec, float* g_lambda, char* g_push_flag, int opt)
{
	const int nchannels = 2, vnum = NDIM + 4;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int *lu = nullptr;
	float Loss = 0.0f, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};
	float *dataim = nullptr, *varim = nullptr;
	float *xvec = nullptr, *maxJump = nullptr;

	if (Idx < NFits) {
		// Load the data
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		lu = g_lu + Idx * nchannels * 2;
		xvec = g_xvec + Idx * vnum;
		maxJump = g_maxJump + Idx * vnum;
		init_xvec_2ch(boxsz, splineszz, dataim, xvec);
		init_maxJump_2ch(boxsz, splineszz, xvec, maxJump);
		
		// initialize the Loss, grad, and Hessian
		get_lgH_2ch(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, xvec, warpdeg, lu, coeff_R2T, &Loss, grad, Hessian, opt);
		
		g_Loss[Idx] = Loss;
		for (unsigned int i = 0; i < vnum; i++)
			g_grad[Idx * vnum + i] = grad[i];
		for (unsigned int i = 0; i < vnum * vnum; i++)
			g_Hessian[Idx * vnum * vnum + i] = Hessian[i];

		// Initializition lambda and pvec.
		g_lambda[Idx] = INIT_LAMBDA;
		for (unsigned int i = 0; i < vnum; i++)
			g_pvec[Idx * vnum + i] = INIT_ERR;
		
		// initialize the push_flag.
		g_push_flag[Idx] = 1;
	}
	__syncthreads();
}



/*! @brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
			Initialize the g_xvec, g_maxJump	
			Initialize the g_Loss, g_grad, g_Hessian
			Initialize the g_lambda at INIT_LAMBDA for Levenberg-Marquardt optimization.
			Initialize the g_pvec at INIT_ERR
			Initialize the g_push_flag at 1
*/
__global__ void Init_BP(int NFits, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF,
	int warpdeg, int* g_lu, float* coeff_R2T, 
	float* g_xvec, float* g_maxJump, float* g_Loss, float* g_grad, float* g_Hessian, float* g_pvec, float* g_lambda, char* g_push_flag, int opt)
{
	const int nchannels = 2, vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int *lu = nullptr;
	float Loss = 0.0f, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};
	float *dataim = nullptr, *varim = nullptr;
	float *xvec = nullptr, *maxJump = nullptr;

	if (Idx < NFits) {
		// initialize the xvec
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		lu = g_lu + Idx * nchannels * 2;
		xvec = g_xvec + Idx * vnum;
		maxJump = g_maxJump + Idx * vnum;
		init_xvec_BP(boxsz, splineszz, dataim, xvec);
		init_maxJump_BP(boxsz, splineszz, xvec, maxJump);
		
		// initialize the Loss, grad, and Hessian
		get_lgH_BP(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, xvec, warpdeg, lu, coeff_R2T, &Loss, grad, Hessian, opt);
		
		g_Loss[Idx] = Loss;
		for (unsigned int i = 0; i < vnum; i++)
			g_grad[Idx * vnum + i] = grad[i];
		for (unsigned int i = 0; i < vnum * vnum; i++)
			g_Hessian[Idx * vnum * vnum + i] = Hessian[i];

		// Initializition lambda and pvec.
		g_lambda[Idx] = INIT_LAMBDA;
		for (unsigned int i = 0; i < vnum; i++)
			g_pvec[Idx * vnum + i] = INIT_ERR;
		
		// initialize the push_flag.
		g_push_flag[Idx] = 1;
	}
	__syncthreads();
}



/*!	@brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
			Single iteration of a Levenberg-Marguardt update of the xvec.
*/
__global__ void LMupdate_1ch(int NFits, char* g_push_flag, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF, 
	float* g_maxJump, float* g_lambda, float* g_xvec, float* g_grad, float* g_Hessian, float* g_pvec, float* g_Loss, int opt)
{
	const int vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	char push_flag = 0, decomp_flag = 0;
	
	float Loss = 0.0f, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};
	float xvec[vnum] = {0.0f}, pvec[vnum] = {0.0f}, maxJump[vnum] = {0.0f}, L[vnum * vnum] = {0.0f};

	float *dataim = nullptr, *varim = nullptr;
	float lambda = 0.0f, mu = 1.0f;
	
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
		if ( decomp_flag == 1) {
			// calculate the pvec H * pvec = grad
			Cholesky_solve(L, grad, vnum, pvec);
			
			// update and refine xvec and pvec
			for (unsigned int i = 0; i < vnum; i++) {
				if (pvec[i] / g_pvec[Idx * vnum + i] < -0.5f)
					maxJump[i] *= 0.5f; 
				pvec[i] = pvec[i] / (1.0f + fabs(pvec[i] / maxJump[i]));
				xvec[i] = xvec[i] - pvec[i];
			}
			boxConstrain_1ch(boxsz, splineszz, xvec);

			// calculate the loss, grad, and Heissian with the new xvec
			get_lgH_1ch(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, xvec, &Loss, grad, Hessian, opt);
			
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
			Single iteration of a Levenberg-Marguardt update of the xvec.
*/
__global__ void LMupdate_2ch(int NFits, char* g_push_flag, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF,
	int warpdeg, int* g_lu, float* coeff_R2T,
	float* g_maxJump, float* g_lambda, float* g_xvec, float* g_grad, float* g_Hessian, float* g_pvec, float* g_Loss, int opt)
{
	const int nchannels = 2, vnum = NDIM + 4;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	char push_flag = 0, decomp_flag = 0;
	
	int *lu = nullptr;
	float Loss = 0.0f, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};
	float xvec[vnum] = {0.0f}, pvec[vnum] = {0.0f}, maxJump[vnum] = {0.0f}, L[vnum * vnum] = {0.0f};

	float *dataim = nullptr, *varim = nullptr; 
	float lambda = 0.0f, mu = 1.0f;
	
	if (Idx < NFits)
		push_flag = g_push_flag[Idx];
	__syncthreads();

	if (push_flag) {
		// LOAD DATA FROM THE GLOBAL MEMORY
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		lu = g_lu + Idx * nchannels * 2;
		lambda = g_lambda[Idx];
		mu = 1.0f + lambda;
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
		memcpy(grad, g_grad + Idx * vnum, vnum * sizeof(float));
		memcpy(maxJump, g_maxJump + Idx * vnum, vnum * sizeof(float));
		memcpy(Hessian, g_Hessian + Idx * vnum * vnum, vnum * vnum * sizeof(float));

		// CHOLESKY DECOMPOSITION OF THE HEISSIAN
		decomp_flag = Cholesky_decomp(Hessian, vnum, L);
		
		// DECOMPOSIBLE
		if (decomp_flag == 1) {
			// calculate the pvec H * pvec = grad
			Cholesky_solve(L, grad, vnum, pvec);
			
			// update and refine xvec and pvec
			for (unsigned int i = 0; i < vnum; i++) {
				if (pvec[i] / g_pvec[Idx * vnum + i] < -0.5f)
					maxJump[i] *= 0.5f; 
				pvec[i] = pvec[i] / (1.0f + fabs(pvec[i] / maxJump[i]));
				xvec[i] = xvec[i] - pvec[i];
			}
			boxConstrain_2ch(boxsz, splineszz, xvec);
			
			// calculate the Loss, grad, and Heissian with the new xvec
			get_lgH_2ch(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, xvec, warpdeg, lu, coeff_R2T, &Loss, grad, Hessian, opt);

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
			Single iteration of a Levenberg-Marguardt update of the xvec.
*/
__global__ void LMupdate_BP(int NFits, char* g_push_flag, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF,
	int warpdeg, int* g_lu, float* coeff_R2T,
	float* g_maxJump, float* g_lambda, float* g_xvec, float* g_grad, float* g_Hessian, float* g_pvec, float* g_Loss, int opt)
{
	const int nchannels = 2, vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	char push_flag = 0, decomp_flag = 0;
	
	int *lu = nullptr;
	float Loss = 0.0f, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};
	float xvec[vnum] = {0.0f}, pvec[vnum] = {0.0f}, maxJump[vnum] = {0.0f}, L[vnum * vnum] = {0.0f};

	float *dataim = nullptr, *varim = nullptr; 
	float lambda = 0.0f, mu = 1.0f;
	
	if (Idx < NFits)
		push_flag = g_push_flag[Idx];
	__syncthreads();

	if (push_flag) {
		// LOAD DATA FROM THE GLOBAL MEMORY
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		lu = g_lu + Idx * nchannels * 2;
		lambda = g_lambda[Idx];
		mu = 1.0f + lambda;
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
		memcpy(grad, g_grad + Idx * vnum, vnum * sizeof(float));
		memcpy(maxJump, g_maxJump + Idx * vnum, vnum * sizeof(float));
		memcpy(Hessian, g_Hessian + Idx * vnum * vnum, vnum * vnum * sizeof(float));

		// CHOLESKY DECOMPOSITION OF THE HEISSIAN
		decomp_flag = Cholesky_decomp(Hessian, vnum, L);
		
		// DECOMPOSIBLE
		if (decomp_flag == 1) {
			// calculate the pvec H * pvec = grad
			Cholesky_solve(L, grad, vnum, pvec);
			
			// update and refine xvec and pvec
			for (unsigned int i = 0; i < vnum; i++) {
				if (pvec[i] / g_pvec[Idx * vnum + i] < -0.5f)
					maxJump[i] *= 0.5f; 
				pvec[i] = pvec[i] / (1.0f + fabs(pvec[i] / maxJump[i]));
				xvec[i] = xvec[i] - pvec[i];
			}
			boxConstrain_BP(boxsz, splineszz, xvec);
			
			// calculate the Loss, grad, and Heissian with the new xvec
			get_lgH_BP(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, xvec, warpdeg, lu, coeff_R2T, &Loss, grad, Hessian, opt);

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
__global__ void getCRLB_1ch(int NFits, int boxsz, int splineszx, int splineszz, float* g_var, float* coeff_PSF, 
	float* g_xvec, float* g_Loss, float* g_Hessian, float* g_CRLB, int opt)
{
	const int vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	char cholesky_info = 0;
	
	float xvec[vnum] = {0.0f}, FIM[vnum * vnum] = {0.0f}, L[vnum * vnum] = {0.0f}; 
	float *varim = nullptr;

	if (Idx < NFits){
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));

		// construct the Fisher Information Matrix (FIM)
		if (opt == 0) {
			varim = (g_var) ? g_var + Idx * boxsz * boxsz : nullptr;
			get_FIM_1ch(boxsz, splineszx, splineszz, varim, coeff_PSF, xvec, FIM);
		}
		else
			for (unsigned int i = 0; i < vnum * vnum; i++)
				FIM[i] = g_Hessian[Idx * vnum * vnum + i];
		
		cholesky_info = Cholesky_decomp(FIM, vnum, L);
		if (cholesky_info == 1) {
			Cholesky_invert(L, vnum, FIM);
			for (unsigned int i = 0; i < vnum; i++)
				g_CRLB[Idx * vnum + i] = (opt == 0) ? FIM[i * vnum + i] : g_Loss[Idx] / (float)boxsz / (float)boxsz * FIM[i * vnum + i];
		}
	}	
	__syncthreads();
}



/*!	@brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
			calculate the CRLB	
*/
__global__ void getCRLB_2ch(int NFits, int boxsz, int splineszx, int splineszz, float* g_var, float* coeff_PSF, 
	int warpdeg, int* g_lu, float* coeff_R2T, 
	float* g_xvec, float* g_Loss, float* g_Hessian, float* g_CRLB, int opt)
{
	const int nchannels = 2, vnum = NDIM + 4;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	char cholesky_info = 0;

	int *lu = nullptr;
	float xvec[vnum] = {0.0f}, FIM[vnum * vnum] = {0.0f}, L[vnum * vnum] = {0.0f};
	float *varim = nullptr; 

	if (Idx < NFits) {
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
		
		// construct the Fisher Information Matrix (FIM)
		if (opt == 0) {
			varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
			lu = g_lu + Idx * nchannels * 2;
			get_FIM_2ch(boxsz, splineszx, splineszz, varim, coeff_PSF, xvec, warpdeg, lu, coeff_R2T, FIM);
		}
		else
			for (unsigned int i = 0; i < vnum * vnum; i++)
				FIM[i] = g_Hessian[Idx * vnum * vnum + i];
		
		cholesky_info = Cholesky_decomp(FIM, vnum, L);
		if (cholesky_info == 1) {
			Cholesky_invert(L, vnum, FIM);
			for (unsigned int i = 0; i < vnum; i++)
				g_CRLB[Idx * vnum + i] = (opt == 0) ? FIM[i * vnum + i] : g_Loss[Idx] / (float)nchannels / (float)boxsz / (float)boxsz * FIM[i * vnum + i];
		}
	}	
	__syncthreads();
}



/*!	@brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
			calculate the CRLB	
*/
__global__ void getCRLB_BP(int NFits, int boxsz, int splineszx, int splineszz, float* g_var, float* coeff_PSF, 
	int warpdeg, int* g_lu, float* coeff_R2T, 
	float* g_xvec, float* g_Loss, float* g_Hessian, float* g_CRLB, int opt)
{
	const int nchannels = 2, vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	char cholesky_info = 0;

	int *lu = nullptr;
	float xvec[vnum] = {0.0f}, FIM[vnum * vnum] = {0.0f}, L[vnum * vnum] = {0.0f};
	float *varim = nullptr; 

	if (Idx < NFits) {
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
		
		// construct the Fisher Information Matrix (FIM)
		if (opt == 0) {
			varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
			lu = g_lu + Idx * nchannels * 2;
			get_FIM_BP(boxsz, splineszx, splineszz, varim, coeff_PSF, xvec, warpdeg, lu, coeff_R2T, FIM);
		}
		else
			for (unsigned int i = 0; i < vnum * vnum; i++)
				FIM[i] = g_Hessian[Idx * vnum * vnum + i];
		
		cholesky_info = Cholesky_decomp(FIM, vnum, L);
		if (cholesky_info == 1) {
			Cholesky_invert(L, vnum, FIM);
			for (unsigned int i = 0; i < vnum; i++)
				g_CRLB[Idx * vnum + i] = (opt == 0) ? FIM[i * vnum + i] : g_Loss[Idx] / (float)nchannels / (float)boxsz / (float)boxsz * FIM[i * vnum + i];
		}
	}	
	__syncthreads();
}

#endif