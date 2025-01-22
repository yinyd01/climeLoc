#pragma once
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "matrix.cuh"
#include "functionals.cuh"

#ifndef KERNELS_1_CUH
#define KERNELS_1_CUH

/*! @brief	kernel functions for dualview PSFfit_MULPA
	@param		nnum:				const int, 1, the number of emitters
	@param		nchannels:			const int, {1, 2}, the number of channels
	@param[in]	NFits: 				int, the number of nchannel-PSF
	@param[in]	boxsz: 				int, the size of the PSF square
	@param[in]	splineszx: 			int, the size of the spline cube (x-y-axis)
	@param[in]	splineszz: 			int, the size of the spline cube (z-axis)
	@param[in]	g_data: 			(NFits * nchannels * boxsz * boxsz) float, PSF data
	@param[in]	g_var: 				(NFits * nchannels * boxsz * boxsz) float, pixel-dependent variance, NULL for pixel-independent variance
	@param[in]	g_coeff_PSF: 		(nchannels * splineszx * splineszx * splineszz * 64) float, spline coefficient for the nchannel 3d PSF
	@param[in]	warpdeg:			int, the degree of the polynomial function that warps 2d locations from one channel to the other
	@param[in]	g_lu:				(NFits * nchannels * 2) int, [lc, uc] the left and upper corner of the PSF square in each channel (image coordinates)
	@param[in]	coeff_R2T:			((nchannels - 1) * 2 * warpdeg * warpdeg) float, the [coeffx_B2A, coeffy_B2A] coefficients of the polynomial warping from refernce channel to each of the target channel	
	@param[out]	g_xvec:				(NFits * vnum) float, see definitions.h
	@param[out]	g_CRLB:				(NFits * vnum) float, the CRLB corresponding to each of the variables in xvec, respectively
	@param[out] g_Loss:				(NFits) float, Loss value for the optimization of a PSF fitting.
	@param[out] g_nnum:				(NFits) unsigned int, number of PSFs of a fit.
	@param		g_iter_push: 		(NFits) char, flag for whether iteration should proceed.
	@param		g_maxJump: 			(NFits * vnum) float, maxJump control vector for each of the parameter for xvec.
	@param		g_grad: 			(NFits * vnum) float, gradient vector of w.r.t the xvec.
	@param		g_Hessian: 			(NFits * vnum * vnum) float, Hessian matrix w.r.t the xvec.
	@param		g_pvec: 			(NFits * vnum) float, increment of xvec for a single iteration.
	@param		g_lambda: 			(NFits) float, lambda value for Levenberg-Marquardt optimization.
*/



/*! @brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>> Initialize the g_xvec (assuming there is only one emitter) */
__global__ void PreInit_1ch(int NFits, int boxsz, int splineszz, float* g_data, float* g_xvec)
{
	const int nchannels = 1, vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *xvec = nullptr;

	if (Idx < NFits) {	
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		xvec = g_xvec + Idx * vnum;
		init_xvec_1ch(boxsz, splineszz, dataim, xvec);
	}
	__syncthreads();
}



/*! @brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>> Initialize the g_xvec (assuming there is only one emitter) */
__global__ void PreInit_2ch(int NFits, int boxsz, int splineszz, float* g_data, float* g_xvec)
{
	const int nchannels = 2, vnum = NDIM + 4;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *xvec = nullptr;

	if (Idx < NFits) {	
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		xvec = g_xvec + Idx * vnum;
		init_xvec_2ch(boxsz, splineszz, dataim, xvec);
	}
	__syncthreads();
}



/*! @brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>> Initialize the g_xvec (assuming there is only one emitter) */
__global__ void PreInit_BP(int NFits, int boxsz, int splineszz, float* g_data, float* g_xvec)
{
	const int nchannels = 2, vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *xvec = nullptr;

	if (Idx < NFits) {	
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		xvec = g_xvec + Idx * vnum;
		init_xvec_BP(boxsz, splineszz, dataim, xvec);
	}
	__syncthreads();
}



/*! @brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>> Initialization with a given xvec
			Initialize the g_maxJump, g_Loss, g_grad, g_Hessian
			Initialize the g_lambda at INIT_LAMBDA for Levenberg-Marquardt optimization.
			Initialize the g_pvec at INIT_ERR
			Initialize the g_iter_push at 1
*/
__global__ void Init_1_1ch(int NFits, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF, float* g_xvec, 
	float* g_maxJump, float* g_Loss, float* g_grad, float* g_Hessian, float* g_pvec, float* g_lambda, char* g_iter_push)
{
	const int nnum = 1, nchannels = 1, vnum0 = NDIM + nchannels;															
	const int vnum = nnum * vnum0 + nchannels;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *varim = nullptr;
	float *xvec = nullptr, *maxJump = nullptr;

	int offset_x[nnum] = {0}, offset_y[nnum] = {0}, offset_z[nnum] = {0};
	float deltaf[nnum * 64] = {0.0f}, ddeltaf_dx[nnum * 64] = {0.0f}, ddeltaf_dy[nnum * 64] = {0.0f}, ddeltaf_dz[nnum * 64] = {0.0f};
	float loc[nnum * NDIM] = {0.0f}, psf[nnum] = {0.0f}, dpsf_dloc[nnum * NDIM] = {0.0f}, deriv1[vnum] = {0.0f};
	
	float Loss = 0.0f, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};

	if (Idx < NFits) {
		// load data
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		xvec = g_xvec + Idx * vnum;
		
		// initialize the maxJump vector
		maxJump = g_maxJump + Idx * vnum;
		init_maxJump_1ch(nnum, boxsz, splineszz, xvec, maxJump);
		
		get_lgH_1ch(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, nnum, xvec,
			loc, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, psf, dpsf_dloc, deriv1,
			&Loss, grad, Hessian);

		g_Loss[Idx] = Loss;
		memcpy(g_grad + Idx * vnum, grad, vnum * sizeof(float));
		memcpy(g_Hessian + Idx * vnum * vnum, Hessian, vnum * vnum * sizeof(float));

		// Initializition lambda and pvec.
		g_lambda[Idx] = INIT_LAMBDA;
		for (unsigned int i = 0; i < vnum; i++)
			g_pvec[Idx * vnum + i] = INIT_ERR;
		
		// initialize the push_flag.
		g_iter_push[Idx] = 1;
	}
	__syncthreads();
}



__global__ void Init_1_2ch(int NFits, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF, 
	int warpdeg, int* g_lu, float* coeff_R2T, float* g_xvec, 
	float* g_maxJump, float* g_Loss, float* g_grad, float* g_Hessian, float* g_pvec, float* g_lambda, char* g_iter_push)
{
	const int nnum = 1, nchannels = 2, vnum0 = NDIM + nchannels;
	const int vnum = nnum * vnum0 + nchannels;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *varim = nullptr;
	float *xvec = nullptr, *maxJump = nullptr;

	int *lu = nullptr;
	float loc_ref[nnum * NDIM] = {0.0f}, trans_deriv[nnum * 4] = {0.0f};
	
	int offset_x[nnum] = {0}, offset_y[nnum] = {0}, offset_z[nnum] = {0};
	float deltaf[nnum * 64] = {0.0f}, ddeltaf_dx[nnum * 64] = {0.0f}, ddeltaf_dy[nnum * 64] = {0.0f}, ddeltaf_dz[nnum * 64] = {0.0f};
	float loc[nnum * NDIM] = {0.0f}, psf[nnum] = {0.0f}, dpsf_dloc[nnum * NDIM] = {0.0f}, deriv1[vnum] = {0.0f};
	
	float Loss = 0.0f, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};

	if (Idx < NFits) {
		// load the data
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		lu = g_lu + Idx * nchannels * 2;
		xvec = g_xvec + Idx * vnum;
		
		// initialize the maxJump vector
		maxJump = g_maxJump + Idx * vnum;
		init_maxJump_2ch(nnum, boxsz, splineszz, xvec, maxJump);

		get_lgH_2ch(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, nnum, xvec, 
			warpdeg, lu, coeff_R2T, loc_ref, trans_deriv,
			loc, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, psf, dpsf_dloc, deriv1,
			&Loss, grad, Hessian);
		
		g_Loss[Idx] = Loss;
		memcpy(g_grad + Idx * vnum, grad, vnum * sizeof(float));
		memcpy(g_Hessian + Idx * vnum * vnum, Hessian, vnum * vnum * sizeof(float));

		// Initializition lambda and pvec.
		g_lambda[Idx] = INIT_LAMBDA;
		for (unsigned int i = 0; i < vnum; i++)
			g_pvec[Idx * vnum + i] = INIT_ERR;
		
		// initialize the push_flag.
		g_iter_push[Idx] = 1;
	}
	__syncthreads();
}



__global__ void Init_1_BP(int NFits, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF, 
	int warpdeg, int* g_lu, float* coeff_R2T, float* g_xvec, 
	float* g_maxJump, float* g_Loss, float* g_grad, float* g_Hessian, float* g_pvec, float* g_lambda, char* g_iter_push)
{
	const int nnum = 1, nchannels = 2, vnum0 = NDIM + 1;
	const int vnum = nnum * vnum0 + 1;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *varim = nullptr;
	float *xvec = nullptr, *maxJump = nullptr;

	int *lu = nullptr;
	float loc_ref[nnum * NDIM] = {0.0f}, trans_deriv[nnum * 4] = {0.0f};
	
	int offset_x[nnum] = {0}, offset_y[nnum] = {0}, offset_z[nnum] = {0};
	float deltaf[nnum * 64] = {0.0f}, ddeltaf_dx[nnum * 64] = {0.0f}, ddeltaf_dy[nnum * 64] = {0.0f}, ddeltaf_dz[nnum * 64] = {0.0f};
	float loc[nnum * NDIM] = {0.0f}, psf[nnum] = {0.0f}, dpsf_dloc[nnum * NDIM] = {0.0f}, deriv1[vnum] = {0.0f};
	
	float Loss = 0.0f, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};

	if (Idx < NFits) {
		// load the data
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		lu = g_lu + Idx * nchannels * 2;
		xvec = g_xvec + Idx * vnum;
		
		// initialize the maxJump vector
		maxJump = g_maxJump + Idx * vnum;
		init_maxJump_BP(nnum, boxsz, splineszz, xvec, maxJump);

		get_lgH_BP(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, nnum, xvec, 
			warpdeg, lu, coeff_R2T, loc_ref, trans_deriv,
			loc, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, psf, dpsf_dloc, deriv1,
			&Loss, grad, Hessian);
		
		g_Loss[Idx] = Loss;
		memcpy(g_grad + Idx * vnum, grad, vnum * sizeof(float));
		memcpy(g_Hessian + Idx * vnum * vnum, Hessian, vnum * vnum * sizeof(float));

		// Initializition lambda and pvec.
		g_lambda[Idx] = INIT_LAMBDA;
		for (unsigned int i = 0; i < vnum; i++)
			g_pvec[Idx * vnum + i] = INIT_ERR;
		
		// initialize the push_flag.
		g_iter_push[Idx] = 1;
	}
	__syncthreads();
}



/*!	@brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>> Single iteration of a Levenberg-Marguardt update of the xvec. */
__global__ void LMupdate_1_1ch(int NFits, char* g_iter_push, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF,
	float* g_xvec, float* g_Loss, float* g_maxJump, float* g_lambda, float* g_grad, float* g_Hessian, float* g_pvec)
{
	const int nnum = 1, nchannels = 1, vnum0 = NDIM + nchannels;
	const int vnum = nnum * vnum0 + nchannels;
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *varim = nullptr;

	char push_flag = 0, decomp_flag = 0;
	
	int offset_x[nnum] = {0}, offset_y[nnum] = {0}, offset_z[nnum] = {0};
	float deltaf[nnum * 64] = {0.0f}, ddeltaf_dx[nnum * 64] = {0.0f}, ddeltaf_dy[nnum * 64] = {0.0f}, ddeltaf_dz[nnum * 64] = {0.0f};
	float loc[nnum * NDIM] = {0.0f}, psf[nnum] = {0.0f}, dpsf_dloc[nnum * NDIM] = {0.0f}, deriv1[vnum] = {0.0f};
	
	float Loss = 0.0f, Loss_old = 0.0f, lambda = 0.0f, mu = 1.0f;
	float xvec[vnum] = {0.0f}, pvec[vnum] = {0.0f}, maxJump[vnum] = {0.0f};
	float grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f}, L[vnum * vnum] = {0.0f};
	
	if (Idx < NFits)
		push_flag = (g_iter_push[Idx] && g_lambda[Idx] < LMBDA_MAX);
	__syncthreads();

	if (push_flag) {
		// LOAD DATA FROM THE GLOBAL MEMORY
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		lambda = g_lambda[Idx];
		mu = 1.0f + lambda;
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
		memcpy(grad, g_grad + Idx * vnum, vnum * sizeof(float));
		memcpy(maxJump, g_maxJump + Idx * vnum, vnum * sizeof(float));
		memcpy(Hessian, g_Hessian + Idx * vnum * vnum, vnum * vnum * sizeof(float));
		Loss_old = g_Loss[Idx];
		
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
			boxConstrain_1ch(nnum, boxsz, splineszz, xvec);
			
			get_lgH_1ch(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, nnum, xvec,
				loc, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, psf, dpsf_dloc, deriv1,
				&Loss, grad, Hessian);
			
			// evaluate the Loss
			if (Loss >= ACCEPTANCE * Loss_old) {
				mu = max((1 + lambda * SCALE_UP) / (1 + lambda), 1.3f);
				g_lambda[Idx] = SCALE_UP * lambda;
				for (unsigned int i = 0; i < vnum; i++)
					g_Hessian[Idx * vnum * vnum + i * vnum + i] = g_Hessian[Idx * vnum * vnum + i * vnum + i] * mu;
			}
			else {
				if (Loss < Loss_old) {
					mu = 1 + SCALE_DOWN * lambda;
					g_lambda[Idx] = SCALE_DOWN * lambda;
				}
				
				memcpy(g_xvec + Idx * vnum, xvec, vnum * sizeof(float));
				memcpy(g_pvec + Idx * vnum, pvec, vnum * sizeof(float));
				memcpy(g_maxJump + Idx * vnum, maxJump, vnum * sizeof(float));
				memcpy(g_grad + Idx * vnum, grad, vnum * sizeof(float));
				memcpy(g_Hessian + Idx * vnum * vnum, Hessian, vnum * vnum * sizeof(float));
				g_iter_push[Idx] = (fabs(Loss - Loss_old) >= (fabs(Loss) * OPTTOL));
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



__global__ void LMupdate_1_2ch(int NFits, char* g_iter_push, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF, 
	int warpdeg, int* g_lu, float* coeff_R2T, 
	float* g_xvec, float* g_Loss, float* g_maxJump, float* g_lambda, float* g_grad, float* g_Hessian, float* g_pvec)
{
	const int nnum = 1, nchannels = 2, vnum0 = NDIM + nchannels;
	const int vnum = nnum * vnum0 + nchannels;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *varim = nullptr; 

	char push_flag = 0, decomp_flag = 0;
	
	int *lu = nullptr;
	float loc_ref[nnum * NDIM] = {0.0f}, trans_deriv[nnum * 4] = {0.0f};
	
	int offset_x[nnum] = {0}, offset_y[nnum] = {0}, offset_z[nnum] = {0};
	float deltaf[nnum * 64] = {0.0f}, ddeltaf_dx[nnum * 64] = {0.0f}, ddeltaf_dy[nnum * 64] = {0.0f}, ddeltaf_dz[nnum * 64] = {0.0f};
	float loc[nnum * NDIM] = {0.0f}, psf[nnum] = {0.0f}, dpsf_dloc[nnum * NDIM] = {0.0f}, deriv1[vnum] = {0.0f};
	
	float Loss = 0.0f, Loss_old = 0.0f, lambda = 0.0f, mu = 1.0f;
	float xvec[vnum] = {0.0f}, pvec[vnum] = {0.0f}, maxJump[vnum] = {0.0f};
	float grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f}, L[vnum * vnum] = {0.0f};
	
	if (Idx < NFits)
		push_flag = (g_iter_push[Idx] && g_lambda[Idx] < LMBDA_MAX);
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
		Loss_old = g_Loss[Idx];
		
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
			boxConstrain_2ch(nnum, boxsz, splineszz, xvec);
			
			get_lgH_2ch(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, nnum, xvec, 
				warpdeg, lu, coeff_R2T, loc_ref, trans_deriv,
				loc, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, psf, dpsf_dloc, deriv1,
				&Loss, grad, Hessian);
			
			// evaluate the Loss
			if (Loss >= ACCEPTANCE * Loss_old) {
				mu = max((1 + lambda * SCALE_UP) / (1 + lambda), 1.3f);
				g_lambda[Idx] = SCALE_UP * lambda;
				for (unsigned int i = 0; i < vnum; i++)
					g_Hessian[Idx * vnum * vnum + i * vnum + i] = g_Hessian[Idx * vnum * vnum + i * vnum + i] * mu;
			}
			else {
				if (Loss < Loss_old) {
					mu = 1 + SCALE_DOWN * lambda;
					g_lambda[Idx] = SCALE_DOWN * lambda;
				}
				
				memcpy(g_xvec + Idx * vnum, xvec, vnum * sizeof(float));
				memcpy(g_pvec + Idx * vnum, pvec, vnum * sizeof(float));
				memcpy(g_maxJump + Idx * vnum, maxJump, vnum * sizeof(float));
				memcpy(g_grad + Idx * vnum, grad, vnum * sizeof(float));
				memcpy(g_Hessian + Idx * vnum * vnum, Hessian, vnum * vnum * sizeof(float));
				g_iter_push[Idx] = (fabs(Loss - Loss_old) >= (fabs(Loss) * OPTTOL));
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



__global__ void LMupdate_1_BP(int NFits, char* g_iter_push, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* coeff_PSF, 
	int warpdeg, int* g_lu, float* coeff_R2T, 
	float* g_xvec, float* g_Loss, float* g_maxJump, float* g_lambda, float* g_grad, float* g_Hessian, float* g_pvec)
{
	const int nnum = 1, nchannels = 2, vnum0 = NDIM + 1;
	const int vnum = nnum * vnum0 + 1;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dataim = nullptr, *varim = nullptr; 

	char push_flag = 0, decomp_flag = 0;
	
	int *lu = nullptr;
	float loc_ref[nnum * NDIM] = {0.0f}, trans_deriv[nnum * 4] = {0.0f};
	
	int offset_x[nnum] = {0}, offset_y[nnum] = {0}, offset_z[nnum] = {0};
	float deltaf[nnum * 64] = {0.0f}, ddeltaf_dx[nnum * 64] = {0.0f}, ddeltaf_dy[nnum * 64] = {0.0f}, ddeltaf_dz[nnum * 64] = {0.0f};
	float loc[nnum * NDIM] = {0.0f}, psf[nnum] = {0.0f}, dpsf_dloc[nnum * NDIM] = {0.0f}, deriv1[vnum] = {0.0f};
	
	float Loss = 0.0f, Loss_old = 0.0f, lambda = 0.0f, mu = 1.0f;
	float xvec[vnum] = {0.0f}, pvec[vnum] = {0.0f}, maxJump[vnum] = {0.0f};
	float grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f}, L[vnum * vnum] = {0.0f};
	
	if (Idx < NFits)
		push_flag = (g_iter_push[Idx] && g_lambda[Idx] < LMBDA_MAX);
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
		Loss_old = g_Loss[Idx];
		
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
			boxConstrain_BP(nnum, boxsz, splineszz, xvec);
			
			get_lgH_BP(boxsz, splineszx, splineszz, dataim, varim, coeff_PSF, nnum, xvec, 
				warpdeg, lu, coeff_R2T, loc_ref, trans_deriv,
				loc, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, psf, dpsf_dloc, deriv1,
				&Loss, grad, Hessian);
			
			// evaluate the Loss
			if (Loss >= ACCEPTANCE * Loss_old) {
				mu = max((1 + lambda * SCALE_UP) / (1 + lambda), 1.3f);
				g_lambda[Idx] = SCALE_UP * lambda;
				for (unsigned int i = 0; i < vnum; i++)
					g_Hessian[Idx * vnum * vnum + i * vnum + i] = g_Hessian[Idx * vnum * vnum + i * vnum + i] * mu;
			}
			else {
				if (Loss < Loss_old) {
					mu = 1 + SCALE_DOWN * lambda;
					g_lambda[Idx] = SCALE_DOWN * lambda;
				}
				
				memcpy(g_xvec + Idx * vnum, xvec, vnum * sizeof(float));
				memcpy(g_pvec + Idx * vnum, pvec, vnum * sizeof(float));
				memcpy(g_maxJump + Idx * vnum, maxJump, vnum * sizeof(float));
				memcpy(g_grad + Idx * vnum, grad, vnum * sizeof(float));
				memcpy(g_Hessian + Idx * vnum * vnum, Hessian, vnum * vnum * sizeof(float));
				g_iter_push[Idx] = (fabs(Loss - Loss_old) >= (fabs(Loss) * OPTTOL));
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



/*!	@brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>> get the crlb */
__global__ void getCRLB_1_1ch(int NFits, int boxsz, int splineszx, int splineszz, float* g_var, float* coeff_PSF, float* g_xvec, float* g_CRLB)
{
	const int nnum = 1, nchannels = 1, vnum0 = NDIM + nchannels;
	const int vnum = nnum * vnum0 + nchannels;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *varim = nullptr;

	char cholesky_info = 0;

	int offset_x[nnum] = {0}, offset_y[nnum] = {0}, offset_z[nnum] = {0};
	float deltaf[nnum * 64] = {0.0f}, ddeltaf_dx[nnum * 64] = {0.0f}, ddeltaf_dy[nnum * 64] = {0.0f}, ddeltaf_dz[nnum * 64] = {0.0f};
	float loc[nnum * NDIM] = {0.0f}, psf[nnum] = {0.0f}, dpsf_dloc[nnum * NDIM] = {0.0f}, deriv1[vnum] = {0.0f};
	
	float xvec[vnum] = {0.0f}, FIM[vnum * vnum] = {0.0f}, L[vnum * vnum] = {0.0f};

	if (Idx < NFits) {
		// load data
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
		
		// construct the Fisher Information Matrix (FIM)
		get_FIM_1ch(boxsz, splineszx, splineszz, varim, coeff_PSF, nnum, xvec,
			loc, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, psf, dpsf_dloc, deriv1, FIM);
		
		cholesky_info = Cholesky_decomp(FIM, vnum, L);
		if (cholesky_info == 1) {
			Cholesky_invert(L, vnum, FIM);
			for (unsigned int i = 0; i < vnum; i++)
				g_CRLB[Idx * vnum + i] = FIM[i * vnum + i];
		}
	}	
	__syncthreads();
}



__global__ void getCRLB_1_2ch(int NFits, int boxsz, int splineszx, int splineszz, float* g_var, float* coeff_PSF, 
	int warpdeg, int* g_lu, float* coeff_R2T, float* g_xvec, float* g_CRLB)
{
	const int nnum = 1, nchannels = 2, vnum0 = NDIM + nchannels;
	const int vnum = nnum * vnum0 + nchannels;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *varim = nullptr;

	char cholesky_info = 0;

	int *lu = nullptr;
	float loc_ref[nnum * NDIM] = {0.0f}, trans_deriv[nnum * 4] = {0.0f};
	
	int offset_x[nnum] = {0}, offset_y[nnum] = {0}, offset_z[nnum] = {0};
	float deltaf[nnum * 64] = {0.0f}, ddeltaf_dx[nnum * 64] = {0.0f}, ddeltaf_dy[nnum * 64] = {0.0f}, ddeltaf_dz[nnum * 64] = {0.0f};
	float loc[nnum * NDIM] = {0.0f}, psf[nnum] = {0.0f}, dpsf_dloc[nnum * NDIM] = {0.0f}, deriv1[vnum] = {0.0f};
	
	float xvec[vnum] = {0.0f}, FIM[vnum * vnum] = {0.0f}, L[vnum * vnum] = {0.0f};

	if (Idx < NFits) {
		// load data
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		lu = g_lu + Idx * nchannels * 2;
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
		
		// construct the Fisher Information Matrix (FIM)
		get_FIM_2ch(boxsz, splineszx, splineszz, varim, coeff_PSF, nnum, xvec, 
			warpdeg, lu, coeff_R2T, loc_ref, trans_deriv,
			loc, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, psf, dpsf_dloc, deriv1, FIM);
		
		cholesky_info = Cholesky_decomp(FIM, vnum, L);
		if (cholesky_info == 1) {
			Cholesky_invert(L, vnum, FIM);
			for (unsigned int i = 0; i < vnum; i++)
				g_CRLB[Idx * vnum + i] = FIM[i * vnum + i];
		}
	}	
	__syncthreads();
}



__global__ void getCRLB_1_BP(int NFits, int boxsz, int splineszx, int splineszz, float* g_var, float* coeff_PSF, 
	int warpdeg, int* g_lu, float* coeff_R2T, float* g_xvec, float* g_CRLB)
{
	const int nnum = 1, nchannels = 2, vnum0 = NDIM + 1;
	const int vnum = nnum * vnum0 + 1;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *varim = nullptr;

	char cholesky_info = 0;

	int *lu = nullptr;
	float loc_ref[nnum * NDIM] = {0.0f}, trans_deriv[nnum * 4] = {0.0f};
	
	int offset_x[nnum] = {0}, offset_y[nnum] = {0}, offset_z[nnum] = {0};
	float deltaf[nnum * 64] = {0.0f}, ddeltaf_dx[nnum * 64] = {0.0f}, ddeltaf_dy[nnum * 64] = {0.0f}, ddeltaf_dz[nnum * 64] = {0.0f};
	float loc[nnum * NDIM] = {0.0f}, psf[nnum] = {0.0f}, dpsf_dloc[nnum * NDIM] = {0.0f}, deriv1[vnum] = {0.0f};
	
	float xvec[vnum] = {0.0f}, FIM[vnum * vnum] = {0.0f}, L[vnum * vnum] = {0.0f};

	if (Idx < NFits) {
		// load data
		varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz : nullptr;
		lu = g_lu + Idx * nchannels * 2;
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
		
		// construct the Fisher Information Matrix (FIM)
		get_FIM_BP(boxsz, splineszx, splineszz, varim, coeff_PSF, nnum, xvec, 
			warpdeg, lu, coeff_R2T, loc_ref, trans_deriv,
			loc, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, psf, dpsf_dloc, deriv1, FIM);
		
		cholesky_info = Cholesky_decomp(FIM, vnum, L);
		if (cholesky_info == 1) {
			Cholesky_invert(L, vnum, FIM);
			for (unsigned int i = 0; i < vnum; i++)
				g_CRLB[Idx * vnum + i] = FIM[i * vnum + i];
		}
	}	
	__syncthreads();
}

#endif