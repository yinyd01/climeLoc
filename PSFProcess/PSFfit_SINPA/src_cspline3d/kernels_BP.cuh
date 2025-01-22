#pragma once
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "matrix.cuh"
#include "functionals.cuh"

#ifndef KERNELS_BP_CUH
#define KERNELS_BP_CUH

/*! @brief	kernel functions for dualview PSFfit_SINPA
	@param		nchannels:			const int, 2, the number of channels 
	@param		vnum:				const int, NDIM + 2, the number of optimizatin parameters
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
__global__ void Init_BP(int NFits, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* g_coeff_PSF,
	int warpdeg, int* g_lu, float* g_coeff_R2T, 
	float* g_xvec, float* g_maxJump, float* g_Loss, float* g_grad, float* g_Hessian, float* g_pvec, float* g_lambda, char* g_push_flag, int opt)
{
	const int nchannels = 2;
	const int vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int *lu_ref = nullptr, *lu_tar = nullptr;
	float *coeff_R2T = nullptr, trans_deriv[4] = {0.0f};
	
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float *coeff_PSF = nullptr, deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float psf = 0.0f, loc[NDIM] = {0.0f}, dpsf_dloc[NDIM] = {0.0f};
	
	float data = 0.0f, model = 0.0f, Loss = 0.0f;
	float deriv1[vnum] = {0.0f}, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};

	float *dataim = nullptr, *varim = nullptr;
	float *xvec = nullptr, *maxJump = nullptr;
	int chID = 0, pxID = 0, i = 0, j = 0;

	if (Idx < NFits) {
		// initialize the xvec
		dataim = g_data + Idx * nchannels * boxsz * boxsz;
		xvec = g_xvec + Idx * vnum;
		init_xvec_BP(boxsz, splineszz, dataim, xvec);
		
		// initialize the maxJump vector
		maxJump = g_maxJump + Idx * vnum;
		init_maxJump_BP(boxsz, splineszz, xvec, maxJump);
		
		// initialize the Loss, grad, and Hessian
		lu_ref = g_lu + Idx * nchannels * 2;
		memset(grad, 0, vnum * sizeof(float));
		memset(Hessian, 0, vnum * vnum * sizeof(float));
		for (chID = 0, Loss = 0.0f; chID < nchannels; chID++) {
			// load data from the global memory
			dataim = g_data + Idx * nchannels * boxsz * boxsz + chID * boxsz * boxsz;
			varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz + chID * boxsz * boxsz : nullptr;
			coeff_PSF = g_coeff_PSF + chID * splineszx * splineszx * splineszz * 64;
			if (chID == 0) {
				memcpy(loc, xvec, NDIM * sizeof(float));
				trans_deriv[0] = 1.0f; trans_deriv[1] = 0.0f;
				trans_deriv[2] = 0.0f; trans_deriv[3] = 1.0f;
			}
			else {
				coeff_R2T = g_coeff_R2T + (chID - 1) * 2 * warpdeg * warpdeg;
				lu_tar = g_lu + Idx * nchannels * 2 + chID * 2;
				TransR2T(warpdeg, coeff_R2T, lu_ref, lu_tar, xvec, loc, trans_deriv);
			}
			DeltaConstruct3D(splineszx, splineszz, loc, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);
			
			for (pxID = 0; pxID < boxsz * boxsz; pxID++) {
				data = dataim[pxID];
				fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, coeff_PSF, offset_x, offset_y, offset_z, 
					deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, &psf, dpsf_dloc);
				get_pixel_model_BP(psf, xvec, &model);
				get_pixel_deriv_BP(psf, dpsf_dloc, trans_deriv, xvec, deriv1);
				if (varim) {
					data += varim[pxID];
					model += varim[pxID];
				}
				
				accum_loss(data, model, &Loss, opt);
				accum_grad(vnum, data, model, deriv1, grad, opt);
				accum_Hessian(vnum, data, model, deriv1, Hessian, opt);
			}
		}
		for (i = 0; i < vnum; i++)
			for (j = i + 1; j < vnum; j++)
				Hessian[j * vnum + i] = Hessian[i * vnum + j];
		
		g_Loss[Idx] = Loss;
		for (i = 0; i < vnum; i++)
			g_grad[Idx * vnum + i] = grad[i];
		for (i = 0; i < vnum * vnum; i++)
			g_Hessian[Idx * vnum * vnum + i] = Hessian[i];

		// Initializition lambda and pvec.
		g_lambda[Idx] = INIT_LAMBDA;
		for (i = 0; i < vnum; i++)
			g_pvec[Idx * vnum + i] = INIT_ERR;
		
		// initialize the push_flag.
		g_push_flag[Idx] = 1;
	}
	__syncthreads();
}



/*!	@brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
			Single iteration of a Levenberg-Marguardt update of the xvec.
*/
__global__ void LMupdate_BP(int NFits, char* g_push_flag, int boxsz, int splineszx, int splineszz, float* g_data, float* g_var, float* g_coeff_PSF,
	int warpdeg, int* g_lu, float* g_coeff_R2T,
	float* g_maxJump, float* g_lambda, float* g_xvec, float* g_grad, float* g_Hessian, float* g_pvec, float* g_Loss, int opt)
{
	const int nchannels = 2;
	const int vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	char push_flag = 0, decomp_flag = 0;
	
	int *lu_ref = nullptr, *lu_tar = nullptr;
	float *coeff_R2T = nullptr, trans_deriv[4] = {0.0f};
	
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float *coeff_PSF = nullptr, deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float psf = 0.0f, loc[NDIM] = {0.0f}, dpsf_dloc[NDIM] = {0.0f};
	
	float data = 0.0f, model = 0.0f, Loss = 0.0f;
	float deriv1[vnum] = {0.0f}, grad[vnum] = {0.0f}, Hessian[vnum * vnum] = {0.0f};
	float xvec[vnum] = {0.0f}, pvec[vnum] = {0.0f}, maxJump[vnum] = {0.0f}, L[vnum * vnum] = {0.0f};

	float *dataim = nullptr, *varim = nullptr; 
	float lambda = 0.0f, mu = 1.0f;
	int chID = 0, pxID = 0, i = 0, j = 0;
	
	if (Idx < NFits)
		push_flag = g_push_flag[Idx];
	__syncthreads();

	if (push_flag) {
		// LOAD DATA FROM THE GLOBAL MEMORY
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
			for (i = 0; i < vnum; i++) {
				if (pvec[i] / g_pvec[Idx * vnum + i] < -0.5f)
					maxJump[i] *= 0.5f; 
				pvec[i] = pvec[i] / (1.0f + fabs(pvec[i] / maxJump[i]));
				xvec[i] = xvec[i] - pvec[i];
			}
			boxConstrain_BP(boxsz, splineszz, xvec);
			
			// calculate the Loss, grad, and Heissian with the new xvec
			lu_ref = g_lu + Idx * nchannels * 2;
			memset(grad, 0, vnum * sizeof(float));
			memset(Hessian, 0, vnum * vnum * sizeof(float));
			for (chID = 0, Loss = 0.0f; chID < nchannels; chID++) {
				dataim = g_data + Idx * nchannels * boxsz * boxsz + chID * boxsz * boxsz;
				varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz + chID * boxsz * boxsz : nullptr;
				coeff_PSF = g_coeff_PSF + chID * splineszx * splineszx * splineszz * 64;

				if (chID == 0) {
					memcpy(loc, xvec, NDIM * sizeof(float));
					trans_deriv[0] = 1.0f; trans_deriv[1] = 0.0f;
					trans_deriv[2] = 0.0f; trans_deriv[3] = 1.0f;
				}
				else {
					coeff_R2T = g_coeff_R2T + (chID - 1) * 2 * warpdeg * warpdeg;
					lu_tar = g_lu + Idx * nchannels * 2 + chID * 2;
					TransR2T(warpdeg, coeff_R2T, lu_ref, lu_tar, xvec, loc, trans_deriv);
				}
				DeltaConstruct3D(splineszx, splineszz, loc, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);

				for (pxID = 0; pxID < boxsz * boxsz; pxID++) {
					data = dataim[pxID];
					fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, coeff_PSF, offset_x, offset_y, offset_z, 
						deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, &psf, dpsf_dloc);
					get_pixel_model_BP(psf, xvec, &model);
					get_pixel_deriv_BP(psf, dpsf_dloc, trans_deriv, xvec, deriv1);
					if (varim) {
						data += varim[pxID];
						model += varim[pxID];
					}
					
					accum_loss(data, model, &Loss, opt);
					accum_grad(vnum, data, model, deriv1, grad, opt);
					accum_Hessian(vnum, data, model, deriv1, Hessian, opt);
				}
			}
			for (i = 0; i < vnum; i++)
				for (j = i + 1; j < vnum; j++)
					Hessian[j * vnum + i] = Hessian[i * vnum + j];

			// evaluate the Loss
			if (Loss >= ACCEPTANCE * g_Loss[Idx]) {
				mu = max((1 + lambda * SCALE_UP) / (1 + lambda), 1.3f);
				g_lambda[Idx] = SCALE_UP * lambda;
				for (i = 0; i < vnum; i++)
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
			for (i = 0; i < vnum; i++)
				g_Hessian[Idx * vnum * vnum + i * vnum + i] = g_Hessian[Idx * vnum * vnum + i * vnum + i] * mu;
		}
	}
	__syncthreads();
}



/*!	@brief	<<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
			calculate the CRLB	
*/
__global__ void getCRLB_BP(int NFits, int boxsz, int splineszx, int splineszz, float* g_var, float* g_coeff_PSF, 
	int warpdeg, int* g_lu, float* g_coeff_R2T, 
	float* g_xvec, float* g_Loss, float* g_Hessian, float* g_CRLB, int opt)
{
	const int nchannels = 2;
	const int vnum = NDIM + 2;
	unsigned int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	char cholesky_info = 0;

	int *lu_ref = nullptr, *lu_tar = nullptr;
	float *coeff_R2T = nullptr, trans_deriv[4] = {0.0f};
	
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float *coeff_PSF = nullptr, deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float psf = 0.0f, loc[NDIM] = {0.0f}, dpsf_dloc[NDIM] = {0.0f};
	
	float model = 0.0f, deriv1[vnum] = {0.0f}, FIM[vnum * vnum] = {0.0f};
	float xvec[vnum] = {0.0f}, L[vnum * vnum] = {0.0f};

	float *varim = nullptr; 
	int chID = 0, pxID = 0, i = 0, j = 0;

	if (Idx < NFits) {
		memcpy(xvec, g_xvec + Idx * vnum, vnum * sizeof(float));
		
		// construct the Fisher Information Matrix (FIM)
		if (opt == 0) {
			lu_ref = g_lu + Idx * nchannels * 2;
			memset(FIM, 0, vnum * vnum * sizeof(float));
			for (chID = 0; chID < nchannels; chID++) {
				varim = (g_var) ? g_var + Idx * nchannels * boxsz * boxsz + chID * boxsz * boxsz : nullptr;
				coeff_PSF = g_coeff_PSF + chID * splineszx * splineszx * splineszz * 64;
				if (chID == 0) {
					memcpy(loc, xvec, NDIM * sizeof(float));
					trans_deriv[0] = 1.0f; trans_deriv[1] = 0.0f;
					trans_deriv[2] = 0.0f; trans_deriv[3] = 1.0f;
				}
				else {
					coeff_R2T = g_coeff_R2T + (chID - 1) * 2 * warpdeg * warpdeg;
					lu_tar = g_lu + Idx * nchannels * 2 + chID * 2;
					TransR2T(warpdeg, coeff_R2T, lu_ref, lu_tar, xvec, loc, trans_deriv);
				}
				DeltaConstruct3D(splineszx, splineszz, loc, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);

				for (pxID = 0; pxID < boxsz * boxsz; pxID++) {
					fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, coeff_PSF, offset_x, offset_y, offset_z, 
						deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, &psf, dpsf_dloc);
					get_pixel_model_BP(psf, xvec, &model);
					get_pixel_deriv_BP(psf, dpsf_dloc, trans_deriv, xvec, deriv1);
					if (varim)
						model += varim[pxID];
					
					accum_FIM(vnum, model, deriv1, FIM);
				}
			}
			for (i = 0; i < vnum; i++)
				for (j = i + 1; j < vnum; j++)
					FIM[j * vnum + i] = FIM[i * vnum + j];
		}
		else
			for (i = 0; i < vnum * vnum; i++)
				FIM[i] = g_Hessian[Idx * vnum * vnum + i];
		
		cholesky_info = Cholesky_decomp(FIM, vnum, L);
		if (cholesky_info == 1) {
			Cholesky_invert(L, vnum, FIM);
			for (i = 0; i < vnum; i++)
				if (opt == 0)
					g_CRLB[Idx * vnum + i] = FIM[i * vnum + i];
				else
					g_CRLB[Idx * vnum + i] = g_Loss[Idx] / (float)nchannels / (float)boxsz / (float)boxsz * FIM[i * vnum + i];
		}
	}	
	__syncthreads();
}


#endif