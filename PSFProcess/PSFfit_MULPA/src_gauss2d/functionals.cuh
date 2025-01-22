#pragma once
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h> 

#ifndef FUNCTIONALS_CUH
#define FUNCTIONALS_CUH


/*!	@brief	Parameters in general
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	dataim:				(nchannels * boxsz * boxsz) float, the PSF data. 
	@param[in]	varim:				(nchannels * boxsz * boxsz) float, the pixel-dependent variance, NULL for pixel-independent variance
	@param[in]	PSFsigmax: 			(nchannels) float, sigma (x-axis) of the Gaussian PSF in each channel
	@param[in]	PSFsigmay: 			(nchannels) float, sigma (y-axis) of the Gaussian PSF in each channel
	@param[in]	lnp:				(2), float, [mu, sigma] of the gaussian distribution of lnI

	@param[in]	nchannels:			int, number of channels
	@param[in] 	nnum:				int, number of emitters
	@param[in]	vnum:				int, the number of parameters in xvec, vnum = nnum * (NDIM + nchannels) + nchannels
	@param[in]	xvec:				(vnum) float, see definitions.h
	
	@param[in]	warpdeg:			int, the degree of the polynomial function that warps 2d locations from one channel to the other
	@param[in]	lu:					(nchannels * 2) int, [lc, uc] the left and upper corner of the PSF square in each channel (image coordinates)
	@param[in]	coeff_R2T:			((nchannels - 1) * 2 * warpdeg * warpdeg) float, the [coeffx_B2A, coeffy_B2A] coefficients of the polynomial warping from refernce channel to each of the target channel	
	@param[in]	loc_ref:			(nnum * NDIM) float, preallocated array for [x, y, z] the subpixel location of the emitter in the reference channel (box coordinates)
	@param[in]	trans_deriv:		(nnum * 4) float, [dxT/dxR, dyT/dxR, dxT/dyR, dyT/dyR] the transformation derivatives of each emitter linking locations from the target channels to the reference channel
	
	@param[in]	loc:				(nnum * NDIM) float, preallocated array for [x, y, z] the subpixel location of the emitter (box coordinates)
	@param[in]	psf:				(nnum) float, preallocated array for the psf value of each emitter at the given pixel in each channel
	@param[in]	dpsf_dloc:			(nnum * NDIM) float, preallocated array for [dpsf_dx, dpsf_dy, dpsf_dz] the derivatives of the psf over the location
	@param[in]	deriv1:				(vnum) float, preallocated array for the derivatives of the model over the xvec
	
	@param[out] loss:				(1) float, pointer to the negative log likeliklihood
	@param[out]	grad:				(vnum) float, gradient
	@param[out]	Hessian:			(vnum * vnum) float, Hessian
	@param[out]	FIM:				(vnum * vnum) float, Fisher Information Matrix
*/



/////////////////////////////////////////////// FUNCTIONALS IN GENERAL //////////////////////////////////////////////
/*! @brief		get the sum of the input PSF square, negative values are ignored. 
	@return 	float, sum of the PSF data.											*/
__device__ static inline float get_psfsum(int boxsz, float* dataim)
{
	float dumSum = 0.0f;
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++)
		if (dataim[pxID] > 0.0f)	
			dumSum += dataim[pxID];
	return dumSum;	
}



/*!	@brief		get the minimum of the input PSF square
	@return 	float, the minimum of the PSF data.	*/
__device__ static inline float get_psfmin(int boxsz, float* dataim)
{
	float dumMin = 1e10f;
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++)
		if (dataim[pxID] < dumMin)
			dumMin = dataim[pxID];
	return dumMin;	
}



/*!	@brief	Transform the subpixel loc in the reference channel to that in a target channels
			the locs are in the box-coordinate system, which needs the lu (left and upper_corner) to translate into image-coordinate
			xT = sum([coeffx_i * yR^(i//deg) * xR^(i%deg) for i in range(deg * deg)])
			yT = sum([coeffy_i * yR^(i//deg) * xR^(i%deg) for i in range(deg * deg)])	*/
__device__ static void TransR2T(int nnum, int warpdeg, float* coeff_R2T, int* lu_ref, int* lu_tar, float* loc_ref, float* loc_tar, float* trans_deriv)
{
	float locx_ref = 0.0f, locy_ref = 0.0f;
	float cx = 0.0f, cy = 0.0f;
	unsigned int n = 0, i = 0, j = 0;
	
	memset(loc_tar, 0, nnum * NDIM * sizeof(float));
    if (trans_deriv)
		memset(trans_deriv, 0, nnum * 4 * sizeof(float));
    
	for (n = 0; n < nnum; n++) {
		locx_ref = (float)lu_ref[0] + loc_ref[n * NDIM];
		locy_ref = (float)lu_ref[1] + loc_ref[n * NDIM + 1];
		for (i = 0, cy = 1.0f; i < warpdeg; i++) {
			for (j = 0, cx = 1.0f; j < warpdeg; j++) {
				loc_tar[n * NDIM] += coeff_R2T[i * warpdeg + j] * cy * cx;
				loc_tar[n * NDIM + 1] += coeff_R2T[warpdeg * warpdeg + i * warpdeg + j] * cy * cx;
				if (trans_deriv) {
					if (j < warpdeg - 1) {
						trans_deriv[n * 4] += coeff_R2T[i * warpdeg + (j + 1)] * (j + 1) * cy * cx; // dxT/dxR
						trans_deriv[n * 4 + 1] += coeff_R2T[warpdeg * warpdeg + i * warpdeg + (j + 1)] * (j + 1) * cy * cx; // dyT/dxR
					}	
					if (i < warpdeg - 1) {
						trans_deriv[n * 4 + 2] += coeff_R2T[(i + 1) * warpdeg + j] * (i + 1) * cy * cx; // dxT/dyR
						trans_deriv[n * 4 + 3] += coeff_R2T[warpdeg * warpdeg + (i + 1) * warpdeg + j] * (i + 1) * cy * cx; // dyT/dyR
					}		
				}
				cx *= locx_ref;
			}
			cy *= locy_ref;
		}
		loc_tar[n * NDIM] -= (float)lu_tar[0];
		loc_tar[n * NDIM + 1] -= (float)lu_tar[1];
	}
	return;
}



/*!	@brief	Construct the model value and deriv1 values at pxID.	*/
__device__ static void fconstruct_gauss2D(int nnum, int pxID, int boxsz, float PSFsigmax, float PSFsigmay, float* loc, float* psf, float* dpsf_dloc)
{
	float PSFx = 0.0f, PSFy = 0.0f, dPSFx_dx = 0.0f, dPSFy_dy = 0.0f;
	float delta_xu = 0.0f, delta_yu = 0.0f, delta_xl = 0.0f, delta_yl = 0.0f;

	for (unsigned int n = 0; n < nnum; n++) {
		delta_xu = ((float)(pxID % boxsz) + 1.0f - loc[n * NDIM]) / PSFsigmax;
		delta_yu = ((float)(pxID / boxsz) + 1.0f - loc[n * NDIM + 1]) / PSFsigmay;
		delta_xl = ((float)(pxID % boxsz) - loc[n * NDIM]) / PSFsigmax;
		delta_yl = ((float)(pxID / boxsz) - loc[n * NDIM + 1]) / PSFsigmay;

		PSFx = 0.5f * (erf(delta_xu / SQRTTWO) - erf(delta_xl / SQRTTWO));
		PSFy = 0.5f * (erf(delta_yu / SQRTTWO) - erf(delta_yl / SQRTTWO));
		psf[n] = PSFx * PSFy;

		if (dpsf_dloc) {
			dPSFx_dx = -invSQRTTWOPI / PSFsigmax * (exp(-0.5f * delta_xu * delta_xu) - exp(-0.5f * delta_xl * delta_xl));
			dPSFy_dy = -invSQRTTWOPI / PSFsigmay * (exp(-0.5f * delta_yu * delta_yu) - exp(-0.5f * delta_yl * delta_yl));
			dpsf_dloc[n * NDIM] = dPSFx_dx * PSFy;
			dpsf_dloc[n * NDIM + 1] = PSFx * dPSFy_dy;
		}
	}
	return;
}



/*!	@brief	accumulat the loss across the pixels in each channel	*/
__device__ static inline void accum_loss_lkh(float data, float model, float* Loss)
{
	const float dum = (data > 0.0f) ? (model - data - data * log(model / data)) : model;
	*Loss += 2.0f * dum;
	return;
}



/*!	@brief	accumulat the gradient across the pixels in both channels	*/
__device__ static inline void accum_grad_lkh(int vnum, float data, float model, float* deriv1, float* grad)
{
	const float dum = (data > 0.0f) ? (1.0f - data / model) : 1.0f;
	for (unsigned int i = 0; i < vnum; i++)
		grad[i] += dum * deriv1[i];
	
	return;
}



/*!	@brief	accumulate the up-triangle part of the Hessian at a pixel	*/
__device__ static inline void accum_Hessian_lkh(int vnum, float data, float model, float* deriv1, float* Hessian)
{
	const float dum = (data > 0.0f) ? (data / model / model) : 0.0f;
	for (unsigned int i = 0; i < vnum; i++) {
		Hessian[i * vnum + i] += dum * deriv1[i] * deriv1[i];
		for (unsigned int j = i + 1; j < vnum; j++)
			Hessian[i * vnum + j] += dum * deriv1[i] * deriv1[j];
	}
	return;
}



/*!	@brief	accumulate the up-triangle part of the Fisher Information Matrix (FIM) at a pixel		*/
__device__ static inline void accum_FIM(int vnum, float model, float* deriv1, float* FIM)
{
	for (unsigned int i = 0; i < vnum; i++)	
		for (unsigned int j = i; j < vnum; j++)
			FIM[i * vnum + j] += deriv1[i] * deriv1[j] / model;
	return;
}



/*!	@brief	copy the diagnal submatrix of an FIM to a new dumFIM with the rest values are 0	*/
__device__ static void diagcpy_FIM(int nnum, int nchannels, float* FIM, float* dumFIM)
{
	const unsigned int vnum0 = NDIM + nchannels;
	const unsigned int vnum = nnum * vnum0 + nchannels;

	memset(dumFIM, 0, vnum * vnum * sizeof(float));
	for (unsigned int n = 0; n < nnum; n++)
		for (unsigned int i = n * vnum0; i < (n + 1) * vnum0; i++)
			for (unsigned int j = n * vnum0; j < (n + 1) * vnum0; j++)
				dumFIM[i * vnum + j] = FIM[i * vnum + j];
	for (unsigned int i = nnum * vnum0; i < vnum; i++)
		for (unsigned int j = nnum * vnum0; j < vnum; j++)
			dumFIM[i * vnum + j] = FIM[i * vnum + j];
	return;
}



/*!	@brief	reform the FIM to avoid singularity caused by approximal localizations	*/
__device__ static void reform_FIM(int nnum, int nchannels, float* xvec, float* dumCRLB, float* FIM)
{
	const unsigned int vnum0 = NDIM + nchannels;
	const unsigned int vnum = nnum * vnum0 + nchannels;
	unsigned int n1 = 0, n2 = 0, i = 0, j = 0, k = 0;
	float alpha = 0.0;

	for (k = 0; k < vnum0; k++)
		for (n1 = 0; n1 < nnum - 1; n1++)
			for (n2 = n1 + 1; n2 < nnum; n2++) {
				i = n1 * vnum0 + k;
				j = n2 * vnum0 + k;
				alpha = (xvec[i] - xvec[j]) * (xvec[i] - xvec[j]) / sqrt(dumCRLB[i] * dumCRLB[j]);
				FIM[i * vnum + j] *= alpha / (alpha + 1.0f);
				FIM[j * vnum + i] *= alpha / (alpha + 1.0f);
			}
	return;
}






/////////////////////////////////////////////// FUNCTIONALS FOR 1-CHANNEL //////////////////////////////////////////////
/*!	@brief		profile the PSF square data	*/
__device__ static void init_xvec_1ch(int boxsz, float* dataim, float* xvec)
{
	float dumx = 0.0f, dumy = 0.0f, dumI = 0.0f, bkg = 0.0f;
	int pxID = 0;

	// get the minimum as the background in both channel
	bkg = get_psfmin(boxsz, dataim);

	// locate the x, y, and z at the center of mass of the data in reference channel (0-th channel)
	for (pxID = 0, dumI = 0.0, dumx = 0.0f, dumy = 0.0f; pxID < boxsz * boxsz; pxID++)
		if (dataim[pxID] > 0) {
			dumI += dataim[pxID] - bkg;
			dumx += ((float)(pxID % boxsz) + 0.5f) * (dataim[pxID] - bkg);
			dumy += ((float)(pxID / boxsz) + 0.5f) * (dataim[pxID] - bkg);
		}
	if (dumI > IMIN) {
		xvec[0] = dumx / dumI;
		xvec[1] = dumy / dumI;	
	}
	else {
		xvec[0] = 0.5f * (float)boxsz;
		xvec[1] = 0.5f * (float)boxsz;
	}

	// get photon numbers in the 2nd channel
	xvec[NDIM] = max(dumI, IMIN);
	xvec[NDIM + 1] = max(bkg, BGMIN);

	return;
}



/*!	@brief	set the maxJump	*/
__device__ static void init_maxJump_1ch(int nnum, int boxsz, float* xvec, float* maxJump)
{
	const int vnum0 = NDIM + 1;
	for (int n = 0; n < nnum; n++) {
		maxJump[n * vnum0] = max(1.0f, 0.1f * (float)boxsz);
		maxJump[n * vnum0 + 1] = max(1.0f, 0.1f * (float)boxsz);
		maxJump[n * vnum0 + NDIM] = max(100.0f, xvec[n * vnum0 + NDIM]);
	}
	maxJump[nnum * vnum0] = max(20.0f, xvec[nnum * vnum0]);
	return;
}



/*!	@brief	box-constrain the xvec	*/
__device__ static void boxConstrain_1ch(int nnum, int boxsz, float* xvec)
{
	const int vnum0 = NDIM + 1;
	for (int n = 0; n < nnum; n++) {
		xvec[n * vnum0] = min(max(xvec[n * vnum0], -1.0f), (float)boxsz + 1.0f);
		xvec[n * vnum0 + 1] = min(max(xvec[n * vnum0 + 1], -1.0f), (float)boxsz + 1.0f);
		xvec[n * vnum0 + NDIM] = max(xvec[n * vnum0 + NDIM], IMIN);
	}
	for (int n = 0; n < nnum; n++)
		if (xvec[n * vnum0 + NDIM] <= IMIN) {
			xvec[nnum * vnum0] = BGMIN;
			break;
		}
	xvec[nnum * vnum0] = max(xvec[nnum * vnum0], BGMIN);
	return;
}



/*!	@brief	calculate the model value at a pixel	*/
__device__ static void get_pixel_model_1ch(int nnum, float* psf, float* xvec, float* model)
{
	const int vnum0 = NDIM + 1;
	*model = 0.0f;
	for (int n = 0; n < nnum; n++)
		*model += xvec[n * vnum0 + NDIM] * psf[n];
	*model += xvec[nnum * vnum0];
	*model = max(*model, FLT_EPSILON);
	return;
}



/*!	@brief	calculate the derivatives at a pixel	*/
__device__ static void get_pixel_deriv_1ch(int nnum, float* psf, float* dpsf_dloc, float* xvec, float* deriv1)
{
	const int vnum0 = NDIM + 1;
	const int vnum = nnum * vnum0 + 1;
	memset(deriv1, 0, vnum * sizeof(float));
	for (int n = 0; n < nnum; n++) {
		deriv1[n * vnum0] = xvec[n * vnum0 + NDIM] * dpsf_dloc[n * NDIM];
		deriv1[n * vnum0 + 1] = xvec[n * vnum0 + NDIM] * dpsf_dloc[n * NDIM + 1];
		deriv1[n * vnum0 + NDIM] = psf[n];
	}
	deriv1[nnum * vnum0] = 1.0f;
	return;
}



/*!	@brief	calculate the loss, gradients, and Hessian	*/
__device__ static void get_lgH_1ch(int boxsz, float* dataim, float* varim, float* PSFsigmax, float* PSFsigmay, int nnum, float* xvec,
	float* loc, float* psf, float* dpsf_dloc, float* deriv1, float* loss, float* grad, float* Hessian)
{
	const int nchannels = 1, vnum0 = NDIM + nchannels;
	const int vnum = nnum * vnum0 + nchannels;
	float data = 0.0f, model = 0.0f;

	for (unsigned int n = 0; n < nnum; n++)
		memcpy(loc + n * NDIM, xvec + n * vnum0, NDIM * sizeof(float));
	
	*loss = 0.0f;
	memset(grad, 0, vnum * sizeof(float));
	memset(Hessian, 0, vnum * vnum * sizeof(float));
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		data = dataim[pxID];
		fconstruct_gauss2D(nnum, pxID, boxsz, PSFsigmax[0], PSFsigmay[0], loc, psf, dpsf_dloc);
		get_pixel_model_1ch(nnum, psf, xvec, &model);
		get_pixel_deriv_1ch(nnum, psf, dpsf_dloc, xvec, deriv1);
		if (varim) {
			data += varim[pxID];
			model += varim[pxID];
		}	

		accum_loss_lkh(data, model, loss);
		accum_grad_lkh(vnum, data, model, deriv1, grad);
		accum_Hessian_lkh(vnum, data, model, deriv1, Hessian);
	}
	for (unsigned int i = 0; i < vnum; i++)
		for (unsigned int j = i + 1; j < vnum; j++)
			Hessian[j * vnum + i] = Hessian[i * vnum + j];
	return;
}



/*!	@brief	calculate the FIM	*/
__device__ static void get_FIM_1ch(int boxsz, float* varim, float* PSFsigmax, float* PSFsigmay, int nnum, float* xvec,
	float* loc, float* psf, float* dpsf_dloc, float* deriv1, float* FIM)
{
	const int nchannels = 1, vnum0 = NDIM + nchannels;
	const int vnum = nnum * vnum0 + nchannels;
	float model = 0.0f;

	for (unsigned int n = 0; n < nnum; n++)
		memcpy(loc + n * NDIM, xvec + n * vnum0, NDIM * sizeof(float));
	
	memset(FIM, 0, vnum * vnum * sizeof(float));
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		fconstruct_gauss2D(nnum, pxID, boxsz, PSFsigmax[0], PSFsigmay[0], loc, psf, dpsf_dloc);
		get_pixel_model_1ch(nnum, psf, xvec, &model);
		get_pixel_deriv_1ch(nnum, psf, dpsf_dloc, xvec, deriv1);
		if (varim)
			model += varim[pxID];

		accum_FIM(vnum, model, deriv1, FIM);
	}
	for (unsigned int i = 0; i < vnum; i++)
		for (unsigned int j = i + 1; j < vnum; j++)
			FIM[j * vnum + i] = FIM[i * vnum + j];
	return;
}





/////////////////////////////////////////////// FUNCTIONALS FOR 2-CHANNEL //////////////////////////////////////////////
/*!	@brief		profile the PSF square data	*/
__device__ static void init_xvec_2ch(int boxsz, float* dataim, float* xvec)
{
	float dumx = 0.0f, dumy = 0.0f, dumI0 = 0.0f, bkg0 = 0.0f, dumI1 = 0.0f, bkg1 = 0.0f;
	int pxID = 0;

	// get the minimum as the background in both channel
	bkg0 = get_psfmin(boxsz, dataim);
	bkg1 = get_psfmin(boxsz, dataim + boxsz * boxsz);

	// locate the x, y, and z at the center of mass of the data in reference channel (0-th channel)
	for (pxID = 0, dumI0 = 0.0, dumx = 0.0f, dumy = 0.0f; pxID < boxsz * boxsz; pxID++)
		if (dataim[pxID] > 0) {
			dumI0 += dataim[pxID] - bkg0;
			dumx += ((float)(pxID % boxsz) + 0.5f) * (dataim[pxID] - bkg0);
			dumy += ((float)(pxID / boxsz) + 0.5f) * (dataim[pxID] - bkg0);
		}
	if (dumI0 > IMIN) {
		xvec[0] = dumx / dumI0;
		xvec[1] = dumy / dumI0;	
	}
	else {
		xvec[0] = 0.5f * (float)boxsz;
		xvec[1] = 0.5f * (float)boxsz;
	}

	// get photon numbers in the 2nd channel
	for (pxID = boxsz * boxsz, dumI1 = 0.0; pxID < 2 * boxsz * boxsz; pxID++)
		if (dataim[pxID] > 0)
			dumI1 += dataim[pxID] - bkg1;
			
	xvec[NDIM] = max(dumI0, IMIN) + max(dumI1, IMIN);
	xvec[NDIM + 1] = max(dumI0, IMIN) / xvec[NDIM];
	xvec[NDIM + 2] = max(bkg0, BGMIN);
	xvec[NDIM + 3] = max(bkg1, BGMIN);
	return;
}



/*!	@brief	set the maxJump	*/
__device__ static void init_maxJump_2ch(int nnum, int boxsz, float* xvec, float* maxJump)
{
	const int vnum0 = NDIM + 2;
	for (int n = 0; n < nnum; n++) {
		maxJump[n * vnum0] = max(1.0f, 0.1f * (float)boxsz);
		maxJump[n * vnum0 + 1] = max(1.0f, 0.1f * (float)boxsz);
		maxJump[n * vnum0 + NDIM] = max(100.0f, xvec[n * vnum0 + NDIM]);
		maxJump[n * vnum0 + NDIM + 1] = max(0.1f, xvec[n * vnum0 + NDIM + 1]);
	}
	maxJump[nnum * vnum0] = max(20.0f, xvec[nnum * vnum0]);
	maxJump[nnum * vnum0 + 1] = max(20.0f, xvec[nnum * vnum0 + 1]);
	return;
}



/*!	@brief	box-constrain the xvec	*/
__device__ static void boxConstrain_2ch(int nnum, int boxsz, float* xvec)
{
	const int vnum0 = NDIM + 2;
	for (int n = 0; n < nnum; n++) {
		xvec[n * vnum0] = min(max(xvec[n * vnum0], -1.0f), (float)boxsz + 1.0f);
		xvec[n * vnum0 + 1] = min(max(xvec[n * vnum0 + 1], -1.0f), (float)boxsz + 1.0f);
		xvec[n * vnum0 + NDIM] = max(xvec[n * vnum0 + NDIM], IMIN);
		xvec[n * vnum0 + NDIM + 1] = min(max(xvec[n * vnum0 + NDIM + 1], 0.0f), 1.0f);
	}
	for (int n = 0; n < nnum; n++)
		if (xvec[n * vnum0 + NDIM] <= IMIN) {
			xvec[nnum * vnum0] = BGMIN;
			xvec[nnum * vnum0 + 1] = BGMIN;
			break;
		}
	xvec[nnum * vnum0] = max(xvec[nnum * vnum0], BGMIN);
	xvec[nnum * vnum0 + 1] = max(xvec[nnum * vnum0 + 1], BGMIN);
	return;
}



/*!	@brief	calculate the model value at a pixel	*/
__device__ static void get_pixel_model_2ch(int nnum, int chID, float* psf, float* xvec, float* model)
{
	const int vnum0 = NDIM + 2;
	float dum_fracN = 0.0f;
	*model = 0.0f;
	for (int n = 0; n < nnum; n++) {
		dum_fracN = (chID == 0) ? xvec[n * vnum0 + NDIM + 1] : 1.0f - xvec[n * vnum0 + NDIM + 1];
		*model += xvec[n * vnum0 + NDIM] * dum_fracN * psf[n];
	}
	*model += xvec[nnum * vnum0 + chID];
	*model = max(*model, FLT_EPSILON);
	return;
}



/*!	@brief	calculate the derivatives at a pixel	*/
__device__ static void get_pixel_deriv_2ch(int nnum, int chID, float* psf, float* dpsf_dloc, float* trans_deriv_R2T, float* xvec, float* deriv1)
{
	const int vnum0 = NDIM + 2;
	float dum_fracN = 0.0f;
	memset(deriv1, 0, (nnum * vnum0 + 2) * sizeof(float));
	for (int n = 0; n < nnum; n++) {
		dum_fracN = (chID == 0) ? xvec[n * vnum0 + NDIM + 1] : 1.0f - xvec[n * vnum0 + NDIM + 1];
		deriv1[n * vnum0] = xvec[n * vnum0 + NDIM] * dum_fracN * (dpsf_dloc[n * NDIM] * trans_deriv_R2T[n * 4] + dpsf_dloc[n * NDIM + 1] * trans_deriv_R2T[n * 4 + 1]);
		deriv1[n * vnum0 + 1] = xvec[n * vnum0 + NDIM] * dum_fracN * (dpsf_dloc[n * NDIM] * trans_deriv_R2T[n * 4 + 2] + dpsf_dloc[n * NDIM + 1] * trans_deriv_R2T[n * 4 + 3]);
		deriv1[n * vnum0 + NDIM] = dum_fracN * psf[n];
		deriv1[n * vnum0 + NDIM + 1] = (chID == 0) ? xvec[n * vnum0 + NDIM] * psf[n] : -xvec[n * vnum0 + NDIM] * psf[n];
	}
	deriv1[nnum * vnum0 + chID] = 1.0f;
	return;
}



/*!	@brief	calculate the loss, gradients, and Hessian	*/
__device__ static void get_lgH_2ch(int boxsz, float* dataim, float* varim, float* PSFsigmax, float* PSFsigmay, int nnum, float* xvec, 
	int warpdeg, int* lu, float* coeff_R2T, float*loc_ref, float* trans_deriv,
	float* loc, float* psf, float* dpsf_dloc, float* deriv1, float* loss, float* grad, float* Hessian)
{
	const int nchannels = 2, vnum0 = NDIM + nchannels;
	const int vnum = nnum * vnum0 + nchannels;
	float data = 0.0f, model = 0.0f;

	int *lu_ref = lu, *lu_tar = lu + 2;
	
	for (unsigned int n = 0; n < nnum; n++)
		memcpy(loc_ref + n * NDIM, xvec + n * vnum0, NDIM * sizeof(float));
	
	*loss = 0.0f;
	memset(grad, 0, vnum * sizeof(float));
	memset(Hessian, 0, vnum * vnum * sizeof(float));
	for (unsigned int chID = 0; chID < nchannels; chID++) {
		
		if (chID == 0) {
			memcpy(loc, loc_ref, nnum * NDIM * sizeof(float));
			memset(trans_deriv, 0, nnum * 4 * sizeof(float));
			for (unsigned int n = 0; n < nnum; n++) {
				trans_deriv[n * 4] = 1.0f; 		
				trans_deriv[n * 4 + 3] = 1.0f;
			}	
		}	
		else 
			TransR2T(nnum, warpdeg, coeff_R2T, lu_ref, lu_tar, loc_ref, loc, trans_deriv);

		for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
			data = dataim[chID * boxsz * boxsz + pxID];
			fconstruct_gauss2D(nnum, pxID, boxsz, PSFsigmax[chID], PSFsigmay[chID], loc, psf, dpsf_dloc);
			get_pixel_model_2ch(nnum, chID, psf, xvec, &model);
			get_pixel_deriv_2ch(nnum, chID, psf, dpsf_dloc, trans_deriv, xvec, deriv1);
			if (varim) {
				data += varim[chID * boxsz * boxsz + pxID];
				model += varim[chID * boxsz * boxsz + pxID];
			}	

			accum_loss_lkh(data, model, loss);
			accum_grad_lkh(vnum, data, model, deriv1, grad);
			accum_Hessian_lkh(vnum, data, model, deriv1, Hessian);
		}
	}
	for (unsigned int i = 0; i < vnum; i++)
		for (unsigned int j = i + 1; j < vnum; j++)
			Hessian[j * vnum + i] = Hessian[i * vnum + j];
	return;
}



/*!	@brief	calculate the FIM		*/
__device__ static void get_FIM_2ch(int boxsz, float* varim, float* PSFsigmax, float* PSFsigmay, int nnum, float* xvec, 
	int warpdeg, int* lu, float* coeff_R2T, float*loc_ref, float* trans_deriv,
	float* loc, float* psf, float* dpsf_dloc, float* deriv1, float* FIM)
{
	const int nchannels = 2, vnum0 = NDIM + nchannels;
	const int vnum = nnum * vnum0 + nchannels;
	float model = 0.0f;

	int *lu_ref = lu, *lu_tar = lu + 2;
	
	for (unsigned int n = 0; n < nnum; n++)
		memcpy(loc_ref + n * NDIM, xvec + n * vnum0, NDIM * sizeof(float));
	
	memset(FIM, 0, vnum * vnum * sizeof(float));
	for (unsigned int chID = 0; chID < nchannels; chID++) {
		
		if (chID == 0) {
			memcpy(loc, loc_ref, nnum * NDIM * sizeof(float));
			memset(trans_deriv, 0, nnum * 4 * sizeof(float));
			for (unsigned int n = 0; n < nnum; n++) {
				trans_deriv[n * 4] = 1.0f; 		
				trans_deriv[n * 4 + 3] = 1.0f;
			}	
		}	
		else 
			TransR2T(nnum, warpdeg, coeff_R2T, lu_ref, lu_tar, loc_ref, loc, trans_deriv);

		for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
			fconstruct_gauss2D(nnum, pxID, boxsz, PSFsigmax[chID], PSFsigmay[chID], loc, psf, dpsf_dloc);
			get_pixel_model_2ch(nnum, chID, psf, xvec, &model);
			get_pixel_deriv_2ch(nnum, chID, psf, dpsf_dloc, trans_deriv, xvec, deriv1);
			if (varim)
				model += varim[chID * boxsz * boxsz + pxID];

			accum_FIM(vnum, model, deriv1, FIM);
		}
	}
	for (unsigned int i = 0; i < vnum; i++)
		for (unsigned int j = i + 1; j < vnum; j++)
			FIM[j * vnum + i] = FIM[i * vnum + j];
	return;
}





///////////////////////////////////// FUNCTIONALS FOR ADDING NEW EMITTER IN THE FIRST CHANNEL /////////////////////////////////////
/*!	@brief		check if a testing px is in the trap of the existing localizations
	@return		bool, true if the testing pxID is within the 2-pixel range of one of the localizations
*/
__device__ static bool isInTrap(int boxsz, int pxID, int nnum, float* locs)
{
	const int row = pxID / boxsz;
	const int col = pxID % boxsz;
	int trap_xmin = 0, trap_xmax = 0, trap_ymin = 0, trap_ymax = 0; 
	for (int n = 0; n < nnum; n++) {
		trap_xmin = (int)locs[n * NDIM] - 2;
		trap_xmax = (int)locs[n * NDIM] + 2;
		trap_ymin = (int)locs[n * NDIM + 1] - 2;
		trap_ymax = (int)locs[n * NDIM + 1] + 2;
		if (row >= trap_ymin && row <= trap_ymax && col >= trap_xmin && col <= trap_xmax)
			return true;
	}
	return false;
}



/*!	@brief	get thee peak position of the residual image	
	@param[out]	x_resPeak:	(1), float, pointer to the x-axis residual peak position 
	@param[out] y_resPeak: 	(1), float, pointer to the y-axis residual peak position
*/
__device__ static void get_resPeakPos(int nchannels, int boxsz, float* dataim, float* PSFsigmax, float* PSFsigmay, int nnum, float* xvec,
	float* loc, float* psf, float* x_resPeak, float* y_resPeak)
{
	int vnum0 = NDIM + nchannels;
	float model = 0.0f, resp = 0.0f;
	
	for (unsigned int n = 0; n < nnum; n++)
		memcpy(loc + n * NDIM, xvec + n * vnum0, NDIM * sizeof(float));
	
	resp = 0.0f;
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		fconstruct_gauss2D(nnum, pxID, boxsz, PSFsigmax[0], PSFsigmay[0], loc, psf, NULL);
		if (nchannels == 1)
			get_pixel_model_1ch(nnum, psf, xvec, &model);
		else
			get_pixel_model_2ch(nnum, 0, psf, xvec, &model);
		if (resp < dataim[pxID] - model) {
			resp = dataim[pxID] - model;
			*x_resPeak = (float)(pxID % boxsz) + 0.5f; 
			*y_resPeak = (float)(pxID / boxsz) + 0.5f;
		}
	}
	return;
}



/*!	@brief	get the mass center
	@param[out]	x_mc:		(1), float, pointer to the x-axis mass center 
	@param[out] y_mc: 		(1), float, pointer to the y-axis mass center
*/
__device__ static void get_massCenter(int nchannels, int nnum, float* xvec, float* x_mc, float* y_mc)
{
	int vnum0 = NDIM + nchannels, n = 0;
	float photon = 0.0f, tmpxc = 0.0f, tmpyc = 0.0f, tmpI = 0.0f;

	for (n = 0, tmpxc = 0.0f, tmpyc = 0.0f, tmpI = 0.0f; n < nnum; n++) {
		photon = (nchannels == 2) ? xvec[n * vnum0 + NDIM] * xvec[n * vnum0 + NDIM + 1] : xvec[n * vnum0 + NDIM];
		photon = max(photon, 1.0f);
		tmpxc += photon * xvec[n * vnum0];
		tmpyc += photon * xvec[n * vnum0 + 1];
		tmpI += photon;
	}
	*x_mc = tmpxc / tmpI;
	*y_mc = tmpyc / tmpI;
	return;
}



/*!	@brief	get the nearest neighbor from the xvec for the input location
	@param[in]	x_in:		float, the input x-axis location 
	@param[in] 	y_in: 		float, the input y-axis location
	@param[out]	ind:		int, pointer to the index of the nearest neighbor in the xvec
	@param[out]	nnd:		float, pointer to the nearest neighbor distance
*/
__device__ static void get_nn(int nchannels, int nnum, float* xvec, float x_in, float y_in, int* ind, float* nnd)
{
	int vnum0 = NDIM + nchannels;
	float dist2 = 0.0f;

	*nnd = 1e10f;
	for (int n = 0; n < nnum; n++) {
		dist2 = (xvec[n * vnum0] - x_in) * (xvec[n * vnum0] - x_in) + (xvec[n * vnum0 + 1] - y_in) * (xvec[n * vnum0 + 1] - y_in);
		if (*nnd > dist2) {
			*nnd = dist2;
			*ind = n;
		}	
	}
	*nnd = sqrt(*nnd);
	return;
}



/*!	@brief	add the new emitter via push and pull
	@param[in]	x_resPeak:		float, the x-axis peak position of the residual image 
	@param[in] 	y_resPeak: 		float, the y-axis peak position of the residual image
	@param[in]	x_massc:		float, the x-axis mass center of the previous locations 
	@param[in] 	y_massc: 		float, the y-axis mass center of the previous locations
	@param[out]	x_new:			float, the x-axis location for the new emitter
	@param[out]	y_new:			float, the y-axis location for the new emitter
*/
__device__ static void get_newLoc(int boxsz, float PSFsigmax, float PSFsigmay, float x_resPeak, float y_resPeak, float x_massc, float y_massc, float* x_new, float* y_new)
{
	const float pplen = 0.5f * sqrt(0.5f * (PSFsigmax * PSFsigmax + PSFsigmay * PSFsigmay)); 
	float dist = 0.0f;
	
	// push and pull
	dist = sqrt((x_massc - x_resPeak) * (x_massc - x_resPeak) + (y_massc - y_resPeak) * (y_massc - y_resPeak));
	if ((x_resPeak < PSFsigmax) || (x_resPeak > (float)boxsz - PSFsigmax) || (y_resPeak < PSFsigmay) || (y_resPeak > (float)boxsz - PSFsigmay)) {
		*x_new = min(max(x_resPeak + pplen * (x_resPeak - x_massc) / dist, -1.0f), (float)boxsz + 1.0f);
		*y_new = min(max(y_resPeak + pplen * (y_resPeak - y_massc) / dist, -1.0f), (float)boxsz + 1.0f);
	}
	else {
		*x_new = x_resPeak - min(dist - 1.0f, pplen) * (x_resPeak - x_massc) / dist;
		*y_new = y_resPeak - min(dist - 1.0f, pplen) * (y_resPeak - y_massc) / dist;
	}
	return;
}

#endif