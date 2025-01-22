#pragma once
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

#ifndef FUNCTIONALS_CUH
#define FUNCTIONALS_CUH



/*! @brief		get the sum of the input PSF square, negative values are ignored. 
	@param[in]	boxsz:				size of the PSF square. 
	@param[in]	dataim:				pointer to the PSF data. 
	@return 	sum of the PSF data.	
*/
__device__ static inline float get_psfsum(int boxsz, float* dataim)
{
	float dumSum = 0.0f;
	for (int pxID = 0; pxID < boxsz * boxsz; pxID++)
		if (dataim[pxID] > 0.0f)	
			dumSum += dataim[pxID];
	return dumSum;	
}



/*!	@brief		get the minimum of the input PSF square
	@param[in]	boxsz:				size of the PSF square. 
	@param[in]	dataim:				pointer to the PSF data. 
	@return 	the minimum of the PSF data.			
*/
__device__ static inline float get_psfmin(int boxsz, float* dataim)
{
	float dumMin = 1e10f;
	for (int pxID = 0; pxID < boxsz * boxsz; pxID++)
		if (dataim[pxID] < dumMin)
			dumMin = dataim[pxID];
	return dumMin;	
}



/*!	@brief		get the 3rd-minimum (medium of the first 5 minima) of the input PSF square
	@param[in]	boxsz:				size of the PSF square. 
	@param[in]	dataim:				pointer to the PSF data. 
	@return 	the 3rd-minimum of the PSF data.			
*/
__device__ static inline float get_psf3rdmin(int boxsz, float* dataim)
{
	float dumMin_1st = 1e10f, dumMin_2nd = 1e10f, dumMin_3rd = 1e10f;
	for (int pxID = 0; pxID < boxsz * boxsz; pxID++)
	{
		if (dataim[pxID] < dumMin_1st)
		{
			dumMin_3rd = dumMin_2nd;
			dumMin_2nd = dumMin_1st;
			dumMin_1st = dataim[pxID];
		}
		else if (dataim[pxID] < dumMin_2nd)
		{
			dumMin_3rd = dumMin_2nd;
			dumMin_2nd = dataim[pxID];
		}
		else if (dataim[pxID] < dumMin_3rd)
			dumMin_3rd = dataim[pxID];
	}
	return dumMin_3rd;	
}



/*!	@brief		profile the PSF square data
	@param[in]	boxsz:				int, size of the PSF square. 
	@param[in]	zrange:				int, axial range for fit
	@param[in]	dataim:				(nchannels * boxsz * boxsz) float, the n-channel PSF data.
	@param[out]	xvec:				(NDIM + nchannels + nchannels) float, see definitions.h		
*/
__device__ static void init_xvec_1ch(int boxsz, int zrange, float* dataim, float* xvec)
{
	float dumx = 0.0f, dumy = 0.0f, dumI = 0.0f, bkg = 0.0f;
	int pxID = 0;
	
	// Get the photon numbers and bkg for the dataim
	bkg = get_psfmin(boxsz, dataim);
	
	// locate the x and y at the center of mass of the dataim
	for (pxID = 0, dumI = 0.0f, dumx = 0.0f, dumy = 0.0f; pxID < boxsz * boxsz; pxID++)
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
	xvec[2] = 0.5f * (float)zrange;
	xvec[NDIM] = max(dumI, IMIN);
	xvec[NDIM + 1] = max(bkg, BGMIN);
	return;
}



/*!	@brief		profile the PSF square data
	@param[in]	boxsz:				int, size of the PSF square. 
	@param[in]	zrange:				int, axial range for fit
	@param[in]	dataim:				(nchannels * boxsz * boxsz) float, the n-channel PSF data.
	@param[out]	xvec:				(NDIM + nchannels + nchannels) float, see definitions.h		
*/
__device__ static void init_xvec_2ch(int boxsz, int zrange, float* dataim, float* xvec)
{
	float dumx = 0.0f, dumy = 0.0f, dumI0 = 0.0f, bkg0 = 0.0f, dumI1 = 0.0f, bkg1 = 0.0f;
	int pxID = 0;

	// Get the photon numbers and bkg for the dataim
	bkg0 = get_psfmin(boxsz, dataim);
	bkg1 = get_psfmin(boxsz, dataim + boxsz * boxsz);

	// locate the x and y at the center of mass of the dataim
	for (pxID = 0, dumI0 = 0.0f, dumx = 0.0f, dumy = 0.0f; pxID < boxsz * boxsz; pxID++)
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
	xvec[2] = 0.5f * (float)zrange;
	
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



/*!	@brief	set the maxJump
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	zrange:			int, the size of the splines cube (axis-axis)
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out] maxJump:			(vnum) float, maxJump according to the xvec
*/
__device__ static void init_maxJump_1ch(int boxsz, int zrange, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[2] = max(2.0f, 0.05f * (float)zrange);
	maxJump[NDIM] = max(100.0f, xvec[NDIM]);
	maxJump[NDIM + 1] = max(20.0f, xvec[NDIM + 1]);
	return;
}



/*!	@brief	set the maxJump
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	zrange:				int, the size of the splines cube (axis-axis)
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out] maxJump:			(vnum) float, maxJump according to the xvec
*/
__device__ static void init_maxJump_2ch(int boxsz, int zrange, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[2] = max(2.0f, 0.05f * (float)zrange);
	maxJump[NDIM] = max(100.0f, xvec[NDIM]);
	maxJump[NDIM + 1] = max(0.1f, xvec[NDIM + 1]);
	maxJump[NDIM + 2] = max(20.0f, xvec[NDIM + 2]);
	maxJump[NDIM + 3] = max(20.0f, xvec[NDIM + 3]);	
	return;
}



/*!	@brief	Transform the subpixel loc in the reference channel to that in a target channels
			the locs are in the box-coordinate system, which needs the lu (left and upper_corner) to translate into image-coordinate
			xT = sum([coeffx_i * yR^(i//deg) * xR^(i%deg) for i in range(deg * deg)])
			yT = sum([coeffy_i * yR^(i//deg) * xR^(i%deg) for i in range(deg * deg)])
	@param[in]	warpdeg:			int, the degree of the polynomial function that warps 2d locations from one channel to the other
	@param[in]	coeff_R2T:			(2 * warpdeg * warpdeg) float, [coeffx_R2T, coeffy_R2T] the coefficients of the polynomial warping from the ref channel to others
	@param[in]	lu_ref:				(2) int, [lc, uc] the left and upper corner of the PSF square in the reference channel (image coordinates)
	@param[in]	lu_tar:				(2) int, [lc, uc] the left and upper corner of the PSF square in a target channel (image coordinates)
	@param[in]	loc_ref:			(NDIM) float, [locx, locy, locz] the subpixel location of the emitter in the reference channel (box coordinates)
	@param[out]	loc_tar:			(NDIM) float, [locx, locy, locz] the subpixel location of the emitter in a target channel (box coordinates)
	@param[out]	trans_deriv:		[4] float, [dxT/dxR, dyT/dxR, dxT/dyR, dyT/dyR] the transformation derivatives linking locations from the target channels to the reference channel					
*/
__device__ static void TransR2T(int warpdeg, float* coeff_R2T, int* lu_ref, int* lu_tar, float* loc_ref, float* loc_tar, float* trans_deriv)
{
	const float locx_ref = loc_ref[0] + (float)lu_ref[0];
	const float locy_ref = loc_ref[1] + (float)lu_ref[1];
	float cx = 0.0f, cy = 0.0f;
	int i = 0, j = 0;
	
	memset(loc_tar, 0, NDIM * sizeof(float));
    if (trans_deriv)
		memset(trans_deriv, 0, 4 * sizeof(float));
    
	for (i = 0, cy = 1.0f; i < warpdeg; i++) {
		for (j = 0, cx = 1.0f; j < warpdeg; j++) {
			loc_tar[0] += coeff_R2T[i * warpdeg + j] * cy * cx;
			loc_tar[1] += coeff_R2T[warpdeg * warpdeg + i * warpdeg + j] * cy * cx;
			if (trans_deriv) {
				if (j < warpdeg - 1) {
					trans_deriv[0] += coeff_R2T[i * warpdeg + (j + 1)] * (j + 1) * cy * cx; // dxT/dxR
					trans_deriv[1] += coeff_R2T[warpdeg * warpdeg + i * warpdeg + (j + 1)] * (j + 1) * cy * cx; // dyT/dxR
				}	
				if (i < warpdeg - 1) {
					trans_deriv[2] += coeff_R2T[(i + 1) * warpdeg + j] * (i + 1) * cy * cx; // dxT/dyR
					trans_deriv[3] += coeff_R2T[warpdeg * warpdeg + (i + 1) * warpdeg + j] * (i + 1) * cy * cx; // dyT/dyR
				}		
			}
			cx *= locx_ref;
		}
		cy *= locy_ref;
	}
	loc_tar[0] -= (float)lu_tar[0];
	loc_tar[1] -= (float)lu_tar[1];
	loc_tar[2] = loc_ref[2];
	return;
}



/*!	@brief	Construct the model value and deriv1 values at pxID.
	@param[in]	pxID:				int, the index of the working pixel
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	PSFsigmax:			float, sigma of the Gaussian PSF model (x-axis)
	@param[in] 	PSFsigmay:			float, sigma of the Gaussian PSF model (y-axis)
	@param[in]	loc:				(2) float, [locx, locy] location of the emitter
	@param[out]	psf:				float pointer to the psf value at the given pixel
	@param[out]	dpsf_dloc:			(4) float, [dpsf_dx, dpsf_dy, dpsf_dsx, dfsp_dsy] the derivatives of the psf over the location and the sigmas	 
*/
__device__ static void fconstruct_gauss2D(int pxID, int boxsz, float PSFsigmax, float PSFsigmay, float* loc, float* psf, float* dpsf_dloc)
{
	float PSFx = 0.0f, PSFy = 0.0f, dPSFx_dx = 0.0f, dPSFy_dy = 0.0f, dPSFx_dsx = 0.0f, dPSFy_dsy = 0.0f;
	float delta_xu = 0.0f, delta_yu = 0.0f, delta_xl = 0.0f, delta_yl = 0.0f;

	delta_xu = ((float)(pxID % boxsz) + 1.0f - loc[0]) / PSFsigmax;
	delta_yu = ((float)(pxID / boxsz) + 1.0f - loc[1]) / PSFsigmay;
	delta_xl = ((float)(pxID % boxsz) - loc[0]) / PSFsigmax;
	delta_yl = ((float)(pxID / boxsz) - loc[1]) / PSFsigmay;

	PSFx = 0.5f * (erf(delta_xu / SQRTTWO) - erf(delta_xl / SQRTTWO));
	PSFy = 0.5f * (erf(delta_yu / SQRTTWO) - erf(delta_yl / SQRTTWO));
	*psf = PSFx * PSFy;

	if (dpsf_dloc) {
		dPSFx_dx = -invSQRTTWOPI / PSFsigmax * (exp(-0.5f * delta_xu * delta_xu) - exp(-0.5f * delta_xl * delta_xl));
		dPSFy_dy = -invSQRTTWOPI / PSFsigmay * (exp(-0.5f * delta_yu * delta_yu) - exp(-0.5f * delta_yl * delta_yl));
		dPSFx_dsx = -invSQRTTWOPI / PSFsigmax * (delta_xu * exp(-0.5f * delta_xu * delta_xu) - delta_xl * exp(-0.5f * delta_xl * delta_xl));
		dPSFy_dsy = -invSQRTTWOPI / PSFsigmay * (delta_yu * exp(-0.5f * delta_yu * delta_yu) - delta_yl * exp(-0.5f * delta_yl * delta_yl));

		dpsf_dloc[0] = dPSFx_dx * PSFy;
		dpsf_dloc[1] = PSFx * dPSFy_dy;
		dpsf_dloc[2] = dPSFx_dsx * PSFy;
		dpsf_dloc[3] = PSFx * dPSFy_dsy;
	}
	return;
}



/*!	@brief	Construct the model value and deriv1 values at pxID.
	@param[in]	pxID:				int, the index of the working pixel
	@param[in]	boxsz:				int, the size of the PSF square
	@param[in]	astigs:				(9) float, [PSFsigmax0, PSFsigmay0, shiftx, shifty, dof, Ax, Bx, Ay, By] the astigmatic parameters			
	@param[in]	loc:				(NDIM) float, [x, y, z] location of the emitter (box coordinates)
	@param[out]	psf:				float pointer to the psf value at the given pixel
	@param[out]	dpsf_dloc:			(NDIM) float, [dpsf_dx, dpsf_dy, dpsf_dz] the derivatives of the psf over the location 	 
*/
__device__ static void fconstruct_gauss3D(int pxID, int boxsz, float* astigs, float* loc, float* psf, float* dpsf_dloc)
{
	const float PSFsigmax0 = astigs[0];
	const float PSFsigmay0 = astigs[1];
	const float shiftx = astigs[2];
	const float shifty = astigs[3];
	const float d = astigs[4];
	const float Ax = astigs[5];
	const float Bx = astigs[6];
	const float Ay = astigs[7];
	const float By = astigs[8];

	float zx = 0.0f, zy = 0.0f, sx = 0.0f, sy = 0.0f, alpha_x = 0.0f, alpha_y = 0.0f;
	float dalphax_dz = 0.0f, dalphay_dz = 0.0f, dsx_dz = 0.0f, dsy_dz = 0.0f;
	float PSFx = 0.0f, PSFy = 0.0f, dPSFx_dx = 0.0f, dPSFy_dy = 0.0f, dPSFx_dsx = 0.0f, dPSFy_dsy = 0.0f;
	float delta_xu = 0.0f, delta_yu = 0.0f, delta_xl = 0.0f, delta_yl = 0.0f;
	
	zx = loc[2] - shiftx;
	zy = loc[2] - shifty;

	alpha_x = sqrt(max(1.0f + zx*zx / d/d + Ax * zx*zx*zx / d/d/d + Bx * zx*zx*zx*zx / d/d/d/d, 1.0f));
	alpha_y = sqrt(max(1.0f + zy*zy / d/d + Ay * zy*zy*zy / d/d/d + By * zy*zy*zy*zy / d/d/d/d, 1.0f));

	sx = PSFsigmax0 * alpha_x;
	sy = PSFsigmay0 * alpha_y;

	delta_xu = ((float)(pxID % boxsz) + 1.0f - loc[0]) / sx;
	delta_yu = ((float)(pxID / boxsz) + 1.0f - loc[1]) / sy;
	delta_xl = ((float)(pxID % boxsz) - loc[0]) / sx;
	delta_yl = ((float)(pxID / boxsz) - loc[1]) / sy;

	PSFx = 0.5f * (erf(delta_xu / SQRTTWO) - erf(delta_xl / SQRTTWO));
	PSFy = 0.5f * (erf(delta_yu / SQRTTWO) - erf(delta_yl / SQRTTWO));
	*psf = PSFx * PSFy;

	if (dpsf_dloc) {
		dalphax_dz = 2.0f * zx / d/d + 3.0f * Ax * zx*zx / d/d/d + 4.0f * Bx * zx*zx*zx / d/d/d/d;
		dalphay_dz = 2.0f * zy / d/d + 3.0f * Ay * zy*zy / d/d/d + 4.0f * By * zy*zy*zy / d/d/d/d;
		
		dsx_dz = 0.5f * PSFsigmax0 / alpha_x * dalphax_dz;
		dsy_dz = 0.5f * PSFsigmay0 / alpha_y * dalphay_dz;

		dPSFx_dx = -invSQRTTWOPI / sx * (exp(-0.5f * delta_xu * delta_xu) - exp(-0.5f * delta_xl * delta_xl));
		dPSFy_dy = -invSQRTTWOPI / sy * (exp(-0.5f * delta_yu * delta_yu) - exp(-0.5f * delta_yl * delta_yl));
		dPSFx_dsx = -invSQRTTWOPI / sx * (delta_xu * exp(-0.5f * delta_xu * delta_xu) - delta_xl * exp(-0.5f * delta_xl * delta_xl));
		dPSFy_dsy = -invSQRTTWOPI / sy * (delta_yu * exp(-0.5f * delta_yu * delta_yu) - delta_yl * exp(-0.5f * delta_yl * delta_yl));

		dpsf_dloc[0] = dPSFx_dx * PSFy;
		dpsf_dloc[1] = PSFx * dPSFy_dy;
		dpsf_dloc[2] = dPSFx_dsx * dsx_dz * PSFy + PSFx * dPSFy_dsy * dsy_dz;
	}
	return;
}



/*!	@brief	box-constrain the xvec
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	zrange:			int, the size of the splines cube (axis-axis)
	@param[out] xvec:				(vnum) float, see definitions.h
*/
__device__ static void boxConstrain_1ch(int boxsz, int zrange, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
	xvec[2] = min(max(xvec[2], 0.0f), (float)zrange);
	xvec[NDIM] = max(xvec[NDIM], IMIN);
	xvec[NDIM + 1] = max(xvec[NDIM + 1], BGMIN);
	return;
}



/*!	@brief	box-constrain the xvec
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	zrange:				int, the size of the splines cube (axis-axis)
	@param[out] xvec:				(vnum) float, see definitions.h
*/
__device__ static void boxConstrain_2ch(int boxsz, int zrange, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
	xvec[2] = min(max(xvec[2], 0.0f), (float)zrange);
	xvec[NDIM] = max(xvec[NDIM], IMIN);
	xvec[NDIM + 1] = min(max(xvec[NDIM + 1], 0.0f), 1.0f);
	xvec[NDIM + 2] = max(xvec[NDIM + 2], BGMIN);
	xvec[NDIM + 3] = max(xvec[NDIM + 3], BGMIN);
	return;
}



/*!	@brief	calculate the model at a pixel for each channel
	@param[in]	psf:				float, the psf value at the pixel in the given channel
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out]	model:				(1) pointer to the model
*/
__device__ static void get_pixel_model_1ch(float psf, float* xvec, float* model)
{
	*model = max(xvec[NDIM] * psf + xvec[NDIM + 1], BGMIN);
	return;
}



/*!	@brief	calculate the model at a pixel for each channel
	@param[in]	chID:				int, channel ID
	@param[in]	psf:				float, the psf value at the pixel in the given channel
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out]	model:				(1) pointer to the model
*/
__device__ static void get_pixel_model_2ch(int chID, float psf, float* xvec, float* model)
{
	const float dum_fracN = (chID == 0) ? xvec[NDIM + 1] : 1.0f - xvec[NDIM + 1];
	*model = max(xvec[NDIM] * dum_fracN * psf + xvec[NDIM + 2 + chID], BGMIN);
	return;
}



/*!	@brief	calculate the derivatives at a pixel for each channel
	@param[in]	psf:				float, the psf value at the pixel in the given channel
	@param[in]	dpsf_dloc:			(NDIM) float, [dpsf_dx, dpsf_dy] the derivatives of the psf over the location at the pixel in the given channel
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out]	deriv1:				(vnum) float, the derivatives of the model over the xvec
*/
__device__ static void get_pixel_deriv_1ch(float psf, float* dpsf_dloc, float* xvec, float* deriv1)
{
	memset(deriv1, 0, (NDIM + 2) * sizeof(float));
	deriv1[0] = xvec[NDIM] * dpsf_dloc[0];
	deriv1[1] = xvec[NDIM] * dpsf_dloc[1];
	deriv1[2] = xvec[NDIM] * dpsf_dloc[2];
	deriv1[NDIM] = psf;
	deriv1[NDIM + 1] = 1.0f;
	return;
}



/*!	@brief	calculate the derivatives at a pixel for each channel
	@param[in]	chID:				int, channel ID
	@param[in]	psf:				float, the psf value at the pixel in the given channel
	@param[in]	dpsf_dloc:			(NDIM) float, [dpsf_dx, dpsf_dy] the derivatives of the psf over the location at the pixel in the given channel
	@param[in]	trans_deriv_R2T:	(4) float, [dxT/dxR, dyT/dxR, dxT/dyR, dyT/dyR] the transformation derivatives linking locations from the ref channel to a target channels
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out]	deriv1:				(vnum) float, the derivatives of the model over the xvec
*/
__device__ static void get_pixel_deriv_2ch(int chID, float psf, float* dpsf_dloc, float* trans_deriv_R2T, float* xvec, float* deriv1)
{
	const float dum_fracN = (chID == 0) ? xvec[NDIM + 1] : 1.0f - xvec[NDIM + 1];
	memset(deriv1, 0, (NDIM + 4) * sizeof(float));
	deriv1[0] = xvec[NDIM] * dum_fracN * (dpsf_dloc[0] * trans_deriv_R2T[0] + dpsf_dloc[1] * trans_deriv_R2T[1]);
	deriv1[1] = xvec[NDIM] * dum_fracN * (dpsf_dloc[0] * trans_deriv_R2T[2] + dpsf_dloc[1] * trans_deriv_R2T[3]);
	deriv1[2] = xvec[NDIM] * dum_fracN * dpsf_dloc[2];
	deriv1[NDIM] = dum_fracN * psf;
	deriv1[NDIM + 1] = (chID == 0) ? xvec[NDIM] * psf : -xvec[NDIM] * psf;
	deriv1[NDIM + 2 + chID] = 1.0f;
	return;
}



/*!	@brief	accumulat the loss across the pixels
	@param[in]	data:				the data at a pixel
	@param[in] 	model:				the model at a pixel
	@param[out]	loss:				pointer to the accumulating loss value
	@param[in]	opt:				optimization method. 0 for MLE and 1 for LSQ
*/
__device__ static inline void accum_loss(float data, float model, float* Loss, int opt)
{
	if (opt == 0) {
		if (data > 0)
			*Loss += 2.0f * (model - data - data * log(model / data));
		else
			*Loss += 2.0f * model;
	}
	else
		*Loss += (model - data) * (model - data);
	return;
}



/*!	@brief	accumulat the gradient across the pixels
	@param[in]	vnum:				number of parameters in xvec
	@param[in]	data:				the data at a pixel
	@param[in]	model:				the model at a pixel
	@param[in] 	deriv1:				(vnum) the derivatives of the model over the xvec
	@param[out]	grad:				(vnum) the accumulating gradient vector
	@param[in]	opt:				optimization method. 0 for MLE and 1 for LSQ
*/
__device__ static inline void accum_grad(int vnum, float data, float model, float* deriv1, float* grad, int opt)
{
	if (opt == 0) {
		data = (data > 0.0f) ? data : 0.0f;
		for (int i = 0; i < vnum; i++)
			grad[i] += (1.0f - data / model) * deriv1[i];
	}
	else
		for (int i = 0; i < vnum; i++)
			grad[i] += 2.0f * (model - data) * deriv1[i];
	return;
}



/*!	@brief	accumulate the up-triangle part of the Hessian at a pixel
	@param[in]	vnum:				number of parameters in xvec
	@param[in]	data:				the data at a pixel
	@param[in]	model:				the model at a pixel
	@param[in]	deriv1:				(vnum) the derivatives of the model over the xvec
	@param[out]	Hessian:			(vnum * vnum) the accumulating Hessian matrix
	@param[in]	opt:				optimization method. 0 for MLE and 1 for LSQ
*/
__device__ static inline void accum_Hessian(int vnum, float data, float model, float* deriv1, float* Hessian, int opt)
{
	int i = 0, j = 0;
	if (opt == 0) {
		data = (data > 0.0f) ? data : 0.0f;
		for (i = 0; i < vnum; i++)
			for (j = i; j < vnum; j++)
				Hessian[i * vnum + j] += (data / model / model) * deriv1[i] * deriv1[j];
	}
	else
		for (i = 0; i < vnum; i++)
			for (j = 0; j < vnum; j++) 
				Hessian[i * vnum + j] += 2.0f * deriv1[i] * deriv1[j];
	return;
}



/*!	@brief	accumulate the up-triangle part of the Fisher Information Matrix (FIM) at a pixel (opt = MLE only)
	@param[in]	vnum:				number of parameters in xvec
	@param[in] 	model:				the model at a pixel
	@param[in] 	deriv1:				(vnum) the derivatives of the model over the xvec
	@param[out]	FIM:				(vnum * vnum) the accumulating FIM matrix
*/
__device__ static inline void accum_FIM(int vnum, float model, float* deriv1, float* FIM)
{
	int i = 0, j = 0;
	for (i = 0; i < vnum; i++)	
		for (j = i; j < vnum; j++)
			FIM[i * vnum + j] += deriv1[i] * deriv1[j] / model;
	return;
}

#endif