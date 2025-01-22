#pragma once
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h> 

#ifndef FUNCTIONALS_CUH
#define FUNCTIONALS_CUH



/*! @brief		get the sum of the input PSF square, negative values are ignored. 
	@param[in]	boxsz:				int, size of the PSF square. 
	@param[in]	dataim:				(boxsz * boxsz) float, the PSF data. 
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
	@param[in]	boxsz:				int, size of the PSF square. 
	@param[in]	dataim:				(boxsz * boxsz) float, the PSF data. 
	@return 	dumMin:				float, the minimum of the PSF data.			
*/
__device__ static inline float get_psfmin(int boxsz, float* dataim)
{
	float dumMin = 1e10f;
	for (int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		if (dataim[pxID] < dumMin) 
			dumMin = dataim[pxID];
	}
	return dumMin;	
}



/*!	@brief		get the 3rd-minimum (medium of the first 5 minima) of the input PSF square
	@param[in]	boxsz:				int, size of the PSF square. 
	@param[in]	dataim:				(boxsz * boxsz) float, the PSF data. 
	@return 	dumMin_3rd:			float, the 3rd-minimum of the PSF data.			
*/
__device__ static inline float get_psf3rdmin(int boxsz, float* dataim)
{
	float dumMin_1st = 1e10f, dumMin_2nd = 1e10f, dumMin_3rd = 1e10f;
	for (int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		if (dataim[pxID] < dumMin_1st) {
			dumMin_3rd = dumMin_2nd;
			dumMin_2nd = dumMin_1st;
			dumMin_1st = dataim[pxID];
		}
		else if (dataim[pxID] < dumMin_2nd) {
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
	@param[in]	dataim:				(nchannels * boxsz * boxsz) float, the n-channel PSF data.
	@param[out]	xvec:				(NDIM + nchannels + nchannels) float, see definitions.h		
*/
__device__ static void init_xvec_1ch(int boxsz, float* dataim, float* xvec)
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

	xvec[NDIM] = max(dumI, IMIN);
	xvec[NDIM + 1] = max(bkg, BGMIN);
	return;
}



/*!	@brief		profile the PSF square data
	@param[in]	boxsz:				int, size of the PSF square. 
	@param[in]	dataim:				(nchannels * boxsz * boxsz) float, the n-channel PSF data.
	@param[out]	xvec:				(NDIM + nchannels + nchannels) float, see definitions.h		
*/
__device__ static void init_xvec_2ch(int boxsz, float* dataim, float* xvec)
{
	float dumx = 0.0f, dumy = 0.0f, dumI0 = 0.0f, bkg0 = 0.0f, dumI1 = 0.0f, bkg1 = 0.0f;
	int pxID = 0;

	// Get the photon numbers and bkg for the dataim
	bkg0 = get_psfmin(boxsz, dataim);
	bkg1 = get_psfmin(boxsz, dataim + boxsz * boxsz);

	// locate the x and y at the center of mass of the dataim
	// calculate the variance in the dataim
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
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out] maxJump:			(vnum) float, maxJump according to the xvec
*/
__device__ static void init_maxJump_1ch(int boxsz, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[NDIM] = max(100.0f, xvec[NDIM]);
	maxJump[NDIM + 1] = max(20.0f, xvec[NDIM + 1]);
	return;
}



/*!	@brief	set the maxJump
	@param[in]	boxsz:				int, size of the PSF square
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out] maxJump:			(vnum) float, maxJump according to the xvec
*/
__device__ static void init_maxJump_2ch(int boxsz, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
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
	@param[in]	loc_ref:			(NDIM) float, [locx, locy] the subpixel location of the emitter in the reference channel (box coordinates)
	@param[out]	loc_tar:			(NDIM) float, [locx, locy] the subpixel location of the emitter in a target channel (box coordinates)
	@param[out]	trans_deriv:		(4) float, [dxT/dxR, dyT/dxR, dxT/dyR, dyT/dyR] the transformation derivatives linking locations from the target channels to the reference channel					
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
	return;
}



/*!	@brief	Translating the actural pixels idx to spline idx by mapping the emitter center to the origin of the spline coordinates;	
			Build the delta matrix that matches the arrangement of the cspline coefficient for dot multiplication
			This matrix is built as:
				[[dy^0*dx^0, ..., dy^0*dx^3], [dy^1*dx^0, ..., dy^1*dx^3], ..., [dy^3*dx^0, ..., dy^3*dx^3]]
			It matches the coeff a_xyz at an cspline_idx[idy, idx] as:
				[[a_00, a_01, ..., a_03], [a_10, a_11, ..., a_13], ..., [a_30, a_31, ..., a_33]]
	@param[in]	splinesz:			int, size of the splines square
	@param[in]	loc:				(NDIM) float, [x, y] the subpixel location of the emitter (box coordinates)
	@param[out]	offset_x:			int pointer to the x-offset translating the loc from the box coordination to the spline coordination
	@param[out] offset_y:			int pointer to the y-offset translating the loc from the box coordination to the spline coordination
	@param[out]	deltaf:				(16) float, the deltas for the corresponding cspline coefficients
	@param[out] ddeltaf_dx:			(16) float, the deltas for the derivatives over the x-axis of the corresponding cspline coefficients
	@param[out] ddeltaf_dy:			(16) float, the deltas for the derivatives over the y-axis of the corresponding cspline coefficients  			
*/
__device__ static void DeltaConstruct2D(int splinesz, float* loc, int* offset_x, int* offset_y, float* deltaf, float* ddeltaf_dx, float* ddeltaf_dy)
{
	int i = 0, j = 0;
	float delta_x = 0.0f, delta_y = 0.0f, cx = 1.0f, cy = 1.0f;

	*offset_x = splinesz / 2 - (int)floor(loc[0]);
	*offset_y = splinesz / 2 - (int)floor(loc[1]);
	delta_x = 1.0f - (loc[0] - floor(loc[0]));
	delta_y = 1.0f - (loc[1] - floor(loc[1]));

	for (i = 0, cy = 1.0f; i < 4; i++) {
		for (j = 0, cx = 1.0f; j < 4; j++) {
			deltaf[i * 4 + j] = cy * cx;
			if (ddeltaf_dx && ddeltaf_dy) {
				if (j < 3) ddeltaf_dx[i * 4 + j + 1] = ((float)j + 1) * cy * cx;
				if (i < 3) ddeltaf_dy[(i + 1) * 4 + j] = ((float)i + 1) * cy * cx;
				if (j == 3) ddeltaf_dx[i * 4] = 0.0f;
				if (i == 3) ddeltaf_dy[j] = 0.0f;
			}
			cx *= delta_x;
		}
		cy *= delta_y;
	}
	return;
}



/*!	@brief	Construct the model value and deriv1 values at the pxID in each channel.
	@param[in]	pxID:				int, the index of the working pixel in each channel
	@param[in]	boxsz:				int, size of the PSF square
	@param[in] 	splinesz:			int, size of the splines square
	@param[in]	coeff_PSF:			(splinesz * splinesz * 16) float, the cubic spline coefficients of the experimental PSF for each channel
	@param[in]	offset_x:			int, the x-offset translating the location from the box-coordination to the spline-coordination
	@param[in] 	offset_y:			int, the y-offset translating the location from the box-coordination to the spline-coordination 
	@param[in]	deltaf:				(16) float, the deltas for the corresponding cspline coefficients
	@param[in] 	ddeltaf_dx:			(16) float, the deltas for the derivatives over the x-axis of the corresponding cspline coefficients
	@param[in] 	ddeltaf_dy:			(16) float, the deltas for the derivatives over the y-axis of the corresponding cspline coefficients
	@param[out]	psf:				float pointer to the psf value at the given pixel in each channel
	@param[out]	dpsf_dloc:			(NDIM) float, [dpsf_dx, dpsf_dy] the derivatives of the psf over the location 	 
*/
__device__ static void fconstruct_spline2D(int pxID, int boxsz, int splinesz, float* coeff_PSF, 
	int offset_x, int offset_y, float* deltaf, float* ddeltaf_dx, float* ddeltaf_dy, float* psf, float* dpsf_dloc)
{
	int spindx = 0, spindy = 0, splineIdx = 0, i = 0;
	float dum = 0.0f;
	
	if (dpsf_dloc && ddeltaf_dx && ddeltaf_dy)
		memset(dpsf_dloc, 0, NDIM * sizeof(float));

	spindx = (pxID % boxsz) + offset_x;
	spindy = (pxID / boxsz) + offset_y;

	spindx = min(max(spindx, 0), splinesz - 1);
	spindy = min(max(spindy, 0), splinesz - 1);

	splineIdx = spindy * splinesz + spindx;
	for (i = 0, dum = 0.0f; i < 16; i++) {
		dum += deltaf[i] * coeff_PSF[splineIdx * 16 + i];
		if (dpsf_dloc && ddeltaf_dx && ddeltaf_dy) {
			dpsf_dloc[0] -= ddeltaf_dx[i] * coeff_PSF[splineIdx * 16 + i];
			dpsf_dloc[1] -= ddeltaf_dy[i] * coeff_PSF[splineIdx * 16 + i];
		}
	}
	*psf = max(dum, FLT_EPSILON);
	
	return;
}



/*!	@brief	box-constrain the xvec
	@param[in]	boxsz:				int, size of the PSF square
	@param[out] xvec:				(vnum) float, see definitions.h
*/
__device__ static void boxConstrain_1ch(int boxsz, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
	xvec[NDIM] = max(xvec[NDIM], IMIN);
	xvec[NDIM + 1] = max(xvec[NDIM + 1], BGMIN);
	return;
}



/*!	@brief	box-constrain the xvec
	@param[in]	boxsz:				int, size of the PSF square
	@param[out] xvec:				(vnum) float, see definitions.h
*/
__device__ static void boxConstrain_2ch(int boxsz, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
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
	deriv1[NDIM] = dum_fracN * psf;
	deriv1[NDIM + 1] = (chID == 0) ? xvec[NDIM] * psf : -xvec[NDIM] * psf;
	deriv1[NDIM + 2 + chID] = 1.0f;
	return;
}



/*!	@brief	accumulat the loss across the pixels in each channel
	@param[in]	data:				float, the data at a pixel
	@param[in] 	model:				float, the model at a pixel
	@param[out]	loss:				pointer to the accumulating loss value
	@param[in]	opt:				int, optimization method. 0 for MLE and 1 for LSQ
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



/*!	@brief	accumulat the gradient across the pixels in both channels
	@param[in]	vnum:				int, the number of parameters in xvec
	@param[in]	data:				float, the data at a pixel
	@param[in] 	model:				float, the model at a pixel
	@param[in]	deriv1:				(vnum) float, the derivatives of the model over the xvec
	@param[out]	grad:				(vnum) float, the accumulating gradient vector
	@param[in]	opt:				int, optimization method. 0 for MLE and 1 for LSQ
*/
__device__ static inline void accum_grad(int vnum, float data, float model, float* deriv1, float* grad, int opt)
{
	float dum = 0.0f;
	if (opt == 0)
		dum = (data > 0.0f) ? (1.0f - data / model) : 1.0f;
	else
		dum = 2.0f * (model - data);
		
	for (int i = 0; i < vnum; i++)
		grad[i] += dum * deriv1[i];
	
	return;
}



/*!	@brief	accumulate the up-triangle part of the Hessian at a pixel
	@param[in]	vnum:				int, the number of parameters in xvec
	@param[in]	data:				float, the data at a pixel
	@param[in] 	model:				float, the model at a pixel
	@param[in]	deriv1:				(vnum) float, the derivatives of the model over the xvec
	@param[out]	Hessian:			(vnum * vnum) float, the accumulating Hessian matrix
	@param[in]	opt:				int, optimization method. 0 for MLE and 1 for LSQ
*/
__device__ static inline void accum_Hessian(int vnum, float data, float model, float* deriv1, float* Hessian, int opt)
{
	float dum = 0.0f;
	if (opt == 0)
		dum = (data > 0.0f) ? (data / model / model) : 0.0f;
	else
		dum = 2.0f;

	for (unsigned int i = 0; i < vnum; i++) {
		Hessian[i * vnum + i] += dum * deriv1[i] * deriv1[i];
		for (unsigned int j = i + 1; j < vnum; j++)
			Hessian[i * vnum + j] += dum * deriv1[i] * deriv1[j];
	}
	return;
}



/*!	@brief	accumulate the up-triangle part of the Fisher Information Matrix (FIM) at a pixel
	@param[in]	vnum:				int, the number of parameters in xvec
	@param[in] 	model:				float, the model at a pixel
	@param[in]	deriv1:				(vnum) float, the derivatives of the model over the xvec
	@param[out]	FIM:				(vnum * vnum) float, the accumulating FIM matrix
*/
__device__ static inline void accum_FIM(int vnum, float model, float* deriv1, float* FIM)
{
	for (unsigned int i = 0; i < vnum; i++)	
		for (unsigned int j = i; j < vnum; j++)
			FIM[i * vnum + j] += deriv1[i] * deriv1[j] / model;
	return;
}

#endif