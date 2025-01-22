#pragma once
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h> 
#include "definitions.h"

#ifndef FUNCTIONALS_CUH
#define FUNCTIONALS_CUH


/*!	@brief	Parameters in general
	@param[in]	boxsz:				int, size of the PSF square
	@param[in] 	splineszx:			int, size of the splines cube (lateral-axis)
	@param[in]	splineszz:			int, size of the splines cube (axial-axis)
	@param[in]	dataim:				(nchannels * boxsz * boxsz) float, the PSF data. 
	@param[in]	varim:				(nchannels * boxsz * boxsz) float, the pixel-dependent variance, NULL for pixel-independent variance
	@param[in]	coeff_PSF:			(nchannels * splineszx * splineszx * splineszz* 64) float, cubic spline coefficients of the experimental PSF

	@param[in]	warpdeg:			int, the degree of the polynomial function that warps 2d locations from one channel to the other
	@param[in]	lu:					(nchannels * 2) int, [lc, uc] the left and upper corner of the PSF square in each channel (image coordinates)
	@param[in]	coeff_R2T:			((nchannels - 1) * 2 * warpdeg * warpdeg) float, the [coeffx_B2A, coeffy_B2A] coefficients of the polynomial warping from refernce channel to each of the target channel

	@param[in]	nchannels:			int, number of channels
	@param[in]	vnum:				int, the number of parameters in xvec
	@param[in]	xvec:				(vnum) float, see definitions.h
	@param[out] maxJump:			(vnum) float, maxJump according to the xvec
	@param[out] loss:				(1) float, pointer to the negative log likeliklihood
	@param[out]	grad:				(vnum) float, gradient
	@param[out]	Hessian:			(vnum * vnum) float, Hessian
	@param[out]	FIM:				(vnum * vnum) float, Fisher Information Matrix
	@param[in]	opt:				optimization method. 0 for MLE and 1 for LSQ
*/


/*! @brief		get the sum of the input PSF square, negative values are ignored. 
	@return 	sum of the PSF data.	
*/
__device__ static inline float get_psfsum(int boxsz, float* dataim, float offset)
{
	float dumSum = 0.0f;
	for (int pxID = 0; pxID < boxsz * boxsz; pxID++)
		if (dataim[pxID] > offset)	
			dumSum += dataim[pxID] - offset;
	return dumSum;	
}



/*! @brief		get the minimum of the input PSF square. 
	@return 	minimum of the PSF data.	
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
	@return 	the 3rd-minimum of the PSF data.			
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



/*!	@brief	Transform the subpixel loc in the reference channel to that in a target channels
			the locs are in the box-coordinate system, which needs the lu (left and upper_corner) to translate into image-coordinate
			xT = sum([coeffx_i * yR^(i//deg) * xR^(i%deg) for i in range(deg * deg)])
			yT = sum([coeffy_i * yR^(i//deg) * xR^(i%deg) for i in range(deg * deg)])
	@param[in]	lu_ref:				(2) int, [lc, uc] the left and upper corner of the PSF square in the reference channel (image coordinates)
	@param[in]	lu_tar:				(2) int, [lc, uc] the left and upper corner of the PSF square in a target channel (image coordinates)
	@param[in]	loc_ref:			(NDIM) float, [locx, locy, locz] the subpixel location of the emitter in the reference channel (box coordinates)
	@param[out]	loc_tar:			(NDIM) float, [locx, locy, locz] the subpixel location of the emitter in a target channel (box coordinates)
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
	loc_tar[2] = loc_ref[2];
	return;
}



/*!	@brief	Translating actural pixels idx to spline idx by mapping the emitter center to the origin of the spline coordinates;	
			Build the delta matrix that matches the arrangement of the cspline coefficient for dot multiplication
			This matrix is built as:
				[[dy^0*dx^0, ..., dy^0*dx^3], [dy^1*dx^0, ..., dy^1*dx^3], ..., [dy^3*dx^0, ..., dy^3*dx^3]]
			It matches the coeff a_xyz at an cspline_idx[idy, idx] as:
				[[a_00, a_01, ..., a_03], [a_10, a_11, ..., a_13], ..., [a_30, a_31, ..., a_33]]
	@param[in]	loc:				(NDIM) float, [x, y, z] the subpixel location of the emitter (box coordinates)
	@param[out]	offset_x:			int pointer to the x-offset translating the loc from the box coordination to the spline coordination
	@param[out] offset_y:			int pointer to the y-offset translating the loc from the box coordination to the spline coordination
	@param[out] offset_z:			int pointer to the z-offset translating the loc from the box coordination to the spline coordination
	@param[out]	deltaf:				(64) float, the deltas for the corresponding cspline coefficients
	@param[out] ddeltaf_dx:			(64) float, the deltas for the derivatives over the x-axis of the corresponding cspline coefficients
	@param[out] ddeltaf_dy:			(64) float, the deltas for the derivatives over the y-axis of the corresponding cspline coefficients  	
	@param[out] ddeltaf_dz:			(64) float, the deltas for the derivatives over the z-axis of the corresponding cspline coefficients
*/
__device__ static void DeltaConstruct3D(int splineszx, int splineszz, float* loc,
	int* offset_x, int* offset_y, int* offset_z, float *deltaf, float *ddeltaf_dx, float *ddeltaf_dy, float *ddeltaf_dz)
{
	int i = 0, j = 0, k = 0;
	float cz = 1.0f, cy = 1.0f, cx = 1.0f;
	float delta_z = 0.0f, delta_y = 0.0f, delta_x = 0.0f;

	*offset_x = splineszx / 2 - (int)floor(loc[0]);
	*offset_y = splineszx / 2 - (int)floor(loc[1]);
	*offset_z = splineszz / 2 - (int)floor(loc[2]);
	delta_x = 1.0f - (loc[0] - floor(loc[0]));
	delta_y = 1.0f - (loc[1] - floor(loc[1]));
	delta_z = 1.0f - (loc[2] - floor(loc[2]));

	for (i = 0, cz = 1.0f; i < 4; i++) {
		for (j = 0, cy = 1.0f; j < 4; j++) {
			for (k = 0, cx = 1.0f; k < 4; k++) {
				deltaf[i*16 + j*4 + k] = cz * cy * cx;
				if (ddeltaf_dz && ddeltaf_dy && ddeltaf_dx) {
					if (k < 3) ddeltaf_dx[i*16 + j*4 + k+1] = ((float)k + 1) * cz * cy * cx;
					if (j < 3) ddeltaf_dy[i*16 + (j+1)*4 + k] = ((float)j + 1) * cz * cy * cx;
					if (i < 3) ddeltaf_dz[(i+1)*16 + j*4 + k] = ((float)i + 1) * cz * cy * cx;
					if (k == 3) ddeltaf_dx[i*16 + j*4] = 0.0f;
					if (j == 3) ddeltaf_dy[i*16 + k] = 0.0f;
					if (i == 3) ddeltaf_dz[j*4 + k] = 0.0f;	
				}
				cx *= delta_x;
			}
			cy *= delta_y;
		}
		cz *= delta_z;
	}
	return;
}



/*!	@brief	Construct the model value and deriv1 values at pxID.
	@param[in]	pxID:				int, the index of the working pixel in each channel
	@param[in]	offset_x:			int, the x-offset translating the location from the box-coordination to the spline-coordination
	@param[in] 	offset_y:			int, the y-offset translating the location from the box-coordination to the spline-coordination
	@param[in] 	offset_z:			int, the z-offset translating the location from the box-coordination to the spline-coordination
	@param[in]	deltaf:				(64) float, the deltas for the corresponding cspline coefficients
	@param[in] 	ddeltaf_dx:			(64) float, the deltas for the derivatives over the x-axis of the corresponding cspline coefficients
	@param[in] 	ddeltaf_dy:			(64) float, the deltas for the derivatives over the y-axis of the corresponding cspline coefficients
	@param[in] 	ddeltaf_dz:			(64) float, the deltas for the derivatives over the z-axis of the corresponding cspline coefficients
	@param[out]	psf:				float pointer to the psf value at the given pixel in each channel
	@param[out]	dpsf_dloc:			(NDIM) float, [dpsf_dx, dpsf_dy, dpsf_dz] the derivatives of the psf over the location	 
*/
__device__ static void fconstruct_spline3D(int pxID, int boxsz, int splineszx, int splineszz, float* coeff_PSF, 
	int offset_x, int offset_y, int offset_z, float* deltaf, float* ddeltaf_dx, float* ddeltaf_dy, float* ddeltaf_dz,
	float* psf, float* dpsf_dloc)
{
	int spindx = 0, spindy = 0, spindz = 0, splineIdx = 0, i = 0;
	float dum = 0.0f;

	if (dpsf_dloc && ddeltaf_dx && ddeltaf_dy && ddeltaf_dz)
		memset(dpsf_dloc, 0, NDIM * sizeof(float));

	spindx = (pxID % boxsz) + offset_x;
	spindy = (pxID / boxsz) + offset_y;
	spindz = splineszz / 2 + offset_z;

	spindx = min(max(spindx, 0), splineszx - 1);
	spindy = min(max(spindy, 0), splineszx - 1);
	spindz = min(max(spindz, 0), splineszz - 1);

	splineIdx = spindz * splineszx*splineszx + spindy * splineszx + spindx;
	for (i = 0, dum = 0.0f; i < 64; i++) {
		dum += deltaf[i] * coeff_PSF[splineIdx * 64 + i];
		if (dpsf_dloc && ddeltaf_dx && ddeltaf_dy && ddeltaf_dz) {
			dpsf_dloc[0] -= ddeltaf_dx[i] * coeff_PSF[splineIdx * 64 + i];
			dpsf_dloc[1] -= ddeltaf_dy[i] * coeff_PSF[splineIdx * 64 + i];
			dpsf_dloc[2] -= ddeltaf_dz[i] * coeff_PSF[splineIdx * 64 + i];
		}
	}
	*psf = max(dum, FLT_EPSILON);
	return;
}



/*!	@brief	accumulat the loss across the pixels
	@param[in]	data:				the data at a pixel
	@param[in] 	model:				the model at a pixel
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
	@param[in]	data:				the data at a pixel
	@param[in]	model:				the model at a pixel
	@param[in] 	deriv1:				(vnum) float, the derivatives of the model over the xvec
*/
__device__ static inline void accum_grad(int vnum, float data, float model, float* deriv1, float* grad, int opt)
{
	if (opt == 0) {
		data = (data > 0.0f) ? data : 0.0f;
		for (unsigned int i = 0; i < vnum; i++)
			grad[i] += (1.0f - data / model) * deriv1[i];
	}
	else
		for (unsigned int i = 0; i < vnum; i++)
			grad[i] += 2.0f * (model - data) * deriv1[i];
	return;
}



/*!	@brief	accumulate the up-triangle part of the Hessian at a pixel
	@param[in]	data:				the data at a pixel
	@param[in]	model:				the model at a pixel
	@param[in]	deriv1:				(vnum) float, the derivatives of the model over the xvec
*/
__device__ static inline void accum_Hessian(int vnum, float data, float model, float* deriv1, float* Hessian, int opt)
{
	if (opt == 0) {
		data = (data > 0.0f) ? data : 0.0f;
		for (unsigned int i = 0; i < vnum; i++)
			for (unsigned int j = i; j < vnum; j++)
				Hessian[i * vnum + j] += (data / model / model) * deriv1[i] * deriv1[j];
	}
	else
		for (unsigned int i = 0; i < vnum; i++)
			for (unsigned int j = 0; j < vnum; j++) 
				Hessian[i * vnum + j] += 2.0f * deriv1[i] * deriv1[j];
	return;
}



/*!	@brief	accumulate the up-triangle part of the Fisher Information Matrix (FIM) at a pixel (opt = MLE only)
	@param[in] 	model:				the model at a pixel
	@param[in] 	deriv1:				(vnum) float, the derivatives of the model over the xvec
*/
__device__ static inline void accum_FIM(int vnum, float model, float* deriv1, float* FIM)
{
	for (unsigned int i = 0; i < vnum; i++)	
		for (unsigned int j = i; j < vnum; j++)
			FIM[i * vnum + j] += deriv1[i] * deriv1[j] / model;
	return;
}





//////////////////////////////// FUNCTIONALS FOR 1CH ///////////////////////////////////////
/*!	@brief		profile the PSF square data		*/
__device__ static void init_xvec_1ch(int boxsz, int splineszz, float* dataim, float* xvec)
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

	xvec[2] = 0.5f * (float)splineszz;
	xvec[NDIM] = max(dumI, IMIN);
	xvec[NDIM + 1] = max(bkg, BGMIN);
	return;
}



/*!	@brief	set the maxJump		*/
__device__ static void init_maxJump_1ch(int boxsz, int splineszz, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[2] = max(2.0f, 0.05f * (float)splineszz);
	maxJump[NDIM] = max(100.0f, xvec[NDIM]);
	maxJump[NDIM + 1] = max(20.0f, xvec[NDIM + 1]);
	return;
}



/*!	@brief	box-constrain the xvec		*/
__device__ static void boxConstrain_1ch(int boxsz, int splineszz, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
	xvec[2] = min(max(xvec[2], 0.0f), (float)splineszz);
	xvec[NDIM] = max(xvec[NDIM], IMIN);
	xvec[NDIM + 1] = max(xvec[NDIM + 1], BGMIN);
	return;
}



/*!	@brief	calculate the model at a pixel for each channel
	@param[in]	psf:				float, the psf value at the pixel in the given channel
	@param[out]	model:				(1) pointer to the model
*/
__device__ static void get_pixel_model_1ch(float psf, float* xvec, float* model)
{
	*model = max(xvec[NDIM] * psf + xvec[NDIM + 1], BGMIN);
	return;
}



/*!	@brief	calculate the derivatives at a pixel for each channel
	@param[in]	psf:				float, the psf value at the pixel in the given channel
	@param[in]	dpsf_dloc:			(NDIM) float, [dpsf_dx, dpsf_dy] the derivatives of the psf over the location at the pixel in the given channel
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



/*!	@brief	calculate the -log(likelihood), gradients, and Hessian		*/
__device__ static void get_lgH_1ch(int boxsz, int splineszx, int splineszz, float* dataim, float* varim, float* coeff_PSF, float* xvec,
	float* loss, float* grad, float* Hessian, int opt)
{
	const int vnum = NDIM + 2;
	
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float psf = 0.0f, loc[NDIM] = {0.0f}, dpsf_dloc[NDIM] = {0.0f};
	float data = 0.0f, model = 0.0f, deriv1[vnum] = {0.0f};
	
	memcpy(loc, xvec, NDIM * sizeof(float));
	
	*loss = 0.0f;
	memset(grad, 0, vnum * sizeof(float));
	memset(Hessian, 0, vnum * vnum * sizeof(float));
	DeltaConstruct3D(splineszx, splineszz, loc, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		data = dataim[pxID];
		fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, coeff_PSF, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, &psf, dpsf_dloc);
		get_pixel_model_1ch(psf, xvec, &model);
		get_pixel_deriv_1ch(psf, dpsf_dloc, xvec, deriv1);
		if (varim) {
			data += varim[pxID];
			model += varim[pxID];
		}			
		accum_loss(data, model, loss, opt);
		accum_grad(vnum, data, model, deriv1, grad, opt);
		accum_Hessian(vnum, data, model, deriv1, Hessian, opt);
	}
	for (unsigned int i = 0; i < vnum; i++)
		for (unsigned int j = i + 1; j < vnum; j++)
			Hessian[j * vnum + i] = Hessian[i * vnum + j];
	return;
}



/*!	@brief	calculate the FIM for opt == MLE		*/
__device__ static void get_FIM_1ch(int boxsz, int splineszx, int splineszz, float* varim, float* coeff_PSF, float* xvec, float* FIM)
{
	const int vnum = NDIM + 2;
	
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float psf = 0.0f, loc[NDIM] = {0.0f}, dpsf_dloc[NDIM] = {0.0f};
	float model = 0.0f, deriv1[vnum] = {0.0f};
	
	memcpy(loc, xvec, NDIM * sizeof(float));

	memset(FIM, 0, vnum * vnum * sizeof(float));
	DeltaConstruct3D(splineszx, splineszz, loc, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, coeff_PSF, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, &psf, dpsf_dloc);
		get_pixel_model_1ch(psf, xvec, &model);
		get_pixel_deriv_1ch(psf, dpsf_dloc, xvec, deriv1);
		if (varim) 
			model += varim[pxID];
				
		accum_FIM(vnum, model, deriv1, FIM);
	}
	for (unsigned int i = 0; i < vnum; i++)
		for (unsigned int j = i + 1; j < vnum; j++)
			FIM[j * vnum + i] = FIM[i * vnum + j];
	return;
}






//////////////////////////////// FUNCTIONALS FOR 2CH ///////////////////////////////////////
/*!	@brief		profile the PSF square data		*/
__device__ static void init_xvec_2ch(int boxsz, int splineszz, float* dataim, float* xvec)
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
	xvec[2] = 0.5f * (float)splineszz;
	
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



/*!	@brief	set the maxJump		*/
__device__ static void init_maxJump_2ch(int boxsz, int splineszz, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[2] = max(2.0f, 0.05f * (float)splineszz);
	maxJump[NDIM] = max(100.0f, xvec[NDIM]);
	maxJump[NDIM + 1] = max(0.1f, xvec[NDIM + 1]);
	maxJump[NDIM + 2] = max(20.0f, xvec[NDIM + 2]);
	maxJump[NDIM + 3] = max(20.0f, xvec[NDIM + 3]);	
	return;
}



/*!	@brief	box-constrain the xvec		*/
__device__ static void boxConstrain_2ch(int boxsz, int splineszz, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
	xvec[2] = min(max(xvec[2], 0.0f), (float)splineszz);
	xvec[NDIM] = max(xvec[NDIM], IMIN);
	xvec[NDIM + 1] = min(max(xvec[NDIM + 1], 0.0f), 1.0f);
	xvec[NDIM + 2] = max(xvec[NDIM + 2], BGMIN);
	xvec[NDIM + 3] = max(xvec[NDIM + 3], BGMIN);
	return;
}



/*!	@brief	calculate the model at a pixel for each channel
	@param[in]	chID:				int, channel ID
	@param[in]	psf:				float, the psf value at the pixel in the given channel
	@param[out]	model:				(1) pointer to the model
*/
__device__ static void get_pixel_model_2ch(int chID, float psf, float* xvec, float* model)
{
	const float dum_fracN = (chID == 0) ? xvec[NDIM + 1] : 1.0f - xvec[NDIM + 1];
	*model = max(xvec[NDIM] * dum_fracN * psf + xvec[NDIM + 2 + chID], BGMIN);
	return;
}



/*!	@brief	calculate the derivatives at a pixel for each channel
	@param[in]	chID:				int, channel ID
	@param[in]	psf:				float, the psf value at the pixel in the given channel
	@param[in]	dpsf_dloc:			(NDIM) float, [dpsf_dx, dpsf_dy] the derivatives of the psf over the location at the pixel in the given channel
	@param[in]	trans_deriv_R2T:	(4) float, [dxT/dxR, dyT/dxR, dxT/dyR, dyT/dyR] the transformation derivatives linking locations from the ref channel to a target channels
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



/*!	@brief	calculate the loss, gradients, and Hessian	*/
__device__ static void get_lgH_2ch(int boxsz, int splineszx, int splineszz, float* dataim, float* varim, float* coeff_PSF, float* xvec, 
	int warpdeg, int* lu, float* coeff_R2T, float* loss, float* grad, float* Hessian, int opt)
{
	const int nchannels = 2, vnum = NDIM + nchannels + nchannels;
	
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float loc_ref[NDIM] = {0.0f}, loc[NDIM] = {0.0f}, trans_deriv[4] = {0.0f}, psf = 0.0f, dpsf_dloc[NDIM] = {0.0f};
	float data = 0.0f, model = 0.0f, deriv1[vnum] = {0.0f};
	
	int *lu_ref = lu, *lu_tar = lu + 2;
	float *p_coeff_PSF = nullptr;

	memcpy(loc_ref, xvec, NDIM * sizeof(float));
	
	*loss = 0.0f;
	memset(grad, 0, vnum * sizeof(float));
	memset(Hessian, 0, vnum * vnum * sizeof(float));
	for (unsigned int chID = 0; chID < nchannels; chID++) {
		
		if (chID == 0) {
			memcpy(loc, loc_ref, NDIM * sizeof(float));
			memset(trans_deriv, 0, 4 * sizeof(float));
			trans_deriv[0] = 1.0f; 		
			trans_deriv[3] = 1.0f;	
		}	
		else 
			TransR2T(warpdeg, coeff_R2T, lu_ref, lu_tar, loc_ref, loc, trans_deriv);

		p_coeff_PSF = coeff_PSF + chID * splineszx * splineszx * splineszz * 64;
		DeltaConstruct3D(splineszx, splineszz, loc, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);
		for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
			data = dataim[chID * boxsz * boxsz + pxID];
			fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, p_coeff_PSF, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, &psf, dpsf_dloc);
			get_pixel_model_2ch(chID, psf, xvec, &model);
			get_pixel_deriv_2ch(chID, psf, dpsf_dloc, trans_deriv, xvec, deriv1);
			if (varim) {
				data += varim[chID * boxsz * boxsz + pxID];
				model += varim[chID * boxsz * boxsz + pxID];
			}	

			accum_loss(data, model, loss, opt);
			accum_grad(vnum, data, model, deriv1, grad, opt);
			accum_Hessian(vnum, data, model, deriv1, Hessian, opt);
		}
	}
	for (unsigned int i = 0; i < vnum; i++)
		for (unsigned int j = i + 1; j < vnum; j++)
			Hessian[j * vnum + i] = Hessian[i * vnum + j];
	return;
}



/*!	@brief	calculate the FIM		*/
__device__ static void get_FIM_2ch(int boxsz, int splineszx, int splineszz, float* varim, float* coeff_PSF, float* xvec, 
	int warpdeg, int* lu, float* coeff_R2T, float* FIM)
{
	const int nchannels = 2, vnum = NDIM + nchannels + nchannels;
	
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float loc_ref[NDIM] = {0.0f}, loc[NDIM] = {0.0f}, trans_deriv[4] = {0.0f}, psf = 0.0f, dpsf_dloc[NDIM] = {0.0f};
	float model = 0.0f, deriv1[vnum] = {0.0f};

	int *lu_ref = lu, *lu_tar = lu + 2;
	float *p_coeff_PSF = nullptr;
	
	memcpy(loc_ref, xvec, NDIM * sizeof(float));
	memset(FIM, 0, vnum * vnum * sizeof(float));
	for (unsigned int chID = 0; chID < nchannels; chID++) {
		
		if (chID == 0) {
			memcpy(loc, loc_ref, NDIM * sizeof(float));
			memset(trans_deriv, 0, 4 * sizeof(float));
			trans_deriv[0] = 1.0f; 		
			trans_deriv[3] = 1.0f;	
		}	
		else 
			TransR2T(warpdeg, coeff_R2T, lu_ref, lu_tar, loc_ref, loc, trans_deriv);

		p_coeff_PSF = coeff_PSF + chID * splineszx * splineszx * splineszz * 64;
		DeltaConstruct3D(splineszx, splineszz, loc, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);
		for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
			fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, p_coeff_PSF, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, &psf, dpsf_dloc);
			get_pixel_model_2ch(chID, psf, xvec, &model);
			get_pixel_deriv_2ch(chID, psf, dpsf_dloc, trans_deriv, xvec, deriv1);
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





//////////////////////////////// FUNCTIONALS FOR BP ///////////////////////////////////////
/*!	@brief		profile the PSF square data		*/
__device__ static void init_xvec_BP(int boxsz, int splineszz, float* dataim, float* xvec)
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
	xvec[2] = 0.5f * (float)splineszz;
	
	// get photon numbers in the 2nd channel
	for (pxID = boxsz * boxsz, dumI1 = 0.0; pxID < 2 * boxsz * boxsz; pxID++)
		if (dataim[pxID] > 0)
			dumI1 += dataim[pxID] - bkg1;

	xvec[NDIM] = max(dumI0 + dumI1, IMIN);
	xvec[NDIM + 1] = max(0.5f * (bkg0 + bkg1), BGMIN);
	return;
}



/*!	@brief	set the maxJump		*/
__device__ static void init_maxJump_BP(int boxsz, int splineszz, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[2] = max(2.0f, 0.05f * (float)splineszz);
	maxJump[NDIM] = max(100.0f, xvec[NDIM]);
	maxJump[NDIM + 1] = max(20.0f, xvec[NDIM + 1]);
	return;
}



/*!	@brief	box-constrain the xvec		*/
__device__ static void boxConstrain_BP(int boxsz, int splineszz, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
	xvec[2] = min(max(xvec[2], 0.0f), (float)splineszz);
	xvec[NDIM] = max(xvec[NDIM], IMIN);
	xvec[NDIM + 1] = max(xvec[NDIM + 1], BGMIN);
	return;
}



/*!	@brief	calculate the model at a pixel for each channel
	@param[in]	psf:				float, the psf value at the pixel in the given channel
	@param[out]	model:				(1) pointer to the model
*/
__device__ static void get_pixel_model_BP(float psf, float* xvec, float* model)
{
	*model = max(xvec[NDIM] * 0.5f * psf + xvec[NDIM + 1], BGMIN);
	return;
}



/*!	@brief	calculate the derivatives at a pixel for each channel
	@param[in]	psf:				float, the psf value at the pixel in the given channel
	@param[in]	dpsf_dloc:			(NDIM) float, [dpsf_dx, dpsf_dy] the derivatives of the psf over the location at the pixel in the given channel
	@param[in]	trans_deriv_R2T:	(4) float, [dxT/dxR, dyT/dxR, dxT/dyR, dyT/dyR] the transformation derivatives linking locations from the ref channel to a target channels
	@param[out]	deriv1:				(vnum) float, the derivatives of the model over the xvec
*/
__device__ static void get_pixel_deriv_BP(float psf, float* dpsf_dloc, float* trans_deriv_R2T, float* xvec, float* deriv1)
{
	memset(deriv1, 0, (NDIM + 2) * sizeof(float));
	deriv1[0] = xvec[NDIM] * 0.5f * (dpsf_dloc[0] * trans_deriv_R2T[0] + dpsf_dloc[1] * trans_deriv_R2T[1]);
	deriv1[1] = xvec[NDIM] * 0.5f * (dpsf_dloc[0] * trans_deriv_R2T[2] + dpsf_dloc[1] * trans_deriv_R2T[3]);
	deriv1[2] = xvec[NDIM] * 0.5f * dpsf_dloc[2];
	deriv1[NDIM] = 0.5f * psf;
	deriv1[NDIM + 1] = 1.0f;		
	return;
}



/*!	@brief	calculate the loss, gradients, and Hessian	*/
__device__ static void get_lgH_BP(int boxsz, int splineszx, int splineszz, float* dataim, float* varim, float* coeff_PSF, float* xvec, 
	int warpdeg, int* lu, float* coeff_R2T, float* loss, float* grad, float* Hessian, int opt)
{
	const int nchannels = 2, vnum = NDIM + 2;
	
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float loc_ref[NDIM] = {0.0f}, loc[NDIM] = {0.0f}, trans_deriv[4] = {0.0f}, psf = 0.0f, dpsf_dloc[NDIM] = {0.0f};
	float data = 0.0f, model = 0.0f, deriv1[vnum] = {0.0f};
	
	int *lu_ref = lu, *lu_tar = lu + 2;
	float *p_coeff_PSF = nullptr;
	
	memcpy(loc_ref, xvec, NDIM * sizeof(float));
	
	*loss = 0.0f;
	memset(grad, 0, vnum * sizeof(float));
	memset(Hessian, 0, vnum * vnum * sizeof(float));
	for (unsigned int chID = 0; chID < nchannels; chID++) {
		
		if (chID == 0) {
			memcpy(loc, loc_ref, NDIM * sizeof(float));
			memset(trans_deriv, 0, 4 * sizeof(float));
			trans_deriv[0] = 1.0f; 		
			trans_deriv[3] = 1.0f;	
		}	
		else 
			TransR2T(warpdeg, coeff_R2T, lu_ref, lu_tar, loc_ref, loc, trans_deriv);

		p_coeff_PSF = coeff_PSF + chID * splineszx * splineszx * splineszz * 64;
		DeltaConstruct3D(splineszx, splineszz, loc, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);
		for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
			data = dataim[chID * boxsz * boxsz + pxID];
			fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, p_coeff_PSF, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, &psf, dpsf_dloc);
			get_pixel_model_BP(psf, xvec, &model);
			get_pixel_deriv_BP(psf, dpsf_dloc, trans_deriv, xvec, deriv1);
			if (varim) {
				data += varim[chID * boxsz * boxsz + pxID];
				model += varim[chID * boxsz * boxsz + pxID];
			}	

			accum_loss(data, model, loss, opt);
			accum_grad(vnum, data, model, deriv1, grad, opt);
			accum_Hessian(vnum, data, model, deriv1, Hessian, opt);
		}
	}
	for (unsigned int i = 0; i < vnum; i++)
		for (unsigned int j = i + 1; j < vnum; j++)
			Hessian[j * vnum + i] = Hessian[i * vnum + j];
	return;
}



/*!	@brief	calculate the FIM		*/
__device__ static void get_FIM_BP(int boxsz, int splineszx, int splineszz, float* varim, float* coeff_PSF, float* xvec, 
	int warpdeg, int* lu, float* coeff_R2T, float* FIM)
{
	const int nchannels = 2, vnum = NDIM + 2;
	
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float loc_ref[NDIM] = {0.0f}, loc[NDIM] = {0.0f}, trans_deriv[4] = {0.0f}, psf = 0.0f, dpsf_dloc[NDIM] = {0.0f};
	float model = 0.0f, deriv1[vnum] = {0.0f};

	int *lu_ref = lu, *lu_tar = lu + 2;
	float *p_coeff_PSF = nullptr;
	
	memcpy(loc_ref, xvec, NDIM * sizeof(float));
	memset(FIM, 0, vnum * vnum * sizeof(float));
	for (unsigned int chID = 0; chID < nchannels; chID++) {
		
		if (chID == 0) {
			memcpy(loc, loc_ref, NDIM * sizeof(float));
			memset(trans_deriv, 0, 4 * sizeof(float));
			trans_deriv[0] = 1.0f; 		
			trans_deriv[3] = 1.0f;	
		}	
		else 
			TransR2T(warpdeg, coeff_R2T, lu_ref, lu_tar, loc_ref, loc, trans_deriv);

		p_coeff_PSF = coeff_PSF + chID * splineszx * splineszx * splineszz * 64;
		DeltaConstruct3D(splineszx, splineszz, loc, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);
		for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
			fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, p_coeff_PSF, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, &psf, dpsf_dloc);
			get_pixel_model_BP(psf, xvec, &model);
			get_pixel_deriv_BP(psf, dpsf_dloc, trans_deriv, xvec, deriv1);
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
#endif