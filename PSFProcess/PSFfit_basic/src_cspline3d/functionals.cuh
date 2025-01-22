#pragma once
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h> 
#include "definitions.h"

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



/*! @brief		get the minimum of the input PSF square. 
	@param[in]	boxsz:				size of the PSF square. 
	@param[in]	dataim:				pointer to the PSF data. 
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



/*!	@brief		profile the PSF in each of the channels
	@param[in]	boxsz:				int, size of the PSF square. 
	@param[in]	splineszz:			int, size of the spline coefficients (z-axis)
	@param[in]	dataim:				pointer to the PSF data. 
	@param[out]	xvec:				(VNUM) float, see definitions.h.		
*/
__device__ static void init_xvec(int boxsz, int splineszz, float* dataim, float* xvec)
{
	float dumx = 0.0f, dumy = 0.0f, dumSum = 0.0f, dumMin = 0.0f;
	int pxID = 0;
	
	// Get the photon numbers and bkg for the dataim
	dumSum = get_psfsum(boxsz, dataim);
	dumMin = get_psfmin(boxsz, dataim);
	
	// locate the x and y at the center of mass of the dataim
	for (pxID = 0, dumx = 0.0f, dumy = 0.0f; pxID < boxsz * boxsz; pxID++)
		if (dataim[pxID] > 0) {
			dumx += ((float)(pxID % boxsz) + 0.5f) * dataim[pxID];
			dumy += ((float)(pxID / boxsz) + 0.5f) * dataim[pxID];
		}	
	
	if (dumSum - dumMin > IMIN) {
		dumSum -= dumMin;
		xvec[0] = dumx / dumSum;
		xvec[1] = dumy / dumSum;
	}
	else {
		dumMin = IMIN;
		xvec[0] = 0.5f * (float)boxsz;
		xvec[1] = 0.5f * (float)boxsz;
	}
	xvec[2] = 0.5f * (float)splineszz;
	xvec[NDIM] = dumSum;
	xvec[NDIM + 1] = max(dumMin, BGMIN);
	return;
}



/*!	@brief	set the maxJump
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	splineszz:			int, the size of the splines cube (axis-axis)
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out] maxJump:			(vnum) float, maxJump according to the xvec
*/
__device__ static void init_maxJump(int boxsz, int splineszz, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[2] = max(2.0f, 0.05f * (float)splineszz);
	maxJump[NDIM] = max(100.0f, xvec[NDIM]);
	maxJump[NDIM + 1] = max(20.0f, xvec[NDIM + 1]);
	return;
}



/*!	@brief	box-constrain the xvec
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	splineszz:			int, the size of the splines cube (axis-axis)
	@param[out] xvec:				(vnum) float, see definitions.h
*/
__device__ static void boxConstrain(int boxsz, int splineszz, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
	xvec[2] = min(max(xvec[2], 0.0f), (float)splineszz);
	xvec[NDIM] = max(xvec[NDIM], IMIN);
	xvec[NDIM + 1] = max(xvec[NDIM + 1], BGMIN);
	return;
}



/*!	@brief	Translating actural pixels idx to spline idx by mapping the emitter center to the origin of the spline coordinates;	
			Build the delta matrix that matches the arrangement of the cspline coefficient for dot multiplication
			This matrix is built as:
				[[dy^0*dx^0, ..., dy^0*dx^3], [dy^1*dx^0, ..., dy^1*dx^3], ..., [dy^3*dx^0, ..., dy^3*dx^3]]
			It matches the coeff a_xyz at an cspline_idx[idy, idx] as:
				[[a_00, a_01, ..., a_03], [a_10, a_11, ..., a_13], ..., [a_30, a_31, ..., a_33]]
	@param[in]	splineszx:			int, size of the splines square
	@param[in]	splineszz:			int, size of the splines cube (axial-axis)
	@param[in]	xvec:				(vnum) float, [x, y, z, I, b]
	@param[out]	offset_x:			int pointer to the x-offset translating the xvec from the box coordination to the spline coordination
	@param[out] offset_y:			int pointer to the y-offset translating the xvec from the box coordination to the spline coordination
	@param[out] offset_z:			int pointer to the z-offset translating the xvec from the box coordination to the spline coordination
	@param[out]	deltaf:				(64) float, the deltas for the corresponding cspline coefficients
	@param[out] ddeltaf_dx:			(64) float, the deltas for the derivatives over the x-axis of the corresponding cspline coefficients
	@param[out] ddeltaf_dy:			(64) float, the deltas for the derivatives over the y-axis of the corresponding cspline coefficients  	
	@param[out] ddeltaf_dz:			(64) float, the deltas for the derivatives over the z-axis of the corresponding cspline coefficients
*/
__device__ static void DeltaConstruct3D(int splineszx, int splineszz, float* xvec,
	int* offset_x, int* offset_y, int* offset_z, float *deltaf, float *ddeltaf_dx, float *ddeltaf_dy, float *ddeltaf_dz)
{
	int i = 0, j = 0, k = 0;
	float cz = 1.0f, cy = 1.0f, cx = 1.0f;
	float delta_z = 0.0f, delta_y = 0.0f, delta_x = 0.0f;

	*offset_x = splineszx / 2 - (int)floor(xvec[0]);
	*offset_y = splineszx / 2 - (int)floor(xvec[1]);
	*offset_z = splineszz / 2 - (int)floor(xvec[2]);
	delta_x = 1.0f - (xvec[0] - floor(xvec[0]));
	delta_y = 1.0f - (xvec[1] - floor(xvec[1]));
	delta_z = 1.0f - (xvec[2] - floor(xvec[2]));

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
	@param[in]	boxsz:				int, size of the PSF square
	@param[in] 	splineszx:			int, size of the splines cube (lateral-axis)
	@param[in]	splineszz:			int, size of the splines cube (axial-axis)
	@param[in]	coeff_PSF:			(splineszx * splineszx * splineszz* 64) float, cubic spline coefficients of the experimental PSF
	@param[in]	offset_x:			int, the x-offset translating the location from the box-coordination to the spline-coordination
	@param[in] 	offset_y:			int, the y-offset translating the location from the box-coordination to the spline-coordination
	@param[in] 	offset_z:			int, the z-offset translating the location from the box-coordination to the spline-coordination
	@param[in]	deltaf:				(64) float, the deltas for the corresponding cspline coefficients
	@param[in] 	ddeltaf_dx:			(64) float, the deltas for the derivatives over the x-axis of the corresponding cspline coefficients
	@param[in] 	ddeltaf_dy:			(64) float, the deltas for the derivatives over the y-axis of the corresponding cspline coefficients
	@param[in] 	ddeltaf_dz:			(64) float, the deltas for the derivatives over the z-axis of the corresponding cspline coefficients
	@param[in]	xvec:				(vnum) float, [x, y, z, I, b]
	@param[out]	psf:				float pointer to the psf value at the given pixel in each channel
	@param[out]	deriv1:				(vnum float, the derivatives of the model over the xvec 
*/
__device__ static void fconstruct_spline3D(int pxID, int boxsz, int splineszx, int splineszz, float* coeff_PSF, 
	int offset_x, int offset_y, int offset_z, float* deltaf, float* ddeltaf_dx, float* ddeltaf_dy, float* ddeltaf_dz,
	float* xvec, float* model, float* deriv1)
{
	const int spindx = min(max((pxID % boxsz) + offset_x, 0), splineszx - 1);
	const int spindy = min(max((pxID / boxsz) + offset_y, 0), splineszx - 1);
	const int spindz = min(max(splineszz / 2 + offset_z, 0), splineszz - 1);
	const int splineIdx = spindz * splineszx*splineszx + spindy * splineszx + spindx;
	
	float psf = 0.0f, dpsf_dloc[NDIM] = {0.0f};
	memset(deriv1, 0, (NDIM + 2) * sizeof(float));
	for (unsigned int i = 0; i < 64; i++) {
		psf += deltaf[i] * coeff_PSF[splineIdx * 64 + i];
		dpsf_dloc[0] -= ddeltaf_dx[i] * coeff_PSF[splineIdx * 64 + i];
		dpsf_dloc[1] -= ddeltaf_dy[i] * coeff_PSF[splineIdx * 64 + i];
		dpsf_dloc[2] -= ddeltaf_dz[i] * coeff_PSF[splineIdx * 64 + i];
	}
	*model = max(xvec[NDIM] * psf + xvec[NDIM + 1], FLT_EPSILON);
	for (unsigned int i = 0; i < NDIM; i++)
		deriv1[i] = xvec[NDIM] * dpsf_dloc[i];
	deriv1[NDIM] = psf;
	deriv1[NDIM + 1] = 1.0f;
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
			*Loss += 2.0f * (model - data - data * log(model) + data * log(data));
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
	@param[in]	opt:				optimization method. 0 for MLE and 1 for LSQ
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
	@param[in]	opt:				optimization method. 0 for MLE and 1 for LSQ
*/
__device__ static inline void accum_Hessian(int vnum, float data, float model, float* deriv1, float* Hessian, int opt)
{
	float dum = 0.0f;
	int i = 0, j = 0;
	if (opt == 0)
		dum = (data > 0.0f) ? (data / model / model) : 0.0f;
	else
		dum = 2.0f;

	for (i = 0; i < vnum; i++) {
		Hessian[i * vnum + i] += dum * deriv1[i] * deriv1[i];
		for (j = i + 1; j < vnum; j++)
			Hessian[i * vnum + j] += dum * deriv1[i] * deriv1[j];
	}
	return;
}



/*!	@brief	accumulate the up-triangle part of the Fisher Information Matrix (FIM) at a pixel (opt = MLE only)
	@param[in]	vnum:				int, the number of parameters in xvec
	@param[in] 	model:				float, the model at a pixel
	@param[in]	deriv1:				(vnum) float, the derivatives of the model over the xvec
	@param[out]	FIM:				(vnum * vnum) float, the accumulating FIM matrix
*/
__device__ static inline void accum_FIM(int vnum, float model, float* deriv1, float* FIM)
{
	int i = 0, j = 0;
	for (i = 0; i < vnum; i++)	
		for (j = i; j < vnum; j++)
			FIM[i * vnum + j] += deriv1[i] * deriv1[j] / model;
	return;
}



/*!	@brief	calculate the -log(likelihood), gradients, and Hessian
	@param[in]	boxsz:				int, size of the PSF square
	@param[in] 	splineszx:			int, size of the splines cube (lateral-axis)
	@param[in]	splineszz:			int, size of the splines cube (axial-axis)
	@param[in]	dataim:				pointer to the PSF data. 
	@param[in]	varim:				pointer to the camvar. 
	@param[in]	coeff_PSF:			(splineszx * splineszx * splineszz* 64) float, cubic spline coefficients of the experimental PSF
	@param[in]	xvec:				(vnum) float, see definitions.h
	@param[out] nllkh:				(1) float, pointer to the negative log likeliklihood
	@param[out]	grad:				(vnum) float, gradient
	@param[out]	Hessian:			(vnum * vnum) float, Hessian
	@param[in]	opt:				optimization method. 0 for MLE and 1 for LSQ
*/
__device__ static void get_lgH(int boxsz, int splineszx, int splineszz, float* dataim, float* varim, float* coeff_PSF, float* xvec,
	float* nllkh, float* grad, float* Hessian, int opt)
{
	const int vnum = NDIM + 2;
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float data = 0.0f, model = 0.0f, Loss = 0.0f, deriv1[VNUM] = {0.0f};
	int pxID = 0, i = 0, j = 0;
	
	memset(grad, 0, vnum * sizeof(float));
	memset(Hessian, 0, vnum * vnum * sizeof(float));
	DeltaConstruct3D(splineszx, splineszz, xvec, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);
	for (pxID = 0, Loss = 0.0f; pxID < boxsz * boxsz; pxID++) {
		data = dataim[pxID];
		fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, coeff_PSF, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, xvec, &model, deriv1);
		if (varim) {
			data += varim[pxID];
			model += varim[pxID];
		}			
		accum_loss(data, model, &Loss, opt);
		accum_grad(vnum, data, model, deriv1, grad, opt);
		accum_Hessian(vnum, data, model, deriv1, Hessian, opt);
	}
	for (i = 0; i < vnum; i++)
		for (j = i + 1; j < vnum; j++)
			Hessian[j * vnum + i] = Hessian[i * vnum + j];
	*nllkh = Loss;
	
	return;
}



/*!	@brief	calculate the FIM for opt == MLE
	@param[in]	boxsz:				int, size of the PSF square
	@param[in] 	splineszx:			int, size of the splines cube (lateral-axis)
	@param[in]	splineszz:			int, size of the splines cube (axial-axis)
	@param[in]	varim:				pointer to the camvar. 
	@param[in]	coeff_PSF:			(splineszx * splineszx * splineszz* 64) float, cubic spline coefficients of the experimental PSF
	@param[in]	xvec:				(vnum) float, see definitions.h
	@param[out]	FIM:				(vnum * vnum) float, the Fisher Information Matrix	
*/
__device__ static void get_FIM(int boxsz, int splineszx, int splineszz, float* varim, float* coeff_PSF, float* xvec, float* FIM)
{
	const int vnum = NDIM + 2;
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float deltaf[64] = {0.0f}, ddeltaf_dx[64] = {0.0f}, ddeltaf_dy[64] = {0.0f}, ddeltaf_dz[64] = {0.0f};
	float model = 0.0f, deriv1[VNUM] = {0.0f};
	
	memset(FIM, 0, vnum * vnum * sizeof(float));
	DeltaConstruct3D(splineszx, splineszz, xvec, &offset_x, &offset_y, &offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz);
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		fconstruct_spline3D(pxID, boxsz, splineszx, splineszz, coeff_PSF, offset_x, offset_y, offset_z, deltaf, ddeltaf_dx, ddeltaf_dy, ddeltaf_dz, xvec, &model, deriv1);
		if (varim) 
			model += varim[pxID];
				
		accum_FIM(vnum, model, deriv1, FIM);
	}
	for (unsigned int i = 0; i < vnum; i++)
		for (unsigned int j = i + 1; j < vnum; j++)
			FIM[j * vnum + i] = FIM[i * vnum + j];
	
	return;
}

#endif