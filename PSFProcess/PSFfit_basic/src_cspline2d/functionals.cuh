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
	@param[in]	dataim:				pointer to the PSF data. 
	@param[out]	xvec:				(VNUM) float, see definitions.h.		
*/
__device__ static void init_xvec(int boxsz, float* dataim, float* xvec)
{
	float dumx = 0.0f, dumy = 0.0f;
	float dumSum = get_psfsum(boxsz, dataim);
	float dumMin = get_psfmin(boxsz, dataim);
	int pxID = 0;
	
	// locate the x and y at the center of mass
	if (dumSum - dumMin > IMIN) {
		for (pxID = 0, dumx = 0.0f, dumy = 0.0f; pxID < boxsz * boxsz; pxID++)
			if (dataim[pxID] > 0) {
				dumx += ((float)(pxID % boxsz) + 0.5f) * dataim[pxID];
				dumy += ((float)(pxID / boxsz) + 0.5f) * dataim[pxID];
			}
		xvec[0] = dumx / (dumSum - dumMin);
		xvec[1] = dumy / (dumSum - dumMin);
	}
	else {
		xvec[0] = 0.5f * (float)boxsz;
		xvec[1] = 0.5f * (float)boxsz;
	}

	// photon and background
	xvec[NDIM] = max(dumSum - dumMin, IMIN);
	xvec[NDIM + 1] = max(dumMin, BGMIN);
	return;
}



/*!	@brief	set the maxJump
	@param[in]	boxsz:				int, size of the PSF square
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out] maxJump:			(vnum) float, maxJump according to the xvec
*/
__device__ static void init_maxJump(int boxsz, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[NDIM] = max(100.0f, xvec[NDIM]);
	maxJump[NDIM + 1] = max(20.0f, xvec[NDIM + 1]);
	return;
}



/*!	@brief	box-constrain the xvec
	@param[in]	boxsz:				int, size of the PSF square
	@param[out] xvec:				(vnum) float, see definitions.h
*/
__device__ static void boxConstrain(int boxsz, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
	xvec[NDIM] = max(xvec[NDIM], IMIN);
	xvec[NDIM + 1] = max(xvec[NDIM + 1], BGMIN);
	return;
}



/*!	@brief	Translating the actural pixels idx to spline idx by mapping the emitter center to the origin of the spline coordinates;	
			Build the delta matrix that matches the arrangement of the cspline coefficient for dot multiplication
			This matrix is built as:
				[[dy^0*dx^0, ..., dy^0*dx^3], [dy^1*dx^0, ..., dy^1*dx^3], ..., [dy^3*dx^0, ..., dy^3*dx^3]]
			It matches the coeff a_xyz at an cspline_idx[idy, idx] as:
				[[a_00, a_01, ..., a_03], [a_10, a_11, ..., a_13], ..., [a_30, a_31, ..., a_33]]
	@param[in]	splinesz:			int, size of the splines square
	@param[in]	xvec:				(vnum) float, [x, y, I, b]
	@param[out]	offset_x:			int pointer to the x-offset translating the loc from the box coordination to the spline coordination
	@param[out] offset_y:			int pointer to the y-offset translating the loc from the box coordination to the spline coordination
	@param[out]	deltaf:				(16) float, the deltas for the corresponding cspline coefficients
	@param[out] ddeltaf_dx:			(16) float, the deltas for the derivatives over the x-axis of the corresponding cspline coefficients
	@param[out] ddeltaf_dy:			(16) float, the deltas for the derivatives over the y-axis of the corresponding cspline coefficients  			
*/
__device__ static void DeltaConstruct2D(int splinesz, float* xvec, int* offset_x, int* offset_y, float* deltaf, float* ddeltaf_dx, float* ddeltaf_dy)
{
	int i = 0, j = 0;
	float delta_x = 0.0f, delta_y = 0.0f, cx = 1.0f, cy = 1.0f;

	*offset_x = splinesz / 2 - (int)floor(xvec[0]);
	*offset_y = splinesz / 2 - (int)floor(xvec[1]);
	delta_x = 1.0f - (xvec[0] - floor(xvec[0]));
	delta_y = 1.0f - (xvec[1] - floor(xvec[1]));

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
	@param[in]	xvec:				(vnum) float, [x, y, I, b]
	@param[out]	model:				float pointer to the psf value at the given pixel in each channel
	@param[out]	deriv1:				(vnum) float, the derivatives of the model over the xvec 	 
*/
__device__ static void fconstruct_spline2D(int pxID, int boxsz, int splinesz, float* coeff_PSF, 
	int offset_x, int offset_y, float* deltaf, float* ddeltaf_dx, float* ddeltaf_dy, float* xvec, float* model, float* deriv1)
{
	const int spindx = min(max((pxID % boxsz) + offset_x, 0), splinesz - 1);
	const int spindy = min(max((pxID / boxsz) + offset_y, 0), splinesz - 1);
	const int splineIdx = spindy * splinesz + spindx;
	
	float psf = 0.0f, dpsf_dloc[NDIM] = {0.0f};
	
	memset(deriv1, 0, (NDIM + 2) * sizeof(float));
	for (unsigned int i = 0; i < 16; i++) {
		psf += deltaf[i] * coeff_PSF[splineIdx * 16 + i];
		dpsf_dloc[0] -= ddeltaf_dx[i] * coeff_PSF[splineIdx * 16 + i];
		dpsf_dloc[1] -= ddeltaf_dy[i] * coeff_PSF[splineIdx * 16 + i];
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
	@param[in] 	splinesz:			int, size of the splines square
	@param[in]	dataim:				pointer to the PSF data. 
	@param[in]	varim:				pointer to the camvar. 
	@param[in]	coeff_PSF:			(splinesz * splinesz * 16) float, the cubic spline coefficients of the experimental PSF for each channel
	@param[in]	xvec:				(vnum) float, see definitions.h
	@param[out] nllkh:				(1) float, pointer to the negative log likeliklihood
	@param[out]	grad:				(vnum) float, gradient
	@param[out]	Hessian:			(vnum * vnum) float, Hessian
	@param[in]	opt:				optimization method. 0 for MLE and 1 for LSQ
*/
__device__ static void get_lgH(int boxsz, int splinesz, float* dataim, float* varim, float* coeff_PSF, float* xvec,
	float* nllkh, float* grad, float* Hessian, int opt)
{
	const int vnum = NDIM + 2;
	int offset_x = 0, offset_y = 0;
	float deltaf[16] = {0.0f}, ddeltaf_dx[16] = {0.0f}, ddeltaf_dy[16] = {0.0f};
	float data = 0.0f, model = 0.0f, Loss = 0.0f, deriv1[VNUM] = {0.0f};
	int pxID = 0, i = 0, j = 0;
	
	memset(grad, 0, vnum * sizeof(float));
	memset(Hessian, 0, vnum * vnum * sizeof(float));
	DeltaConstruct2D(splinesz, xvec, &offset_x, &offset_y, deltaf, ddeltaf_dx, ddeltaf_dy);
	for (pxID = 0, Loss = 0.0f; pxID < boxsz * boxsz; pxID++) {
		data = dataim[pxID];
		fconstruct_spline2D(pxID, boxsz, splinesz, coeff_PSF, offset_x, offset_y, deltaf, ddeltaf_dx, ddeltaf_dy, xvec, &model, deriv1);
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
	@param[in] 	splinesz:			int, size of the splines square
	@param[in]	varim:				pointer to the camvar. 
	@param[in]	coeff_PSF:			(splinesz * splinesz * 16) float, the cubic spline coefficients of the experimental PSF for each channel
	@param[in]	xvec:				(vnum) float, see definitions.h
	@param[out]	FIM:				(vnum * vnum) float, the Fisher Information Matrix	
*/
__device__ static void get_FIM(int boxsz, int splinesz, float* varim, float* coeff_PSF, float* xvec, float* FIM)
{
	const int vnum = NDIM + 2;
	int offset_x = 0, offset_y = 0;
	float deltaf[16] = {0.0f}, ddeltaf_dx[16] = {0.0f}, ddeltaf_dy[16] = {0.0f};
	float model = 0.0f, deriv1[VNUM] = {0.0f};
	
	memset(FIM, 0, vnum * vnum * sizeof(float));
	DeltaConstruct2D(splinesz, xvec, &offset_x, &offset_y, deltaf, ddeltaf_dx, ddeltaf_dy);
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		fconstruct_spline2D(pxID, boxsz, splinesz, coeff_PSF, offset_x, offset_y, deltaf, ddeltaf_dx, ddeltaf_dy, xvec, &model, deriv1);
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