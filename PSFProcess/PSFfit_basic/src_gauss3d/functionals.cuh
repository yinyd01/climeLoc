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
	@param[in]	zrange:				int, fitting range (z-axis)
	@param[in]	dataim:				pointer to the PSF data. 
	@param[out]	xvec:				(VNUM) float, see definitions.h.		
*/
__device__ static void init_xvec(int boxsz, int zrange, float* dataim, float* xvec)
{
	float dumx = 0.0f, dumy = 0.0f, dumSum = 0.0f, dumMin = 0.0f;
	int pxID = 0;
	
	// Get the photon numbers and bkg for the dataim
	dumSum = get_psfsum(boxsz, dataim);
	dumMin = get_psfmin(boxsz, dataim);
	
	// locate the x and y at the center of mass of the dataim
	// calculate the variance in the dataim
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
	xvec[2] = 0.5f * (float)zrange;
	xvec[NDIM] = dumSum;
	xvec[NDIM + 1] = max(dumMin, BGMIN);
	return;
}



/*!	@brief	set the maxJump
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	zrange:				int, fitting range (axial-axis)
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out] maxJump:			(vnum) float, maxJump according to the xvec
*/
__device__ static void init_maxJump(int boxsz, int zrange, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[2] = max(2.0f, 0.05f * (float)zrange);
	maxJump[NDIM] = max(100.0f, xvec[NDIM]);
	maxJump[NDIM + 1] = max(20.0f, xvec[NDIM + 1]);
	return;
}



/*!	@brief	box-constrain the xvec
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	zrange:				int, fitting range (axial-axis)
	@param[out] xvec:				(vnum) float, see definitions.h
*/
__device__ static void boxConstrain(int boxsz, int zrange, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
	xvec[2] = min(max(xvec[2], 0.0f), (float)zrange);
	xvec[NDIM] = max(xvec[NDIM], IMIN);
	xvec[NDIM + 1] = max(xvec[NDIM + 1], BGMIN);
	return;
}



/*!	@brief	Construct the model value and deriv1 values at pxID.
	@param[in]	pxID:				int, the index of the working pixel
	@param[in]	boxsz:				int, the size of the PSF square
	@param[in]	astigs:				(9) float, [PSFsigmax0, PSFsigmay0, shiftx, shifty, dof, Ax, Bx, Ay, By] the astigmatic parameters			
	@param[in]	xvec:				(vnum) float, [x, y, z, I, b]
	@param[out]	model:				(1) float, pointer to the psf value at the given pixel
	@param[out]	deriv1:				(vnum) float, the derivatives of the model over the xvec 	 
*/
__device__ static void fconstruct_gauss3D(int pxID, int boxsz, float* astigs, float* xvec, float* model, float* deriv1)
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

	const float zx = xvec[2] - shiftx;
	const float zy = xvec[2] - shifty;

	const float alpha_x = sqrt(max(1.0f + zx*zx / d/d + Ax * zx*zx*zx / d/d/d + Bx * zx*zx*zx*zx / d/d/d/d, 1.0f));
	const float alpha_y = sqrt(max(1.0f + zy*zy / d/d + Ay * zy*zy*zy / d/d/d + By * zy*zy*zy*zy / d/d/d/d, 1.0f));
	const float sx = PSFsigmax0 * alpha_x;
	const float sy = PSFsigmay0 * alpha_y;

	const float delta_xu = ((float)(pxID % boxsz) + 1.0f - xvec[0]) / sx;
	const float delta_yu = ((float)(pxID / boxsz) + 1.0f - xvec[1]) / sy;
	const float delta_xl = ((float)(pxID % boxsz) - xvec[0]) / sx;
	const float delta_yl = ((float)(pxID / boxsz) - xvec[1]) / sy;

	const float PSFx = 0.5f * (erf(delta_xu / SQRTTWO) - erf(delta_xl / SQRTTWO));
	const float PSFy = 0.5f * (erf(delta_yu / SQRTTWO) - erf(delta_yl / SQRTTWO));
	const float psf = PSFx * PSFy;

	const float dalphax_dz = 2.0f * zx / d/d + 3.0f * Ax * zx*zx / d/d/d + 4.0f * Bx * zx*zx*zx / d/d/d/d;
	const float dalphay_dz = 2.0f * zy / d/d + 3.0f * Ay * zy*zy / d/d/d + 4.0f * By * zy*zy*zy / d/d/d/d;
	const float dsx_dz = 0.5f * PSFsigmax0 / alpha_x * dalphax_dz;
	const float dsy_dz = 0.5f * PSFsigmay0 / alpha_y * dalphay_dz;

	const float dPSFx_dx = -invSQRTTWOPI / sx * (exp(-0.5f * delta_xu * delta_xu) - exp(-0.5f * delta_xl * delta_xl));
	const float dPSFy_dy = -invSQRTTWOPI / sy * (exp(-0.5f * delta_yu * delta_yu) - exp(-0.5f * delta_yl * delta_yl));
	const float dPSFx_dsx = -invSQRTTWOPI / sx * (delta_xu * exp(-0.5f * delta_xu * delta_xu) - delta_xl * exp(-0.5f * delta_xl * delta_xl));
	const float dPSFy_dsy = -invSQRTTWOPI / sy * (delta_yu * exp(-0.5f * delta_yu * delta_yu) - delta_yl * exp(-0.5f * delta_yl * delta_yl));

	float dpsf_dloc[NDIM] = {0.0f};
	dpsf_dloc[0] = dPSFx_dx * PSFy;
	dpsf_dloc[1] = PSFx * dPSFy_dy;
	dpsf_dloc[2] = dPSFx_dsx * dsx_dz * PSFy + PSFx * dPSFy_dsy * dsy_dz;

	*model = max(xvec[NDIM] * psf + xvec[NDIM + 1], BGMIN);
	memset(deriv1, 0, (NDIM + 2) * sizeof(float));
	deriv1[0] = xvec[NDIM] * dpsf_dloc[0];
	deriv1[1] = xvec[NDIM] * dpsf_dloc[1];
	deriv1[2] = xvec[NDIM] * dpsf_dloc[2];
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
	@param[in]	zrange:				int, fitting range (axial-axis)
	@param[in]	dataim:				pointer to the PSF data. 
	@param[in]	varim:				pointer to the camvar. 
	@param[in]	astigs:				(9) float, astigmatic parameters
	@param[in]	xvec:				(vnum) float, see definitions.h
	@param[out] nllkh:				(1) float, pointer to the negative log likeliklihood
	@param[out]	grad:				(vnum) float, gradient
	@param[out]	Hessian:			(vnum * vnum) float, Hessian
	@param[in]	opt:				optimization method. 0 for MLE and 1 for LSQ
*/
__device__ static void get_lgH(int boxsz, float* dataim, float* varim, float* astigs, float* xvec,
	float* nllkh, float* grad, float* Hessian, int opt)
{
	const int vnum = NDIM + 2;
	float data = 0.0f, model = 0.0f, Loss = 0.0f, deriv1[VNUM] = {0.0f};
	int pxID = 0, i = 0, j = 0;
	
	memset(grad, 0, vnum * sizeof(float));
	memset(Hessian, 0, vnum * vnum * sizeof(float));
	for (pxID = 0, Loss = 0.0f; pxID < boxsz * boxsz; pxID++) {
		data = dataim[pxID];
		fconstruct_gauss3D(pxID, boxsz, astigs, xvec, &model, deriv1);
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
	@param[in]	varim:				pointer to the camvar. 
	@param[in]	astigs:				(9) float, astigmatic parameters
	@param[in]	xvec:				(vnum) float, see definitions.h
	@param[out]	FIM:				(vnum * vnum) float, the Fisher Information Matrix	
*/
__device__ static void get_FIM(int boxsz, float* varim, float* astigs, float* xvec, float* FIM)
{
	const int vnum = NDIM + 2;
	float model = 0.0f, deriv1[VNUM] = {0.0f};
	
	memset(FIM, 0, vnum * vnum * sizeof(float));
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		fconstruct_gauss3D(pxID, boxsz, astigs, xvec, &model, deriv1);
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