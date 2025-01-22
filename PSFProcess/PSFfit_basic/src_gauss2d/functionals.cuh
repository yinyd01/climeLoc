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
	@param[in]	fixs:				int, 1 if sigmax and sigmay are fixed at a given value, 0 if they are free to fit	
	@param[in]	boxsz:				int, size of the PSF square. 
	@param[in]	dataim:				pointer to the PSF data. 
	@param[out]	xvec:				(VNUM) float, see definitions.h.		
*/
__device__ static void init_xvec(int fixs, int boxsz, float* dataim, float* xvec)
{
	float dumx = 0.0f, dumy = 0.0f, dumxx = 0.0f, dumyy = 0.0f;
	float dumSum = get_psfsum(boxsz, dataim);
	float dumMin = get_psfmin(boxsz, dataim);
	int pxID = 0;
	
	// locate the x and y at the center of mass, and initialize the PSFsigmax
	if (dumSum - dumMin > IMIN) {
		for (pxID = 0, dumx = 0.0f, dumy = 0.0f, dumxx = 0.0f, dumyy = 0.0f; pxID < boxsz * boxsz; pxID++)
			if (dataim[pxID] > 0) {
				dumx += ((float)(pxID % boxsz) + 0.5f) * dataim[pxID];
				dumy += ((float)(pxID / boxsz) + 0.5f) * dataim[pxID];
				if (fixs == 0) {
					dumxx += ((float)(pxID % boxsz) + 0.5f) * ((float)(pxID % boxsz) + 0.5f) * dataim[pxID];
					dumyy += ((float)(pxID / boxsz) + 0.5f) * ((float)(pxID / boxsz) + 0.5f) * dataim[pxID];
				}
			}
		xvec[0] = dumx / (dumSum - dumMin);
		xvec[1] = dumy / (dumSum - dumMin);
		if (fixs == 0) {
			xvec[NDIM + 2] = sqrt(dumxx / (dumSum - dumMin) - xvec[0] * xvec[0]);
			xvec[NDIM + 3] = sqrt(dumyy / (dumSum - dumMin) - xvec[1] * xvec[1]);
		}
	}
	else {
		xvec[0] = 0.5f * (float)boxsz;
		xvec[1] = 0.5f * (float)boxsz;
		if (fixs == 0) {
			xvec[NDIM + 2] = 0.2f * (float)boxsz;
			xvec[NDIM + 3] = 0.2f * (float)boxsz;
		}
	}

	// photon and background
	xvec[NDIM] = max(dumSum - dumMin, IMIN);
	xvec[NDIM + 1] = max(dumMin, BGMIN);
	return;
}



/*!	@brief	set the maxJump
	@param[in]	fixs:				int, 1 if sigmax and sigmay are fixed at a given value, 0 if they are free to fit
	@param[in]	boxsz:				int, size of the PSF square
	@param[in] 	xvec:				(vnum) float, see definitions.h
	@param[out] maxJump:			(vnum) float, maxJump according to the xvec
*/
__device__ static void init_maxJump(int fixs, int boxsz, float* xvec, float* maxJump)
{
	maxJump[0] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[1] = max(1.0f, 0.1f * (float)boxsz);
	maxJump[NDIM] = max(100.0f, xvec[NDIM]);
	maxJump[NDIM + 1] = max(20.0f, xvec[NDIM + 1]);
	if (fixs == 0) {
		maxJump[NDIM + 2] = max(0.3f, 0.05f * (float)boxsz);
		maxJump[NDIM + 3] = max(0.3f, 0.05f * (float)boxsz);
	}
	return;
}



/*!	@brief	box-constrain the xvec
	@param[in]	fixs:				int, 1 if sigmax and sigmay are fixed at a given value, 0 if they are free to fit
	@param[in]	PSFsigmax:			float, sigma of the Gaussian PSF model (x-axis)
	@param[in] 	PSFsigmay:			float, sigma of the Gaussian PSF model (y-axis)
	@param[in]	boxsz:				int, size of the PSF square
	@param[out] xvec:				(vnum) float, see definitions.h
*/
__device__ static void boxConstrain(int fixs, int boxsz, float PSFsigmax, float PSFsigmay, float* xvec)
{
	xvec[0] = min(max(xvec[0], 0.0f), (float)boxsz);
	xvec[1] = min(max(xvec[1], 0.0f), (float)boxsz);
	xvec[NDIM] = max(xvec[NDIM], IMIN);
	xvec[NDIM + 1] = max(xvec[NDIM + 1], BGMIN);
	if (fixs == 0) {
		xvec[NDIM + 2] = min(max(xvec[NDIM + 2], 0.5f * PSFsigmax), 10.0f * PSFsigmax);
		xvec[NDIM + 3] = min(max(xvec[NDIM + 3], 0.5f * PSFsigmay), 10.0f * PSFsigmay);
	}
	return;
}



/*!	@brief	Construct the model value and deriv1 values at pxID.
	@param[in]	fixs:				int, 1 if sigmax and sigmay are fixed at a given value, 0 if they are free to fit
	@param[in]	pxID:				int, the index of the working pixel
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	PSFsigmax:			float, sigma of the Gaussian PSF model (x-axis)
	@param[in] 	PSFsigmay:			float, sigma of the Gaussian PSF model (y-axis)
	@param[in]	xvec:				(VNUM) float, see definitions.h
	@param[out]	model:				float pointer to the model value at the given pixel
	@param[out]	deriv1:				(VNUM) float, derivatives of the model over each of the parameter in xvec 	 
*/
__device__ static void fconstruct_gauss2D(int fixs, int pxID, int boxsz, float PSFsigmax, float PSFsigmay, float* xvec, float* model, float* deriv1)
{
	const int vnum = (fixs == 1) ? NDIM + 2 : NDIM + 4;
	float dPSFx_dx = 0.0f, dPSFy_dy = 0.0f, dPSFx_dsx = 0.0f, dPSFy_dsy = 0.0f;
	
	const float delta_xu = ((float)(pxID % boxsz) + 1.0f - xvec[0]) / PSFsigmax;
	const float delta_yu = ((float)(pxID / boxsz) + 1.0f - xvec[1]) / PSFsigmay;
	const float delta_xl = ((float)(pxID % boxsz) - xvec[0]) / PSFsigmax;
	const float delta_yl = ((float)(pxID / boxsz) - xvec[1]) / PSFsigmay;

	const float PSFx = 0.5f * (erf(delta_xu / SQRTTWO) - erf(delta_xl / SQRTTWO));
	const float PSFy = 0.5f * (erf(delta_yu / SQRTTWO) - erf(delta_yl / SQRTTWO));
	
	*model = max(xvec[NDIM] * PSFx * PSFy + xvec[NDIM + 1], BGMIN);

	memset(deriv1, 0, vnum * sizeof(float));
	dPSFx_dx = -invSQRTTWOPI / PSFsigmax * (exp(-0.5f * delta_xu * delta_xu) - exp(-0.5f * delta_xl * delta_xl));
	dPSFy_dy = -invSQRTTWOPI / PSFsigmay * (exp(-0.5f * delta_yu * delta_yu) - exp(-0.5f * delta_yl * delta_yl));
	deriv1[0] = xvec[NDIM] * dPSFx_dx * PSFy;
	deriv1[1] = xvec[NDIM] * PSFx * dPSFy_dy;
	deriv1[NDIM] = PSFx * PSFy;
	deriv1[NDIM + 1] = 1.0f;

	if (fixs == 0) {
		dPSFx_dsx = -invSQRTTWOPI / PSFsigmax * (delta_xu * exp(-0.5f * delta_xu * delta_xu) - delta_xl * exp(-0.5f * delta_xl * delta_xl));
		dPSFy_dsy = -invSQRTTWOPI / PSFsigmay * (delta_yu * exp(-0.5f * delta_yu * delta_yu) - delta_yl * exp(-0.5f * delta_yl * delta_yl));
		deriv1[NDIM + 2] = xvec[NDIM] * dPSFx_dsx * PSFy;
		deriv1[NDIM + 3] = xvec[NDIM] * PSFx * dPSFy_dsy;
	} 
	
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
	@param[in]	fixs:				int, 1 if sigmax and sigmay are fixed at a given value, 0 if they are free to fit
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	dataim:				pointer to the PSF data. 
	@param[in]	varim:				pointer to the camvar. 
	@param[in]	PSFsigmax:			float, sigma of the Gaussian PSF model (x-axis)
	@param[in] 	PSFsigmay:			float, sigma of the Gaussian PSF model (y-axis)
	@param[in]	xvec:				(vnum) float, see definitions.h
	@param[out] nllkh:				(1) float, pointer to the negative log likeliklihood
	@param[out]	grad:				(vnum) float, gradient
	@param[out]	Hessian:			(vnum * vnum) float, Hessian
	@param[in]	opt:				optimization method. 0 for MLE and 1 for LSQ
*/
__device__ static void get_lgH(int fixs, int boxsz, float* dataim, float* varim, float PSFsigmax, float PSFsigmay, float* xvec,
	float* nllkh, float* grad, float* Hessian, int opt)
{
	const int vnum = (fixs == 1) ? NDIM + 2 : NDIM + 4;
	float data = 0.0f, model = 0.0f, Loss = 0.0f, deriv1[VMAX] = {0.0f};
	int pxID = 0, i = 0, j = 0;
	
	memset(grad, 0, vnum * sizeof(float));
	memset(Hessian, 0, vnum * vnum * sizeof(float));
	for (pxID = 0, Loss = 0.0f; pxID < boxsz * boxsz; pxID++) {
		data = dataim[pxID];
		if (fixs == 1)
			fconstruct_gauss2D(fixs, pxID, boxsz, PSFsigmax, PSFsigmay, xvec, &model, deriv1);
		else
			fconstruct_gauss2D(fixs, pxID, boxsz, xvec[NDIM + 2], xvec[NDIM + 3], xvec, &model, deriv1);
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
	@param[in]	fixs:				int, 1 if sigmax and sigmay are fixed at a given value, 0 if they are free to fit
	@param[in]	boxsz:				int, size of the PSF square
	@param[in]	varim:				pointer to the camvar. 
	@param[in]	PSFsigmax:			float, sigma of the Gaussian PSF model (x-axis)
	@param[in] 	PSFsigmay:			float, sigma of the Gaussian PSF model (y-axis)
	@param[in]	xvec:				(vnum) float, see definitions.h
	@param[out]	FIM:				(vnum * vnum) float, the Fisher Information Matrix	
*/
__device__ static void get_FIM(int fixs, int boxsz, float* varim, float PSFsigmax, float PSFsigmay, float* xvec, float* FIM)
{
	const int vnum = (fixs == 1) ? NDIM + 2 : NDIM + 4;
	float model = 0.0f, deriv1[VMAX] = {0.0f};
	
	memset(FIM, 0, vnum * vnum * sizeof(float));
	for (unsigned int pxID = 0; pxID < boxsz * boxsz; pxID++) {
		if (fixs == 1)
			fconstruct_gauss2D(fixs, pxID, boxsz, PSFsigmax, PSFsigmay, xvec, &model, deriv1);
		else
			fconstruct_gauss2D(fixs, pxID, boxsz, xvec[NDIM + 2], xvec[NDIM + 3], xvec, &model, deriv1);
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