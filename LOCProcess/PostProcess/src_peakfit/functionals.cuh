#pragma once
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "definitions.h"

#ifndef FUNCTIONALS_CUH
#define FUNCTIONALS_CUH


/*! @brief  initialize the parameters of a 2D gaussian profile for 2D gaussian fit
    @param[in]  imHeight:   int, height of the image.
    @param[in]  imWidth:    int, width of the image.        
    @param[in]  imdata:     float pointer to the image data for fit.
    @param[out] xvec:       float pointer to the gaussian parameters [xc, yc, sx, sy, rho, intensity, bkg]. 
*/
__device__ static void _INIT_PROFILE(int imHeight, int imWidth, float* imdata, float* xvec)
{
    float dumI = 0.0f, dumx = 0.0f, dumy = 0.0f, dumb = 1e10f, dumsx = 0.0f, dumsy = 0.0f, dumrho = 0.0f;
    unsigned int i = 0;

    // intensity and background
    for (i = 0, dumI = 0.0f, dumb = 1e10f; i < imHeight * imWidth; i++) {
        dumI += imdata[i];
        dumb = (dumb < imdata[i]) ? dumb : imdata[i];
    }
    dumI -= dumb * imHeight * imWidth;
    if (dumI == 0.0f) {
        xvec[0] = 0.5f * (float)imWidth;
        xvec[1] = 0.5f * (float)imHeight;
        xvec[2] = 0.25f * (float)imWidth;
        xvec[3] = 0.25f * (float)imHeight;
        xvec[4] = 0.0f;
        xvec[5] = 0.0f;
        xvec[6] = dumb;
        return;
    }

    // center
    for (i = 0, dumx = 0.0f, dumy = 0.0f; i < imHeight * imWidth; i++) {
        dumx += ((float)(i % imWidth) + 0.5f) * (imdata[i] - dumb);
        dumy += ((float)(i / imWidth) + 0.5f) * (imdata[i] - dumb);
    }
    dumx /= dumI;
    dumy /= dumI;
     
    // variance
    for (i = 0, dumsx = 0.0f, dumsy = 0.0f, dumrho = 0.0f; i < imHeight * imWidth; i++) {
        dumsx += ((float)(i % imWidth) + 0.5f - dumx) * ((float)(i % imWidth) + 0.5f - dumx) * (imdata[i] - dumb);
        dumsy += ((float)(i / imWidth) + 0.5f - dumy) * ((float)(i / imWidth) + 0.5f - dumy) * (imdata[i] - dumb);
        dumrho += ((float)(i % imWidth) + 0.5f - dumx) * ((float)(i / imWidth) + 0.5f - dumy) * (imdata[i] - dumb);
    }
    dumsx /= dumI;
    dumsy /= dumI;
    dumrho /= dumI;
    dumsx = sqrt(dumsx);
    dumsy = sqrt(dumsy);
    dumrho /= dumsx * dumsy;
    
    xvec[0] = dumx;
    xvec[1] = dumy;
    xvec[2] = dumsx;
    xvec[3] = dumsy;
    xvec[4] = dumrho;
    xvec[5] = dumI;
    xvec[6] = dumb;
    return;
}



/*! @brief construct the model value and deriv1 values at pxID.
	@param[in]  pxID:       int, the index of the running pixel
	@param[in]  imHeight:   int, the height of the image
    @param[in]  imWidth:    int, the width of the image
    @param[in]  xvec:       float pointer to the gaussian parameters [xc, yc, sx, sy, rho, intensity, bkg]			
	@param[out] model: 		float pointer to the value of the Gaussian model at pxID.
	@param[out] deriv1: 	pointer to the derivatives of the Gaussian parameters at pxID.
*/
__device__ static void _fconstruct(int pxID, int imHeight, int imWidth, float* xvec, float* model, float* deriv1)
{
    float u = 0.0f, v = 0.0f, p = 0.0f, q = 0.0f, mu = 0.0f;

    xvec[4] = max(min(xvec[4], 0.99f), 0.0f);
    u = ((float)(pxID % imWidth) + 0.5f - xvec[0]) / xvec[2];
    v = ((float)(pxID / imWidth) + 0.5f - xvec[1]) / xvec[3];
    p = -0.5f / (1.0f - xvec[4] * xvec[4]) * (u * u - 2 * xvec[4] * u * v + v * v);
    q = 1.0f / xvec[2] / xvec[3] / sqrt(1.0f - xvec[4] * xvec[4]);
    
    mu = 0.5f / PI * q * exp(p);
    *model = max(xvec[5] * mu + xvec[6], 0.0f);

    if (deriv1) {
		deriv1[0] = xvec[5] * mu / xvec[2] * (u - xvec[4] * v) / (1.0f - xvec[4] * xvec[4]);
		deriv1[1] = xvec[5] * mu / xvec[3] * (v - xvec[4] * u) / (1.0f - xvec[4] * xvec[4]);
		deriv1[2] = xvec[5] * mu / xvec[2] * ((u * u - xvec[4] * u * v) / (1.0f - xvec[4] * xvec[4]) - 1.0f);
		deriv1[3] = xvec[5] * mu / xvec[3] * ((v * v - xvec[4] * u * v) / (1.0f - xvec[4] * xvec[4]) - 1.0f);
		deriv1[4] = xvec[5] * mu * (2 * xvec[4] * p + u * v + xvec[4]) / (1.0f - xvec[4] * xvec[4]);
        deriv1[5] = mu;
		deriv1[6] = 1.0f;
	}
    return;
}

#endif