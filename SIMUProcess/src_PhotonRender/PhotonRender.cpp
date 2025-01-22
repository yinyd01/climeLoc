#include <string.h>
#include <math.h>
#include "DLL_Macros.h"
#include "definitions.h"

/*! @brief blur the emitters via the cspline calibrated PSF
	@param[in]	splineszx:		int, size of the cspline sqaure/cube (lateral-axis)
	@param[in]	splineszz:		int, size of the cspline cube (axial-axis)
	@param[in]	coeff_PSF:		(splineszx * splineszx * splineszz * 64) or (splineszx * splineszx * 16) float, the cubic spline coefficient of the PSF
	@param[in]	nspots:			int, number of coordinate			
	@param[in]	locs:			(nspots * ndim) float, the localizations of the emitters
	@param[in]	phot:			(nspots) float, the phot number of each emitter
	@param[in]	imHeight:		int, the height of the image to render
	@param[in]	imWidth:		int, the width of the image to renger
	@param[out]	N_photon:		(ImHeight * ImWidth) float array, the pointer to the image to render
*/


CDLL_EXPORT void PhotRender_2d(int splineszx, float* coeff_PSF, int nspots, float* locs, float* phot, 
	int imHeight, int imWidth, float* N_photon)
{
	const int ndim = 2;
	int offset_x = 0, offset_y = 0;
	float _x = 0.0f, _y = 0.0f;
	float delta_x = 0.0f, delta_y = 0.0f, cx = 1.0f, cy = 1.0f, deltaf[16] = {0.0f}, dum = 0.0f;
	int spIdx = 0, spIdy = 0; 
	int pxIdx = 0, pxIdy = 0, i = 0, j = 0;

	memset(N_photon, 0, imHeight * imWidth * sizeof(float));
	
	for (unsigned int Ind = 0; Ind < nspots; Ind++) {
		_x = locs[Ind * ndim];
		_y = locs[Ind * ndim + 1];
		offset_x = (int)floor(_x) - splineszx / 2;
		offset_y = (int)floor(_y) - splineszx / 2;
		if (offset_x + splineszx - 1 < 0 || offset_x >= imWidth || offset_y + splineszx - 1 < 0 || offset_y >= imHeight)
			continue;
		
		delta_x = 1.0f - (_x - floor(_x));
		delta_y = 1.0f - (_y - floor(_y));
		memset(deltaf, 0, 16 * sizeof(float));
		for (i = 0, cy = 1.0f; i < 4; i++) {
			for (j = 0, cx = 1.0f; j < 4; j++) {
				deltaf[i * 4 + j] = cy * cx;
				cx *= delta_x;
			}
			cy *= delta_y;
		}
			
		for (spIdy = 0; spIdy < splineszx; spIdy++) 
			for (spIdx = 0; spIdx < splineszx; spIdx++) {
				pxIdy = spIdy + offset_y;
				pxIdx = spIdx + offset_x;
				if (pxIdx >= 0 && pxIdx < imWidth && pxIdy >= 0 && pxIdy < imHeight) {
					for (i = 0, dum = 0.0f; i < 16; i++)
						dum += coeff_PSF[(spIdy * splineszx + spIdx) * 16 + i] * deltaf[i];
					N_photon[pxIdy * imWidth + pxIdx] += (dum > 0.0f) ? phot[Ind] * dum : 0.0f;
				}
			}
	}
}



CDLL_EXPORT void PhotRender_3d(int splineszx, int splineszz, float* coeff_PSF, int nspots, float* locs, float* phot,
	int imHeight, int imWidth, float* N_photon)
{
	const int ndim = 3;
	int offset_x = 0, offset_y = 0, offset_z = 0;
	float _x = 0.0f, _y = 0.0f, _z = 0.0f;
	float delta_x = 0.0f, delta_y = 0.0f, delta_z = 0.0f, cx = 1.0f, cy = 1.0f, cz = 1.0f, deltaf[64] = {0.0f}, dum = 0.0f;
	int spIdx = 0, spIdy = 0, spIdz = 0, spInd = 0; 
	int pxIdx = 0, pxIdy = 0, i = 0, j = 0, k = 0;

	memset(N_photon, 0, imHeight * imWidth * sizeof(float));
	
	for (unsigned int Ind = 0; Ind < nspots; Ind++) {
		_x = locs[Ind * ndim];
		_y = locs[Ind * ndim + 1];
		_z = locs[Ind * ndim + 2];
		if (_z < 0 || _z >= (float)splineszz)
			continue;

		offset_x = (int)floor(_x) - splineszx / 2;
		offset_y = (int)floor(_y) - splineszx / 2;
		offset_z = splineszz / 2 - (int)floor(_z);
		if (offset_x + splineszx - 1 < 0 || offset_x >= imWidth || offset_y + splineszx - 1 < 0 || offset_y >= imHeight)
			continue;
		
		delta_x = 1.0f - (_x - floor(_x));
		delta_y = 1.0f - (_y - floor(_y));
		delta_z = 1.0f - (_z - floor(_z));
		memset(deltaf, 0, 64 * sizeof(float));
		for (i = 0, cz = 1.0f; i < 4; i++) {
			for (j = 0, cy = 1.0f; j < 4; j++) {
				for (k = 0, cx = 1.0f; k < 4; k++) {
					deltaf[i * 16 + j * 4 + k] = cz * cy * cx;
					cx *= delta_x;
				}
				cy *= delta_y;
			}
			cz *= delta_z;
		}
			
		for (spIdy = 0; spIdy < splineszx; spIdy++) 
			for (spIdx = 0; spIdx < splineszx; spIdx++) {
				pxIdy = spIdy + offset_y;
				pxIdx = spIdx + offset_x;
				if (pxIdx >= 0 && pxIdx < imWidth && pxIdy >= 0 && pxIdy < imHeight) {
					spIdz = offset_z + splineszz / 2;
					spInd = spIdz * splineszx * splineszx + spIdy * splineszx + spIdx; 
					for (i = 0, dum = 0.0f; i < 64; i++)
						dum += coeff_PSF[spInd * 64 + i] * deltaf[i];
					N_photon[pxIdy * imWidth + pxIdx] += (dum > 0.0f) ? phot[Ind] * dum : 0.0f;
				}		
			}
	}
}