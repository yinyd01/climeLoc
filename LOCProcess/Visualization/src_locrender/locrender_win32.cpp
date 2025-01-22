#include <string.h>
#include <math.h>
#include "DLL_Macros.h"
#include "definitions.h"

/*! @brief	Render single-channel image from coordinates 
	@param[in]	nchannels:		int, number of dimention
	@param[in]	nspots:			int, number of coordinates
	@param[in]	ncbits:			int, number of entris for the colormap			
	@param[in]	locx:			(nspots,) float array, coordinate-x 
	@param[in]	locy:			(nspots,) float array, coordinate-y
	@param[in]	crlbx:			(nspots,) float array, crlb at x-axis
	@param[in]	crlby:			(nspots,) float array, crlb at y-axis
	@param[in]	photon:			(nspots,) float array, photons of each coordinates
	@param[in]	coder:			(nspots,) float array, the coder of the luts (e.g. locz, loss...)
	@param[in]	luts:			(3 * ncmap,) float array, the 3-channel RGB colormap
	@param[in]	alpha:			float, the transparency parameter
								alpha <= 0: 	pixel accumulation
								alpha > 1:		pixel overlay
								0 <alpha <= 1:	pixel transparent
	@param[in]	imHeight:		int, the height of the image to render
	@param[in]	imWidth:		int, the width of the image to renger
	@param[out]	im_gray:		(ImHeight * ImWidth,) float array, the pointer to the image to render
	@param[out]	im_rgb:			(ImHeight * ImWidth * 3,) foult array, the pointer to the image to render
	NOTE:
	the locs, crlbs are at the unit of the pixel size of the target image to be rendered
	the luts should be pre-normalized to [0, 1]
*/



static void direct_photon_gray_kernel(int nspots, float* locx, float* locy, float* photon, 
	float alpha, int imHeight, int imWidth, float* im_gray)
{
	int pxIdx = 0, pxIdy = 0, pxID = 0;
	float dum0 = 0.0f, dum1 = 0.0f;

	for (unsigned int Ind = 0; Ind < nspots; Ind++) {
		pxIdx = (int)locx[Ind];
		pxIdy = (int)locy[Ind];
		if (pxIdx >= 0 && pxIdx < imWidth && pxIdy >= 0 && pxIdy < imHeight) {
			pxID = pxIdy * imWidth + pxIdx;
			if (alpha <= 0.0f)
				im_gray[pxID] += photon[Ind];
			else {
				dum0 = alpha * photon[Ind];
				dum1 = min(1.0f, dum0);	
				im_gray[pxID] = dum0 + (1.0f - dum1) * im_gray[pxID];
			}
		}
	}
	return;
}


static void direct_blink_gray_kernel(int nspots, float* locx, float* locy, 
	float alpha, int imHeight, int imWidth, float* im_gray)
{
	int pxIdx = 0, pxIdy = 0, pxID = 0;
	float dum0 = 0.0f, dum1 = 0.0f;

	for (unsigned int Ind = 0; Ind < nspots; Ind++) {
		pxIdx = (int)locx[Ind];
		pxIdy = (int)locy[Ind];
		if (pxIdx >= 0 && pxIdx < imWidth && pxIdy >= 0 && pxIdy < imHeight) {
			pxID = pxIdy * imWidth + pxIdx;
			if (alpha <= 0.0f)
				im_gray[pxID] += 1.0f;
			else {
				dum0 = alpha;
				dum1 = min(1.0f, dum0);
				im_gray[pxID] = dum0 + (1.0f - dum1) * im_gray[pxID];
			}
		}
	}
	return;
}


static void blur_photon_gray_kernel(int nspots, float* locx, float* locy, float* sigs, float* photons,
	float alpha, int imHeight, int imWidth, float* im_gray)
{
	int pxIdx = 0, pxIdy = 0, pxID = 0;
	float dum0 = 0.0f, dum1 = 0.0f;

	float xc = 0.0f, yc = 0.0f, sig = 0.0f, photon = 0.0f; 
	float hw = 0.0f, dumx = 0.0f, dumy = 0.0f, tmp = 0.0f;
    int minx = 0, miny = 0, maxx = 0, maxy = 0;

	for (unsigned int Ind = 0; Ind < nspots; Ind++) {
		xc = locx[Ind];
		yc = locy[Ind];
		sig = max(sigs[Ind], MINSIG);
		photon = photons[Ind];

		hw = sig * sigScaler;
		minx = max((int)(xc - hw), 0);
        maxx = min((int)(xc + hw + 1.0f), imWidth);
        miny = max((int)(yc - hw), 0);
        maxy = min((int)(yc + hw + 1.0f), imHeight);

        for (pxIdy = miny; pxIdy < maxy; pxIdy++) 
			for (pxIdx = minx; pxIdx < maxx; pxIdx++) {
				pxID = pxIdy * imWidth + pxIdx;
				dumx = ((float)pxIdx + 0.5f - xc) / sig;
				dumy = ((float)pxIdy + 0.5f - yc) / sig;
				tmp = photon / (2.0f*PI*sig*sig) * exp(-0.5f*(dumx*dumx + dumy*dumy));
				if (alpha <= 0.0f)
					im_gray[pxID] += tmp;
				else {
					dum0 = alpha * tmp;
					dum1 = min(1.0f, dum0);
					im_gray[pxID] = dum0 + (1.0f - dum1) * im_gray[pxID];
				}
			}
	}
	return;
}


static void blur_blink_gray_kernel(int nspots, float* locx, float* locy, float* sigs,
	float alpha, int imHeight, int imWidth, float* im_gray)
{
	int pxIdx = 0, pxIdy = 0, pxID = 0;
	float dum0 = 0.0f, dum1 = 0.0f;

	float xc = 0.0f, yc = 0.0f, sig = 0.0f; 
	float hw = 0.0f, dumx = 0.0f, dumy = 0.0f, tmp = 0.0f;
    int minx = 0, miny = 0, maxx = 0, maxy = 0;
	
	for (unsigned int Ind = 0; Ind < nspots; Ind++) {
		xc = locx[Ind];
		yc = locy[Ind];
		sig = max(sigs[Ind], MINSIG);

		hw = sig * sigScaler;
		minx = max((int)(xc - hw), 0);
        maxx = min((int)(xc + hw + 1.0f), imWidth);
        miny = max((int)(yc - hw), 0);
        maxy = min((int)(yc + hw + 1.0f), imHeight);

        for (pxIdy = miny; pxIdy < maxy; pxIdy++) 
			for (pxIdx = minx; pxIdx < maxx; pxIdx++) {
				pxID = pxIdy * imWidth + pxIdx;
				dumx = ((float)pxIdx + 0.5f - xc) / sig;
				dumy = ((float)pxIdy + 0.5f - yc) / sig;
				tmp = 1.0f / (2.0f*PI*sig*sig) * exp(-0.5f*(dumx*dumx + dumy*dumy));
				if (alpha <= 0.0f)
					im_gray[pxID] += tmp;
				else {
					dum0 = alpha * tmp;
					dum1 = min(1.0f, dum0);
					im_gray[pxID] = dum0 + (1.0f - dum1) * im_gray[pxID];
				}
			}
	}
	return;
}





/*! @brief	brief description multi-channel image from coordinates 
	@param[in]	nchannels:		int, number of dimention
	@param[in]	nspots:			int, number of coordinates from all the n-channels
	@param[in]	ncbits:			int, number of entris for the colormap			
	@param[in]	ch_reg:			(nspots,) int array, channel registration for all the spots
	@param[in]	locx:			(nspots,) float array, coordinate-x 
	@param[in]	locy:			(nspots,) float array, coordinate-y
	@param[in]	crlbx:			(nspots,) float array, crlb at x-axis
	@param[in]	crlby:			(nspots,) float array, crlb at y-axis
	@param[in]	photon:			(nspots,) float array, photons of each coordinates
	@param[in]	coder:			(nspots,) float array, the coder of the luts (e.g. locz, loss...)
	@param[in]	luts:			(nchannels * 3 * ncbits,) float array, the 3-channel RGB colormap
	@param[in]	alpha:			float, the transparency parameter
								alpha <= 0: 	pixel accumulation
								alpha > 1:		pixel overlay
								0 <alpha <= 1:	pixel transparent
	@param[in]	imHeight:		int, the height of the image to render
	@param[in]	imWidth:		int, the width of the image to renger
	@param[out]	im_rgb:			(ImHeight * ImWidth * 3,) float array, the pointer to the image to render
	NOTE:
	the locs, crlbs are at the unit of the pixel size of the target image to be rendered
	the luts should be pre-normalized to [0, 1]
*/



static void direct_photon_rgb_kernel(int nchannels, int nspots, int ncbits, int* ch_reg, float* locx, float* locy, float* photon, float* coder, 
	float* luts, float alpha, int imHeight, int imWidth, float* im_rgb)
{
	int chID = 0, cInd = 0;
	int pxIdx = 0, pxIdy = 0, pxIdc = 0, pxID = 0;
	float dum0 = 0.0f, dum1 = 0.0f;

	for (unsigned int Ind = 0; Ind < nspots; Ind++) {
		pxIdx = (int)locx[Ind];
		pxIdy = (int)locy[Ind];
		chID = ch_reg[Ind];
		if (chID >=0 && chID < nchannels && coder[Ind] > 0.0f && pxIdx >= 0 && pxIdx < imWidth && pxIdy >= 0 && pxIdy < imHeight) {
			cInd = (int)(coder[Ind] * (float)(ncbits - 1));
			for (pxIdc = 0; pxIdc < 3; pxIdc++) {
				pxID = (pxIdy * imWidth + pxIdx) * 3 + pxIdc;
				if (alpha <= 0.0f)
					im_rgb[pxID] += photon[Ind] * luts[chID * 3 * ncbits + pxIdc * ncbits + cInd];
				else {
					dum0 = alpha * photon[Ind];
					dum1 = min(1.0f, dum0 * 5.0f);
					im_rgb[pxID] = dum0 * luts[chID * 3 * ncbits + pxIdc * ncbits + cInd] + (1.0f - dum1) * im_rgb[pxID];
				}
			}
		}
	}
	return;
}


static void direct_blink_rgb_kernel(int nchannels, int nspots, int ncbits, int* ch_reg, float* locx, float* locy, float* coder, 
	float* luts, float alpha, int imHeight, int imWidth, float* im_rgb)
{
	int chID = 0, cInd = 0; 
	int pxIdx = 0, pxIdy = 0, pxIdc = 0, pxID = 0;
	float dum0 = 0.0f, dum1 = 0.0f;
	
	for (unsigned int Ind = 0; Ind < nspots; Ind++) {
		pxIdx = (int)locx[Ind];
		pxIdy = (int)locy[Ind];
		chID = ch_reg[Ind];
		if (chID >= 0 && chID < nchannels && coder[Ind] > 0.0f && pxIdx >= 0 && pxIdx < imWidth && pxIdy >= 0 && pxIdy < imHeight) {
			cInd = (int)(coder[Ind] * (float)(ncbits - 1));
			for (pxIdc = 0; pxIdc < 3; pxIdc++) {
				pxID = (pxIdy * imWidth + pxIdx) * 3 + pxIdc;
				if (alpha <= 0.0f)
					im_rgb[pxID] += luts[chID * 3 * ncbits + pxIdc * ncbits + cInd];
				else {
					dum0 = alpha;
					dum1 = min(1.0f, dum0 * 5.0f);
					im_rgb[pxID] = dum0 * luts[chID * 3 * ncbits + pxIdc * ncbits + cInd] + (1.0f - dum1) * im_rgb[pxID];
				}
			}	
		}
	}
	return;
}


static void blur_photon_rgb_kernel(int nchannels, int nspots, int ncbits, int* ch_reg, float* locx, float* locy, float* sigs, float* photons, float* coder, 
	float* luts, float alpha, int imHeight, int imWidth, float* im_rgb)
{
	int chID = 0, cInd = 0; 
	int pxIdx = 0, pxIdy = 0, pxIdc = 0, pxID = 0;
	float dum0 = 0.0f, dum1 = 0.0f;

	float xc = 0.0f, yc = 0.0f, sig = 0.0f, photon = 0.0f; 
	float hw = 0.0f, dumx = 0.0f, dumy = 0.0f, tmp = 0.0f;
    int minx = 0, miny = 0, maxx = 0, maxy = 0;
	
	for (unsigned int Ind = 0; Ind < nspots; Ind++) {
		xc = locx[Ind];
		yc = locy[Ind];
		chID = ch_reg[Ind];
		if (chID >= 0 && chID < nchannels && coder[Ind] > 0.0 && xc >= 0 && xc < imWidth && yc >= 0 && yc < imHeight) {
			cInd = (int)(coder[Ind] * (float)(ncbits - 1));
			
			sig = max(sigs[Ind], MINSIG);
			photon = photons[Ind];

			hw = sig * sigScaler;
			minx = max((int)(xc - hw), 0);
			maxx = min((int)(xc + hw + 1.0f), imWidth);
			miny = max((int)(yc - hw), 0);
			maxy = min((int)(yc + hw + 1.0f), imHeight);

			for (pxIdy = miny; pxIdy < maxy; pxIdy++) 
				for (pxIdx = minx; pxIdx < maxx; pxIdx++) {
					dumx = ((float)pxIdx + 0.5f - xc) / sig;
					dumy = ((float)pxIdy + 0.5f - yc) / sig;
					tmp = photon / (2.0f*PI*sig*sig) * exp(-0.5f*(dumx*dumx + dumy*dumy));
					for (pxIdc = 0; pxIdc < 3; pxIdc++) {
						pxID = (pxIdy * imWidth + pxIdx) * 3 + pxIdc;
						if (alpha <= 0.0f)
							im_rgb[pxID] += tmp * luts[chID * 3 * ncbits + pxIdc * ncbits + cInd];
						else {
							dum0 = alpha * tmp;
							dum1 = min(1.0f, dum0 * 5.0f);
							im_rgb[pxID] = dum0 * luts[chID * 3 * ncbits + pxIdc * ncbits + cInd] + (1.0f - dum1) * im_rgb[pxID];
						}
					}
				}
		}
	}
	return;
}


static void blur_blink_rgb_kernel(int nchannels, int nspots, int ncbits, int* ch_reg, float* locx, float* locy, float* sigs, float* coder,
	float* luts, float alpha, int imHeight, int imWidth, float* im_rgb)
{
	int chID = 0, cInd = 0;
	int pxIdx = 0, pxIdy = 0, pxIdc = 0, pxID = 0;
	float dum0 = 0.0f, dum1 = 0.0f;

	float xc = 0.0f, yc = 0.0f, sig = 0.0f; 
	float hw = 0.0f, dumx = 0.0f, dumy = 0.0f, tmp = 0.0f;
    int minx = 0, miny = 0, maxx = 0, maxy = 0;
	
	for (unsigned int Ind = 0; Ind < nspots; Ind++) {
		xc = locx[Ind];
		yc = locy[Ind];
		chID = ch_reg[Ind];
		if (chID >= 0 && chID < nchannels && coder[Ind] > 0.0f && xc >= 0 && xc < imWidth && yc >= 0 && yc < imHeight) {
			cInd = (int)(coder[Ind] * (float)(ncbits - 1));

			sig = max(sigs[Ind], MINSIG);
			
			hw = sig * sigScaler;
			minx = max((int)(xc - hw), 0);
			maxx = min((int)(xc + hw + 1.0f), imWidth);
			miny = max((int)(yc - hw), 0);
			maxy = min((int)(yc + hw + 1.0f), imHeight);

			for (pxIdy = miny; pxIdy < maxy; pxIdy++) 
				for (pxIdx = minx; pxIdx < maxx; pxIdx++) {
					dumx = ((float)pxIdx + 0.5f - xc) / sig;
					dumy = ((float)pxIdy + 0.5f - yc) / sig;
					tmp = 1.0f / (2.0f*PI*sig*sig) * exp(-0.5f*(dumx*dumx + dumy*dumy));
					for (pxIdc = 0; pxIdc < 3; pxIdc++) {
						pxID = (pxIdy * imWidth + pxIdx) * 3 + pxIdc;
						if (alpha <= 0.0f)
							im_rgb[pxID] += tmp * luts[chID * 3 * ncbits + pxIdc * ncbits + cInd];
						else {
							dum0 = alpha * tmp;
							dum1 = min(1.0f, dum0 * 5.0f);
							im_rgb[pxID] = dum0 * luts[chID * 3 * ncbits + pxIdc * ncbits + cInd] + (1.0f - dum1) * im_rgb[pxID];
						} 
					}
				}
		}
	}
	return;
}





// render wrappers
CDLL_EXPORT void loc_render_gray(int nspots, float* locx, float* locy, float* sigs, float* photon,
	float alpha, int imHeight, int imWidth, float* im_gray, const int isblur, const char* intensity)
{
	if (isblur == 0) {
		if (strcmp(intensity, "photon") == 0)
			direct_photon_gray_kernel(nspots, locx, locy, photon, alpha, imHeight, imWidth, im_gray);
		else if (strcmp(intensity, "blink") == 0)
			direct_blink_gray_kernel(nspots, locx, locy, alpha, imHeight, imWidth, im_gray); 
	}
	else  {
		if (strcmp(intensity, "photon") == 0) 
			blur_photon_gray_kernel(nspots, locx, locy, sigs, photon, alpha, imHeight, imWidth, im_gray);
		else if (strcmp(intensity, "blink") == 0)
			blur_blink_gray_kernel(nspots, locx, locy, sigs, alpha, imHeight, imWidth, im_gray);
	}
	return;
}



CDLL_EXPORT void loc_render_rgb(int nchannels, int nspots, int ncbits, int* ch_reg, float* locx, float* locy, float* sigs, float* photon, float* coder,
	float* luts, float alpha, int imHeight, int imWidth, float* im_rgb, const int isblur, const char* intensity)
{
	if (isblur == 0) {
		if (strcmp(intensity, "photon") == 0)
			direct_photon_rgb_kernel(nchannels, nspots, ncbits, ch_reg, locx, locy, photon, coder, luts, alpha, imHeight, imWidth, im_rgb);
		else if (strcmp(intensity, "blink") == 0)
			direct_blink_rgb_kernel(nchannels, nspots, ncbits, ch_reg, locx, locy, coder, luts, alpha, imHeight, imWidth, im_rgb); 
	}
	else {
		if (strcmp(intensity, "photon") == 0)
			blur_photon_rgb_kernel(nchannels, nspots, ncbits, ch_reg, locx, locy, sigs, photon, coder, luts, alpha, imHeight, imWidth, im_rgb);
		else if (strcmp(intensity, "blink") == 0)
			blur_blink_rgb_kernel(nchannels, nspots, ncbits, ch_reg, locx, locy, sigs, coder, luts, alpha, imHeight, imWidth, im_rgb);
	}
	return;
}