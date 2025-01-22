#include <string.h>
#include <math.h>
#include "DLL_Macros.h"
#include "definitions.h"

/*! @brief	Render image from coordinates 
	@param[in]	nspots:			int, number of coordinates
	@param[in]	g_locx:			(nspots) float, coordinate-x 
	@param[in]	g_locy:			(nspots) float, coordinate-y
	@param[in]	g_crlbx:		(nspots) float, crlb at x-axis
	@param[in]	g_crlby:		(nspots) float, crlb at y-axis
	@param[in]	g_photon:		(nspots) float, photons of each coordinates
	@param[in]	imHeight:		int, the height of the image to render
	@param[in]	imWidth:		int, the width of the image to renger
	@param[out]	g_im:			(ImHeight * ImWidth) float, the rendered image
*/


/*! @brief	direct render the image with given coordinates and photons */
static void histI_render_kernel(int nspots, float* locx, float* locy, float* photon, int imHeight, int imWidth, float* im)
{
	int pxIdx = 0, pxIdy = 0;
	for (unsigned int Idx = 0; Idx < nspots; Idx++) {
		pxIdx = (int)locx[Idx];
		pxIdy = (int)locy[Idx];
		if (pxIdx >= 0 && pxIdx < imWidth && pxIdy >= 0 && pxIdy < imHeight)
			im[pxIdy * imWidth + pxIdx] += photon[Idx];
	}
}



/*! @brief  direct render the image with given coordinates */
static void histN_render_kernel(int nspots, float* locx, float* locy, int imHeight, int imWidth, float* im)
{
	int pxIdx = 0, pxIdy = 0;
	for (unsigned int Idx = 0; Idx < nspots; Idx++) {
		pxIdx = (int)locx[Idx];
		pxIdy = (int)locy[Idx];
		if (pxIdx >= 0 && pxIdx < imWidth && pxIdy >= 0 && pxIdy < imHeight)
			im[pxIdy * imWidth + pxIdx] += 1.0f;
	}
}



/*! @brief 	render the image with a gaussian kernel with given coordinates, crlbs, and photons */
static void gaussI_render_kernel(int nspots, float* locx, float* locy, float* crlbx, float* crlby, float* photons, int imHeight, int imWidth, float* im)
{
	int pxIdx = 0, pxIdy = 0;
	float xc = 0.0f, yc = 0.0f, sigx = 0.0f, sigy = 0.0f, photon = 0.0f; 
	float hwx = 0.0f, hwy = 0.0f, dumx = 0.0f, dumy = 0.0f;
    int minx = 0, miny = 0, maxx = 0, maxy = 0;
	
	for (unsigned int Idx = 0; Idx < nspots; Idx++) {
		xc = locx[Idx];
		yc = locy[Idx];
		sigx = sqrt(max(crlbx[Idx], MINCRLB));
		sigy = sqrt(max(crlby[Idx], MINCRLB));
		photon = photons[Idx];

		hwx = sigx * sigScaler;
		hwy = sigy * sigScaler;

		minx = max((int)(xc - hwx), 0);
        maxx = min((int)(xc + hwx + 1.0f), imWidth);
        miny = max((int)(yc - hwy), 0);
        maxy = min((int)(yc + hwy + 1.0f), imHeight);

        for (pxIdy = miny; pxIdy < maxy; pxIdy++) 
			for (pxIdx = minx; pxIdx < maxx; pxIdx++) {
				dumx = ((float)pxIdx + 0.5f - xc) / sigx;
				dumy = ((float)pxIdy + 0.5f - yc) / sigy;
				im[pxIdy * imWidth + pxIdx] += photon / (2.0f*PI*sigx*sigy) * exp(-0.5f*(dumx*dumx + dumy*dumy));
			}
	}
}



/*! @brief 	render the image with a gaussian kernel with given coordinates, crlbs, and photons */
static void gaussN_render_kernel(int nspots, float* locx, float* locy, float* crlbx, float* crlby, int imHeight, int imWidth, float* im)
{
	int pxIdx = 0, pxIdy = 0;
	float xc = 0.0f, yc = 0.0f, sigx = 0.0f, sigy = 0.0f; 
	float hwx = 0.0f, hwy = 0.0f, dumx = 0.0f, dumy = 0.0f;
    int minx = 0, miny = 0, maxx = 0, maxy = 0;
	
	for (unsigned int Idx = 0; Idx < nspots; Idx++) {
		xc = locx[Idx];
		yc = locy[Idx];
		sigx = sqrt(crlbx[Idx]);
		sigy = sqrt(crlby[Idx]);

		hwx = sigx * sigScaler;
		hwy = sigy * sigScaler;

		minx = max((int)(xc - hwx), 0);
        maxx = min((int)(xc + hwx + 1.0f), imWidth);
        miny = max((int)(yc - hwy), 0);
        maxy = min((int)(yc + hwy + 1.0f), imHeight);

        for (pxIdy = miny; pxIdy < maxy; pxIdy++) 
			for (pxIdx = minx; pxIdx < maxx; pxIdx++) {
				dumx = ((float)pxIdx + 0.5f - xc) / sigx;
				dumy = ((float)pxIdy + 0.5f - yc) / sigy;
				im[pxIdy * imWidth + pxIdx] += 1.0f / (2.0f*PI*sigx*sigy) * exp(-0.5f*(dumx*dumx + dumy*dumy));
			}
	}
}



/*! @brief 	render wrapper */
CDLL_EXPORT void loc_render(int nspots, float* locx, float* locy, float* crlbx, float* crlby, float* photon,
	int imHeight, int imWidth, float* im, const char* opt)
{
	if (strcmp(opt, "histI") == 0)
		histI_render_kernel(nspots, locx, locy, photon, imHeight, imWidth, im);
	else if (strcmp(opt, "histN") == 0)
		histN_render_kernel(nspots, locx, locy, imHeight, imWidth, im);
	else if (strcmp(opt, "gaussI") == 0)
		gaussI_render_kernel(nspots, locx, locy, crlbx, crlby, photon, imHeight, imWidth, im);
	else
		gaussN_render_kernel(nspots, locx, locy, crlbx, crlby, imHeight, imWidth, im);
}