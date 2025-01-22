#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "DLL_Macros.h"
#include "definitions.h"



/*!
	@brief return the minimum of an int array
	@param[in]	n:			int, number of elements in an array
	@param[in]	arr:		(n) int array, the input array
	@param[out]	min_val:	int, pointer to the minimum of the input array
*/
static inline void arr_min(int n, uint16_t* arr, uint16_t* min_val)
{
	*min_val = arr[0];
	for (unsigned int i = 1; i < n; i++)
		if (*min_val > arr[i])
			*min_val = arr[i];
	return;
}



/*!
	@brief return the maxmimum of an int array
	@param[in]	n:			int, number of elements in an array
	@param[in]	arr:		(n) int array, the input array
	@param[out]	max_val:	int, pointer to the minimum of the input array
*/
static inline void arr_max(int n, uint16_t* arr, uint16_t* max_val)
{
	*max_val = arr[0];
	for (unsigned int i = 1; i < n; i++)
		if (*max_val < arr[i])
			*max_val = arr[i];
	return;
}



/*!
	@brief draw color square rois onto an n-channel gray-scale uint16 image (flattened 3d array) 
	@param[in]	nchannels:		int, number of channels, maximum 4
	@param[in]	imszh:			int, height of the image
	@param[in]	imszw:			int, width of the image
	@param[in]	imin:			(nchannels * imszh * imszw,) uint16, the input n-channel gray-scale uint16 2d image
	@param[in]	boxsz:			int, size of the roi, should be odd
	@param[in]	nrois:			int, number of rois to draw  
	@param[in]	ind_corner:		(nchannels * 2 * nrois,) int, [[lefts], [uppers]] of each channel of the roi left-upper corner
	@param[in]	ind_center:		(nchannels * 2 * nrois,) int, [[xcenters], [ycenters]] of each channel of the roi center
	@param[out] imout:			(nchannels * imszh * imszw * 3,) uint8, the out put image with red rois drawn on the input image
*/
CDLL_EXPORT void im_drawer(int nchannels, int imszh, int imszw, uint16_t* imin, int boxsz, 
							int nrois, int* ind_corner, int* ind_center, uint8_t* imout)
{
	uint16_t im_min = 0, im_max = 0;
	int chID = 0, Ind = 0, pxIdx = 0, pxIdy = 0, pxID = 0;
	float scalar = 0.0f, dum = 0.0f;
	
	nchannels = (nchannels > 4) ? 4 : nchannels;
	for (chID = 0; chID < nchannels; chID++)
	{
		arr_min(imszh*imszw, imin+chID*imszh*imszw, &im_min);
		arr_max(imszh*imszw, imin+chID*imszh*imszw, &im_max);
		if (im_max <= im_min)
			continue;
		
		// copy images from imin to imout
		scalar = (float)(im_max - im_min);
		for (unsigned int i = 0; i < imszh * imszw; i++)
		{
			pxID = chID * imszh * imszw + i;
			dum = 255.0 * (float)(imin[pxID] - im_min) / scalar;
			for (unsigned int j = 0; j < 3; j++)
				imout[pxID * 3 + j] = (uint8_t)dum;
		}

		// draw roi edge and center one-by-one
		for (Ind = 0; Ind < nrois; Ind++)
		{
			pxIdy = ind_corner[chID * 2 * nrois + nrois + Ind] - 1;
			if (pxIdy >= 0 && pxIdy < imszh)
				for (unsigned int i = 0; i < boxsz; i++)
				{
					pxIdx = ind_corner[chID * 2 * nrois + Ind] + i;
					if (pxIdx >= 0 && pxIdx < imszw)
					{	
						for (unsigned int j = 0; j < 3; j++)
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + j] = 0;
						if (chID < 3)
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + chID] = 255;
						else
						{
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + 1] = 255;
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + 2] = 255;
						}
					}	
				}
			
			pxIdx = ind_corner[chID * 2 * nrois + Ind] - 1;
			if (pxIdx >= 0 && pxIdx < imszw)
				for (unsigned int i = 0; i < boxsz; i++)
				{
					pxIdy = ind_corner[chID * 2 * nrois + nrois + Ind] + i;
					if (pxIdy >= 0 && pxIdy < imszh)
					{	
						for (unsigned int j = 0; j < 3; j++)
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + j] = 0;
						if (chID < 3)
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + chID] = 255;
						else
						{
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + 1] = 255;
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + 2] = 255;
						}
					}	
				}
			
			pxIdy = ind_corner[chID * 2 * nrois + nrois + Ind] + boxsz;
			if (pxIdy >= 0 && pxIdy < imszh)
				for (unsigned int i = 0; i < boxsz; i++)
				{
					pxIdx = ind_corner[chID * 2 * nrois + Ind] + i;
					if (pxIdx >= 0 && pxIdx < imszw)
					{	
						for (unsigned int j = 0; j < 3; j++)
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + j] = 0;
						if (chID < 3)
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + chID] = 255;
						else
						{
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + 1] = 255;
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + 2] = 255;
						}
					}	
				}
			
			pxIdx = ind_corner[chID * 2 * nrois + Ind] + boxsz;
			if (pxIdx >= 0 && pxIdx < imszw)
				for (unsigned int i = 0; i < boxsz; i++)
				{
					pxIdy = ind_corner[chID * 2 * nrois + nrois + Ind] + i;
					if (pxIdy >= 0 && pxIdy < imszh)
					{	
						for (unsigned int j = 0; j < 3; j++)
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + j] = 0;
						if (chID < 3)
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + chID] = 255;
						else
						{
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + 1] = 255;
							imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + 2] = 255;
						}
					}	
				}
			
			pxIdx = ind_center[chID * 2 * nrois + Ind];
			pxIdy = ind_center[chID * 2 * nrois + nrois + Ind];
			if (pxIdx > ind_corner[chID * 2 * nrois + Ind] && pxIdx < ind_corner[chID * 2 * nrois + Ind] + boxsz && 
				pxIdy > ind_corner[chID * 2 * nrois + nrois + Ind] && pxIdy < ind_corner[chID * 2 * nrois + nrois + Ind] + boxsz)
			{
				for (unsigned int j = 0; j < 3; j++)
					imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + j] = 0;
				if (chID < 3)
					imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + chID] = 255;
				else
				{
					imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + 1] = 255;
					imout[(chID * imszh * imszw + pxIdy * imszw + pxIdx) * 3 + 2] = 255;
				}
			}	
		}
	}
	return;
}



/*!
	@brief collect the psfs from the input images 
	@param[in]	nchannels:		int, number of channels, maximum 4
	@param[in]	imszh:			int, height of the image
	@param[in]	imszw:			int, width of the image
	@param[in]	imin:			(nchannels * imszh * imszw,) uint16, the input n-channel gray-scale uint16 2d image
	@param[in]	boxsz:			int, size of the roi, should be odd
	@param[in]	nrois:			int, number of rois to draw  
	@param[in]	ind_corner:		(nchannels * 2 * nrois,) int, [[lefts], [uppers]] of each channel of the roi left-upper corner
	@param[in]	ind_center:		(nchannels * 2 * nrois,) int, [[xcenters], [ycenters]] of each channel of the roi center
	@param[out] psfout:			(nchannels * nrois * boxsz * boxsz * 3,) uint8, the out put image with red rois drawn on the input image
*/
CDLL_EXPORT void psf_extract(int nchannels, int imszh, int imszw, uint16_t* imin, int boxsz, 
								int nrois, int* ind_corner, int* ind_center, uint8_t* psfout)
{
	uint16_t im_min = 0, im_max = 0, *dum_im = nullptr;
	int chID = 0, Ind = 0, ind_corner_x = 0, ind_corner_y = 0, ind_center_x = 0, ind_center_y = 0, pxID = 0;
	float scalar = 0.0f, dum = 0.0f;
	
	nchannels = (nchannels > 4) ? 4 : nchannels;
	dum_im = (uint16_t*)malloc(boxsz * boxsz * sizeof(uint16_t));

	for (chID = 0; chID < nchannels; chID++)
		for (Ind = 0; Ind < nrois; Ind++)
		{
			ind_corner_x = ind_corner[chID * 2 * nrois + Ind];
			ind_corner_y = ind_corner[chID * 2 * nrois + nrois + Ind];
			if (ind_corner_x > 0 && ind_corner_x < imszw - boxsz && ind_corner_y > 0 && ind_corner_y < imszh - boxsz)
			{
				memset(dum_im, 0, boxsz * boxsz * sizeof(uint16_t));
				im_min = 65535;
				im_max = 0;
				for (unsigned int i = 0; i < boxsz * boxsz; i++)
				{
					pxID = chID * imszh * imszw + (ind_corner_y + i / boxsz) * imszw + (ind_corner_x + i % boxsz);
					dum_im[i] = imin[pxID];
					im_min = min(im_min, imin[pxID]);
					im_max = max(im_max, imin[pxID]);
				}

				scalar = (float)(im_max - im_min);
				for (unsigned int i = 0; i < boxsz * boxsz; i++)
				{
					dum = 255.0f * (float)(dum_im[i] - im_min) / scalar;
					pxID = chID * nrois * boxsz * boxsz + Ind * boxsz * boxsz + i;
					for (unsigned int j = 0; j < 3; j++)
						psfout[pxID * 3 + j] = (uint8_t)dum;
				}
			}
			
			ind_center_x = ind_center[chID * 2 * nrois + Ind];
			ind_center_y = ind_center[chID * 2 * nrois + nrois + Ind];
			if (ind_center_x <= ind_corner_x || ind_center_x >= ind_corner_x + boxsz || ind_center_y <= ind_corner_y || ind_center_y >= ind_corner_y + boxsz)
				continue;
			pxID = chID * nrois * boxsz * boxsz + Ind * boxsz * boxsz + (ind_center_y - ind_corner_y) * boxsz + ind_center_x - ind_corner_x;
			for (unsigned int j = 0; j < 3; j++)
				psfout[pxID * 3 + j] = 0;
			if (chID < 3)
				psfout[pxID * 3 + chID] = 255;
			else
			{
				psfout[pxID * 3 + 1] = 255;
				psfout[pxID * 3 + 2] = 255;
			}
		}
	
	free(dum_im);
	return;
}