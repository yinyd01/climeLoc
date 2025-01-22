#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "DLL_Macros.h"
#include "definitions.h"
#include "functionals_3d.h"



/*! @brief		fast search of the local maxima of an 2d-image.
                the local maximum is defined as that the pixel value is higher than its surrounding 8 pixels 
	@param[in]	nMax:				int, max number of possible maxima, usually is total number of pixels divided by 9. 
	@param[in]  imszy:              int, size of the image (y-axis)
    @param[in]  imszx:              int, size of the image (x-axis)
    @param[in]  img:                (imszy * imszx) float, flattened image for local maxima search
    @param[out] nspots:             (1) int, pointer to the number of spots in the final maxima_list
    @param[out] maxima_List:        (nMax * 3) float, [[x, y, intensity],...] list of the local maxima
*/
CDLL_EXPORT void local_maxima_2d(int nMax, int imszy, int imszx, float* img, int* nspots, float* maxima_List)
{
    const int ndim = 2, vnum0 = ndim + 1;
    int ind_max = 0;
    memset(maxima_List, 0, nMax * vnum0 * sizeof(float));
    
    // 4 corners
    if ((img[0*imszx+0] > img[0*imszx+1]) && (img[0*imszx+0] > img[1*imszx+0]) && (img[0*imszx+0] > img[1*imszx+1])) {
        maxima_List[ind_max * vnum0 + 0] = 0.0f;
        maxima_List[ind_max * vnum0 + 1] = 0.0f;
        maxima_List[ind_max * vnum0 + 2] = img[0 * imszx + 0];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    }
    if ((img[0*imszx+imszx-1] > img[0*imszx+imszx-2]) && (img[0*imszx+imszx-1] > img[1*imszx+imszx-1]) && (img[0*imszx+imszx-1] > img[1*imszx+imszx-2])) {
        maxima_List[ind_max * vnum0 + 0] = (float)(imszx-1);
        maxima_List[ind_max * vnum0 + 1] = 0.0f;
        maxima_List[ind_max * vnum0 + 2] = img[0 * imszx + imszx-1];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    }
    if ((img[(imszy-1)*imszx+0] > img[(imszy-1)*imszx+1]) && (img[(imszy-1)*imszx+0] > img[(imszy-2)*imszx+0]) && (img[(imszy-1)*imszx+0] > img[(imszy-2)*imszx+1])) {
        maxima_List[ind_max * vnum0 + 0] = 0.0f;
        maxima_List[ind_max * vnum0 + 1] = (float)(imszy-1);
        maxima_List[ind_max * vnum0 + 2] = img[(imszy-1) * imszx + 0];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    }
    if ((img[(imszy-1)*imszx+imszx-1] > img[(imszy-1)*imszx+imszx-2]) && (img[(imszy-1)*imszx+imszx-1] > img[(imszy-2)*imszx+imszx-1]) && (img[(imszy-1)*imszx+imszx-1] > img[(imszy-2)*imszx+imszx-2])) {
        maxima_List[ind_max * vnum0 + 0] = (float)(imszx-1);
        maxima_List[ind_max * vnum0 + 1] = (float)(imszy-1);
        maxima_List[ind_max * vnum0 + 2] = img[(imszy-1) * imszx + imszx-1];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    }

    // 4 edges
    for(unsigned int indx = 1; indx < imszx - 1; indx++)    
        if ((img[0*imszx+indx] > img[0*imszx + indx-1]) && (img[0*imszx+indx] > img[0*imszx + indx+1]) &&
            (img[0*imszx+indx] > img[1*imszx + indx-1]) && (img[0*imszx+indx] > img[1*imszx + indx]) && (img[0*imszx+indx] > img[1*imszx + indx+1])) {
            
            maxima_List[ind_max * vnum0 + 0] = (float)indx;
            maxima_List[ind_max * vnum0 + 1] = 0.0f;
            maxima_List[ind_max * vnum0 + 2] = img[0 * imszx + indx];
            
            ind_max++;
            indx++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    for(unsigned int indx = 1; indx < imszx - 1; indx++)    
        if ((img[(imszy-1)*imszx+indx] > img[(imszy-1)*imszx + indx-1]) && (img[(imszy-1)*imszx+indx] > img[(imszy-1)*imszx + indx+1]) &&
            (img[(imszy-1)*imszx+indx] > img[(imszy-2)*imszx + indx-1]) && (img[(imszy-1)*imszx+indx] > img[(imszy-2)*imszx + indx]) && (img[(imszy-1)*imszx+indx] > img[(imszy-2)*imszx + indx+1])) {
            
            maxima_List[ind_max * vnum0 + 0] = (float)indx;
            maxima_List[ind_max * vnum0 + 1] = (float)(imszy - 1);
            maxima_List[ind_max * vnum0 + 2] = img[(imszy-1) * imszx + indx];
            
            ind_max++;
            indx++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    for(unsigned int indy = 1; indy < imszy - 1; indy++)    
        if ((img[indy*imszx+0] > img[(indy-1)*imszx + 0]) && (img[indy*imszx+0] > img[(indy-1)*imszx + 1]) &&
            (img[indy*imszx+0] > img[indy*imszx + 1]) &&
            (img[indy*imszx+0] > img[(indy+1)*imszx + 0]) && (img[indy*imszx+0] > img[(indy+1)*imszx + 1])) {
            
            maxima_List[ind_max * vnum0 + 0] = 0.0f;
            maxima_List[ind_max * vnum0 + 1] = (float)indy;
            maxima_List[ind_max * vnum0 + 2] = img[indy * imszx + 0];
            
            ind_max++;
            indy++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    for(unsigned int indy = 1; indy < imszy - 1; indy++)    
        if ((img[indy*imszx+imszx-1] > img[(indy-1)*imszx + imszx-2]) && (img[indy*imszx+imszx-1] > img[(indy-1)*imszx + imszx-1]) &&
            (img[indy*imszx+imszx-1] > img[indy*imszx + imszx-2]) &&
            (img[indy*imszx+imszx-1] > img[(indy+1)*imszx + imszx-2]) && (img[indy*imszx+imszx-1] > img[(indy+1)*imszx + imszx-1])) {
            
            maxima_List[ind_max * vnum0 + 0] = (float)(imszx-1);
            maxima_List[ind_max * vnum0 + 1] = (float)indy;
            maxima_List[ind_max * vnum0 + 2] = img[indy * imszx + imszx-1];
            
            ind_max++;
            indy++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }

    // inner canvas
    for(unsigned int indy = 1; indy < imszy - 1; indy++) 
        for(unsigned int indx = 1; indx < imszx - 1; indx++)    
            if ((img[indy*imszx+indx] > img[(indy-1)*imszx + indx-1]) && (img[indy*imszx+indx] > img[(indy-1)*imszx + indx]) && (img[indy*imszx+indx] > img[(indy-1)*imszx + indx+1]) &&
                (img[indy*imszx+indx] > img[indy*imszx + indx-1]) && (img[indy*imszx+indx] > img[indy*imszx + indx+1]) &&
                (img[indy*imszx+indx] > img[(indy+1)*imszx + indx-1]) && (img[indy*imszx+indx] > img[(indy+1)*imszx + indx]) && (img[indy*imszx+indx] > img[(indy+1)*imszx + indx+1])) {
                
                maxima_List[ind_max * vnum0 + 0] = (float)indx;
                maxima_List[ind_max * vnum0 + 1] = (float)indy;
                maxima_List[ind_max * vnum0 + 2] = img[indy * imszx + indx];
                
                ind_max++;
                indx++;
                if (ind_max == nMax) {
                    *nspots = nMax;
                    return;
                } 
            } 
    *nspots = ind_max;
    return;  
}



/*! @brief		fast search of the local maxima of an 3d-image.
                the local maximum is defined as that the pixel value is higher than its surrounding 26 pixels 
	@param[in]	nMax:				int, max number of possible maxima, usually is total number of voxels divided by 27. 
	@param[in]  imszz:              int, size of the image (z-axis)
    @param[in]  imszy:              int, size of the image (y-axis)
    @param[in]  imszx:              int, size of the image (x-axis)
    @param[in]  img:                (imszz * imszy * imszx) float, flattened image for local maxima search
    @param[out] nspots:             (1) int, pointer to the number of spots in the final maxima_list
    @param[out] maxima_List:        (nMax * 4) float, [[x, y, z, intensity],...] list of the local maxima
*/
CDLL_EXPORT void local_maxima_3d(int nMax, int imszz, int imszy, int imszx, float* img, int* nspots, float* maxima_List)
{
    const int ndim = 3, vnum0 = ndim + 1;
    int ind_max = 0;
    memset(maxima_List, 0, nMax * vnum0 * sizeof(float));

    float neighbors[26] = {0.0f};
    int indz = 0, indy = 0, indx = 0;
    bool ismax = true;
    
    // inner space
    for (unsigned int indz = 1; indz < imszz - 1; indz++)
        for (unsigned int indy = 1; indy < imszy - 1; indy++)
            for (unsigned int indx = 1; indx < imszx - 1; indx++) {
                ismax = ismax_inner(imszz, imszy, imszx, indz, indy, indx, img, neighbors);
                if (ismax) {
                    maxima_List[ind_max * vnum0 + 0] = (float)indx;
                    maxima_List[ind_max * vnum0 + 1] = (float)indy;
                    maxima_List[ind_max * vnum0 + 2] = (float)indz;
                    maxima_List[ind_max * vnum0 + 3] = img[indz*imszy*imszx + indy*imszx + indx];
                    ind_max++;
                    indx++;
                    if (ind_max == nMax) {
                        *nspots = nMax;
                        return;
                    } 
                }        
            }

    // 6 faces
    for (unsigned int indy = 1; indy < imszy - 1; indy++)
        for (unsigned int indx = 1; indx < imszx - 1; indx++) {
            ismax = ismax_u(imszz, imszy, imszx, indy, indx, img, neighbors);
            if (ismax) {
                maxima_List[ind_max * vnum0 + 0] = (float)indx;
                maxima_List[ind_max * vnum0 + 1] = (float)indy;
                maxima_List[ind_max * vnum0 + 2] = 0.0f;
                maxima_List[ind_max * vnum0 + 3] = img[0*imszy*imszx + indy*imszx + indx];
                ind_max++;
                indx++;
                if (ind_max == nMax) {
                    *nspots = nMax;
                    return;
                } 
            }
        }
    for (unsigned int indy = 1; indy < imszy - 1; indy++)
        for (unsigned int indx = 1; indx < imszx - 1; indx++) {
            ismax = ismax_b(imszz, imszy, imszx, indy, indx, img, neighbors);
            if (ismax) {
                maxima_List[ind_max * vnum0 + 0] = (float)indx;
                maxima_List[ind_max * vnum0 + 1] = (float)indy;
                maxima_List[ind_max * vnum0 + 2] = (float)(imszz - 1);
                maxima_List[ind_max * vnum0 + 3] = img[(imszz-1)*imszy*imszx + indy*imszx + indx];
                ind_max++;
                indx++;
                if (ind_max == nMax) {
                    *nspots = nMax;
                    return;
                } 
            }           
        }
    for (unsigned int indz = 1; indz < imszz - 1; indz++)
        for (unsigned int indx = 1; indx < imszx - 1; indx++) {
            ismax = ismax_n(imszz, imszy, imszx, indz, indx, img, neighbors);
            if (ismax) {
                maxima_List[ind_max * vnum0 + 0] = (float)indx;
                maxima_List[ind_max * vnum0 + 1] = 0.0f;
                maxima_List[ind_max * vnum0 + 2] = (float)indz;
                maxima_List[ind_max * vnum0 + 3] = img[indz*imszy*imszx + 0*imszx + indx];
                ind_max++;
                indx++;
                if (ind_max == nMax) {
                    *nspots = nMax;
                    return;
                } 
            }
        }
    for (unsigned int indz = 1; indz < imszz - 1; indz++)
        for (unsigned int indx = 1; indx < imszx - 1; indx++) {
            ismax = ismax_s(imszz, imszy, imszx, indz, indx, img, neighbors);
            if (ismax) {
                maxima_List[ind_max * vnum0 + 0] = (float)indx;
                maxima_List[ind_max * vnum0 + 1] = (float)(imszy - 1);
                maxima_List[ind_max * vnum0 + 2] = (float)indz;
                maxima_List[ind_max * vnum0 + 3] = img[indz*imszy*imszx + (imszy-1)*imszx + indx];
                ind_max++;
                indx++;
                if (ind_max == nMax) {
                    *nspots = nMax;
                    return;
                } 
            }           
        }
    for (unsigned int indz = 1; indz < imszz - 1; indz++)
        for (unsigned int indy = 1; indy < imszy - 1; indy++) {
            ismax = ismax_w(imszz, imszy, imszx, indz, indy, img, neighbors);
            if (ismax) {
                maxima_List[ind_max * vnum0 + 0] = 0.0f;
                maxima_List[ind_max * vnum0 + 1] = (float)indy;
                maxima_List[ind_max * vnum0 + 2] = (float)indz;
                maxima_List[ind_max * vnum0 + 3] = img[indz*imszy*imszx + indy*imszx + 0];
                ind_max++;
                indy++;
                if (ind_max == nMax) {
                    *nspots = nMax;
                    return;
                } 
            }
        }
    for (unsigned int indz = 1; indz < imszz - 1; indz++)
        for (unsigned int indy = 1; indy < imszy - 1; indy++) {
            ismax = ismax_e(imszz, imszy, imszx, indz, indy, img, neighbors);
            if (ismax) {
                maxima_List[ind_max * vnum0 + 0] = (float)(imszx - 1);
                maxima_List[ind_max * vnum0 + 1] = (float)indy;
                maxima_List[ind_max * vnum0 + 2] = (float)indz;
                maxima_List[ind_max * vnum0 + 3] = img[indz*imszy*imszx + indy*imszx + imszx-1];
                ind_max++;
                indy++;
                if (ind_max == nMax) {
                    *nspots = nMax;
                    return;
                } 
            }           
        }

    // 12 edges
    for (unsigned int indx = 1; indx < imszx - 1; indx++) {
        ismax = ismax_un(imszz, imszy, imszx, indx, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = (float)indx;
            maxima_List[ind_max * vnum0 + 1] = 0.0f;
            maxima_List[ind_max * vnum0 + 2] = 0.0f;
            maxima_List[ind_max * vnum0 + 3] = img[0*imszy*imszx + 0*imszx + indx];
            ind_max++;
            indx++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    }
    for (unsigned int indx = 1; indx < imszx - 1; indx++) {
        ismax = ismax_us(imszz, imszy, imszx, indx, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = (float)indx;
            maxima_List[ind_max * vnum0 + 1] = (float)(imszy - 1);
            maxima_List[ind_max * vnum0 + 2] = 0.0f;
            maxima_List[ind_max * vnum0 + 3] = img[0*imszy*imszx + (imszy-1)*imszx + indx];
            ind_max++;
            indx++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    }
    for (unsigned int indx = 1; indx < imszx - 1; indx++) {
        ismax = ismax_bn(imszz, imszy, imszx, indx, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = (float)indx;
            maxima_List[ind_max * vnum0 + 1] = 0.0f;
            maxima_List[ind_max * vnum0 + 2] = (float)(imszz - 1);
            maxima_List[ind_max * vnum0 + 3] = img[(imszz-1)*imszy*imszx + 0*imszx + indx];
            ind_max++;
            indx++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    }
    for (unsigned int indx = 1; indx < imszx - 1; indx++) {
        ismax = ismax_bs(imszz, imszy, imszx, indx, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = (float)indx;
            maxima_List[ind_max * vnum0 + 1] = (float)(imszy - 1);
            maxima_List[ind_max * vnum0 + 2] = (float)(imszz - 1);
            maxima_List[ind_max * vnum0 + 3] = img[(imszz-1)*imszy*imszx + (imszy-1)*imszx + indx];
            ind_max++;
            indx++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }      
    }
    for (unsigned int indy = 1; indy < imszy - 1; indy++) {
        ismax = ismax_uw(imszz, imszy, imszx, indy, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = 0.0f;
            maxima_List[ind_max * vnum0 + 1] = (float)indy;
            maxima_List[ind_max * vnum0 + 2] = 0.0f;
            maxima_List[ind_max * vnum0 + 3] = img[0*imszy*imszx + indy*imszx + 0];
            ind_max++;
            indy++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    }
    for (unsigned int indy = 1; indy < imszy - 1; indy++) {
        ismax = ismax_ue(imszz, imszy, imszx, indy, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = (float)(imszx - 1);
            maxima_List[ind_max * vnum0 + 1] = (float)indy;
            maxima_List[ind_max * vnum0 + 2] = 0.0f;
            maxima_List[ind_max * vnum0 + 3] = img[0*imszy*imszx + indy*imszx + imszx-1];
            ind_max++;
            indy++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    }
    for (unsigned int indy = 1; indy < imszy - 1; indy++) {
        ismax = ismax_bw(imszz, imszy, imszx, indy, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = 0.0f;
            maxima_List[ind_max * vnum0 + 1] = (float)indy;
            maxima_List[ind_max * vnum0 + 2] = (float)(imszz - 1);
            maxima_List[ind_max * vnum0 + 3] = img[(imszz-1)*imszy*imszx + indy*imszx + 0];
            ind_max++;
            indy++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    }
    for (unsigned int indy = 1; indy < imszy - 1; indy++) {
        ismax = ismax_be(imszz, imszy, imszx, indy, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = (float)(imszx - 1);
            maxima_List[ind_max * vnum0 + 1] = (float)indy;
            maxima_List[ind_max * vnum0 + 2] = (float)(imszz - 1);
            maxima_List[ind_max * vnum0 + 3] = img[(imszz-1)*imszy*imszx + indy*imszx + imszx-1];
            ind_max++;
            indy++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        } 
    }
    for (unsigned int indz = 1; indz < imszz - 1; indz++) {
        ismax = ismax_nw(imszz, imszy, imszx, indz, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = 0.0f;
            maxima_List[ind_max * vnum0 + 1] = 0.0f;
            maxima_List[ind_max * vnum0 + 2] = (float)indz;
            maxima_List[ind_max * vnum0 + 3] = img[indz*imszy*imszx + 0*imszx + 0];
            ind_max++;
            indz++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    }
    for (unsigned int indz = 1; indz < imszz - 1; indz++) {
        ismax = ismax_ne(imszz, imszy, imszx, indz, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = (float)(imszx - 1);
            maxima_List[ind_max * vnum0 + 1] = 0.0f;
            maxima_List[ind_max * vnum0 + 2] = (float)indz;
            maxima_List[ind_max * vnum0 + 3] = img[indz*imszy*imszx + 0*imszx + imszx-1];
            ind_max++;
            indz++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        } 
    }
    for (unsigned int indz = 1; indz < imszz - 1; indz++) {
        ismax = ismax_sw(imszz, imszy, imszx, indz, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = 0.0f;
            maxima_List[ind_max * vnum0 + 1] = (float)(imszy - 1);
            maxima_List[ind_max * vnum0 + 2] = (float)indz;
            maxima_List[ind_max * vnum0 + 3] = img[indz*imszy*imszx + (imszy-1)*imszx + 0];
            ind_max++;
            indz++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        }
    }
    for (unsigned int indz = 1; indz < imszz - 1; indz++) {
        ismax = ismax_se(imszz, imszy, imszx, indz, img, neighbors);
        if (ismax) {
            maxima_List[ind_max * vnum0 + 0] = (float)(imszx - 1);
            maxima_List[ind_max * vnum0 + 1] = (float)(imszy - 1);
            maxima_List[ind_max * vnum0 + 2] = (float)indz;
            maxima_List[ind_max * vnum0 + 3] = img[indz*imszy*imszx + (imszy-1)*imszx + imszx-1];
            ind_max++;
            indz++;
            if (ind_max == nMax) {
                *nspots = nMax;
                return;
            } 
        } 
    }

    // 8 corners
    ismax = ismax_unw(imszz, imszy, imszx, img, neighbors);
    if (ismax) {
        maxima_List[ind_max * vnum0 + 0] = 0.0f;
        maxima_List[ind_max * vnum0 + 1] = 0.0f;
        maxima_List[ind_max * vnum0 + 2] = 0.0f;
        maxima_List[ind_max * vnum0 + 3] = img[0*imszy*imszx + 0*imszx + 0];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    } 

    ismax = ismax_une(imszz, imszy, imszx, img, neighbors);
    if (ismax) {
        maxima_List[ind_max * vnum0 + 0] = (float)(imszx-1);
        maxima_List[ind_max * vnum0 + 1] = 0.0f;
        maxima_List[ind_max * vnum0 + 2] = 0.0f;
        maxima_List[ind_max * vnum0 + 3] = img[0*imszy*imszx + 0*imszx + imszx-1];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    } 

    ismax = ismax_usw(imszz, imszy, imszx, img, neighbors);
    if (ismax) {
        maxima_List[ind_max * vnum0 + 0] = 0.0f;
        maxima_List[ind_max * vnum0 + 1] = (float)(imszy-1);
        maxima_List[ind_max * vnum0 + 2] = 0.0f;
        maxima_List[ind_max * vnum0 + 3] = img[0*imszy*imszx + (imszy-1)*imszx + 0];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    } 

    ismax = ismax_use(imszz, imszy, imszx, img, neighbors);
    if (ismax) {
        maxima_List[ind_max * vnum0 + 0] = (float)(imszx-1);
        maxima_List[ind_max * vnum0 + 1] = (float)(imszy-1);
        maxima_List[ind_max * vnum0 + 2] = 0.0f;
        maxima_List[ind_max * vnum0 + 3] = img[0*imszy*imszx + (imszy-1)*imszx + imszx-1];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    } 

    ismax = ismax_bnw(imszz, imszy, imszx, img, neighbors);
    if (ismax) {
        maxima_List[ind_max * vnum0 + 0] = 0.0f;
        maxima_List[ind_max * vnum0 + 1] = 0.0f;
        maxima_List[ind_max * vnum0 + 2] = (float)(imszz-1);
        maxima_List[ind_max * vnum0 + 3] = img[(imszz-1)*imszy*imszx + 0*imszx + 0];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    } 

    ismax = ismax_bne(imszz, imszy, imszx, img, neighbors);
    if (ismax) {
        maxima_List[ind_max * vnum0 + 0] = (float)(imszx-1);
        maxima_List[ind_max * vnum0 + 1] = 0.0f;
        maxima_List[ind_max * vnum0 + 2] = (float)(imszz-1);
        maxima_List[ind_max * vnum0 + 3] = img[(imszz-1)*imszy*imszx + 0*imszx + imszx-1];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    } 

    ismax = ismax_bsw(imszz, imszy, imszx, img, neighbors);
    if (ismax) {
        maxima_List[ind_max * vnum0 + 0] = 0.0f;
        maxima_List[ind_max * vnum0 + 1] = (float)(imszy-1);
        maxima_List[ind_max * vnum0 + 2] = (float)(imszz-1);
        maxima_List[ind_max * vnum0 + 3] = img[(imszz-1)*imszy*imszx + (imszy-1)*imszx + 0];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    } 

    ismax = ismax_bse(imszz, imszy, imszx, img, neighbors);
    if (ismax) {
        maxima_List[ind_max * vnum0 + 0] = (float)(imszx-1);
        maxima_List[ind_max * vnum0 + 1] = (float)(imszy-1);
        maxima_List[ind_max * vnum0 + 2] = (float)(imszz-1);
        maxima_List[ind_max * vnum0 + 3] = img[(imszz-1)*imszy*imszx + (imszy-1)*imszx + imszx-1];
        ind_max++;
        if (ind_max == nMax) {
            *nspots = nMax;
            return;
        } 
    } 
    
    *nspots = ind_max;
    return;  
}