#include "DLL_Macros.h"


/*! @brief non-maximum suppression (nms)
    @param[in]  nspots:             int, number of spots
    @param[in]  maxRdius:           int, radius to suppress surrondings
    @param[in]  indx:               (nspots) int, index (x-axis) of the raw local maxima
    @param[in]  indy:               (nspots) int, index (y-axis) of the raw local maxima
    @param[in]  Intensity:          (nspots) float, the intensity read from im[indy, indx]
    @param[out] nms_list:           (nspots) int, register each non-maximums as false
    NOTE:
        indx, indy, and Intensity should be descently sorted according to the Intensity
        nms_list should be initialized to 'all true' before passing to the kernel 
*/
CDLL_EXPORT void nms(int nspots, int maxRadius, int* indx, int* indy, float* Intensity, int* nms_list)
{
    for (unsigned int i = 0; i < nspots; i++) {
        if (nms_list[i] == 0)
            continue;
        
        for (unsigned int j = i + 1; j < nspots; j++)
            if (indx[j]>=indx[i]-maxRadius && indx[j]<=indx[i]+maxRadius && indy[j]>=indy[i]-maxRadius && indy[j]<=indy[i]+maxRadius) 
                nms_list[j] = 0;
    }
    return;
}