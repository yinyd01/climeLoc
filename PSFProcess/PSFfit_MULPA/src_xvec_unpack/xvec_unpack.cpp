#include <string.h>
#include <stdlib.h>
#include "DLL_Macros.h"


/*! @brief  in-place filterout the outside emitters (box coordinates)
    @param[in]  NFits:      int, number of the fits
    @param[in]  vmax:       int, maximum number of parameters. vnum0 = ndim + nchannels, vmax = NMAX * vnum0 + nchannels
    @param[in]  ndim:       int, number of dimension
    @param[in]  nchannels:  int, number of imaging channels
    @param[in]  nnum:       (Nfits) int, the number of emitters for each fit
    @param[in]  xvec:       (NFits * vmax) float, the xvec from the mulpa fit
    @param[in]  crlb:       (NFits * vmax) float, the crlb from the mulpa fit
*/
CDLL_EXPORT void xvec_inroi(int NFits, int boxsz, int vmax, int ndim, int nchannels, int* nnum, float* xvec, float* crlb)
{
    const unsigned int vnum0 = ndim + nchannels;
    int ind_bkg = 0;
    float x = 0.0f, y = 0.0f;
    for (unsigned int i = 0; i < NFits; i++) {
        ind_bkg = nnum[i] * vnum0;
        for (unsigned int n = 0; n < nnum[i]; n++) {
            x = xvec[i * vmax + n * vnum0];
            y = xvec[i * vmax + n * vnum0 + 1];
            if (x < 0.0f || x > (float)boxsz || y < 0.0f || y > (float)boxsz) {
                if (n < nnum[i] - 1) {
                    memcpy(xvec + i * vmax + n * vnum0, xvec + i * vmax + (n+1) * vnum0, (nnum[i]-n-1) * vnum0 * sizeof(float));
                    memcpy(crlb + i * vmax + n * vnum0, crlb + i * vmax + (n+1) * vnum0, (nnum[i]-n-1) * vnum0 * sizeof(float));
                    nnum[i] -= 1;
                    n -= 1;
                }
                else
                    nnum[i] -= 1;
            }
        }
        memcpy(xvec + i * vmax + nnum[i] * vnum0, xvec + i * vmax + ind_bkg, nchannels * sizeof(float));
        memcpy(crlb + i * vmax + nnum[i] * vnum0, crlb + i * vmax + ind_bkg, nchannels * sizeof(float));
    }
    return;
}




/*! @brief  unpack the xvec from MULPA fit
    @param[in]  NFits:      int, number of the fits
    @param[in]  vmax:       int, maximum number of parameters. vnum0 = ndim + nchannels, vmax = NMAX * vnum0 + nchannels
    @param[in]  ndim:       int, number of dimension
    @param[in]  nchannels:  int, number of imaging channels
    @param[in]  nsum:       int, total number of emitters of all the fits (=nnum.sum())
    @param[in]  nnum:       (Nfits) int, the number of emitters for each fit
    @param[in]  xvec:       (NFits * vmax) float, the xvec from the mulpa fit
    @param[in]  crlb:       (NFits * vmax) float, the crlb from the mulpa fit
    @param[out] xvec_o:     ((vnum0 + nchannels) * nsum) float, the rearranged xvec on output
    @param[out] crlb_o:     ((vnum0 + nchannels) * nsum) float, the rearranged crlb on output
*/
CDLL_EXPORT void xvec_unpack2col(int NFits, int vmax, int ndim, int nchannels, int nsum, int* nnum, 
    float* xvec, float* crlb, float* xvec_o, float* crlb_o)
{
    const unsigned int vnum0 = ndim + nchannels;
    
    unsigned int n_accum = 0;
    for (unsigned int i = 0; i < NFits; i++) {
        for (unsigned int n = 0; n < nnum[i]; n++) {
            for (unsigned int j = 0; j < vnum0; j++) {
                xvec_o[j * nsum + n_accum + n] = xvec[i * vmax + n * vnum0 + j];
                crlb_o[j * nsum + n_accum + n] = crlb[i * vmax + n * vnum0 + j];
            }
            for (unsigned int j = 0; j < nchannels; j++) {
                xvec_o[(vnum0 + j) * nsum + n_accum + n] = xvec[i * vmax + nnum[i] * vnum0 + j];
                crlb_o[(vnum0 + j) * nsum + n_accum + n] = crlb[i * vmax + nnum[i] * vnum0 + j];
            }
        }
        n_accum += nnum[i];
    }
    return;
}



/*! @brief  unpack the xvec from MULPA fit
    @param[in]  NFits:      int, number of the fits
    @param[in]  vmax:       int, maximum number of parameters. vnum0 = ndim + nchannels, vmax = NMAX * vnum0 + nchannels
    @param[in]  ndim:       int, number of dimension
    @param[in]  nchannels:  int, number of imaging channels
    @param[in]  nnum:       (Nfits) int, the number of emitters for each fit
    @param[in]  xvec:       (NFits * vmax) float, the xvec from the mulpa fit
    @param[in]  crlb:       (NFits * vmax) float, the crlb from the mulpa fit
    @param[out] xvec_o:     (nsum * vnum) float, the rearranged xvec on output
    @param[out] crlb_o:     (nsum * vnum) float, the rearranged crlb on output
*/
CDLL_EXPORT void xvec_unpack2row(int NFits, int vmax, int ndim, int nchannels, int* nnum, 
    float* xvec, float* crlb, float* xvec_o, float* crlb_o)
{
    const unsigned int vnum0 = ndim + nchannels;
    const unsigned int vnum = vnum0 + nchannels;
    
    unsigned int n_accum = 0;
    for (unsigned int i = 0; i < NFits; i++) {
        for (unsigned int n = 0; n < nnum[i]; n++) {
            memcpy(xvec_o + (n_accum + n) * vnum, xvec + i * vmax + n * vnum0, vnum0 * sizeof(float));
            memcpy(crlb_o + (n_accum + n) * vnum, crlb + i * vmax + n * vnum0, vnum0 * sizeof(float));
            memcpy(xvec_o + (n_accum + n) * vnum + vnum0, xvec + i * vmax + nnum[i] * vnum0, nchannels * sizeof(float));
            memcpy(crlb_o + (n_accum + n) * vnum + vnum0, crlb + i * vmax + nnum[i] * vnum0, nchannels * sizeof(float));
        }
        n_accum += nnum[i];
    }
    return;
}