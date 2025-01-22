#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "DLL_Macros.h"
#include "definitions.h"
#include "kdtree.h"


/*! @brief   count the spots (in the src_locs set) within rings
    @param[in]  ndim:           int, number of dimention
    @param[in]  windowsz:       (ndim) int, [szx, szy, szz] size of the window for cross-correlation 
    @param[in]  test_loc:       (ndim) float, [x, y, z] of the testing localization
    @param[in]  test_photon:    float, the photon number of the testing localization
    @param[in]  kdt_node:       *kd_node, pointer to the root of a kdtree that stors the pool localizations
    @param[in]  src_nspots:     int, number of the localizations in the pool
    @param[in]  src_locs:       (nspots * ndim) float, [[x, y, z],...] coordinates of the pool localizations 
    @param[in]  src_photons:    (nspots) float, photon number of each localization in the pool
    @param[out] c_counts:       (windowsz ** ndim) the A-cross-B-correlation profile
*/
static void ccts_construct(int ndim, int* windowsz, float* test_loc, float test_photon, 
    kd_node* kdt_node, int src_nspots, float* src_locs, float* src_photons, float* c_counts)
{
    int outWindowChk = 0, dumaxis = 0, dumidx = 0, ax = 0, i = 0;
    int pxID[MAXDIM] = { 0 }, ccId = 0; 
    float tmp_pos = 0.0f;

    dumaxis = kdt_node->axis;
    if (dumaxis == -1) {
        for(i = 0; i < kdt_node->nspots; i++) {
            dumidx = kdt_node->indices[i];
            for (ax = 0, outWindowChk = 0; ax < ndim; ax++) {
                tmp_pos = src_locs[dumidx * ndim + ax] - test_loc[ax] + (float)windowsz[ax] / 2.0f;
                if (tmp_pos < 0.0f || tmp_pos >= (float)windowsz[ax]) {
                    outWindowChk = 1;
                    break;
                }
                else
                    pxID[ax] = (int)tmp_pos;
            }
                
            if (outWindowChk == 0) {
                for (ax = ndim - 1, ccId = 0; ax >= 0; ax--)
                    ccId = ccId * windowsz[ax] + pxID[ax];
                c_counts[ccId] += test_photon * src_photons[dumidx];
            }     
        }
        return;
    }
    else { 
        if (kdt_node->left && test_loc[dumaxis] <= kdt_node->divider + (float)windowsz[dumaxis] / 2.0f)
            ccts_construct(ndim, windowsz, test_loc, test_photon, kdt_node->left, src_nspots, src_locs, src_photons, c_counts);
        if (kdt_node->right && test_loc[dumaxis] > kdt_node->divider - (float)windowsz[dumaxis] / 2.0f)
            ccts_construct(ndim, windowsz, test_loc, test_photon, kdt_node->right, src_nspots, src_locs, src_photons, c_counts);
    }
}



/*! @brief   computes the cross-correlation (upto 3 dimentions) via sparse matrix
    @param[in]  ndim:       int, number of dimention
    @param[in]  windowsz:   (ndim) int, [szx, szy, szz] size of the window for cross-correlation 
    @param[in]  nspotsA:    int, number of detections in channel A
    @param[in]  locsA:      (nspotsA * ndim) float, localizations [[x, y, z],...] of each localization in channel A
    @param[in]  photonsA:   (nspotsA) float, the photon number of each localization in channel A 
    @param[in]  nspotsB:    int, number of detections in channel B
    @param[in]  locsB:      (nspotsB * ndim) float, localizations [[x, y, z],...] of each localization in channel B
    @param[in]  photonsB:   (nspotsB) float, the photon number of each localization in channel B
    @param[in]  leafLim:    int, maximum number of element of a leaf node
    @param[out] cc:         (windowsz ** ndim) the A-cross-B-correlation profile
*/
CDLL_EXPORT void kernel(int ndim, int* windowsz, int nspotsA, float* locsA, float* photonsA, 
    int nspotsB, float* locsB, float* photonsB, int leafLim, float* cc)
{
    float test_loc[MAXDIM] = {0.0f};

    // creat a kd-tree for locsB
    int *foo_indices = (int*)malloc(nspotsB * sizeof(int));
    float *foo_locs = (float*)malloc(nspotsB * ndim * sizeof(float));
    kd_node *kdtroot_B = (kd_node*)malloc(sizeof(kd_node));

    memcpy(foo_locs, locsB, nspotsB * ndim * sizeof(float));
    for (unsigned int i = 0; i < nspotsB; i++)
        foo_indices[i] = i;
    kdtroot_B = kdtree_construct(0, nspotsB-1, 0, ndim, nspotsB, foo_locs, foo_indices, leafLim);

    // cross-correlation
    for (unsigned int Idx_A = 0; Idx_A < nspotsA; Idx_A++) {
        memcpy(test_loc, locsA + Idx_A * ndim, ndim * sizeof(float));
        ccts_construct(ndim, windowsz, test_loc, photonsA[Idx_A], kdtroot_B, nspotsB, locsB, photonsB, cc);
    }

    // free up space on heap
    free(foo_locs);
    free(foo_indices);
    kdtree_destruct(kdtroot_B);
}