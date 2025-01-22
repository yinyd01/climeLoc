#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "DLL_Macros.h"
#include "definitions.h"



/*! @brief  Merge the localizations appear at the same place (localizations are up to 3d)
            spots appearing within the tolerance distance are considered as 'at the same place'
            dist_tol are fixed constants
    @note:  locs should be sorted by locx   
    @param[in]  nspots:     int, number of localizations
    @param[in]  ndim:       int, number of dimension
    @param[in]  locs:       (nspots * ndim) float, the [[x, y, z],...] of each localization
    @param[in]  dist_tol:   (ndim) float, maximum distance allowed to consider same-place
    @param[out] labels:     (nspots) int, labels of the clustered localizations (0 for non-clustered)
*/
CDLL_EXPORT void loc_merge_const(int nspots, int ndim, float* locs, float* dist_tol, int *labels)
{
    int Ind_0 = 0, Ind_1 = 0, clusterID = 0;
    float xyz[NDIM] = {0.0f};
    bool isin = true;
    
    memset(labels, 0, nspots * sizeof(int));
    for (Ind_0 = 0, clusterID = 0; Ind_0 < nspots; Ind_0++) {       
        
        // find a new entry that have not been merged: labels(of this) = 0
        if (labels[Ind_0] > 0)  
            continue;
        
        // store coordinates
        clusterID += 1;
        labels[Ind_0] = clusterID;
        memcpy(xyz, locs + Ind_0 * ndim, ndim * sizeof(float)); 
        
        for (Ind_1 = Ind_0 + 1; Ind_1 < nspots; Ind_1++) {
        
            if (labels[Ind_1] > 0)
                continue;

            // fast index to the possible entries might be close to x and y (locx is ascendingly presorted)
            if (locs[Ind_1 * ndim] < xyz[0] - dist_tol[0])  
                continue;
            
            isin = true;
            for (unsigned int dimID = 0; dimID < ndim; dimID++) 
                if (abs(locs[Ind_1 * ndim + dimID] - xyz[dimID]) > dist_tol[dimID]) {
                    isin = false;
                    break;
                }
            
            if (isin) {
                labels[Ind_1] = clusterID;
                for (unsigned int dimID = 0; dimID < ndim; dimID++)
                    xyz[dimID] = 0.5f * (locs[Ind_1 * ndim + dimID] + xyz[dimID]);
            }
        }
    }
    return;
}



/*! @brief  Merge the localizations appear at the same place (localizations are upto 3d)
            spots appearing within the searching radius are considered as 'at the same place'
            searching radius is determined by the variance of the localizations dX = 5 * var
            dist_tol are the upper limit of dX
    @note:  locs should be sorted by locx 
    @param[in]  nspots:     int, number of localizations
    @param[in]  ndim:       int, number of dimensions
    @param[in]  locs:       (nspots * ndim) float, the [[x, y, z],...] of each localization
    @param[in]  locvar:     (nspots * ndim) float, the variance of each localization
    @param[in]  dist_max:   (ndim) float, maximum distance tolerance allowed to consider same-place
    @param[out] labels:     (nspots) int, label the localizations as emitters (0 for non-clustered)
*/
CDLL_EXPORT void loc_merge_var(int nspots, int ndim, float* locs, float* locvar, float* dist_max, int *labels)
{
    int Ind_0 = 0, Ind_1 = 0, clusterID = 0; 
    float xyz[NDIM] = {0.0f}, var[NDIM] ={0.0f}, dist = 0.0f, dist_tol = 0.0f;
    bool isin = true;

    memset(labels, 0, nspots * sizeof(int));
    for (Ind_0 = 0, clusterID = 0; Ind_0 < nspots; Ind_0++) {       
        
        // find a new entry that have not been connected: labels(of this) = 0
        if (labels[Ind_0] > 0) 
            continue;
        
        // store coordinates
        clusterID += 1;
        labels[Ind_0] = clusterID;
        memcpy(xyz, locs + Ind_0 * ndim, ndim * sizeof(float));
        memcpy(var, locvar + Ind_0 * ndim, ndim * sizeof(float));
        
        for (Ind_1 = Ind_0 + 1; Ind_1 < nspots; Ind_1++) {
            
            if (labels[Ind_1] > 0)
                continue;

            // fast index to the possible entries might be close to x and y (locx is ascendingly presorted)
            if (locs[Ind_1 * ndim] < xyz[0] - dist_max[0]) 
                continue;
            
            isin = true;
            for (unsigned int dimID = 0; dimID < ndim; dimID++) {
                dist = abs(locs[Ind_1 * ndim + dimID] - xyz[dimID]);
                dist_tol = min(sqrt(0.5f * (locvar[Ind_1 * ndim + dimID] + var[dimID])), dist_max[dimID]);
                if (dist > dist_tol) {
                    isin = false;
                    break;
                }
            }
            
            if (isin) {
                labels[Ind_1] = clusterID;
                for (unsigned int dimID = 0; dimID < ndim; dimID++)
                    xyz[dimID] = 0.5f * (locs[Ind_1 * ndim + dimID] + xyz[dimID]);
            }
        }
    }
    return;
}



/*! @brief  Connect the localizations appear at the same place in consecutive frames (locations are upto 3d)
            spots appearing within the dist_tol are considered as 'at the same place'
            spots appearing within consecutive frames with less than gap_tol gaps are considered as 'in consecutive frames'
            dist_tol are fixed constant
    @note:  locs should be primarily sorted by frm and secondarily sorted by locs
    @param[in]  nspots:     int, number of localizations
    @param[in]  ndim:       int, number of dimensions
    @param[in]  frm:        (nspots) int, the frame of each localization
    @param[in]  locs:       (nspots * ndim) float, the [[x, y, z],...] of each localization
    @param[in]  dist_tol:   (ndim) float, maximum radius allowed to consider same-place
    @param[in]  gap_tol:    int, maximum frame gap allowed to consider consecutivity
    @param[out] labels:     (nspots) int, labels of the clustered localizations (0 for non-clustered)
*/
CDLL_EXPORT void loc_connect_const(int nspots, int ndim, int* frm, float* locs, float* dist_tol, int gap_tol, int *labels)
{
    int Ind_0 = 0, Ind_1 = 0, clusterID = 0, frh = 0, frtest = 0;
    int ConnFlag = 0, gap = 0; 
    float xyz[NDIM] = {0.0f}, dist = 0.0f;
    bool isin = true;
    
    Ind_0 = 0;
    clusterID = 0;
    memset(labels, 0, nspots * sizeof(int));
    while (Ind_0 < nspots) {       
        //find a new entry that have not been connected: labels(of this) = 0
        while ((labels[Ind_0] > 0) && (Ind_0 < nspots)) 
            Ind_0 += 1;
          
        if (Ind_0 == nspots)
            break;
        
        //store coordinates
        clusterID += 1;
        labels[Ind_0] = clusterID;
        
        frh = frm[Ind_0];
        memcpy(xyz, locs + Ind_0 * ndim, ndim * sizeof(float));
        
        gap = 0;
        Ind_1 = Ind_0;
        while ( (gap <= gap_tol) && (Ind_1 < nspots) && (frm[Ind_1] >= frh) ) {
            ConnFlag = 0;
        
            // index to the next frame
            while ((frm[Ind_1] == frh) && (Ind_1 < nspots)) 
                Ind_1 += 1;        
            frtest = frh + 1;
            
            // fast index to the possible entries might be close to x and y (locx is ascendingly presorted)
            while ((Ind_1 < nspots) && (locs[Ind_1 * ndim] < xyz[0] - dist_tol[0]) && (frm[Ind_1] == frtest)) 
                Ind_1 += 1;
             
            while ((Ind_1 < nspots) && (locs[Ind_1 * ndim] < xyz[0] + dist_tol[0]) && (frm[Ind_1] == frtest)) {
                isin = true;
                for (unsigned int dimID = 0; dimID < ndim; dimID++) {
                    dist = abs(locs[Ind_1 * ndim + dimID] - xyz[dimID]);
                    if (dist > dist_tol[dimID]) {
                        isin = false;
                        break;
                    }
                }

                if (isin && (labels[Ind_1] == 0)) {
                    ConnFlag = 1;
                    labels[Ind_1] = clusterID;
                    for (unsigned int dimID = 0; dimID < ndim; dimID++)
                        xyz[dimID] = 0.5f * (locs[Ind_1 * ndim + dimID] + xyz[dimID]);
                    frh = frm[Ind_1];
                    gap = 0;
                    break;
                }
                else
                    Ind_1 += 1;
            }
            if (ConnFlag == 0) {
                frh = frtest;
                gap += 1; 
            }
        }
    }
    return;
}



/*! @brief  Connect the localizations appear at the same place in consecutive frames (localizations are upto 3d)
            spots appearing within the searching radius are considered as 'at the same place'
            spots appearing within consecutive frames with less than gap_tol gaps are considered as 'in consecutive frames'
            searching radius is determined by the the variance of the localizations dX = 5 * var
            maxR is the upper limit of dX
    @note:  locs should be primarily sorted by frm and secondarily sorted by locs
    @param[in]  nspots:     int, number of localizations
    @param[in]  ndim:       int, number of dimensions
    @param[in]  frm:        (nspots) int, the frame of each localization
    @param[in]  locs:       (nspots * ndim) float, the [[x, y, z],...] of each localization
    @param[in]  locvar:     (nspots * ndim) float, the variance of each localization 0.5 * (varx + vary)
    @param[in]  dist_max:   (ndim) float, maximum diatance tolorence allowed to consider same-place
    @param[in]  gap_tol:    int, maximum frame gap allowed to consider consecutivity
    @param[out] labels:     (nspots) int, label the localizations as emitters (0 for non-clustered)
*/
CDLL_EXPORT void loc_connect_var(int nspots, int ndim, int* frm, float* locs, float* locvar, float* dist_max, int gap_tol, int *labels)
{
    int Ind_0 = 0, Ind_1 = 0, clusterID = 0, frh = 0, frtest = 0;
    int ConnFlag = 0, gap = 0; 
    float xyz[NDIM] = {0.0f}, var[NDIM] = {0.0f}, dist = 0.0f, dist_tol = 0.0f;
    bool isin = true;

    Ind_0 = 0;
    clusterID = 0;
    memset(labels, 0, nspots * sizeof(int));
    while (Ind_0 < nspots) {       
        //find a new entry that have not been connected: labels(of this) = 0
        while ((labels[Ind_0] > 0) && (Ind_0 < nspots)) 
            Ind_0 += 1;
          
        if (Ind_0 == nspots)
            break;
        
        //store coordinates
        clusterID += 1;
        labels[Ind_0] = clusterID;
        
        frh = frm[Ind_0];
        memcpy(xyz, locs + Ind_0 * ndim, ndim * sizeof(float));
        memcpy(var, locvar + Ind_0 * ndim, ndim * sizeof(float));
        
        gap = 0;
        Ind_1 = Ind_0;
        while ( (gap <= gap_tol) && (Ind_1 < nspots) && (frm[Ind_1] >= frh) ) {
            ConnFlag = 0;
        
            // index to the next frame
            while ((frm[Ind_1] == frh) && (Ind_1 < nspots)) 
                Ind_1 += 1;        
            frtest = frh + 1;
            
            // fast index to the possible entries might be close to x and y (locx is ascendingly presorted)
            while ((Ind_1 < nspots) && (locs[Ind_1 * ndim] < xyz[0] - dist_max[0]) && (frm[Ind_1] == frtest)) 
                Ind_1 += 1;
             
            while ((Ind_1 < nspots) && (locs[Ind_1 * ndim] < xyz[0] + dist_max[0]) && (frm[Ind_1] == frtest)) {
                isin = true;
                for (unsigned int dimID = 0; dimID < ndim; dimID++) {
                    dist = abs(locs[Ind_1 * ndim + dimID] - xyz[dimID]);   
                    dist_tol = min(sqrt(0.5f * (locvar[Ind_1 * ndim + dimID] + var[dimID])), dist_max[dimID]);
                    if (dist > dist_tol) {
                        isin = false;
                        break;
                    }
                }
                
                if (isin && (labels[Ind_1] == 0)) {
                    ConnFlag = 1;
                    labels[Ind_1] = clusterID;
                    for (unsigned int dimID = 0; dimID < ndim; dimID++)
                        xyz[dimID] = 0.5f * (locs[Ind_1 * ndim + dimID] + xyz[dimID]);
                    frh = frm[Ind_1];
                    gap = 0;
                    break;
                }
                else
                    Ind_1 += 1;
            }
            if (ConnFlag==0) {
                frh = frtest;
                gap += 1; 
            }
        }
    }
    return;
}