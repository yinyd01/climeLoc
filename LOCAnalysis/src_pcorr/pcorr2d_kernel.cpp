#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "DLL_Macros.h"

#include "definitions.h"
#include "kdtree.h"
#include "functionals.h"

/*! @brief that compute the pair-correlation (upto 2 dimentions) via sparse matrix
    @param[in]  imHeight:           int, height of the canvas
    @param[in]  imWidth:            int, width of the canvas
    @param[in]  ndim:               int, number of dimention (const int ndim = 2)
    @param[in]  nbins:              int, number of radial bins for radial profile
    @param[in]  binsz:              int, size of each radial bin
    @param[in]  rbinEdges:          (nbins+1) float, user defined radial bins for radial profile 
        
    @param[in]  nspotsA:            int, number of detections in channel A
    @param[in]  locsA:              (ndim * nspotsA) float, each nspots contains the localizations of each detection in channel A along one dimention (arranged as x, y, z,...)  
    @param[in]  photonA:            (nspotsA) float, the photon number of each detection in channel A 
    @param[in]  nspotsB:            int, number of detections in channel B
    @param[in]  locsB:              (ndim * nspotsB) float, each nspots contains the localizations of each detection in channel B along one dimention (arranged as x, y, z,...)
    @param[in]  photonB:            (nspotsB) float, the photon number of each detection in channel B
    
    @param[out] r_norm:             (nbins) float, container for the areas of the rings centered at each locA (radius defined by rbinEdges)
    @param[out] r_counts:           (nbins) float, container for number of emitters within the rings centered at each locA (radius defined by rbinEdges)            
    @param[out] pc:                 (nbins) float, the AcorrB pair-correlation profile
    NOTE:
        imHeight, imWidth, binsz, rbinEdges, locsA(B) should be in the same unit (nm is recommended)
        locsB are stored in a kd-tree (kdtree.h) for each locA to search neighbours
        rings may expand outside the canvas, only in-canvas area of the rings are considered (function S_incanvas in functionals.h)
*/



static void rnorm_edge_construct(int nbins, float* rbinEdges, float imHeight, float imWidth, float* test_loc, float* r_norm)
{
    // calculate the ring areas defined by the rbinEdges
    // rings centered at the test_loc and the radius are defined by rbinEdges[nbins+1]
    
    const float maxR = rbinEdges[nbins];
    float Si = 0.0f, So = 0.0f;
    int Idx_r = 0;

    if (test_loc[0] >= maxR && test_loc[0] <= imWidth - maxR && test_loc[1] >= maxR && test_loc[1] <= imHeight - maxR)
        for (Idx_r = 0; Idx_r < nbins; Idx_r++)
            r_norm[Idx_r] = PI * (rbinEdges[Idx_r + 1] + rbinEdges[Idx_r]) * (rbinEdges[Idx_r + 1] - rbinEdges[Idx_r]);
    else
        for (Idx_r = 0; Idx_r < nbins; Idx_r++)
        {
            Si = S_incanvas(imHeight, imWidth, test_loc, rbinEdges[Idx_r]);
            So = S_incanvas(imHeight, imWidth, test_loc, rbinEdges[Idx_r + 1]);
            r_norm[Idx_r] = So - Si;
        }
}



static void rnorm_binsz_construct(int nbins, float binsz, float imHeight, float imWidth, float* test_loc, float* r_norm)
{
    // calculate the ring areas defined by the rbinEdges
    // rings centered at the test_loc and the radius defined by nbins * binsz
    
    const float maxR = nbins * binsz;
    float Si = 0.0f, So = 0.0f;
    int Idx_r = 0;

    if (test_loc[0] >= maxR && test_loc[0] <= imWidth - maxR && test_loc[1] >= maxR && test_loc[1] <= imHeight - maxR)
        for (Idx_r = 0; Idx_r < nbins; Idx_r++)
            r_norm[Idx_r] = PI * (2.0f * (float)Idx_r + 1.0f) * binsz * binsz;
    else
        for (Idx_r = 0; Idx_r < nbins; Idx_r++)
        {
            Si = S_incanvas(imHeight, imWidth, test_loc, (float)Idx_r * binsz);
            So = S_incanvas(imHeight, imWidth, test_loc, (float)(Idx_r + 1) * binsz);
            r_norm[Idx_r] = So - Si;
        }
}



static void rcts_edge_construct(int ndim, int nbins, float* rbinEdges, float* test_loc, float test_photon, 
    kd_node* kdt_node, int src_nspots, float* src_locs, float* src_photons, float* r_counts)
{
    // count the spots (in the src_locs set) within rings
    // rings centered at the test_loc and radius defined by rbinEdges[nbins+1] 
    
    const float maxR = rbinEdges[nbins]; 
    float dist = 0.0f;
    int dumaxis = 0, dumidx = 0;
    int ax = 0, i = 0, Idx_r = 0; 
    
    dumaxis = kdt_node->axis;
    if (dumaxis == -1) // leaf node
    {   
        for(i = 0; i < kdt_node->nspots; i++)
        {
            dumidx = kdt_node->indices[i];
            for (ax = 0, dist = 0.0f; ax < ndim; ax++)
                dist += (src_locs[ax * src_nspots + dumidx] - test_loc[ax]) * (src_locs[ax * src_nspots + dumidx] - test_loc[ax]);
            dist = sqrtf(dist);
            
            if (dist < maxR)
            {
                Idx_r = biSearch(nbins, rbinEdges, dist);
                r_counts[Idx_r] += test_photon * src_photons[dumidx];
            }     
        }
        return;
    }
    else // non-leaf node
    {
        if (kdt_node->left && test_loc[dumaxis] <= kdt_node->divider + maxR)
            rcts_edge_construct(ndim, nbins, rbinEdges, test_loc, test_photon, kdt_node->left, src_nspots, src_locs, src_photons, r_counts);
        if (kdt_node->right && test_loc[dumaxis] > kdt_node->divider - maxR)
            rcts_edge_construct(ndim, nbins, rbinEdges, test_loc, test_photon, kdt_node->right, src_nspots, src_locs, src_photons, r_counts);
    }
}



static void rcts_binsz_construct(int ndim, int nbins, float binsz, float* test_loc, float test_photon, 
    kd_node* kdt_node, int src_nspots, float* src_locs, float* src_photons, float* r_counts)
{
    // count the spots (in the src_locs set) within rings
    // rings centered at the test_loc and radius defined by rbinEdges[nbins+1] 
    
    const float maxR = nbins * binsz; 
    float dist = 0.0f;
    int dumaxis = 0, dumidx = 0;
    int ax = 0, i = 0, Idx_r = 0; 
    
    dumaxis = kdt_node->axis;
    if (dumaxis == -1) // leaf node
    {   
        for(i = 0; i < kdt_node->nspots; i++)
        {
            dumidx = kdt_node->indices[i];
            for (ax = 0, dist = 0.0f; ax < ndim; ax++)
                dist += (src_locs[ax * src_nspots + dumidx] - test_loc[ax]) * (src_locs[ax * src_nspots + dumidx] - test_loc[ax]);
            dist = sqrtf(dist);
            
            if (dist < maxR)
            {
                Idx_r = (int)(dist / binsz);
                r_counts[Idx_r] += test_photon * src_photons[dumidx];
            }     
        }
        return;
    }
    else // non-leaf node
    {
        if (kdt_node->left && test_loc[dumaxis] <= kdt_node->divider + maxR)
            rcts_binsz_construct(ndim, nbins, binsz, test_loc, test_photon, kdt_node->left, src_nspots, src_locs, src_photons, r_counts);
        if (kdt_node->right && test_loc[dumaxis] > kdt_node->divider - maxR)
            rcts_binsz_construct(ndim, nbins, binsz, test_loc, test_photon, kdt_node->right, src_nspots, src_locs, src_photons, r_counts);
    }
}



CDLL_EXPORT void kernel_edge(float imHeight, float imWidth, int nbins, float* rbinEdges, int nspotsA, float* locsA, float* photonsA, 
    int nspotsB, float* locsB, float* photonsB, int leafLim, float* pc)
{
    // calculate the pair-correlation between setA and setB
    
    const int ndim = 2;
    float test_loc[MAXDIM] = { 0.0f }, dist = 0.0f;
    int Idx_A = 0, ax = 0, i = 0;

    // creat a kd-tree for locsB
    int *foo_indices = (int*)malloc(nspotsB * sizeof(int));
    float *foo_locs = (float*)malloc(ndim * nspotsB * sizeof(float));
    kd_node *kdtroot_B = (kd_node*)malloc(sizeof(kd_node));

    memcpy(foo_locs, locsB, ndim * nspotsB * sizeof(float));
    for (i = 0; i < nspotsB; i++)
        foo_indices[i] = i;
    kdtroot_B = kdtree_construct(0, nspotsB-1, 0, ndim, nspotsB, foo_locs, foo_indices, leafLim);
    
    // pair-correlation
    float *r_norm = (float*)malloc(nbins * sizeof(float));
    float *r_counts = (float*)malloc(nbins * sizeof(float));
    for (Idx_A = 0; Idx_A < nspotsA; Idx_A++)
    {
        for (ax = 0; ax < ndim; ax++)
            test_loc[ax] = locsA[ax * nspotsA + Idx_A];
        
        // area of the rings
        memset(r_norm, 0, nbins * sizeof(float));
        rnorm_edge_construct(nbins, rbinEdges, imHeight, imWidth, test_loc, r_norm);
        
        // counts of the spots in rings
        memset(r_counts, 0, nbins * sizeof(float));
        rcts_edge_construct(ndim, nbins, rbinEdges, test_loc, photonsA[Idx_A], kdtroot_B, nspotsB, locsB, photonsB, r_counts);
        
        for (i = 0; i < nbins; i++)
            if (r_norm[i] > 0)
                pc[i] += r_counts[i] / r_norm[i];
    }

    // free up space on heap
    free(foo_locs);
    free(foo_indices);
    kdtree_destruct(kdtroot_B);
    free(r_norm);
    free(r_counts);
}



CDLL_EXPORT void kernel_binsz(float imHeight, float imWidth, int nbins, float binsz, int nspotsA, float* locsA, float* photonsA, 
    int nspotsB, float* locsB, float* photonsB, int leafLim, float* pc)
{
    // calculate the A-cross-B correlation via sparse matrix
    
    const int ndim = 2;
    float test_loc[MAXDIM] = { 0.0f }, dist = 0.0f;
    int Idx_A = 0, ax = 0, i = 0;

    // creat a kd-tree for locsB
    int *foo_indices = (int*)malloc(nspotsB * sizeof(int));
    float *foo_locs = (float*)malloc(ndim * nspotsB * sizeof(float));
    kd_node *kdtroot_B = (kd_node*)malloc(sizeof(kd_node));

    memcpy(foo_locs, locsB, ndim * nspotsB * sizeof(float));
    for (i = 0; i < nspotsB; i++)
        foo_indices[i] = i;
    kdtroot_B = kdtree_construct(0, nspotsB-1, 0, ndim, nspotsB, foo_locs, foo_indices, leafLim);

    // pair-correlation
    float *r_norm = (float*)malloc(nbins * sizeof(float));
    float *r_counts = (float*)malloc(nbins * sizeof(float));
    for (Idx_A = 0; Idx_A < nspotsA; Idx_A++)
    {
        for (ax = 0; ax < ndim; ax++)
            test_loc[ax] = locsA[ax * nspotsA + Idx_A];
            
        // area of the rings
        memset(r_norm, 0, nbins * sizeof(float));
        rnorm_binsz_construct(nbins, binsz, imHeight, imWidth, test_loc, r_norm);
        
        // counts of the spots in rings
        memset(r_counts, 0, nbins * sizeof(float));
        rcts_binsz_construct(ndim, nbins, binsz, test_loc, photonsA[Idx_A], kdtroot_B, nspotsB, locsB, photonsB, r_counts);
        
        for (i = 0; i < nbins; i++)
            if (r_norm[i] > 0)
                pc[i] += r_counts[i] / r_norm[i];
    }

    // free up space on heap
    free(foo_locs);
    free(foo_indices);
    kdtree_destruct(kdtroot_B);
    free(r_norm);
    free(r_counts);
}