#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "definitions.h"
#include "DLL_Macros.h"
#include "nbchain.h"
#include "kdtree.h"


/*! @brief  dbscan clustering algorithm, the neighbors are searched in a kdtree of the input coordinates recursively
    @param  ndim:           int, the number of dimentions
    @param  nspots:         int, the number of the localizations
    @param  locs:           (nspots * ndim) float, [[x, y, z],...] the input localizations, each nspots contains the localizations of each detection along one dimention
    @param  epsilon:        (ndim) float, the epsilon radius for dbscan
    @param  minpts:         int, the number of the minimal points to define a cluster
    @param  cluster_idx:    int, the index of a cluster
    @param  loc_idx:        int, the index of a localization
    @param  classification: (nspots) int, the cluster_id for each spots (noise and unclassied is also labeled as NOISE and UNCLASSIFIED, respectively)
    @param  iscore:         (nspots) int, label if the localization is core or not
*/



/*! @brief  recursion of the construct the epsilon neighbor-chain for the loc_idx-th loc    */
static void nbs_search(int ndim, int nspots, int loc_idx, float* locs, kd_node* kdt_node, float* epsilon, nbchain_t* nbs)
{
    float test_loc[MAXDIM] = {0.0f};
    bool isin = true;
    int dumaxis = 0, dumidx = 0;
    int ax = 0, i = 0;
    
    memcpy(test_loc, locs + loc_idx * ndim, ndim * sizeof(float));
    dumaxis = kdt_node->axis;
    if (dumaxis == -1) {   
        for(i = 0; i < kdt_node->nspots; i++) {
            dumidx = kdt_node->indices[i];
            for (ax = 0, isin = true; ax < ndim; ax++)
                if (fabs(locs[dumidx * ndim + ax] - test_loc[ax]) > epsilon[ax]) {
                    isin = false;
                    break;
                }
            if (isin)
                nbchain_append(nbs, dumidx);            
        }
        return;
    }
    else {
        if (kdt_node->left && test_loc[dumaxis] <= kdt_node->divider + epsilon[dumaxis])
            nbs_search(ndim, nspots, loc_idx, locs, kdt_node->left, epsilon, nbs);
        if (kdt_node->right && test_loc[dumaxis] > kdt_node->divider - epsilon[dumaxis])
            nbs_search(ndim, nspots, loc_idx, locs, kdt_node->right, epsilon, nbs);
    }
    return;
}



/*! @brief  get the epsilon-neighbors of the input loc_idx-th localization    */
static nbchain_t *get_nbs(int ndim, int nspots, int loc_idx, kd_node* kdtroot, float* locs, float* epsilon)
{
    nbchain_t *nbs = nbchain_init();
    if (nbs)
        nbs_search(ndim, nspots, loc_idx, locs, kdtroot, epsilon, nbs);
    return nbs;
}  



/*! @brief  Spread the cluster_idx-th cluster at the loc_idx-th localization
            the loc_idx-th localization should belong to the cluster-idx-th cluster */
static void cluster_expand(int ndim, int nspots, int cluster_idx, int loc_idx, nbchain_t* seeds, kd_node* kdtroot, float* locs, float* epsilon, int minpts, 
    int* classification, bool* iscore)
{
    int dumidx = 0;
    nbnode_t *tmpnode = nullptr;
    
    nbchain_t *subseeds = get_nbs(ndim, nspots, loc_idx, kdtroot, locs, epsilon);
    if (subseeds->nspots >= minpts) {
        iscore[loc_idx] = true;
        
        tmpnode = subseeds->head;
        while (tmpnode) {
            dumidx = tmpnode->loc_idx;
            if (classification[dumidx] == NOISE || classification[dumidx] == UNCLASSIFIED) {
                if (classification[dumidx] == UNCLASSIFIED)
                    nbchain_append(seeds, dumidx);
                classification[dumidx] = cluster_idx;
            }
            tmpnode = tmpnode->next;
        }
    }
    nbchain_destruct(subseeds);
    return;
}



CDLL_EXPORT void dbscan(int ndim, int nspots, float *locs, float* epsilon, int minpts, int* classification, bool* iscore)
{
    int cluster_idx = 0, i = 0, j = 0;
    nbnode_t *tmpnode = nullptr;
    nbchain_t *seeds = nullptr;
    kd_node *kdtroot = kdtree_build(ndim, nspots, locs);

    memset(classification, 0, nspots * sizeof(int));
    memset(iscore, 0, nspots * sizeof(bool));
    for (i = 0, cluster_idx = 1; i < nspots; i++) {
        // visited
        if (classification[i] != UNCLASSIFIED)
            continue;

        // get the neighbor set
        seeds = get_nbs(ndim, nspots, i, kdtroot, locs, epsilon);
        
        // noise (maybe board point)
        if (seeds->nspots < minpts) {
            classification[i] = NOISE;
            nbchain_destruct(seeds);
            continue;
        }

        // core point
        classification[i] = cluster_idx;
        iscore[i] = true;
        
        tmpnode = seeds->head;
        while(tmpnode) {
            classification[tmpnode->loc_idx] = cluster_idx;
            tmpnode = tmpnode->next;
        }
        
        tmpnode = seeds->head;
        while(tmpnode) {
            cluster_expand(ndim, nspots, cluster_idx, tmpnode->loc_idx, seeds, kdtroot, locs, epsilon, minpts, classification, iscore);
            tmpnode = tmpnode->next;
        }
        
        nbchain_destruct(seeds);
        cluster_idx += 1;
    }
    kdtree_destruct(kdtroot);
    
    return;
}