#pragma once
#include <stdlib.h>
#include <string.h>
#include "kdtree.h"
#include "nbchain.h"


static void nbs_search(int ndim, int nspots, int loc_idx, float* locs, kdnode_t* kdt_node, float epsilon, nbchain_t* nbs)
{
    // recursion of the construct the epsilon neighbor-chain for the loc_idx-th loc
    
    float test_loc[MAXDIM] = { 0.0f }, dist = 0.0f;
    int dumaxis = 0, dumidx = 0;
    int ax = 0, i = 0;
    
    for (ax = 0; ax < ndim; ax++)
        test_loc[ax] = locs[ax * nspots + loc_idx];

    dumaxis = kdt_node->axis;
    if (dumaxis == -1) // leaf node
    {   
        for(i = 0; i < kdt_node->nspots; i++)
        {
            dumidx = kdt_node->indices[i];
            for (ax = 0, dist = 0.0f; ax < ndim; ax++)
                dist += (locs[ax * nspots + dumidx] - test_loc[ax]) * (locs[ax * nspots + dumidx] - test_loc[ax]);
            dist = sqrtf(dist);
            
            if (dist < epsilon)
                nbchain_append(nbs, dumidx);            
        }
        return;
    }
    else // non-leaf node
    {
        if (kdt_node->left && test_loc[dumaxis] <= kdt_node->divider + epsilon)
            nbs_search(ndim, nspots, loc_idx, locs, kdt_node->left, epsilon, nbs);
        if (kdt_node->right && test_loc[dumaxis] > kdt_node->divider - epsilon)
            nbs_search(ndim, nspots, loc_idx, locs, kdt_node->right, epsilon, nbs);
    }
    return;
}



static nbchain_t *get_nbs(int ndim, int nspots, int loc_idx, kdnode_t* kdtroot, float* locs, float epsilon)
{
    // get the epsilon-neighbors of the input loc_idx-th localization

    nbchain_t *nbs = nbchain_init();
    if (nbs)
        nbs_search(ndim, nspots, loc_idx, locs, kdtroot, epsilon, nbs);
    
    return nbs;
} 



static void cluster_expand(int ndim, int nspots, int loc_Idx, float* locs, kdnode_t* kdtroot, float epsilon, int minpts,
    float* core_distance, float* reachable_distance, bool* processed)
{
    
}