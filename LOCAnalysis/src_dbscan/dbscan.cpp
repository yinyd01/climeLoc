#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "definitions.h"
#include "DLL_Macros.h"
#include "nbchain.h"
#include "kdtree.h"


/*
    dbscan clustering algorithm
    the neighbors are searched in a kdtree of the input coordinates recursively
    Arguments in General:
        ndim:           number of dimentions
        nspots:         number of the localizations
        locs:           [ndim * nspots] the input localizations, each nspots contains the localizations of each detection along one dimention
        epsilon:        the epsilon radius for dbscan
        minpts:         the number of the minimal points to define a cluster

        cluster_idx:    the index of a cluster
        loc_idx:        the index of a localization

        classification: [nspots] the cluster_id for each spots (noise and unclassied is also labeled as NOISE and UNCLASSIFIED, respectively)
        iscore:         [nspots] label if the localization is core or not
*/


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



static void cluster_expand(int ndim, int nspots, int cluster_idx, int loc_idx, nbchain_t* seeds, kdnode_t* kdtroot, float* locs, float epsilon, int minpts, 
    int* classification, bool* iscore)
{
    // Spread the cluster_idx-th cluster at the loc_idx-th localization
    // the loc_idx-th localization should belong to the cluster-idx-th cluster
    
    int dumidx = 0;
    nbnode_t *tmpnode = nullptr;
    
    nbchain_t *subseeds = get_nbs(ndim, nspots, loc_idx, kdtroot, locs, epsilon);
    if (subseeds->nspots >= minpts)
    {
        iscore[loc_idx] = true;
        
        tmpnode = subseeds->head;
        while (tmpnode) 
        {
            dumidx = tmpnode->loc_idx;
            if (classification[dumidx] == NOISE || classification[dumidx] == UNCLASSIFIED) 
            {
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



CDLL_EXPORT void dbscan(int ndim, int nspots, float *locs, float epsilon, int minpts, int* classification, bool* iscore)
{
    // main function for dbscan
    
    int cluster_idx = 0, i = 0, j = 0;
    nbnode_t *tmpnode = nullptr;
    nbchain_t *seeds = nullptr;
    kdnode_t *kdtroot = kdtree_build(ndim, nspots, locs);

    memset(classification, 0, nspots * sizeof(int));
    memset(iscore, 0, nspots * sizeof(bool));
    for (i = 0, cluster_idx = 1; i < nspots; i++)
    {
        // visited
        if (classification[i] != UNCLASSIFIED)
            continue;

        // get the neighbor set
        seeds = get_nbs(ndim, nspots, i, kdtroot, locs, epsilon);
        
        // noise (maybe board point)
        if (seeds->nspots < minpts)
        {
            classification[i] = NOISE;
            nbchain_destruct(seeds);
            continue;
        }

        // core point
        classification[i] = cluster_idx;
        iscore[i] = true;
        
        tmpnode = seeds->head;
        while(tmpnode)
        {
            classification[tmpnode->loc_idx] = cluster_idx;
            tmpnode = tmpnode->next;
        }
        
        tmpnode = seeds->head;
        while(tmpnode)
        {
            cluster_expand(ndim, nspots, cluster_idx, tmpnode->loc_idx, seeds, kdtroot, locs, epsilon, minpts, classification, iscore);
            tmpnode = tmpnode->next;
        }
        
        nbchain_destruct(seeds);
        cluster_idx += 1;
    }
    kdtree_destruct(kdtroot);
    
    return;
}