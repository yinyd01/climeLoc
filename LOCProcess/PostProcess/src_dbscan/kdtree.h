#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "definitions.h"

#ifndef KDTREE_H
#define KDTREE_H



/*! @brief kdtree for the input coordinates
    @param[in]  ndim:       int, number of dimensions
    @param[in]  nspots:     int, number of coordinates
    @param[in]  coords:     (nspots * ndim) float, the ndim coordinates
    @param[in]  indices:    (nspots) int, indices of the coords
*/

/*! @brief  Swap the i-th and j-th elements of the coords and their original idx */
void swap(int i, int j, int ndim, int nspots, float* coords, int* indices)
{
    float tmp_coord[MAXDIM] = {0.0f};
    int tmp_indice = 0;
    
    memcpy(tmp_coord, coords + i * ndim, ndim * sizeof(float));
    memcpy(coords + i * ndim, coords + j * ndim, ndim * sizeof(float));
    memcpy(coords + j * ndim, tmp_coord, ndim * sizeof(float));

    tmp_indice = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp_indice;    
}



/*! @brief In-place partition the input coords, indices at the median
    @param[in]  start:      int, the starting index to perform the partition-at-median of the coords and indices accordingly (0 for root node) 
    @param[in]  end:        int, the end index to perform the partition-at-median of the coords and indices accordingly ((total-number-of-spots - 1) for root node)
    @param[in]  axis:       int, the axis to determing the pivot
    @return:    md:         the index of the median 
*/
int partition_mid(const int start, const int end, const int axis, int ndim, int nspots, float* coords, int* indices)
{
    int l = start, r = end;
    const int md = (end + start) >> 1;
    float pivot = 0.0;
    int store = 0, i = 0;

    if (end < start)
        return -1;

    while (end > start) {
        pivot = coords[md * ndim + axis];
        swap(md, r, ndim, nspots, coords, indices);
        for (i = l, store = l; i < r; i++) 
            if (coords[i * ndim + axis] <= pivot) {
                if (i != store)
                    swap(i, store, ndim, nspots, coords, indices);
                store++;
            }
        swap(store, r, ndim, nspots, coords, indices);

        if (store == md)
            break;
        if (store > md) 
            r = store - 1;
        else 
            l = store + 1;
    }
    return md;
}



typedef struct node{
    int axis;               // the axis of the current node along which the division is set, -1 for leaf node 
    float divider;          // the division val for the left and right children, 0.0 for leaf node
    int nspots;             // number of indices in this node, 0 for non-leaf node
    int *indices;           // pointer to the original indices (axis_0) of the coords in this node, null for non-leaf node
    struct node *left;      // pointer to the left child, null for leaf node
    struct node *right;     // pointer to the right child, null for leaf node
} kd_node;



/*! @brief  Recursively build a KD-Tree by partitioning-at-median of the coordinates along the given axis
            See kd_node for member properties of the node for non-leaf and leaf nodes.
    @param[in]  start:      int, the starting index to perform the partition-at-median of the coords and indices accordingly (0 for root node) 
    @param[in]  end:        int, the end index to perform the partition-at-median of the coords and indices accordingly ((total-number-of-spots - 1) for root node)
    @param[in]  axis:       int, the axis to determing the divider (0 to start at the root node)
    @param[in]  spotsLim:   int, maxmimum number of spots to make a leaf node
    @return:    this_node:  pointer to the current node.
*/
kd_node* kdtree_construct(int start, int end, int axis, const int ndim, const int nspots, float* coords, int* indices, const int spotsLim) 
{
    kd_node *thisNode = (kd_node*)malloc(sizeof(kd_node));
    int md = 0, dum_nspots = end - start + 1;
    
    if (dum_nspots <= spotsLim) {
        thisNode->axis = -1;
        
        thisNode->divider = 0.0;
        thisNode->left = NULL;
        thisNode->right = NULL;

        thisNode->nspots = dum_nspots;
        thisNode->indices = (int*)malloc(dum_nspots * sizeof(int));
        memcpy(thisNode->indices, indices+start, dum_nspots * sizeof(int));
    }
    else {
        axis %= ndim;
        thisNode->axis = axis;

        md = partition_mid(start, end, axis, ndim, nspots, coords, indices);        
        thisNode->divider = coords[md * ndim + axis];
        thisNode->left = kdtree_construct(start, md, axis+1, ndim, nspots, coords, indices, spotsLim);
        thisNode->right = kdtree_construct(md+1, end, axis+1, ndim, nspots, coords, indices, spotsLim);

        thisNode->nspots = 0;
        thisNode->indices = NULL;
    }
    return thisNode;
}



kd_node *kdtree_build(const int ndim, const int nspots, const float* coords)
{
    // Build a kdtree for the input coordinates
    
    int *indices = (int*)malloc(nspots * sizeof(int));
    float *foo_coords =(float*)malloc(nspots * ndim * sizeof(float));
    kd_node *kdtroot = (kd_node*)malloc(sizeof(kd_node));

    memcpy(foo_coords, coords, nspots * ndim * sizeof(float));
    for (unsigned int i = 0; i < nspots; i++)
        indices[i] = i;
    
    kdtroot = kdtree_construct(0, nspots-1, 0, ndim, nspots, foo_coords, indices, LEAFLIM);
    
    free(indices);
    free(foo_coords);
    return kdtroot;
}



/*! @brief Recursively deallocalte a kdtree*/
void kdtree_destruct(kd_node* node) 
{
    if (node == NULL)
        return; 
    
    kdtree_destruct(node->left);
    kdtree_destruct(node->right);
    
    if (node->axis == -1) // deallocate the indices in the leaf node as well
        free(node->indices);
    free(node);
}



void print_tree(kd_node* node)
{
    if (node == NULL)
        return;
    
    print_tree(node->left);
    print_tree(node->right);
    if (node->axis == -1) {
        printf("axis: %d\t", node->axis);
        printf("indices: ");
        for (int i = 0; i < node->nspots; i++)
            printf("%d, ", node->indices[i]);
        printf("\n");
    }
    else
        printf("axis: %d\tnspots: %d\tdivider: %f\n", node->axis, node->nspots, node->divider);
}

#endif