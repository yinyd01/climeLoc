#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "definitions.h"

#ifndef KDTREE_H
#define KDTREE_H

// nspots:  number of the coordinates
// coords:  [ndim * nspots] each nspots contains the localizations of each detection along one dimention
// indices: [nspots] the indices of the coords

void swap(int i, int j, int ndim, int nspots, float* coords, int* indices)
{
    // Swap the i-th and j-th elements of the coords and their original idx
    
    float tmp_coord = 0.0f;
    int tmp_indice = 0;
    
    for (int ax = 0; ax < ndim; ax++)
    {
        tmp_coord = coords[ax * nspots + i];
        coords[ax * nspots + i] = coords[ax * nspots + j];
        coords[ax * nspots + j] = tmp_coord;
    }

    tmp_indice = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp_indice;    
}


int partition_mid(const int start, const int end, const int axis, int ndim, int nspots, float* coords, int* indices)
{
    /* 
        In-place partition the input coords, indices at the median
        Arguments:
            start:      the starting index to perform the partition-at-median of the coords and indices accordingly (0 for root node) 
            end:        the end index to perform the partition-at-median of the coords and indices accordingly ((total-number-of-spots - 1) for root node)
            axis:       the axis to determing the pivot
        RETURN:
            md:         the index of the median 
    */

    int l = start, r = end;
    const int md = (end + start) >> 1;
    float pivot = 0.0;
    int store = 0, i = 0;

    if (end < start)
        return -1;

    while (end > start) 
    {
        pivot = coords[axis * nspots + md];
        swap(md, r, ndim, nspots, coords, indices);
        for (i = l, store = l; i < r; i++) 
            if (coords[axis * nspots + i] <= pivot) 
            {
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


kd_node* kdtree_construct(int start, int end, int axis, const int ndim, const int nspots, float* coords, int* indices, const int spotsLim) 
{
    /*
        Recursively build a K-d Tree by partitioning-at-median of the coordinates along the given axis
        See kd_node for member properties of the node for non-leaf and leaf nodes.
        Arguments:
            start:      the starting index to perform the partition-at-median of the coords and indices accordingly (0 for root node) 
            end:        the end index to perform the partition-at-median of the coords and indices accordingly ((total-number-of-spots - 1) for root node)
            axis:       the axis to determing the divider (0 to start at the root node)
            spotsLim:   maxmimum number of spots to make a leaf node
        RETURN:
            this_node:  pointer to the current node.
    */
    
    kd_node *thisNode = (kd_node*)malloc(sizeof(kd_node));
    int md = 0, dum_nspots = end - start + 1;
    
    if (dum_nspots <= spotsLim) 
    {
        thisNode->axis = -1;
        
        thisNode->divider = 0.0;
        thisNode->left = NULL;
        thisNode->right = NULL;

        thisNode->nspots = dum_nspots;
        thisNode->indices = (int*)malloc(dum_nspots*sizeof(int));
        memcpy(thisNode->indices, indices+start, dum_nspots*sizeof(int));
    }
    else 
    {
        axis %= ndim;
        thisNode->axis = axis;

        md = partition_mid(start, end, axis, ndim, nspots, coords, indices);        
        thisNode->divider = coords[axis * nspots + md];
        thisNode->left = kdtree_construct(start, md, axis+1, ndim, nspots, coords, indices, spotsLim);
        thisNode->right = kdtree_construct(md+1, end, axis+1, ndim, nspots, coords, indices, spotsLim);

        thisNode->nspots = 0;
        thisNode->indices = NULL;
    }
    return thisNode;
}


void kdtree_destruct(kd_node* node) 
{
    // Recursively deallocalte a kdtree
    
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
    if (node->axis == -1)
    {
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