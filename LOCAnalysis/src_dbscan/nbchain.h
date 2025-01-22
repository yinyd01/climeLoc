#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "definitions.h"

#ifndef NBCHAIN_H
#define NBCHAIN_H

// loc_idx:     index of a localization
// nbnode:      node to store the index of a localization and the pointer to the next node
// nbs:         chain of the nbnodes, to store the pointers to the first and last nbnode, as well as the number of all the nbnodes

typedef struct nbnode_s{
    int loc_idx;
    struct nbnode_s *next;
} nbnode_t;


typedef struct nbchain_s{
    int nspots;
    nbnode_t *head;
    nbnode_t *tail;
} nbchain_t;


nbnode_t *mknode(int loc_idx)
{
    nbnode_t *nbnode = (nbnode_t*)malloc(sizeof(nbnode_t));
    if (nbnode)
    {
        nbnode->loc_idx = loc_idx;
        nbnode->next = nullptr;
    }
    else
        printf("Failed to allocate nbnode for loc_idx %d\n.", loc_idx);
    return nbnode;
}


nbchain_t *nbchain_init()
{
    nbchain_t *nbs = (nbchain_t*)malloc(sizeof(nbchain_t));
    if (nbs)
    {
        nbs->nspots = 0;
        nbs->head = nullptr;
        nbs->tail = nullptr;    
    }
    else
        printf("Failed to allocate nbchain.\n");
    return nbs;
}


void nbchain_append(nbchain_t *nbs, int loc_idx)
{
    nbnode_t *nbnode = mknode(loc_idx);
    
    if (nbnode == nullptr)
        return;
    
    if (nbs->nspots == 0)
    {
        nbs->head = nbnode;
        nbs->tail = nbnode;
    } 
    else 
    {
        nbs->tail->next = nbnode;
        nbs->tail = nbnode;
    }
    nbs->nspots += 1;
    return;
}


void nbchain_destruct(nbchain_t *nbs)
{
    if (nbs == nullptr)
        return;

    nbnode_t *dum = nullptr; 
    nbnode_t *nbnode = nbs->head;
    while (nbnode) 
    {
        dum = nbnode->next;
        free(nbnode);
        nbnode = dum;
    }
    free(nbs);
    return;
}

#endif