#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "definitions.h"

#ifndef SINGLY_LINKED_LIST_H
#define SINGLY_LINKED_LIST_H


/*! @struct node struct of an singly linked list containing the index of a localization, and the pointer to its successor. */
typedef struct nb_node_s{
    int                 loc_idx;    // the index of the localization
    struct nb_node_s    *next;      // pointer to its successor
} nb_node_t;



/*! @struct an singly linked list of the node containg the pointer to its start and the end, and the number of the nodes. */
typedef struct nb_sll_s{
    int                 nnodes;     // number of nodes within the sequence
    nb_node_t           *head;      // pointer to the start of the sequence
    nb_node_t           *tail;      // pointer to the end of the sequence
} nb_sll_t;



/*! @brief  construct a neighbor node. */
nb_node_t *mknode(int loc_idx)
{
    nb_node_t *nb_node = (nb_node_t*)malloc(sizeof(nb_node_t));
    if (nb_node)
    {
        nb_node->loc_idx = loc_idx;
        nb_node->next = nullptr;
    }
    else
        printf("Failed to allocate nb_node for loc_idx %d\n.", loc_idx);
    return nb_node;
}



/*! @brief  initialize the singly linked list of the neighbor nodes. */
nb_sll_t *nb_sll_init()
{
    nb_sll_t *nb_sll = (nb_sll_t*)malloc(sizeof(nb_sll_t));
    if (nb_sll)
    {
        nb_sll->nnodes = 0;
        nb_sll->head = nullptr;
        nb_sll->tail = nullptr;    
    }
    else
        printf("Failed to allocate nbchain.\n");
    return nb_sll;
}



/*! @brief  append a neighbor node to the end of the sll. */
void nb_sll_append(nb_sll_t *nb_sll, int loc_idx)
{
    nb_node_t *nb_node = mknode(loc_idx);
    
    if (nb_node == nullptr)
        return;
    
    if (nb_sll->nnodes == 0)
    {
        nb_sll->head = nb_node;
        nb_sll->tail = nb_node;
    } 
    else 
    {
        nb_sll->tail->next = nb_node;
        nb_sll->tail = nb_node;
    }
    nb_sll->nnodes += 1;
    return;
}



/*! @brief  destruct the sll. */
void nbchain_destruct(nb_sll_t *nb_sll)
{
    if (nb_sll == nullptr)
        return;

    nb_node_t *dum = nullptr; 
    nb_node_t *nb_node = nb_sll->head;
    while (nb_node) 
    {
        dum = nb_node->next;
        free(nb_node);
        nb_node = dum;
    }
    free(nb_sll);
    return;
}

#endif