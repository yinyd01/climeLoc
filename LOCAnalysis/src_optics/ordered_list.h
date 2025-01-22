#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "definitions.h"

#ifndef ORDERED_LIST_H
#define ORDERED_LIST_H


/*! @struct node struct of an order_list containing the index, core distance, reachable distance of a localization, and pointer to its successor. */
typedef struct optics_node_s{
    int                     loc_idx;                // the index of the localization
    float                   core_distance;          // the core_distance of the localization
    float                   reachable_distance;     // the reachable_distance of the localization
    struct optics_node_s    *next;                  // pointer to its successor
} optics_node_t;



/*! @struct an ordered list of the optics_node containg the pointer to its start and the end, and the number of the nodes. */
typedef struct optics_ol_s{
    int                     nnodes;                 // number of nodes within the sequence
    optics_node_t           *head;                  // pointer to the start of the sequence
    optics_node_t           *tail;                  // pointer to the end of the sequence
} optics_ol_t;



/*! @brief  construct an optics node. */
optics_node_t *mknode(int p_loc_idx, float p_core_distance, float p_reachable_distance)
{
    optics_node_t *optics_node = (optics_node_t*)malloc(sizeof(optics_node_t));
    if (optics_node)
    {
        optics_node->loc_idx = p_loc_idx;
        optics_node->core_distance = p_core_distance;
        optics_node->reachable_distance = p_reachable_distance;
        optics_node->next = NULL;
    }
    else
        printf("Failed to allocate optics_node for loc_idx %d\n.", p_loc_idx);
    return optics_node;
}



/*! @brief  initialize the ordered list of the nodes for optics. */
optics_ol_t *optics_ol_init()
{
    optics_ol_t *optics_ol = (optics_ol_t*)malloc(sizeof(optics_ol_t));
    if (optics_ol)
    {
        optics_ol->nnodes = 0;
        optics_ol->head = NULL;
        optics_ol->tail = NULL;    
    }
    else
        printf("Failed to allocate optics_ol.\n");
    return optics_ol;
}



/*! @brief  pop out the first node of the optics_ol. */
void optics_ol_popfirst(optics_ol_t *optics_ol)
{
    optics_node_t *dum = NULL;

    if (optics_ol == NULL || optics_ol->nnodes == 0)
        return;
    
    if (optics_ol->nnodes == 1)
    {
        free(optics_ol->head);
        optics_ol->head = NULL;
        optics_ol->tail = NULL;
        optics_ol->nnodes = 0;
        return;
    }
    
    dum = optics_ol->head;
    optics_ol->head = optics_ol->head->next;
    optics_ol->nnodes -= 1;
    free(dum);  
    return;
}



/*! @brief insert a node into the reachable_distance-ordered optics_ol. */
/*! @note  assumuing there does not exist nodes having the same loc_idx as that of the new node. */
void optics_ol_insert(optics_ol_t* optics_ol, int loc_idx, float core_distance, float reachable_distance)
{
    optics_node_t *optics_node = mknode(loc_idx, core_distance, reachable_distance);
    optics_node_t *dum0 = NULL, *dum1 = NULL;
    
    if (optics_ol == NULL || optics_node == NULL)
        return;
    
    // insert in empty chain
    if (optics_ol->nnodes == 0) // empty chain
    {
        optics_ol->head = optics_node;
        optics_ol->tail = optics_node;
        optics_ol->nnodes = 1;
        return;
    }
    
    // insert in non-empty chain
    dum0 = NULL;
    dum1 = optics_ol->head;
    while (dum1 && reachable_distance >= dum1->reachable_distance)
    {
        dum0 = dum1;
        dum1 = dum1->next;
    }
    if (dum1 == NULL) // append at the end
    {
        dum0->next = optics_node;
        optics_ol->tail = optics_node;
    } 
    else if (dum0 == NULL) // put at the begining
    {
        optics_node->next = dum1;
        optics_ol->head = optics_node;
    }
    else // insert in the middle
    {
        dum0->next = optics_node;
        optics_node->next = dum1;
    }
    optics_ol->nnodes += 1;
    return;
}



/*! @brief  insert a node into the reachable_distance-ordered optics_ol. */
/*! @note   if there exists a node having the same loc_idx with the new node, the new node must have shorter reachable distance, otherwise will not be inserted */
void optics_ol_moveup(optics_ol_t* optics_ol, int loc_idx, float core_distance, float reachable_distance)
{
    optics_node_t *optics_node = mknode(loc_idx, core_distance, reachable_distance);
    optics_node_t *dum0 = NULL, *dum1 = NULL;
    
    if (optics_ol == NULL || optics_node == NULL)
        return;
    
    // insert in empty chain
    if (optics_ol->nnodes == 0) // empty chain
    {
        optics_ol->head = optics_node;
        optics_ol->tail = optics_node;
        optics_ol->nnodes = 1;
        return;
    }
    
    // insert in non-empty chain
    dum0 = NULL;
    dum1 = optics_ol->head;
    while (dum1 && reachable_distance >= dum1->reachable_distance)
    {
        if (loc_idx == dum1->loc_idx)
            return;
        dum0 = dum1;
        dum1 = dum1->next;
    }
    if (dum1 == NULL) // append at the end
    {
        dum0->next = optics_node;
        optics_ol->tail = optics_node;
        optics_ol->nnodes += 1;
        return;
    } 
    if (dum0 == NULL) // put at the begining
    {
        optics_node->next = dum1;
        optics_ol->head = optics_node;
    }
    else // insert in the middle
    {
        dum0->next = optics_node;
        optics_node->next = dum1;
    }
    dum0 = optics_node;
    
    // delete the old node if it exists
    while (dum1 && dum1->loc_idx != optics_node->loc_idx)
    {
        dum0 = dum1;
        dum1 = dum1->next;
    }
    if (dum1 == NULL) // does not exist
    {
        optics_ol->nnodes += 1;
        return;
    }   
    if (dum1 == optics_ol->tail) // tail node
    {
        optics_ol->tail = dum0;
        dum0->next = NULL;
    } 
    else
        dum0->next = dum1->next;
    free(dum1);
    return;
}



/*! @brief  destruct an optics ordered list*/
void optics_ol_destruct(optics_ol_t *optics_ol)
{
    if (optics_ol == NULL)
        return;

    optics_node_t *dum = NULL; 
    optics_node_t *optics_node = optics_ol->head;
    while (optics_node) 
    {
        dum = optics_node->next;
        free(optics_node);
        optics_node = dum;
    }
    free(optics_ol);
    return;
}



/*! @brief  print an optics ordered list*/
static void optics_ol_print(optics_ol_t* chain)
{
    optics_node_t *dum = chain->head;
    while(dum)
    {
        printf("%d:%3.2f-->", dum->loc_idx, dum->reachable_distance);
        dum = dum->next;
    }
    printf("\n");
    return;    
}

#endif