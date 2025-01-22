#pragma once
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "definitions.h"

#ifndef LINKED_LIST_H
#define LINKED_LIST_H


/*! @struct a singly linked list. */
typedef struct sll_s{
    int                     nnodes;                 // number of the nodes int he singly-linked-list
    sll_node_t*             front;                  // pointer to the front node of the singly-linked-list
    sll_node_t*             back;                   // pointer to the end node of the singly-linked-list
} sll_t;



/*! @brief  sll_init an empty sll. */
sll_t* sll_init()
{
    sll_t* newsll = (sll_t*)malloc(sizeof(sll_t));
    if (newsll == NULL)
    {
        printf("failed to allocate a singly linked list.\n");
        return newsll;
    }
    newsll->nnodes = 0;
    newsll->front = NULL;
    newsll->back = NULL;
    return newsll;
}


/*! @brief  make a new sllnode. */
sll_node_t* sll_newnode(bst_node_t* p_bst_node)
{
    sll_node_t *newnode = (sll_node_t*)malloc(sizeof(sll_node_t));
    bst_node_t *m_bst_node = (bst_node_t*)malloc(sizeof(bst_node_t));
    if (newnode == NULL || m_bst_node == NULL)
    {
        printf("failed to allocate a sll node for a singly linked list.\n");
        return NULL;
    }

    m_bst_node->loc_Idx = p_bst_node->loc_Idx;
    m_bst_node->color = p_bst_node->color;
    m_bst_node->parent = p_bst_node->parent;
    m_bst_node->left = p_bst_node->left;
    m_bst_node->right = p_bst_node->right;

    newnode->bst_node = m_bst_node;
    newnode->next = NULL;
    
    return newnode;
} 


/*! @brief  attach a newnode at the begining of the sll (stack push). */
void sll_push_front(sll_t* sllist, bst_node_t* bst_node)
{
    if (sllist == NULL)
        return;

    sll_node_t *newnode = sll_newnode(bst_node);
    if (sllist->front == NULL || sllist->back == NULL)
    {
        sllist->front = newnode;
        sllist->back = newnode;
        sllist->nnodes = 1;
        return;
    }
    
    newnode->next = sllist->front;
    sllist->front = newnode;
    sllist->nnodes += 1;
    return;
}


/*! @brief  append a newnode at the end of the sll (enqueue). */
void sll_push_back(sll_t* sllist, bst_node_t* bst_node)
{
    if (sllist == NULL)
        return;
        
    sll_node_t *newnode = sll_newnode(bst_node);
    if (sllist->front == NULL || sllist->back == NULL)
    {
        sllist->front = newnode;
        sllist->back = newnode;
        sllist->nnodes = 1;
        return;
    }
    
    sllist->back->next = newnode;
    sllist->back = newnode;
    sllist->nnodes += 1;
    return;
}


/*! @brief  get a copy of the front item from the sll. */
sll_node_t* sll_get_front(sll_t* sllist)
{
    if (sllist == NULL || sllist->front == NULL || sllist->back == NULL)
        return NULL;
    
    sll_node_t *front_node = sll_newnode(sllist->front->bst_node);
    front_node->next = sllist->front->next;
    return front_node;
}


/*! @brief  pop out the front item from the sll (stack pop or dequeue). */
void sll_pop_front(sll_t* sllist)
{
    if (sllist == NULL || sllist->front == NULL || sllist->back == NULL)
        return;
    
    sll_node_t *current = sllist->front;
    if (sllist->front == sllist->back)
    {
        sllist->front = NULL;
        sllist->back = NULL;
        sllist->nnodes = 0;
    }
    else
    {
        sllist->front = sllist->front->next;
        sllist->nnodes -= 1;
    }
    free(current);
    current = NULL;   
    return;
}


/*! @brief  delete a node with data == input data*/
void sll_delete_node(sll_t* sllist, int loc_Idx)
{
    if (sllist == NULL || sllist->front == NULL || sllist->back == NULL)
        return;
    if (sllist->front == sllist->back && sllist->front->bst_node->loc_Idx != loc_Idx)
        return;

    sll_node_t *previous = NULL;
    sll_node_t *current = sllist->front;
    
    // only one node and is the one to be delted
    if (sllist->front == sllist->back && sllist->front->bst_node->loc_Idx == loc_Idx)
    {
        sllist->front = NULL;
        sllist->back = NULL;
        sllist->nnodes = 0;
        free(current);
        return;
    }
 
    while(current && current->bst_node->loc_Idx != loc_Idx)
    {
        previous = current;
        current = current->next;
    }

    if (current == NULL)
        printf("failed to find the data %d in the list.\n", loc_Idx);
    else if (current == sllist->front)
    {
        sllist->front = current->next;
        free(current);
        current = NULL;
        sllist->nnodes -= 1;
    }
    else if (current == sllist->back)
    {
        previous->next = NULL;
        sllist->back = previous;
        free(current);
        current = NULL;
        sllist->nnodes -= 1;
    }
    else
    {
        previous->next = current->next;
        free(current);
        current = NULL;
        sllist->nnodes -= 1;
    }
    return;
}


/*! @brief  sll_reverse a sllist. */
void sll_reverse(sll_t* sllist)
{
    if (sllist == NULL || sllist->front == NULL || sllist->front == sllist->back)
        return;

    sll_node_t *previous = NULL;
    sll_node_t *current = sllist->front;
    sll_node_t *preceding = sllist->front->next;

    sllist->back = sllist->front;
    while (preceding)
    {
        current->next = previous;
        previous = current;
        current = preceding;
        preceding = preceding->next;
    }
    current->next = previous;
    sllist->front = current;
    return;
}


/*! @brief  destruct a sllist. */
void sllnode_destruct(sll_node_t* sllnode)
{
    if (sllnode == NULL)
        return;

    sll_node_t *current = NULL;
    while(sllist->front)
    {
        current = sllist->front;
        sllist->front = sllist->front->next;
        free(current);
        current = NULL;
    }
    return;
}


/*! @brief  destruct a sllist. */
void sll_destruct(sll_t* sllist)
{
    if (sllist == NULL)
        return;

    sll_node_t *current = NULL;
    while(sllist->front)
    {
        current = sllist->front;
        sllist->front = sllist->front->next;
        free(current);
        current = NULL;
    }
    return;
}


/*! @brief  print the singly linked list. */
void print_sll(sll_t* sllist)
{
    if (sllist == NULL || sllist->front == NULL)
        return;
    
    sll_node_t *current = sllist->front;
    while (current)
    {
        printf("%d ", current->bst_node->loc_Idx);
        current = current->next;
    }
    printf("\n");
    return;
}

#endif