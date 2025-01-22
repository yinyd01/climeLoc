#pragma once

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define MAXDIM 3
#define LEAFLIM 32

#define UNDEFINE_DIST -1.0

#define UNCLASSIFIED 0
#define NOISE -1
#define SUCCESS 1
#define FAILURE 0

#define PI 3.14159265359f


/*! @struct node for a binary-search-tree. */
typedef struct bst_node_s{
    int                     loc_Idx;                // Index of a localization
    char                    color;                  // label the node color if turns red
    struct bst_node_s*      parent;                 // pointer to its parrent node in a red-black tree
    struct bst_node_s*      left;                   // pointer to its left child in a red-black tree
    struct bst_node_s*      right;                  // pointer to its right child in a red-black tree
} bst_node_t;



/*! @struct node for a singly-linked-list, nesting a bst_node */
typedef struct sll_node_s{
    bst_node_t*             bst_node;               // pointer to the nesting binary-search-tree node
    struct sll_node_s*      next;                   // pointer to its next in a linked-list
} sll_node_t;



#ifndef max
    #define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
    #define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

#endif