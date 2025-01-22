#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "definitions.h"
#include "linked_list.h"

// Complete Binary Tree is a binary tree that fills a full binary tree in level-order


/*! @brief  make a new sllnode. */
bst_node_t* bst_newnode(int p_loc_Idx)
{
    bst_node_t *newnode = (bst_node_t*)malloc(sizeof(bst_node_t));
    if (newnode == NULL)
    {
        printf("failed to allocate a node for a singly linked list.\n");
        return newnode;
    }
    newnode->loc_Idx    = p_loc_Idx;
    newnode->color      = 'B';
    newnode->parent     = NULL;
    newnode->left       = NULL;
    newnode->right      = NULL;
    return newnode;
} 


/*! @brief  pre-order (VLR) traversal of a bst. */
void bst_VLR_transversal(bst_node_t* current)
{
    if (current)
    {
        printf("%d ", current->loc_Idx);
        bst_VLR_transversal(current->left);
        bst_VLR_transversal(current->right);
    }
    return;
}


/*! @brief  in-order (LVR) traversal of a bst. */
void bst_LVR_transversal(bst_node_t* current)
{
    if (current)
    {
        bst_LVR_transversal(current->left);
        printf("%d ", current->loc_Idx);
        bst_LVR_transversal(current->right);
    }
    return;
}


/*! @brief  post-order (LRV) traversal of a bst. */
void bst_LRV_transversal(bst_node_t* current)
{
    if (current)
    {
        bst_LRV_transversal(current->left);
        bst_LRV_transversal(current->right);
        printf("%d ", current->loc_Idx);
    }
    return;
}


/*! @brief  level-order traversal of a bst. */
void bst_lvorder_transversal(bst_node_t* root)
{
    sll_t       *tmp_sll = sll_init();
    sll_node_t  *current_sllnode = NULL;

    sll_push_back(tmp_sll, root);
    while (tmp_sll->front && tmp_sll)
    {
        //current_sllnode = tmp_sll->front;
        current_sllnode = sll_get_front(tmp_sll);
        sll_pop_front(tmp_sll);
        
        printf("%d ", current_sllnode->bst_node->loc_Idx);
        if (current_sllnode->bst_node->left)
            sll_push_back(tmp_sll, current_sllnode->bst_node->left);
        if (current_sllnode->bst_node->right)
            sll_push_back(tmp_sll, current_sllnode->bst_node->right);
        
        //sll_pop_front(tmp_sll); // the current_sllnode (along with smp_sll->front) is freed during pop_front
    }
    
    sll_destruct(tmp_sll);
    free(tmp_sll);
    tmp_sll = NULL;
    
    return;
}


/*! @return the left-most node of the current mode. */
bst_node_t* bst_get_leftmost(bst_node_t* current)
{
    while (current->left)
        current = current->left;
    return current;
} 


/*! @return the right-most node. */
bst_node_t* bst_get_rightmost(bst_node_t* current)
{
    while (current->right)
        current = current->right;
    return current;
}


/*! @return the inorder successor node of the current node. */
bst_node_t* bst_get_successor(bst_node_t* current)
{
    if (current->right)
        return bst_get_leftmost(current->right);

    bst_node_t *successor = current->parent;
    while (successor && current == successor->right)
    {
        current = successor;
        successor = current->parent;
    }      
    return successor;
}


/*! @return the inorder predecessor of the current node. */
bst_node_t* bst_get_predecessor(bst_node_t* current)
{
    if (current->left)
        return bst_get_rightmost(current->left);
    
    bst_node_t *predecessor = current->parent;
    while (predecessor && current == predecessor->left)
    {
        current = predecessor;
        predecessor = current->parent;
    }
    return predecessor;
}


/*! @brief  inorder traversal of a bst (non-recursive). */
void bst_inorder_transversal(bst_node_t* root)
{
    bst_node_t *current = bst_get_leftmost(root);
    while(current)
    {
        printf("%d ", current->loc_Idx);
        current = bst_get_successor(current);
    }
    return;
}


/*! @brief  reverse transversal of a bst. */
void bst_reverse_transversal(bst_node_t* root)
{
    bst_node_t *current = bst_get_rightmost(root);
    while (current)
    {
        printf("%d ", current->loc_Idx);
        current = bst_get_predecessor(current);
    }
    return;
}


/*! @brief  destruct a bst. */
void bst_destruct(bst_node_t* root)
{
    if (root == NULL)
        return;
    
    bst_destruct(root->left);
    bst_destruct(root->right);
    free(root);
    return;
}




int main() 
{
    // bst_node_t instantiation
    bst_node_t *nodeA = bst_newnode(5); 
    bst_node_t *nodeB = bst_newnode(1); 
    bst_node_t *nodeC = bst_newnode(8); 
    bst_node_t *nodeD = bst_newnode(0); 
    bst_node_t *nodeE = bst_newnode(3); 
    bst_node_t *nodeF = bst_newnode(6); 
    bst_node_t *nodeG = bst_newnode(2); 
    bst_node_t *nodeH = bst_newnode(4); 
    bst_node_t *nodeI = bst_newnode(7);
    
    // construct the Binary Tree
    nodeA->left = nodeB; nodeA->right = nodeC; 
    nodeB->left = nodeD; nodeB->right = nodeE; 
    nodeE->left = nodeG; nodeE->right = nodeH; 
    nodeC->left = nodeF; 
    nodeF->right = nodeI;

    // link parent pointer
    nodeB->parent = nodeA; nodeC->parent = nodeA;
    nodeD->parent = nodeB; nodeE->parent = nodeB;
    nodeG->parent = nodeE; nodeH->parent = nodeE;
    nodeF->parent = nodeC; 
    nodeI->parent = nodeF;
    
    printf("hello world\n");
    
    bst_VLR_transversal(nodeA);
    printf("\n");
    bst_LVR_transversal(nodeA);
    printf("\n");
    bst_LRV_transversal(nodeA);
    printf("\n");
    bst_lvorder_transversal(nodeA);
    printf("\n");
    
    bst_node_t *successor = bst_get_successor(nodeA);
    bst_node_t *predecessor = bst_get_predecessor(nodeA);
    printf("pred->val->suc = %d->%d->%d\n", predecessor->loc_Idx, nodeA->loc_Idx, successor->loc_Idx);
    

    printf("inorder transversal: ");
    bst_inorder_transversal(nodeA);
    printf("\n", nodeA->loc_Idx);
    
    printf("reverse transversal: ");
    bst_reverse_transversal(nodeA);
    printf("\n");

    
    bst_destruct(nodeA);    
    
    return 0;
}