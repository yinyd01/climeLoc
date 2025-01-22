#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*! @struct binary search tree node. */
typedef struct node_s{
    int             data;           // node data
    char            color;          // node color (used in red-black tree)
    struct node_s*  parent;         // pointer to its parrent node (NULL for root)
    struct node_s*  left;           // pointer to its left child
    struct node_s*  right;          // pointer to its right child
} node_t;


/*! @brief make a node. */
static node_t* mknode(int p_data)
{
    node_t* newnode = (node_t*)malloc(sizeof(node_t));
    newnode->data = p_data;
    newnode->color = 'B';
    newnode->parent = newnode->left = newnode->right = NULL;
    return newnode;
}


/*! @return the left-most node of the current mode. */
static node_t* get_leftmost(node_t* current_node)
{
    while (current_node->left)
        current_node = current_node->left;
    return current_node;
} 


/*! @return the right-most node. */
static node_t* get_rightmost(node_t* current_node)
{
    while (current_node->right)
        current_node = current_node->right;
    return current_node;
}


/*! @return the inorder successor node of the current node. */
static node_t* get_successor(node_t* current_node)
{
    if (current_node->right)
        return get_leftmost(current_node->right);

    node_t *successor = current_node->parent;
    while (successor && current_node == successor->right)
    {
        current_node = successor;
        successor = current_node->parent;
    }      
    return successor;
}


/*! @return the inorder predecessor of the current node. */
static node_t* get_predecessor(node_t* current_node)
{
    if (current_node->left)
        return get_rightmost(current_node->left);
    
    node_t *predecessor = current_node->parent;
    while (predecessor && current_node == predecessor->left)
    {
        current_node = predecessor;
        predecessor = current_node->parent;
    }
    return predecessor;
}


/*! @brief  inorder traversal of a bst (non-recursive). */
static void inorder_transversal(node_t* root)
{
    node_t *current_node = get_leftmost(root);
    while(current_node)
    {
        printf("%d ", current_node->data);
        current_node = get_successor(current_node);
    }
    return;
}


/*! @brief  reverse transversal of a bst. */
static void reverse_transversal(node_t* root)
{
    node_t *current_node = get_rightmost(root);
    while (current_node)
    {
        printf("%d ", current_node->data);
        current_node = get_predecessor(current_node);
    }
    return;
}


/*! @brief  destruct a bst. */
static void bst_destruct(node_t* root)
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
    // node_t instantiation
    node_t *nodeA = mknode(5); 
    node_t *nodeB = mknode(1); 
    node_t *nodeC = mknode(8); 
    node_t *nodeD = mknode(0); 
    node_t *nodeE = mknode(3); 
    node_t *nodeF = mknode(6); 
    node_t *nodeG = mknode(2); 
    node_t *nodeH = mknode(4); 
    node_t *nodeI = mknode(7);
    
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
    
    node_t *successor = get_successor(nodeA);
    node_t *predecessor = get_predecessor(nodeA);
    printf("pred->val->suc = %d->%d->%d\n", predecessor->data, nodeA->data, successor->data);
    

    printf("inorder transversal: ");
    inorder_transversal(nodeA);
    printf("\n", nodeA->data);
    
    printf("reverse transversal: ");
    reverse_transversal(nodeA);
    printf("\n");

    
    bst_destruct(nodeA);    
    
    return 0;
}