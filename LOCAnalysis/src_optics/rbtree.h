#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define COUNT 10

void dis2d();

/*! @struct A Red-Black tree node structure. */
typedef struct node{
    int             data;       // key data
    char            color;      // color
    struct node*    left;       // pointer to the left child
    struct node     *right;     // pointer to the right child
    struct node     *parent;    // pointer to the parent
} node_t;



/*! @brief  make a new node. */
static node_t* mknode(int p_data)
{
    node_t *newnode = (node_t*)malloc(sizeof(node_t));
    if (newnode != NULL)
    {
        newnode->data = p_data;
        newnode->color = 'R';
        newnode->left = newnode->right = newnode->parent = NULL;
    }
    else
        printf("fialed to allocate a new node.\n");
    return newnode;
}



/*! @brief  recursive function for BST insertion. */
/*! @return the pointer to the newnode. */
static node_t* bst_insert(node_t* trav, node_t* newnode)
{
    if (trav == NULL)
        return newnode;
    
    if (newnode->data < trav->data)
    {
        trav->left = bst_insert(trav->left, newnode);
        trav->left->parent = trav;
    }
    else if (newnode->data > trav->data)
    {   
        trav->right = bst_insert(trav->right, newnode);
        trav->right->parent = trav;
    }
    return trav;
}



/*! @brief  Left Rotation at the pivot. */
static void LeftRotate(node_t* root, node_t* pivot)
{
    if (pivot == NULL || pivot->right == NULL)
        return;
    
    // y stored pointer of right child of pivot
    node_t *y = pivot->right;

    // store y's left subtree's pointer as pivot's right child
    pivot->right = y->left;

    // update parent pointer of pivot's right
    if (pivot->right != NULL)
        pivot->right->parent = pivot;

    // update y's parent pointer
    y->parent = pivot->parent;

    // if pivot's parent is null make y as root of tree
    if (pivot->parent == NULL)
        root = y;

    // store y at the place of pivot
    else if (pivot == pivot->parent->left)
        pivot->parent->left = y;
    else    
        pivot->parent->right = y;

    // make pivot as left child of y
    y->left = pivot;

    //update parent pointer of pivot
    pivot->parent = y;

    return;
}



/*! @brief  Right Rotation at the pivot. */
static void rightRotate(node_t *root, node_t *pivot)
{
    if (pivot == NULL || pivot->left == NULL)
        return;
    
    node_t *x = pivot->left;
    
    pivot->left = x->right;
    if (pivot->left)
        pivot->left->parent = pivot;
    
    x->parent = pivot->parent;
    if (pivot->parent == NULL)
        root = x;
    else if (pivot == pivot->parent->left)
        pivot->parent->left = x;
    else 
        pivot->parent->right = x;
    
    x->right = pivot;
    pivot->parent = x;
    
    return;
}



// Utility function to fixup the Red-Black tree after standard BST insertion
static void insertFixUp(struct node* root,struct node* z)
{
    // iterate until z is not the root and z's parent color is red
    while (z != root && z != root->left && z != root->right && z->parent->color == 'R')
    {
        struct node *y;

        // Find uncle and store uncle in y
        if (z->parent && z->parent->parent && z->parent == z->parent->parent->left)
            y = z->parent->parent->right;
        else
            y = z->parent->parent->left;

        // If uncle is RED, do following
        // (i)  Change color of parent and uncle as BLACK
        // (ii) Change color of grandparent as RED
        // (iii) Move z to grandparent
        if (!y)
            z = z->parent->parent;
        else if (y->color == 'R')
        {
            y->color = 'B';
            z->parent->color = 'B';
            z->parent->parent->color = 'R';
            z = z->parent->parent;
        }

        // Uncle is BLACK, there are four cases (LL, LR, RL and RR)
        else
        {
            // Left-Left (LL) case, do following
            // (i)  Swap color of parent and grandparent
            // (ii) Right Rotate Grandparent
            if (z->parent == z->parent->parent->left &&
                z == z->parent->left)
            {
                char ch = z->parent->color ;
                z->parent->color = z->parent->parent->color;
                z->parent->parent->color = ch;
                rightRotate(root,z->parent->parent);
            }

            // Left-Right (LR) case, do following
            // (i)  Swap color of current node  and grandparent
            // (ii) Left Rotate Parent
            // (iii) Right Rotate Grand Parent
            if (z->parent && z->parent->parent && z->parent == z->parent->parent->left &&
                z == z->parent->right)
            {
                char ch = z->color ;
                z->color = z->parent->parent->color;
                z->parent->parent->color = ch;
                LeftRotate(root,z->parent);
                rightRotate(root,z->parent->parent);
            }

            // Right-Right (RR) case, do following
            // (i)  Swap color of parent and grandparent
            // (ii) Left Rotate Grandparent
            if (z->parent && z->parent->parent &&
                z->parent == z->parent->parent->right &&
                z == z->parent->right)
            {
                char ch = z->parent->color ;
                z->parent->color = z->parent->parent->color;
                z->parent->parent->color = ch;
                LeftRotate(root,z->parent->parent);
            }

            // Right-Left (RL) case, do following
            // (i)  Swap color of current node  and grandparent
            // (ii) Right Rotate Parent
            // (iii) Left Rotate Grand Parent
            if (z->parent && z->parent->parent && z->parent == z->parent->parent->right &&
                z == z->parent->left)
            {
                char ch = z->color ;
                z->color = z->parent->parent->color;
                z->parent->parent->color = ch;
                rightRotate(root,z->parent);
                LeftRotate(root,z->parent->parent);
            }
        }
    }
    root->color = 'B'; //keep root always black
}



/*! @brief  insert a new node into a RedBlack tree.  */
static void insert(node_t* root, int data)
{
    
    node_t *newnode = mknode(data);
    
    // if root is null make z as root
    if (root == NULL)
    {
        newnode->color = 'B';
        root = newnode;
    }
    else
    {
        node_t *y = NULL;
        node_t *x = root;

        // iterative BST insert
        while (x != NULL)
        {
            y = x;
            x = (newnode->data < x->data) ? x->left : x->right;
        } 
        newnode->parent = y;
        if (newnode->data > y->data)
            y->right = newnode;
        else
            y->left = newnode;
        
        // assign the new node as 'RED'
        newnode->color = 'R';

        // call insertFixUp to fix reb-black tree's property if it is voilated due to insertion.
        insertFixUp(root, newnode);
    }
}








/*! @brief  traverse Red-Black tree in inorder fashion. */
static void inorder(node_t* root)
{
    static int last = 0;
    
    if (root == NULL)
        return;
    
    inorder(root->left);
    printf("%d ", root->data);
    if (root->data < last)
        printf("\nPUTE\n");
    
    last = root->data;
    inorder(root->right);
}




int main()
{
struct node* root=NULL;
printf("R&B");
while(1)
{
printf("\nenter operations\n1->insert 2->display 3->2d display\n");
int ch;
scanf("%d",&ch);
switch(ch)
{
case 0:return 0;
case 1:{
printf("enter data to insert\n");
int data;
scanf("%d",&data);
insert(root,data);
}break;
case 2:inorder(root);break;
case 3:dis2d(root,0);break;
default:printf("please enter valid inputs");
}
}

}
void dis2d(struct node *root,int space)
{
if(root)
{
space+=COUNT;
dis2d(root->right,space);
printf("\n\n");
for(int i= COUNT; i<space; i++)
{
printf(" ");
}
printf("%d(%c)",root->data,root->color);
dis2d(root->left,space);
}
}