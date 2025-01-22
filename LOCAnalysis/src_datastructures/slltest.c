#include "definitions.h"
#include "linked_list.h"

bst_node_t* bst_newnode(int p_loc_Idx)
{
    bst_node_t *newnode = (bst_node_t*)malloc(sizeof(bst_node_t));
    if (newnode == NULL)
    {
        printf("failed to allocate a node for a singly linked list.\n");
        return newnode;
    }
    newnode->loc_Idx = p_loc_Idx;
    newnode->color = 'B';
    newnode->parent = NULL;
    newnode->left = NULL;
    newnode->right = NULL;
    return newnode;
} 


int main() {

    sll_t *sllist = sll_init();     
    print_sll(sllist);
    sll_pop_front(sllist);                      
    print_sll(sllist);

    sll_push_back(sllist, bst_newnode(999));
    print_sll(sllist);
    
    sll_pop_front(sllist);                      
    print_sll(sllist);

    sll_push_back(sllist, bst_newnode(5));
    sll_push_back(sllist, bst_newnode(3));
    sll_push_back(sllist, bst_newnode(9));
    sll_push_back(sllist, bst_newnode(7));
    sll_push_back(sllist, bst_newnode(15));
    print_sll(sllist);

    sll_delete_node(sllist, 15);
    print_sll(sllist);

    sll_delete_node(sllist, 5);
    print_sll(sllist);

    sll_delete_node(sllist, 9);
    print_sll(sllist);
    
    
    sll_push_back(sllist, bst_newnode(104));
    sll_push_back(sllist, bst_newnode(90));
    sll_push_back(sllist, bst_newnode(0));
    sll_push_back(sllist, bst_newnode(2));
    sll_push_back(sllist, bst_newnode(44));
    sll_push_back(sllist, bst_newnode(79));
    print_sll(sllist);
    
    
    sll_push_front(sllist, bst_newnode(8));
    print_sll(sllist);

    sll_reverse(sllist);
    print_sll(sllist);

    sll_pop_front(sllist);
    print_sll(sllist);

    sll_pop_front(sllist);
    print_sll(sllist);

    sll_destruct(sllist);
    free(sllist);
    sllist = NULL;
    print_sll(sllist);

    return 0;
}