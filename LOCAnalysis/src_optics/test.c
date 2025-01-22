#include <stdio.h>
#include "ordered_list.h"

int main()
{
    optics_ol_t* nbchain = optics_ol_init();
    
    int idx[10] = {3, 7, 5, 4, 8, 0, 6, 9, 1, 2};
    float core_dist[10] = { 0.0f };
    float reachable_dist[10] = {3.5, 7.5, 5.5, 4.5, 8.5, 0.5, 6.5, 9.5, 1.5, 2.5};
    
    for (unsigned int i = 0; i < 10; i++)
        optics_ol_insert(nbchain, idx[i], core_dist[i], reachable_dist[i]);
    
    printf("before moving up\n");
    optics_ol_print(nbchain);    
    
    printf("after moving up\n");
    optics_ol_moveup(nbchain, 9, 0.0, 10.0);
    optics_ol_print(nbchain);

    optics_ol_destruct(nbchain);
    return 0;

}