#include <stdlib.h>
#include <string.h>
#include <math.h>


/////////////// corners ////////////////
/*! @brief  check if the upper-north-west corner (indz=0, indy=0, indx=0) is the local max of the input image */
static bool ismax_unw(int imszz, int imszy, int imszx, float* img, float* neighbors)
{
    const int indz = 0, zs = 0, ze = 2;
    const int indy = 0, ys = 0, ye = 2;
    const int indx = 0, xs = 0, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx]; 
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 7 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 7; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the upper-north-east corner (indz=0, indy=0, indx=imszx-1) is the local max of the input image */
static bool ismax_une(int imszz, int imszy, int imszx, float* img, float* neighbors)
{
    const int indz = 0, zs = 0, ze = 2;
    const int indy = 0, ys = 0, ye = 2;
    const int indx = imszx-1, xs = -1, xe = 1; 
    
    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 7 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 7; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the upper-south-west corner (indz=0, indy=imszy-1, indx=0) is the local max of the input image */
static bool ismax_usw(int imszz, int imszy, int imszx, float* img, float* neighbors)
{
    const int indz = 0, zs = 0, ze = 2;
    const int indy = imszy-1, ys = -1, ye = 1;
    const int indx = 0, xs = 0, xe = 2;

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 7 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 7; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the upper-south-east corner (indz=0, indy=imszy-1, indx=imszx-1) is the local max of the input image */
static bool ismax_use(int imszz, int imszy, int imszx, float* img, float* neighbors)
{
    const int indz = 0, zs = 0, ze = 2;
    const int indy = imszy-1, ys = -1, ye = 1;
    const int indx = imszx-1, xs = -1, xe = 1; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 7 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 7; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the bottom-north-west corner (indz=imszz-1, indy=0, indx=0) is the local max of the input image */
static bool ismax_bnw(int imszz, int imszy, int imszx, float* img, float* neighbors)
{
    const int indz = imszz-1, zs = -1, ze = 1;
    const int indy = 0, ys = 0, ye = 2;
    const int indx = 0, xs = 0, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 7 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 7; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the bottom-north-east corner (indz=imszz-1, indy=0, indx=imszx-1) is the local max of the input image */
static bool ismax_bne(int imszz, int imszy, int imszx, float* img, float* neighbors)
{
    const int indz = imszz-1, zs = -1, ze = 1;
    const int indy = 0, ys = 0, ye = 2;
    const int indx = imszx-1, xs = -1, xe = 1; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 7 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 7; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the bottom-south-west corner (indz=imszz-1, indy=imszy-1, indx=0) is the local max of the input image */
static bool ismax_bsw(int imszz, int imszy, int imszx, float* img, float* neighbors)
{
    const int indz = imszz-1, zs = -1, ze = 1;
    const int indy = imszy-1, ys = -1, ye = 1;
    const int indx = 0, xs = 0, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 7 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 7; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the bottom-south-east corner (indz=imszz, indy=imszy-1, indx=imszx-1) is the local max of the input image */
static bool ismax_bse(int imszz, int imszy, int imszx, float* img, float* neighbors)
{
    const int indz = imszz-1, zs = -1, ze = 1;
    const int indy = imszy-1, ys = -1, ye = 1;
    const int indx = imszy-1, xs = -1, xe = 1; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 7 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 7; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


//////////////// edges ///////////////////
/*! @brief  check if the upper-north edge (indz=0, indy=0, indx as input) is the local max of the input image */
static bool ismax_un(int imszz, int imszy, int imszx, int indx, float* img, float* neighbors)
{
    const int indz = 0, zs = 0, ze = 2;
    const int indy = 0, ys = 0, ye = 2;
    const int xs = -1, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx]; 
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the upper-south edge (indz=0, indy=imszy-1, indx as input) is the local max of the input image */
static bool ismax_us(int imszz, int imszy, int imszx, int indx, float* img, float* neighbors)
{
    const int indz = 0, zs = 0, ze = 2;
    const int indy = imszy-1, ys = -1, ye = 1;
    const int xs = -1, xe = 2;

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the bottom-north edge (indz=imszz-1, indy=0, indx as input) is the local max of the input image */
static bool ismax_bn(int imszz, int imszy, int imszx, int indx, float* img, float* neighbors)
{
    const int indz = imszz-1, zs = -1, ze = 1;
    const int indy = 0, ys = 0, ye = 2;
    const int xs = -1, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the bottom-south edge (indz=imszz-1, indy=imszy-1, indx as input) is the local max of the input image */
static bool ismax_bs(int imszz, int imszy, int imszx, int indx, float* img, float* neighbors)
{
    const int indz = imszz-1, zs = -1, ze = 1;
    const int indy = imszy-1, ys = -1, ye = 1;
    const int xs = -1, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the upper-west edge (indz=0, indy as input, indx=0) is the local max of the input image */
static bool ismax_uw(int imszz, int imszy, int imszx, int indy, float* img, float* neighbors)
{
    const int indz = 0, zs = 0, ze = 2;
    const int ys = -1, ye = 2;
    const int indx = 0, xs = 0, xe = 2; 
    
    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the upper-east edge (indz=0, indy as input, indx=imszx-1) is the local max of the input image */
static bool ismax_ue(int imszz, int imszy, int imszx, int indy, float* img, float* neighbors)
{
    const int indz = 0, zs = 0, ze = 2;
    const int ys = -1, ye = 2;
    const int indx = imszx-1, xs = -1, xe = 1; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the bottom-west edge (indz=imszz-1, indy as input, indx=0) is the local max of the input image */
static bool ismax_bw(int imszz, int imszy, int imszx, int indy, float* img, float* neighbors)
{
    const int indz = imszz-1, zs = -1, ze = 1;
    const int ys = -1, ye = 2;
    const int indx = 0, xs = 0, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the bottom-east edge (indz=imszz-1, indy as input, indx=imszx-1) is the local max of the input image */
static bool ismax_be(int imszz, int imszy, int imszx, int indy, float* img, float* neighbors)
{
    const int indz = imszz-1, zs = -1, ze = 1;
    const int ys = -1, ye = 2;
    const int indx = imszx-1, xs = -1, xe = 1; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the north-west cylindar (indz as input, indy=0, indx=0) is the local max of the input image */
static bool ismax_nw(int imszz, int imszy, int imszx, int indz, float* img, float* neighbors)
{
    const int zs = -1, ze = 2;
    const int indy = 0, ys = 0, ye = 2;
    const int indx = 0, xs = 0, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the north-east cylindar (indz as input, indy=0, indx=imszx-1) is the local max of the input image */
static bool ismax_ne(int imszz, int imszy, int imszx, int indz, float* img, float* neighbors)
{
    const int zs = -1, ze = 2;
    const int indy = 0, ys = 0, ye = 2;
    const int indx = imszy-1, xs = -1, xe = 1; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the south-west cylindar (indz as input, indy=imszy-1, indx=0) is the local max of the input image */
static bool ismax_sw(int imszz, int imszy, int imszx, int indz, float* img, float* neighbors)
{
    const int zs = -1, ze = 2;
    const int indy = imszy-1, ys = -1, ye = 1;
    const int indx = 0, xs = 0, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the south-east cylindar (indz as input, indy=imszy-1, indx=imszx-1) is the local max of the input image */
static bool ismax_se(int imszz, int imszy, int imszx, int indz, float* img, float* neighbors)
{
    const int zs = -1, ze = 2;
    const int indy = imszy-1, ys = -1, ye = 1;
    const int indx = imszy-1, xs = -1, xe = 1; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx];
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 11 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 11; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


//////////////// face ///////////////////
/*! @brief  check if the upper face (indz=0, indy, indx as input) is the local max of the input image */
static bool ismax_u(int imszz, int imszy, int imszx, int indy, int indx, float* img, float* neighbors)
{
    const int indz = 0, zs = 0, ze = 2;
    const int ys = -1, ye = 2;
    const int xs = -1, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx]; 
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 17 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 17; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the bottom face (indz=imszz-1, indy, indx as input) is the local max of the input image */
static bool ismax_b(int imszz, int imszy, int imszx, int indy, int indx, float* img, float* neighbors)
{
    const int indz = imszz-1, zs = -1, ze = 1;
    const int ys = -1, ye = 2;
    const int xs = -1, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx]; 
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 17 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 17; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the north face (indy=0, indz, indx as input) is the local max of the input image */
static bool ismax_n(int imszz, int imszy, int imszx, int indz, int indx, float* img, float* neighbors)
{
    const int zs = -1, ze = 2;
    const int indy = 0, ys = 0, ye = 2;
    const int xs = -1, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx]; 
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 17 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 17; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the south face (indy=imszy-1, indz, indx as input) is the local max of the input image */
static bool ismax_s(int imszz, int imszy, int imszx, int indz, int indx, float* img, float* neighbors)
{
    const int zs = -1, ze = 2;
    const int indy = imszy-1, ys = -1, ye = 1;
    const int xs = -1, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx]; 
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 17 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 17; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the west face (indx=0, indz, indy as input) is the local max of the input image */
static bool ismax_w(int imszz, int imszy, int imszx, int indz, int indy, float* img, float* neighbors)
{
    const int zs = -1, ze = 2;
    const int ys = -1, ye = 2;
    const int indx = 0, xs = 0, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx]; 
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 17 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 17; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if the east face (indx=imszx-1, indz, indy as input) is the local max of the input image */
static bool ismax_e(int imszz, int imszy, int imszx, int indz, int indy, float* img, float* neighbors)
{
    const int zs = -1, ze = 2;
    const int ys = -1, ye = 2;
    const int indx = imszz-1, xs = -1, xe = 1; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx]; 
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 17 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 17; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}


/*! @brief  check if a inner voxel is the local max of the input image */
static bool ismax_inner(int imszz, int imszy, int imszx, int indz, int indy, int indx, float* img, float* neighbors)
{
    const int zs = -1, ze = 2;
    const int ys = -1, ye = 2;
    const int xs = -1, xe = 2; 

    const float val = img[indz*imszy*imszx + indy*imszx + indx]; 
    int ind_neighbor = 0;
    bool ismax = true;
    memset(neighbors, 0, 26 * sizeof(float));
    for (int i = zs; i < ze; i++)
        for (int j = ys; j < ye; j++)
            for (int k = xs; k < xe; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                neighbors[ind_neighbor] = img[(indz+i)*imszy*imszx + (indy+j)*imszx + (indx+k)];
                ind_neighbor += 1;
            }
    
    for (unsigned int i = 0; i < 26; i++)
        if (val < neighbors[i]) {
            ismax = false;
            break;
        }
    return ismax; 
}