#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "DLL_Macros.h"
#include "definitions.h"

/*
    transform a 2D matrix into a radial profile by summing up elements with same distance-to-center
    The matrix should be center originated (the origin 0 should be at (int)sz/2 for both axes)
    maxR = (min(imHeight, imWidth)-1)/2+1
    Arguments in general:
        imHeight:           Height of the image
        imWidth:            Width of the image
        mat:                [imHeight * imWidth] the input 2D matrix
        radial:             [maxR] the output radial profile
*/

CDLL_EXPORT void kernel_2d(int imHeight, int imWidth, double* mat, double* radial)
{
    int maxR = (min(imHeight, imWidth) - 1) / 2 + 1, ind_r = 0;
    double delta_x = 0.0, delta_y = 0.0, delta_r = 0.0;

    for (int Idx = 0; Idx < imHeight * imWidth; Idx++) {
        delta_x = (double)(Idx % imWidth - imWidth / 2);
        delta_y = (double)(Idx / imWidth - imHeight / 2);
        delta_r = sqrt(delta_x * delta_x + delta_y * delta_y);
    
        ind_r = (int)(delta_r + 0.5); // round(delta_r)
        if (ind_r < maxR)
            radial[ind_r] += mat[Idx];
    }
    return;
}



/*
    transform a 3D matrix into a radial profile by summing up elements with same distance-to-center
    The matrix should be center originated (the origin 0 should be at (int)sz/2 for both axes)
    maxR = (min(imThickness, imHeight, imWidth)-1)/2+1
    Arguments in general:
        imThickness:abort   Thickness of the image
        imHeight:           Height of the image
        imWidth:            Width of the image
        mat:                [imThickness * imHeight * imWidth] the input 3D matrix
        radial:             [maxR] the output radial profile
*/

CDLL_EXPORT void kernel_3d(int imThickness, int imHeight, int imWidth, double* mat, double* radial)
{
    int maxR = (min(imThickness, min(imHeight, imWidth)) - 1) / 2 + 1, ind_r = 0;
    double delta_x = 0.0, delta_y = 0.0, delta_z = 0.0, delta_r = 0.0;
    int dum = 0;

    for (int Idx = 0; Idx < imThickness * imHeight * imWidth; Idx++){
        
        dum = Idx % (imHeight * imWidth);
        delta_x = (double)(dum % imWidth - imWidth / 2);
        delta_y = (double)(dum / imWidth - imHeight / 2);
        
        dum = Idx / (imHeight * imWidth);
        delta_z = (double)(dum - imThickness / 2);

        delta_r = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
    
        ind_r = (int)(delta_r + 0.5); // round(delta_r)
        if (ind_r < maxR)
            radial[ind_r] += mat[Idx];
    }

    return;
}