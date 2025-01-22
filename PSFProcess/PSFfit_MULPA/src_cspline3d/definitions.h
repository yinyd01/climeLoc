#pragma once
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define BLCKSZ 64

#define PI 3.14159265f
#define IMIN 0.001f      
#define IMAX 1e15f
#define BGMIN 0.0f
#define BGMAX 1e5f
#define LMBDA_MAX 1e7f

/*! @brief: construction of the xvec, with respect to nchannels and linkMod
    if (nchannels == 1)
        kernel_1ch:         xvec = [nnum * [x, y, z, N], b] 
    else if (nchahnels == 2)
        kernel_2ch:         xvec = [nnum * [x, y, z, N, rN], b_ch1, b_ch2]    
*/
#define NDIM 3
#define NMAX 7
#define PSFSIG 1.5

#define VNUM1_1CH 5
#define VNUM2_1CH 9
#define VNUM3_1CH 13
#define VNUM4_1CH 17
#define VNUM5_1CH 21

#define VNUM1_2CH 7
#define VNUM2_2CH 12
#define VNUM3_2CH 17
#define VNUM4_2CH 22
#define VNUM5_2CH 27  

#define ACCEPTANCE 1.01f
#define OPTTOL 1e-6

#define INIT_ERR 1e13
#define INIT_LAMBDA 0.1f
#define SCALE_UP 10.0f
#define SCALE_DOWN 0.1f

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#endif