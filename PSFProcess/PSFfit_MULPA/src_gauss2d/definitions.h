#pragma once
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define BLCKSZ 64

#define PI 3.1415927f
#define SQRTTWO 1.4142135623731f
#define invSQRTTWOPI 0.398942280401432678f

#define IMIN 0.001f      
#define IMAX 1e15f
#define BGMIN 0.0f
#define BGMAX 1e5f
#define LMBDA_MAX 1e7f

/*! @brief: construction of the xvec, with respect to nchannels and linkMod
    if (nchannels == 1)
        kernel_1ch:         xvec = [nnum * [x, y, N], b] 
    else if (nchahnels == 2)
        kernel_2ch:         xvec = [nnum * [x, y, N, rN], b_ch1, b_ch2]    
*/
#define NDIM 2
#define NMAX 7

#define VNUM1_1CH 4
#define VNUM2_1CH 7
#define VNUM3_1CH 10
#define VNUM4_1CH 13
#define VNUM5_1CH 16

#define VNUM1_2CH 6
#define VNUM2_2CH 10
#define VNUM3_2CH 14
#define VNUM4_2CH 18
#define VNUM5_2CH 22   

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