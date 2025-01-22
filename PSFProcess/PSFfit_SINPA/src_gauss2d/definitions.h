#pragma once
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define BLCKSZ 64

#define pi 3.1415927f
#define SQRTTWO 1.4142135623731f
#define invSQRTTWOPI 0.398942280401432678f

#define IMIN 0.001f
#define IMAX 1e15f
#define BGMIN 0.0001f
#define BGMAX 1e5f

/*! @brief: construction of the xvec, with respect to nchannels and linkMod
    kernel_1ch:     xvec = [x, y, N, b] 
    kernel_2ch:     xvec = [x, y, N, rN, b_ch1, b_ch2]    
*/
#define NDIM 2
#define VNUM_1CH 4
#define VNUM_2CH 6

#define ACCEPTANCE 1.01f
#define OPTTOL 1e-6
#define INIT_ERR 1e13
#define INIT_LAMBDA 0.1f
#define SCALE_UP 10
#define SCALE_DOWN 0.1f

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#endif