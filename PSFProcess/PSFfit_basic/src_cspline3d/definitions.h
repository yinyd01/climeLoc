#pragma once
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define BLCKSZ 64

#define IMIN 0.001f      
#define IMAX 1e15f
#define BGMIN 0.0f
#define BGMAX 1e5f

// xvec = [x, y, z, I, b]
#define NDIM 3
#define VNUM 5

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