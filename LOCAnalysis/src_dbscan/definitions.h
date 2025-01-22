#pragma once

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define MAXDIM 3
#define LEAFLIM 32

#define UNCLASSIFIED 0
#define NOISE -1
#define SUCCESS 1
#define FAILURE 0

#define PI 3.14159265359f

#ifndef max
    #define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
    #define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#endif