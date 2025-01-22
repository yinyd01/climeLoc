#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "definitions.h"


static int biSearch(const int nbins, float* binEdges, float key)
{
    // binary search the bin which the key should be put in
    // binEdges:    [nbins+1] the edges of each bin

    int l = 0, r = nbins-1, md = 0;
    
    if (key < binEdges[0]) return -1;
    if (key >= binEdges[nbins]) return -1;
    
    while (l <= r)
    {
        md = l + (r - l) / 2;
        if (key >= binEdges[md] && key < binEdges[md+1]) return md;
        if (key >= binEdges[md+1]) l = md + 1;
        else r = md - 1;
    }
    return md;
}



static float S_incanvas(float imHeight, float imWidth, float* center, float r)
{
    // calculate the area-on-canvas of a circle with center=[x, y], radius=r
    // 2d only for so far
    
    float x = center[0];
    float y = center[1];
    float a = 0.0f, b = 0.0f, theta = 0.0f, S = 0.0f;

    if (r == 0.0f)
        return 0.0f;

    if (x < 0.0f || x > imWidth || y < 0.0f || y > imHeight)
        return 0.0f;
    
    // upper-left corner
    if (y >= 0.0f && y < r && x >= 0.0f && x < r)
    {
        a = x;
        b = y;
        if (a*a + b*b < r*r)
        {
            theta = 1.5f*PI - acosf(b/r) - acosf(a/r);
            S = 0.5f*theta*r*r + a*b + 0.5f*sqrtf(r*r-b*b)*b + 0.5f*sqrtf(r*r-a*a)*a;
        }
        else
        {
            theta = 2.0f * (PI - acosf(b/r) - acosf(a/r));
            S = 0.5f*theta*r*r + sqrtf(r*r-b*b)*b + sqrtf(r*r-a*a)*a;
        }
    }

    // upper edge
    else if (y >= 0.0f && y < r && x >= r && x < imWidth - r)
    {
        b = y;
        theta = 2.0f * (PI - acosf(b/r));
        S = 0.5f*theta*r*r + sqrtf(r*r-b*b)*b;
    }

    // upper right corner
    else if (y >= 0.0f && y < r && x >= imWidth - r && x <= imWidth)
    {
        a = imWidth - x;
        b = y;
        if (a*a + b*b < r*r)
        {
            theta = 1.5f*PI - acosf(b/r) - acosf(a/r);
            S = 0.5f*theta*r*r + a*b + 0.5f*sqrtf(r*r-b*b)*b + 0.5f*sqrtf(r*r-a*a)*a;
        }
        else
        {
            theta = 2.0f * (PI - acosf(b/r) - acosf(a/r));
            S = 0.5f*theta*r*r + sqrtf(r*r-b*b)*b + sqrtf(r*r-a*a)*a;
        }
    }
    
    // left edge
    else if (y >= r && y < imHeight - r && x >= 0.0f && x < r)
    {
        a = x;
        theta = 2.0f * (PI - acosf(a/r));
        S = 0.5f*theta*r*r + sqrtf(r*r-a*a)*a;
    }

    // right edge
    else if (y >= r && y < imHeight - r && x >= imWidth - r && x <= imWidth)
    {
        a = imWidth - x;
        theta = 2.0f * (PI - acosf(a/r));
        S = 0.5f*theta*r*r + sqrtf(r*r-a*a)*a;
    }

    // lower left corner
    else if (y >= imHeight - r && y <= imHeight && x >= 0.0f && x < r)
    {
        a = x;
        b = imHeight - y;
        if (a*a + b*b < r*r)
        {
            theta = 1.5f*PI - acosf(b/r) - acosf(a/r);
            S = 0.5f*theta*r*r + a*b + 0.5f*sqrtf(r*r-b*b)*b + 0.5f*sqrtf(r*r-a*a)*a;
        }
        else
        {
            theta = 2.0f * (PI - acosf(b/r) - acosf(a/r));
            S = 0.5f*theta*r*r + sqrtf(r*r-b*b)*b + sqrtf(r*r-a*a)*a;
        }
    }

    // lower edge
    else if (y >= imHeight - r && y <= imHeight && x >= r && x < imWidth - r)
    {
        b = imHeight - y;
        theta = 2.0f * (PI - acosf(b/r));
        S = 0.5f*theta*r*r + sqrtf(r*r-b*b)*b;
    }

    // lower right corner
    else if (y >= imHeight - r && y <= imHeight && x >= imWidth - r && x <= imWidth)
    {
        a = imWidth - x;
        b = imHeight - y;
        if (a*a + b*b < r*r)
        {
            theta = 1.5f*PI - acosf(b/r) - acosf(a/r);
            S = 0.5f*theta*r*r + a*b + 0.5*sqrtf(r*r-b*b)*b + 0.5*sqrtf(r*r-a*a)*a;
        }
        else
        {
            theta = 2.0f * (PI - acosf(b/r) - acosf(a/r));
            S = 0.5f*theta*r*r + sqrtf(r*r-b*b)*b + sqrtf(r*r-a*a)*a;
        }
    }
    
    // central
    else S = PI * r * r;
    
    return S;
}