import numpy as np


def im_seg2d(im, boxsz, indx, indy, offset=0.0, a2d=1.0):
    """
    Segment box-squares with the given x- and y-centers from a 2d image
    INPUT:
        im:             (imszy, imszx) ndarray, the image for segmentation
        boxsz:          int, size of segmented square
        indx:           (nspots,) int ndarray, centers (x-axis) of segmented squares
        indy:           (nspots,) int ndarray, centers (y-axis) of segmented squares
        offset:         (imszy, imszx)  float ndarray or float, the offset for the input image
        a2d:            (imszy, imszx)  float ndarray or float, the a2d for the input image
    RETURN:
        SegStack:       (nspots, boxsz, boxsz) ndarray, segmented square stack
    """
    
    assert boxsz%2 != 0, 'not odd: boxsz={}, should be odd'.format(boxsz)
    boxhsz = boxsz//2    
    
    dum_im = (np.float32(im) - offset) / a2d
    
    nspots = len(indx)
    SegStack = np.zeros((nspots, boxsz, boxsz), dtype=np.float32)
    for i in range(nspots):
        SegStack[i] = dum_im[indy[i]-boxhsz:indy[i]+boxhsz+1, indx[i]-boxhsz:indx[i]+boxhsz+1]
    
    return SegStack



def ims_seg2d(ims, boxsz, indx, indy, indf, offset=0.0, a2d=1.0):
    """
    Segment box-squares with the given x- and y-centers and the frame index from a stack of 2d images 
    INPUT:
        ims:            (nfrm, imszy, imszx) ndarray, images going to be segmented
        boxsz:          int, size of segmented square
        indx:           (nspots,) int ndarray, centers (x-axis) of segmented squares
        indy:           (nspots,) int ndarray, centers (y-axis) of segmented squares
        indf:           (nspots,) int ndarray, frame index (w.r.t. ims) of each indx and indy pair
        offset:         (imszy, imszx)  float ndarray or float, the offset for the input image
        a2d:            (imszy, imszx)  float ndarray or float, the a2d for the input image
    RETURN:
        SegStack:       (nspots, boxsz, boxsz) ndarray, segmented square stack
    """
    
    assert boxsz%2 != 0, 'not odd: boxsz={}, should be odd'.format(boxsz)
    boxhsz = boxsz//2    
    
    dum_ims = (np.float32(ims) - offset) / a2d

    nspots = len(indx)
    SegStack = np.zeros((nspots, boxsz, boxsz), dtype=np.float32)
    for i in range(nspots):
        SegStack[i] = dum_ims[indf[i], indy[i]-boxhsz:indy[i]+boxhsz+1, indx[i]-boxhsz:indx[i]+boxhsz+1]
    
    return SegStack



def ims_seg2df(ims, nfrm, boxsz, indx, indy, indf, offset=0.0, a2d=1.0):
    """
    Segment nfrm-box-cubes with the given x- and y-centers and the frame index from a stack of 2d images 
    INPUT:
        ims:            (nfrm, imszy, imszx) ndarray, images going to be segmented
        nfrm:           int, number of frames for the nfrm-box-cube
        boxsz:          int, size of segmented square
        indx:           (nspots,) int ndarray, centers (x-axis) of segmented squares
        indy:           (nspots,) int ndarray, centers (y-axis) of segmented squares
        indf:           (nspots,) int ndarray, frame index (w.r.t. ims) of each indx and indy pair
        offset:         (imszy, imszx)  float ndarray or float, the offset for the input image
        a2d:            (imszy, imszx)  float ndarray or float, the a2d for the input image
    RETURN:
        SegStack:       (nspots, nfrm, boxsz, boxsz) ndarray, segmented square stack
    """
    
    if boxsz%2 == 0:
        raise ValueError('not odd: boxsz={}, should be odd'.format(boxsz))
    boxhsz = boxsz//2    
    
    dum_ims = (np.float32(ims) - offset) / a2d
    
    nframes = len(dum_ims)
    nspots = len(indx)
    SegStack = np.zeros((nspots, nfrm, boxsz, boxsz), dtype=np.float32)
    for i in range(nspots):
        if indf[i] < nfrm // 2:
            indf[i] = nfrm // 2
        elif indf[i] > nframes - nfrm + nfrm // 2:
            indf[i] = nframes - nfrm + nfrm // 2
        t0 = indf[i] - nfrm // 2
        te = t0 + nfrm
        SegStack[i] = dum_ims[t0:te, indy[i]-boxhsz:indy[i]+boxhsz+1, indx[i]-boxhsz:indx[i]+boxhsz+1]
    
    return SegStack



def ims_seg3d(ims, boxsz, indx, indy, indz, zrange=None, offset=0.0, a2d=1.0):
    """
    Segment box-squared z-stacks with the given x-, y-, and z-centers from a 3D z-stack images
    INPUT:
        ims:            (imszz, imszy, imszx) ndarray, z-stack image going to be segmented
        boxsz:          int, size of segmented square
        indx:           (nspots,) int ndarray, centers (x-axis) of segmented squares
        indy:           (nspots,) int ndarray, centers (y-axis) of segmented squares
        indz:           (nspots,) int ndarray, centers (z-axis) of segmented squares
        zrange:         int, axial size, indz-zrange//2:indz+zrange//2+1 will be segmented
                        (zmin, zmax) typle, zmin:zmax will be segmented (indz ignored)
                        None, all z slices will be segmented (indz ignored)
        offset:         (imszy, imszx) float ndarray or float, the offset for the input image
        a2d:            (imszy, imszx) float ndarray or float, the a2d for the input image
    RETURN:
        SegStack:       (nspots, zrange, boxsz, boxsz) ndarray, stack of the segmented square stack
    """
    
    if boxsz%2 == 0:
        raise ValueError('not odd: boxsz={}, should be odd'.format(boxsz))
    boxhsz = boxsz//2   

    imszz = len(ims)
    dum_ims = (np.float32(ims) - offset) / a2d
    
    nspots = len(indx)
    if isinstance(zrange, (int, np.int32, np.int64)):
        SegStack = np.zeros((nspots, zrange, boxsz, boxsz), dtype=np.float32)
        for i in range(nspots):
            SegStack[i] = dum_ims[indz[i]-zrange//2:indz[i]+zrange//2+1, indy[i]-boxhsz:indy[i]+boxhsz+1, indx[i]-boxhsz:indx[i]+boxhsz+1]
    elif isinstance(zrange, (tuple, list, np.ndarray)):
        zmin, zmax = zrange
        SegStack = np.zeros((nspots, zmax-zmin, boxsz, boxsz), dtype=np.float32)
        for i in range(nspots):
            SegStack[i] = dum_ims[zmin:zmax, indy[i]-boxhsz:indy[i]+boxhsz+1, indx[i]-boxhsz:indx[i]+boxhsz+1]
    else:
        SegStack = np.zeros((nspots, imszz, boxsz, boxsz), dtype=np.float32)
        for i in range(nspots):
            SegStack[i] = dum_ims[:, indy[i]-boxhsz:indy[i]+boxhsz+1, indx[i]-boxhsz:indx[i]+boxhsz+1]
    
    return SegStack