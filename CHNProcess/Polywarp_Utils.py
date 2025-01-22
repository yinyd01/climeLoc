import numpy as np
from scipy.ndimage import map_coordinates
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist



def _isperfectsquare(x):
    return np.int_(np.sqrt(x)) * np.int_(np.sqrt(x)) == x if x > 0 else False

def _isperfectcubic(x):
    return np.int_(np.cbrt(x)) * np.int_(np.cbrt(x)) * np.int_(np.cbrt(x)) == x
    
def _upsample(arr, upscalars, norder):
    """
    UPsampling an array by given upscalars via spline interpolation
    INPUT:
        arr:            (szz, szy, szx) int ndarray, the input array for upsampling
        upscalars:      [upscalarz, upscalary, upscalarx] int ndarray, the arr will be upsampled by upfactor times
        norder:         int, the spline order for upsampling, must be integer between 0 and 5
    RETURN:
        arr_up:         (szz*upscalarz, szy*upscalary, szx*upscalarx) ndarray, the upsampled arr
    NOTE:
        scipy.ndimage.map_coordinates sees arr locats at [0, 1, ... sz-1] other than [0.5, 1.5, ... sz-0.5]
        thus the qvecs should also moves 0.5 back before using map_coordinates
    """
    szs = arr.shape
    upscalars = np.asarray(upscalars, dtype=np.int32)
    upszs = szs * upscalars
    if np.all(upscalars <= 1):
        arr_up = np.copy(arr)
    else:
        vecs = [np.linspace(0, sz, upsz, endpoint=False) + 0.5/upscalar - 0.5 for sz, upsz, upscalar in zip(szs, upszs, upscalars)]
        grids = np.meshgrid(*vecs, indexing='ij')
        arr_up = map_coordinates(arr, grids, order=norder, mode='nearest')
    return arr_up

def _downsize(arr, binsz):    
    """
    Downsizing an array by given binsz via averaging
    INPUT:
        arr:        (szz, szy, szx) int ndarray, the input array, can be n-dim
        binsz:      (binszz, binszy, bisnzx) int ndarray, the blocksize on which averaging is to be performed
    RETURN:
        arr_down:   (szz//binszz, szy//binszy, szx//binszx) ndarray, the downsized array
    """
    binsz = np.asarray(binsz)
    sh = np.column_stack([arr.shape//binsz, binsz]).flatten()
    arr_down = arr.reshape(sh).mean(axis=tuple(range(1, 2*arr.ndim, 2)))

    return arr_down



def _transB2A_build(ndim, deg, coor_chB):
    """
    build a transformation matrix transB2A for polynomial transformation from channel B to channel A, so that coeff_B2A.dot(transB2A)=[x_chA, y_chA]
    transB2A (take ndim=3, deg=3 as an example): 
    [   zB^0*yB^0*xB^0, zB^0*yB^0*xB^1, zB^0*yB^0*xB^2, zB^0*yB^1*xB^0, zB^0*yB^1*xB^1, zB^0*yB^1*xB^2, zB^0*yB^2*xB^0, zB^0*yB^2*xB^1, zB^0*yB^2*xB^2,
        zB^1*yB^0*xB^0, zB^1*yB^0*xB^1, zB^1*yB^0*xB^2, zB^1*yB^1*xB^0, zB^1*yB^1*xB^1, zB^1*yB^1*xB^2, zB^1*yB^2*xB^0, zB^1*yB^2*xB^1, zB^1*yB^2*xB^2,
        zB^2*yB^0*xB^0, zB^2*yB^0*xB^1, zB^2*yB^0*xB^2, zB^2*yB^1*xB^0, zB^2*yB^1*xB^1, zB^2*yB^1*xB^2, zB^2*yB^2*xB^0, zB^2*yB^2*xB^1, zB^2*yB^2*xB^2  ]
    INPUT:
        ndim:           dimentionality must be 2 or 3
        deg:            positive integer, degree of the polynomial function
        coor_chB:       (ndim, ncoors) float ndarray, [[x_chB], [y_chB], [z_chB]] coordinates in channel B that is to be transformed to channel A
    RETURN:
        transB2A:       (deg**ndim, ncoors) float ndarray, the transformation matrix as described above
    """
    ncoors = len(coor_chB[0])
    transB2A = np.zeros((deg**ndim, ncoors), dtype=np.float64)
    
    if ndim == 2:
        dumy = np.ones(ncoors, dtype=np.float64)
        for i in range(deg):
            dumx = np.ones(ncoors, dtype=np.float64)
            for j in range(deg):
                transB2A[i * deg + j] = dumy * dumx
                dumx *= coor_chB[0]
            dumy *= coor_chB[1]
    
    elif ndim == 3:
        dumz = np.ones(ncoors, dtype=np.float64)
        for i in range(deg):
            dumy = np.ones(ncoors, dtype=np.float64)
            for j in range(deg):
                dumx = np.ones(ncoors, dtype=np.float64)
                for k in range(deg):
                    transB2A[i * deg * deg + j * deg + k] = dumz * dumy * dumx
                    dumx *= coor_chB[0]
                dumy *= coor_chB[1]
            dumz *= coor_chB[2]

    return transB2A



def _transB2A_build_div(ndim, deg, coor_chB):
    """
    build a transformation matrix transB2A_divx for polynomial transformation from channel B to channel A, so that coeff_B2A.dot(transB2A_div)=[dx_chA/dx_chB, dy_chA/dx_chB]
    INPUT:
        ndim:           dimentionality must be 2 or 3
        deg:            positive integer, degree of the polynomial function
        coor_chB:       (ndim, ncoors) float ndarray, [[x_chB], [y_chB], [z_chB]] coordinates in channel B that is to be transformed to channel A
    RETURN:
        transB2A_divx:  (deg**ndim, ncoors) float ndarray, the transformation_div matrix over x
        transB2A_divy:  (deg**ndim, ncoors) float ndarray, the transformation_div matrix over y
        transB2A_divz:  (deg**ndim, ncoors) float ndarray, the transformation_div matrix over z
    """
    ncoors = len(coor_chB[0])
    transB2A_divx = np.zeros((deg**ndim, ncoors), dtype=np.float64)
    transB2A_divy = np.zeros((deg**ndim, ncoors), dtype=np.float64)
    
    if ndim == 2:    
        dumy = np.ones(ncoors, dtype=np.float64)
        for i in range(deg - 1):
            dumx = np.ones(ncoors, dtype=np.float64)
            for j in range(deg - 1):
                transB2A_divy[(i + 1) * deg + j] = (i + 1) * dumy * dumx
                transB2A_divx[i * deg + (j + 1)] = (j + 1) * dumy * dumx
                dumx *= coor_chB[0]
            dumy *= coor_chB[1]
        return transB2A_divx, transB2A_divy
    
    elif ndim == 3:
        transB2A_divz = np.zeros((deg**ndim, ncoors), dtype=np.float64)
        dumz = np.ones(ncoors, dtype=np.float64)
        for i in range(deg - 1):
            dumy = np.ones(ncoors, dtype=np.float64)
            for j in range(deg - 1):
                dumx = np.ones(ncoors, dtype=np.float64)
                for k in range(deg - 1):
                    transB2A_divz[(i + 1) * deg * deg + j * deg + k] = (i + 1) * dumz * dumy * dumx
                    transB2A_divy[i * deg * deg + (j + 1) * deg + k] = (j + 1) * dumz * dumy * dumx
                    transB2A_divx[i * deg * deg + j * deg + (k + 1)] = (k + 1) * dumz * dumy * dumx
                    dumx *= coor_chB[0]
                dumy *= coor_chB[1]
            dumz *= coor_chB[2]
        return transB2A_divx, transB2A_divy, transB2A_divz



def get_coeff_B2A(ndim, deg, coor_chB, coor_chA):
    """
    calculate the polynomial coefficients [uvec, vvec] so that uvec.dot(transB2A)=x_chA, vvec.dot(transB2A)=y_chA
    see _transB2A_build for the matrix transB2A     
    INPUT:
        ndim:           int, dimentionality must be 2 or 3
        deg:            pint, ositive integer, degree of the polynomial function
        coor_chB:       (ndim, ncoors) float ndarray, [[x_chB], [y_chB], [z_chB]] coordinates in channel B that is to be transformed to channel A
        coor_chA:       (ndim, ncoors) float ndarray, [[x_chA], [y_chA], [z_chA]] coordinates in channel A where the coor_chB is supposed to be transformed to
    RETURN:
        coeff_B2A:      (ndim, deg**ndim**) float ndarray, [[uvec], [vvec], [wvec]], polynomial coefficients.
    """
    transB2A = _transB2A_build(ndim, deg, coor_chB)
    coeff_B2A = lstsq(transB2A.T, coor_chA.T)[0]
    return coeff_B2A.T



def coor_polywarpB2A(ndim, coor_chB, coeff_B2A):
    """
    Polynomial transformation of the coordinates from channel B to channel A
    INPUT:
        ndim:           int, dimentionality must be 2 or 3
        coor_chB:       (ndim, ncoors) float ndarray, [[x_chB], [y_chB], [z_chB]] coordinates in channel B that is to be transformed to channel A
        coeff_B2A:      (ndim, deg**ndim) float ndarray, [[uvec], [vvec], [wvec]] polynomial coefficients.  
    RETURN:
        coor_chA:       (ndim, ncoors) float ndarray, [[x_chA], [y_chA], [z_chA]] coordinates in channel A where the coor_chB is supposed to be transformed to
    """
    if ndim == 2:
        assert _isperfectsquare(len(coeff_B2A[0])), "number of the polynomial coefficients must be a perfect square"
        deg = np.int_(np.sqrt(len(coeff_B2A[0])))
    elif ndim == 3:
        assert _isperfectcubic(len(coeff_B2A[0])), "number of the polynomial coefficients must be a perfect cubic"
        deg = np.int_(np.cbrt(len(coeff_B2A[0])))

    transB2A = _transB2A_build(ndim, deg, coor_chB)
    return coeff_B2A.dot(transB2A) 



def var_polywarpB2A(ndim, var_chB, coor_chB, coeff_B2A):
    """
    Polynomial transformation of the variance of the coordinates from channel B to channel A
    INPUT:
        ndim:           int, dimentionality must be 2 or 3
        var_chB:        (ndim, ncoors) float ndarray, [[varx_chB], [vary_chB], [varz_chB]] variance of the coordinates in channel B that is to be transformed to channel A
        coor_chB:       (ndim, ncoors) float ndarray, [[x_chB], [y_chB], [z_chB]] coordinates in channel B that is to be transformed to channel A
        coeff_B2A:      (ndim, deg**ndim) float ndarray, [[uvec], [vvec], [wvec]] polynomial coefficients.  
    RETURN:
        coor_chA:       (ndim, ncoors) float ndarray, [[varx_chA], [vary_chA], [varz_chA]] variance of the coordinates in channel A where the coor_chB is supposed to be transformed to
    """
    if ndim == 2:
        assert _isperfectsquare(len(coeff_B2A[0])), "number of the polynomial coefficients must be a perfect square"
        deg = np.int_(np.sqrt(len(coeff_B2A[0])))
    elif ndim == 3:
        assert _isperfectcubic(len(coeff_B2A[0])), "number of the polynomial coefficients must be a perfect cubic"
        deg = np.int_(np.cbrt(len(coeff_B2A[0])))

    transB2A_div = _transB2A_build_div(ndim, deg, coor_chB)
    var_chA = np.zeros_like(var_chB)
    for i in range(ndim):
        var_chA += (coeff_B2A.dot(transB2A_div[i]))**2 * var_chB[i]
    return var_chA 



def im_polywarpB2A(im_chB, coeff_B2A, upscalars=[1, 1], offsetx=0, offsety=0):
    """
    Polynomial transformation of the image from channel B to channel A
    2d image only
    INPUT:
        im_chB:         2D-ndarray, image needs to be transformed
        coeff_B2A:      [uvec, vvec], polynomial coefficients.
        upscalars:      [upscalary, upscalarx], int, upscalar during warping, 1 for no upscaling
        offsetx:        int, the offset (x-axis) of the image
        offsety:        int, the offset (y-axis) of the image   
    RETURN:
        im_chA:         2D-ndarray, transformed image
    """
    
    assert _isperfectsquare(len(coeff_B2A[0])), "number of rows of the polyvecs must be a perfect square"
    deg = np.int_(np.sqrt(len(coeff_B2A[0])))

    ndim = 2
    szy, szx = im_chB.shape
    upszy = szy * upscalars[0]
    upszx = szx * upscalars[1]

    coor_chB = np.vstack([np.tile(np.linspace(offsetx, offsetx+szx, upszx, endpoint=False)+0.5/upscalars[1], upszy), 
                          np.repeat(np.linspace(offsety, offsety+szy, upszy, endpoint=False)+0.5/upscalars[0], upszx)])
    transB2A = _transB2A_build(ndim, deg, coor_chB)
    coor_chA = coeff_B2A.dot(transB2A)    
    indx_chA = np.int32((coor_chA[0] - offsetx) * upscalars[1])
    indy_chA = np.int32((coor_chA[1] - offsety) * upscalars[0])
    
    im_chA = np.zeros((upszy, upszx), dtype=np.float32)
    im_chB = _upsample(np.array(im_chB, dtype=np.float32), upscalars, 3)
    for indy in range(upszy):
        for indx in range(upszx):
            ind = indy * upszx + indx
            if (indy_chA[ind]>=0) and (indy_chA[ind]<upszy) and (indx_chA[ind]>=0) and (indx_chA[ind]<upszx):
                im_chA[indy_chA[ind], indx_chA[ind]] = im_chB[indy, indx]
    im_chA = _downsize(im_chA, upscalars)    
    
    return np.asarray(im_chA, dtype=np.uint16)



def ims_polywarpB2A(ims_chB, coeff_B2A, upscalars=[1, 1], offsetx=0, offsety=0):
    """
    Polynomial transformation of the image from channel B to channel A
    INPUT:
        ims_chB:        3D-ndarray, 2d-image stack needs to be transformed
        coeff_B2A:      [uvec, vvec], polynomial coefficients.
        upscalars:      [upscalary, upscalarx], int, upscalar during warping, 1 for no upscaling
        offsetx:        int, the offset (x-axis) of the image
        offsety:        int, the offset (y-axis) of the image   
    RETURN:
        ims_chA:        3D-ndarray, transformed image
    """
    
    if ims_chB.ndim == 2:
        return im_polywarpB2A(ims_chB, coeff_B2A, upscalars, offsetx, offsety)
    assert ims_chB.ndim == 3, "dimension mismatch: ims_chB.ndim={}, required=2 or 3".format(ims_chB.ndim)
    assert _isperfectsquare(len(coeff_B2A[0])), "number of rows of the polyvecs must be a perfect square"
    deg = np.int_(np.sqrt(len(coeff_B2A[0])))

    ndim = 2
    nfrm, szy, szx = ims_chB.shape
    upszy = szy * upscalars[0]
    upszx = szx * upscalars[1]

    coor_chB = np.vstack([np.tile(np.linspace(offsetx, offsetx+szx, upszx, endpoint=False)+0.5/upscalars[1], upszy), 
                          np.repeat(np.linspace(offsety, offsety+szy, upszy, endpoint=False)+0.5/upscalars[0], upszx)])
    transB2A = _transB2A_build(ndim, deg, coor_chB)
    coor_chA = coeff_B2A.dot(transB2A)    
    indx_chA = np.int32((coor_chA[0] - offsetx) * upscalars[1])
    indy_chA = np.int32((coor_chA[1] - offsety) * upscalars[0])
    
    ims_chA = np.zeros((nfrm, szy, szx), dtype=np.uint16)
    for (i, im_chB) in enumerate(ims_chB):
        im_chB = _upsample(np.array(im_chB, dtype=np.float32), upscalars, 3)
        dum_im = np.zeros((upszy, upszx), dtype=np.float32)
        for indy in range(upszy):
            for indx in range(upszx):
                ind = indy * upszx + indx
                if (indy_chA[ind]>=0) and (indy_chA[ind]<upszy) and (indx_chA[ind]>=0) and (indx_chA[ind]<upszx):
                    dum_im[indy_chA[ind], indx_chA[ind]] = im_chB[indy, indx]
        ims_chA[i] = np.uint16(_downsize(dum_im, upscalars))        
    return ims_chA



def nnd_match(coors_0, coors_1, tolr):
    """
    Match the pairs according to the nearest distance between two sets of coordinates
    Brutal force search by assuming the distance_map is sparse enough
    INPUT:
        coors_0:            [x_0, y_0] coordinates from the 1st channel
        coors_1:            [x_1, y_1] coordinates from the 2nd channel
        tolr:               tolorance distance that distances > tolr won't be taken into account
    RETURN:
        matched_coors_0:    matched index in the 1st channel
        matched_coors_1:    matched index in the 2nd channel
        matched_nnd:        distance of each matched pair
    """
    
    # sort the distances to find the neareast (assuming the distance map is sparse with the tolr)
    dist_map = cdist(coors_0.T, coors_1.T, 'euclidean')
    ind_0, ind_1 = np.where(dist_map <= tolr) 
    dists = dist_map[ind_0, ind_1]
    dum_ind = np.argsort(dists)
    ind_0, ind_1, dists = ind_0[dum_ind], ind_1[dum_ind], dists[dum_ind]
    nCandidates = len(dists)

    matched_ind_0 = np.zeros(nCandidates, dtype=np.int32) 
    matched_ind_1 = np.zeros(nCandidates, dtype=np.int32) 
    matched_nnd = np.zeros(nCandidates, dtype=np.float64)
    nmatched = 0
    for i in range(nCandidates):
        if dists[i] < 0:
            continue
        matched_ind_0[nmatched] = ind_0[i]
        matched_ind_1[nmatched] = ind_1[i]
        matched_nnd[nmatched] = dists[i]
        nmatched += 1
        for j in range(i+1, len(dists)):
            if (ind_0[j] == ind_0[i]) or (ind_1[j] == ind_1[i]):
                dists[j] = -1

    return matched_ind_0[:nmatched], matched_ind_1[:nmatched], matched_nnd[:nmatched]