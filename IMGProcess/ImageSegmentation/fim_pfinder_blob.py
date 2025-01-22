import os
import inspect
import ctypes
import numpy.ctypeslib as ctl
import numpy as np
import cv2
from cv2 import filter2D, GaussianBlur


"""
peak_detection dictionary
filtmethod:     
    blob_DoG:   convolve the raw image with a series Gaussian kernels for DoG blob detection.
    blob_LoG:   convolve the raw image with a series Laplacian Gaussian kernels for LoG blob detection.

segmethod:     
    nms:        the local maxima identified via connectivity are further filtered if they fall into the same box
    grid:       the local maxima identified via connectivity are gathered into box-grids from the image
    None:       the local maxima are identified via connectivity without further filtering            

threshmode and threshval:     
    'dynamic':  thresh = median + (quantile_0.8 - quantile_0.2)/(0.8-0.2) * 0.5 * 2.0 * threshval (threshval = 1.7 in SMAP) 
                (quantile is that of the local maxima identified by connection prior to the thresholding, with or without maximum suppression)
    'std':      thresh = im.mean() + threshval * im.std()
    'constant': thresh = threshval
"""


#===========================================================================
#
#                              Image Normalize 
#
#===========================================================================
def _im_norm(ims_i):
    # transfer data that are subjected to poisson distribution to normal distribution
    ims_o = np.copy(ims_i).astype(np.float64)
    ims_o[ims_o < -0.3750] = 0.0
    ims_o = 2.0 * np.sqrt(ims_o + 0.375)
    return ims_o



#===========================================================================
#
#                               LoG kernel 
#
#===========================================================================
def _getkernel_LoG(sigx, sigy, truncate=4.0):
    """
    get a Laplacian of Gaussian kernel
    INPUTS:
        sigx:       (nslice,) float ndarray or float, the sigma (x-axis) for the Gaussian kernel
        sigy:       (nslice,) float ndarray or float, the sigma (y-axis) for the Gaussian kernel
        truncate:   float, the kernel size is 2 * hknlsz + 1 where hknelsz = ceil(truncate * sigma) 
    """
    if np.isscalar(sigx) and np.isscalar(sigy):
        knlszx = np.int32(np.ceil(sx * truncate))
        knlszy = np.int32(np.ceil(sy * truncate))
        YY, XX = np.meshgrid(np.arange(-knlszy, knlszy+1), np.arange(-knlszx, knlszx+1), indexing='ij')
        return 0.5 / np.pi * np.exp(-0.5*YY**2/sy**2-0.5*XX**2/sx**2) * (1.0/sx**2+1.0/sy**2 -YY**2/sy**4-XX**2/sx**4)
    
    knls = []
    for sx, sy in zip(sigx, sigy):
        knlszx = np.int32(np.ceil(sx * truncate))
        knlszy = np.int32(np.ceil(sy * truncate))
        YY, XX = np.meshgrid(np.arange(-knlszy, knlszy+1), np.arange(-knlszx, knlszx+1), indexing='ij')
        knls.append(0.5 / np.pi * np.exp(-0.5*YY**2/sy**2-0.5*XX**2/sx**2) * (1.0/sx**2+1.0/sy**2 -YY**2/sy**4-XX**2/sx**4))
    return knls



#===========================================================================
#
#                               Local Maxima 
#
#===========================================================================
def _localmaxima(im):
    """
    return the list of the local maxima within (9x9 for 2d or 27x27 for 3d) of the input image
    INPUT:
        im:             (imszz, imszy, imszx) float ndarray, image to find local maximas
    RETURN:
        indices:        (nspots, ndim) int ndarray, indy of the local maxima
        Intensity:      (nspots,) int ndarray, intensity of the local maxima
        nspots:         int, number of the local maxima
    """
    ndim = im.ndim
    assert ndim in {2, 3}, "unsupported ndim: input.ndim={}".format(ndim)
    if ndim == 3 and len(im) == 1:
        im = im[0]
        ndim = 2
    imshape = im.shape
    nMax = np.prod([(imshape[i] - 1) // 2 + 1 for i in range(ndim)])

    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_localmaxima')
    fname = 'localmaxima.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    
    if ndim == 2:
        lclmaxima_kernel = lib.local_maxima_2d
        lclmaxima_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
    elif ndim == 3:
        lclmaxima_kernel = lib.local_maxima_3d
        lclmaxima_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
    
    im = np.ascontiguousarray(im.flatten(), dtype=np.float32)
    nspots_ref = np.ascontiguousarray(np.zeros(1), dtype=np.int32)
    maxima_list = np.ascontiguousarray(np.zeros(nMax * (ndim+1)), dtype=np.float32)
    lclmaxima_kernel(nMax, *imshape, im, nspots_ref, maxima_list)

    nspots, = nspots_ref
    maxima_list = np.reshape(maxima_list, (nMax, (ndim + 1)))
    maxima_list = maxima_list[:nspots]
    return np.int32(maxima_list[:, :ndim]), maxima_list[:, ndim], nspots



def _get_dynamic_thresh(intensities, dynamic_factor):
    """
    get the threshold based on the quantiles
    INPUTS:
        intensities:    (ncand,) float ndarray, the intensities of the local maxima candidates
        pvalue:         dynamic_factor, dynamic_factor to quantile
    RETURN:
        ind:            (ncand,) bool ndarray, index of the survival candidates
    """
    if len(intensities) < 10:
        thresh = np.median(intensities) * dynamic_factor
    else:
        ps = np.array([.2, .5, .8])
        qs = np.quantile(intensities, ps)
        thresh = qs[1] + (qs[-1]-qs[0])/(ps[-1]-ps[0]) * 0.5 * 2.0 * dynamic_factor
    ind = intensities > thresh
    return ind



def _get_numeric_thresh(intensities, threshval):
    """
    get the threshold based on a given threshold
    INPUTS:
        intensities:    (ncand,) float ndarray, the intensities of the local maxima candidates
        threshval:      float, numeric threshold
    RETURN:
        ind:            (ncand,) bool ndarray, index of the survival candidates
    """
    return intensities > threshval



def _nms(boxsz, indx, indy, intensity):
    """
    non-maximum suppression
    INPUT:
        boxsz:          int, boxsz for the of the lclMax kernel, must be odd
        indx:           (nspots,) int ndarray, indices of the local maxima (x-axis)
        indy:           (nspots,) int ndarray, indices of the local maxima (y-axis)
        intensity:      (nspots,) float ndarray, intensity at location (indy, indx)
    RETURN:
        reglist:        registration for non-maxima (as false)
    """
    nspots = len(indx)
    
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_nms')
    fname = 'nms.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    nms_kernel = lib.nms
    nms_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous")]
    
    indx = np.ascontiguousarray(indx, dtype=np.int32)
    indy = np.ascontiguousarray(indy, dtype=np.int32)
    intensity = np.ascontiguousarray(intensity, dtype=np.float32)
    reglist = np.ascontiguousarray(np.ones(nspots), dtype=np.int32)
    nms_kernel(nspots, boxsz, indx, indy, intensity, reglist)
    
    return reglist



def _get_gridc(imsz, boxsz, Indx_i, Indy_i):
    """
    divide the image into boxsz-grids and return the center of the grids that have Indx_i and Indy_i inside
    INPUTS:
        imsz:       (2,) int tuple, (imszy, imszx) size of the image
        boxsz:      int, the size of the square box
        Indx_i:     (nspots,) int ndarray, the input x-location
        Indy_i:     (nspots,) int ndarray, the input y-location
    OUTPUTS:
        Indx_o:     (nboxes,) int ndarray, the output x-location (x-center of the box-grid that has the input location inside)
        Indy_o:     (nboxes,) int ndarray, the output y-location (y-center of the box-grid that has the input location inside)
    """
    imszy, imszx = imsz
    margin_t = imszy % boxsz // 2
    margin_l = imszx % boxsz // 2
    margin_b = margin_t + imszy // boxsz * boxsz
    margin_r = margin_l + imszx // boxsz * boxsz

    dum_ind = (Indx_i >= margin_l) & (Indx_i < margin_r) & (Indy_i >= margin_t) & (Indy_i < margin_b)
    dum_Indx = Indx_i[dum_ind] - margin_l
    dum_Indy = Indy_i[dum_ind] - margin_t
    dum_Indx = dum_Indx // boxsz * boxsz + boxsz // 2 + margin_l
    dum_Indy = dum_Indy // boxsz * boxsz + boxsz // 2 + margin_t   
    Indx_o, Indy_o = np.unique(np.vstack((dum_Indx, dum_Indy)), axis=1)

    return Indx_o, Indy_o



def _im_lclmax(im, boxsz, threshmode, threshval, segmethod):
    """
    Localize the local maximum in an input image with given threshold
    INPUT:
        im:             (imszy, imszx) float ndarray, image to find local maximas, usually the filtered image
        boxsz:          int, boxsz for the of the lclMax kernel, must be odd
        threshmode:     str, {'dynamic', 'std', 'constant'}
        threshval:      float, threshold value corresponding to the threshmode   
    RETURN:
        indx:           (nspots,) int ndarray, indices of the local maxima (x-axis)
        indy:           (nspots,) int ndarray, indices of the local maxima (y-axis)
    """
    assert boxsz % 2 == 1, "The boxsz must be odd, input.boxsz={}".format(boxsz)
    imszy, imszx = im.shape[-2:]

    indicies, intensity, nspots = _localmaxima(im)
    indx = np.int32(indicies[:, 0])
    indy = np.int32(indicies[:, 1])
    if nspots == 0:
        return indx, indy
    
    # thresholding
    if threshmode == 'dynamic':
        ind_survival = _get_dynamic_thresh(intensity, threshval) 
    elif threshmode == 'std':
        ind_survival = _get_numeric_thresh(intensity, threshval*im.std())
    elif threshmode == 'constant':
        ind_survival = _get_numeric_thresh(intensity, threshval) 
    indx, indy, intensity = indx[ind_survival], indy[ind_survival], intensity[ind_survival]
    nspots = len(indy)
    if nspots == 0:
        return indx, indy
    
    # maximum suppression
    if segmethod == 'nms':
        _dumind = np.argsort(-intensity)
        indx, indy, intensity =indx[_dumind], indy[_dumind], intensity[_dumind]
       
        _dumind = _nms(2, indx, indy, intensity)
        #_dumind = _nms(boxsz, indx, indy, intensity)
        indx, indy, intensity = indx[_dumind.astype(bool)], indy[_dumind.astype(bool)], intensity[_dumind.astype(bool)]
    elif segmethod == 'grid':
        indx, indy = _get_gridc((imszy, imszx), boxsz, indx, indy)
        
    return indx, indy



#=======================================================================================================
#
#                               Peak Detection (ignore sCMOS variance)
#
#=======================================================================================================
def ims_blob_DoG(ims, offset, A2D, EMgain, boxsz, min_sigma, max_sigma, sigma_ratio, peak_detection):
    """
    Localize the local maximum in an input image with given threshold (for filtmethod=='blob_DoG')
    for modality in {'2D', 'AS', 'BP'}
    INPUT:
        ims:            (imszy, imszx) or (nfrm, imszy, imszx) float ndarray, image stack going to be segmented
        weight:         (imszy, imszx) float ndarray or None, the weights of each pixel of the image, None for equal weights
        offset:         (imszy, imszx) float ndarray or float, the offset of each pixel of the image, float for equal offset
        A2D:            (imszy, imszx) float ndarray or float, the Analog-to-Digital conversion factor of each pixel of the image, float for equal A2D
        EMgain:         float, the EMgain for the camera
        boxsz:          int, boxsz for the of the lclMax kernel, must be odd
        min_sigma:      (2,) float ndarray or float, minimum (sigmax, sigmay) or sigma for Gaussian kernel
        max_sigma:      (2,) float ndarray or float, maximum (sigmax, sigmay) or sigma for Gaussian kernel
        sigma_ratio:    float, the ratio between the sigma of Gaussian kernels used for computing the Difference-of-Gaussian
        peak_detection: dictionary contains corresponding parameters
    RETURN:
        indx:           (nspots,) int ndarray, indices of the local maxima (x-axis)
        indy:           (nspots,) int ndarray, indices of the local maxima (y-axis)
        indf:           (nspots,) int ndarray, indices of the local maxima (t-axis)
    """
    # parse the input
    filtmethod  = peak_detection['filtmethod']
    segmethod   = peak_detection['segmethod']
    threshmode  = peak_detection['threshmode']
    threshval   = peak_detection['threshval']
    assert sigma_ratio > 1.0, "sigma_ratio should be > 1.0"
    assert filtmethod == 'blob_DoG', "{} is unsupported filtmethod for im_blob_DoG, should only be 'blob_DoG'".format(filtmethod)
    assert threshmode in {'dynamic', 'std', 'constant'}, "{} is unsupported threshmode, should be 'dynamic', 'std', 'constant'".format(filtmethod)
    if ims.ndim == 2:
        ims = ims[np.newaxis,...]

    max_sigma = np.full(2, max_sigma, dtype=np.float32) if np.isscalar(max_sigma) else np.array(max_sigma)
    min_sigma = np.full(2, min_sigma, dtype=np.float32) if np.isscalar(min_sigma) else np.array(min_sigma)
    k = np.int32(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))
    sigma_list = np.array([min_sigma * (sigma_ratio**i) for i in range(k + 1)])
    
    # peak detection
    dum_indx, dum_indy, dum_indf = [], [], []
    for f, im in enumerate(ims):

        # image standarization
        fim0 = (np.array(im, dtype=np.float32) - offset) / (A2D * EMgain)
        imnorm = _im_norm(fim0)
        
        # Difference-of-Gaussian
        fim_dogcube = np.empty((k,)+imnorm.shape, dtype=np.float32)
        fim_previous = GaussianBlur(imnorm, (0, 0), sigmaX=sigma_list[0, 0], sigmaY=sigma_list[0, 1], borderType=cv2.BORDER_REFLECT)
        for i, s in enumerate(sigma_list[1:]):
            fim_current = GaussianBlur(imnorm, (0, 0), sigmaX=s[0], sigmaY=s[1], borderType=cv2.BORDER_REFLECT)
            fim_dogcube[i] = fim_previous - fim_current
            fim_previous = fim_current
        sf = 1.0 / (sigma_ratio - 1.0)
        fim_dogcube *= sf
            
        # find the local maxima
        tmp_indx, tmp_indy = _im_lclmax(fim_dogcube, boxsz, threshmode, threshval, segmethod)
        
        tmp_nspots = len(tmp_indx)
        if tmp_nspots > 0:
            dum_indx.append(tmp_indx)
            dum_indy.append(tmp_indy)
            dum_indf.append(np.zeros(tmp_nspots, dtype=np.int32) + f)
    
    indx = [item for sublist in dum_indx for item in sublist]
    indy = [item for sublist in dum_indy for item in sublist]
    indf = [item for sublist in dum_indf for item in sublist]

    return np.asarray(indx), np.asarray(indy), np.asarray(indf)



def ims_blob_LoG(ims, offset, A2D, EMgain, boxsz, zrange, astigs, nsigs, peak_detection):
    """
    Localize the local maximum in an input image with given threshold (for filtmethod=='blob_LoG')
    for modality in {'2D', 'AS', 'BP'}
    INPUT:
        ims:            (imszy, imszx) or (nfrm, imszy, imszx) float ndarray, image stack going to be segmented
        offset:         (imszy, imszx) float ndarray or float, the offset of each pixel of the image, float for equal offset
        A2D:            (imszy, imszx) float ndarray or float, the Analog-to-Digital conversion factor of each pixel of the image, float for equal A2D
        EMgain:         float, the EMgain for the camera
        boxsz:          int, boxsz for the of the lclMax kernel, must be odd
        zrange:         int, zrange of the PSF calibration
        astigs:         (9,) float ndarray, the astigmatisim [sigmax0, sigmay0, shiftx, shifty, dof, Ax, Bx, Ay, By]
        nsigs:          int, number of sigma for Laplacian-of-Gaussian
        peak_detection: dictionary contains corresponding parameters
    RETURN:
        indx:           (nspots,) int ndarray, indices of the local maxima (x-axis)
        indy:           (nspots,) int ndarray, indices of the local maxima (y-axis)
        indf:           (nspots,) int ndarray, indices of the local maxima (t-axis)
    """
    # parse the input
    filtmethod  = peak_detection['filtmethod']
    segmethod   = peak_detection['segmethod']
    threshmode  = peak_detection['threshmode']
    threshval   = peak_detection['threshval']
    assert filtmethod == 'blob_LoG', "{} is unsupported filtmethod for im_blob_LoG, should only be 'blob_LoG'".format(filtmethod)
    assert threshmode in {'dynamic', 'std', 'constant'}, "{} is unsupported threshmode, should be 'dynamic', 'std', 'constant'".format(filtmethod)
    if ims.ndim == 2:
        ims = ims[np.newaxis,...]
    
    # collect the sigmas and the LoG kernel
    zdata = np.linspace(0.5, zrange-0.5, num=nsigs, endpoint=True)
    zdata[np.argmin(np.abs(zdata - astigs[2]))] = astigs[2]
    zdata[np.argmin(np.abs(zdata - astigs[3]))] = astigs[3]
    zx = (zdata - astigs[2]) / astigs[4]
    zy = (zdata - astigs[3]) / astigs[4]
    sigmax = astigs[0] * np.sqrt(np.maximum(1.0 + zx*zx + astigs[5]*zx*zx*zx + astigs[6]*zx*zx*zx*zx, 1.0))
    sigmay = astigs[1] * np.sqrt(np.maximum(1.0 + zy*zy + astigs[7]*zy*zy*zy + astigs[8]*zy*zy*zy*zy, 1.0))
    knls = _getkernel_LoG(sigmax, sigmay)
    
    # peak detection
    dum_indx, dum_indy, dum_indf = [], [], []
    for f, im in enumerate(ims):

        # image standarization
        fim0 = (np.array(im, dtype=np.float32) - offset) / (A2D * EMgain)
        imnorm = _im_norm(fim0)
        
        # Laplacian of Gaussian
        fim_logcube = np.empty((nsigs,)+imnorm.shape, dtype=np.float32)
        for i, knl in enumerate(knls):
            fim_logcube[i] = filter2D(imnorm, ddepth=-1, kernel=knl, borderType=cv2.BORDER_REFLECT)

        # find the local maxima
        tmp_indx, tmp_indy = _im_lclmax(fim_logcube, boxsz, threshmode, threshval, segmethod)
        
        tmp_nspots = len(tmp_indx)
        if tmp_nspots > 0:
            dum_indx.append(tmp_indx)
            dum_indy.append(tmp_indy)
            dum_indf.append(np.zeros(tmp_nspots, dtype=np.int32) + f)
    
    indx = [item for sublist in dum_indx for item in sublist]
    indy = [item for sublist in dum_indy for item in sublist]
    indf = [item for sublist in dum_indf for item in sublist]

    return np.asarray(indx), np.asarray(indy), np.asarray(indf)



def margin_refine(imsz, boxsz, indx, indy):
    """
    IN-PLACE refine indy and indx that are close to the edge
    INPUTS:
        imsz:       (2) int ndarray, [imszy, imszx] size of the image
        boxsz:      int, size of the box
        indx:       (nspots,) int ndarray, indices (x-axis) to refine
        indy:       (nspots,) int ndarray, indices (y-axis) to refine
    """
    assert boxsz % 2 == 1, "The boxsz must be odd, input.boxsz={}".format(boxsz)
    boxhsz = boxsz//2
    imszy, imszx = imsz
    
    indy[indy < boxhsz] = boxhsz
    indy[indy > imszy-boxhsz-1] = imszy-boxhsz-1
    indx[indx < boxhsz] = boxhsz
    indx[indx > imszx-boxhsz-1] = imszx-boxhsz-1
    return



if __name__ == '__main__':
    pass