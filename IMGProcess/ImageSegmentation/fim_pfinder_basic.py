import os
import inspect
import ctypes
import numpy.ctypeslib as ctl
import numpy as np
from scipy.special import erfinv
from scipy import interpolate
from cv2 import filter2D, GaussianBlur, sepFilter2D


"""
peak_detection dictionary
filtmethod:     
    Gauss:      convolve the raw image with a Gaussian kernel
    DoG:        convolve the raw image with a Gaussian kernel (sig=PSFsigma) minus a larger Gaussian kernel (sig=2.5*PSFsigma), respectively.
    DoA:        convolve the raw image with a 1-st wavelet kernel minus a wavelet kernel with 0 insertions (a-trous), respectively.
    PSF:        convolve the raw image with the (moddle plane of the) PSF kernel
    MIP:        convolve the raw image with each kernel of a kernel stack and projects the maximum intensity (weights are ignored)        

segmethod:     
    nms:        the local maxima identified via connectivity are further filtered if they fall into the same box
    grid:       the local maxima identified via connectivity are gathered into box-grids from the image
    None:       the local maxima are identified via connectivity without further filtering            

threshmode and threshval:     
    'dynamic':  thresh = median + (quantile_0.8 - quantile_0.2)/(0.8-0.2) * 0.5 * 2.0 * threshval (threshval = 1.7 in SMAP) 
                (quantile is that of the local maxima identified by connection prior to the thresholding, with or without maximum suppression)
    'pvalue':   thresh = threshval if filtmethod in {'Gauss', 'DoG', 'DoA'} on probability map
                thresh = sqrt(2) * erfinv(1 - 2 * pval) * sqrt(0.5 * excess) (excess = 2 if EMGain > 1) on standarized image with background subtracted
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


def _prob2val(pvalue, EMexcess):
    val = np.sqrt(2.0) * erfinv(1.0 - 2.0 * pvalue) 
    val *= np.sqrt(0.5 * EMexcess)
    return min(max(1E-7, val), 1E7)



#===========================================================================
#
#                          Background Estimator 
#
#===========================================================================
def _bin1d(data, nbins, padding=True):
    """
    Reshape the 1d data into 2d accroding to the nbins
    INPUTS:
        data:       (ndata) float ndarray, the input image to bin
        nbins:      int, number of bins
        padding:    bool, True to pad the input image
    RETURN:
        result:     (nbins, elementsperbin) float ndarray, the rearranged image accroding to the bins 
    """
    data = np.array(data, dtype=np.float64)
    nbins = np.array(nbins)
    ndata = len(data)
    elementsPerBin = (ndata - 1) // nbins + 1
    toPad = nbins * elementsPerBin - ndata
    
    if toPad != 0:
        if not padding:
            raise ValueError("Enable padding to bin this data")
        
        pad_0 = toPad // 2
        pad_1 = toPad - pad_0
        dataPadded = np.pad(data, (pad_0, pad_1), "constant", constant_values=(np.NaN, np.NaN))
    else:
        dataPadded = data
   
    res = dataPadded.reshape((nbins, elementsPerBin))
    return res



def _bin2d(data, nbins, padding=True):
    """
    Reshape the 2d data into 4d accroding to the 2d-bins
    INPUTS:
        data:       (imszy_i, imszx_i) float ndarray, the input image to bin
        nbins:      (2) int ndarray, [nbinsy, nbinsx]
        padding:    bool, True to pad the input image
    RETURN:
        result:     (nbinsy, elementsperbin_y, nbinsx, elementsperbin_x) float ndarray, the rearranged image accroding to the bins 
    """
    data = np.array(data, dtype=np.float64)
    nbins = np.array(nbins)
    inputShape = np.asarray(data.shape)
    elementsPerBin = (inputShape - 1) // nbins + 1
    toPad = nbins * elementsPerBin - inputShape
    
    if np.any(toPad != 0):
        if not padding:
            raise ValueError("Enable padding to bin this data")
        
        pad_0 = toPad // 2
        pad_1 = toPad - pad_0
        padWidth = ((pad_0[0], pad_1[0]), (pad_0[1], pad_1[1]))
        dataPadded = np.pad(data, padWidth, "constant", constant_values=(np.NaN, np.NaN))
    else:
        dataPadded = data
   
    res = dataPadded.reshape([nbins[0], elementsPerBin[0], nbins[1], elementsPerBin[1]])
    return res



def getbkg_median(im, blocksz, ims_prev=None, ims_next=None, precentile=50):
    """
    get the background of the input frame by block-median
    INPUTS:
        im:         (imszy, imszx) float ndarray, the images for background estimator
        blocksz:    (2) int ndarray, [blockszy, blockszx] block size in pixel
        ims_prev:   (nfrm, imszy, imszx) float ndarray, the image frames before the input im
        ims_next:   (nfrm, imszy, imszx) float ndarray, the image frames after the input im
        precentile: float, 0~100 the precentile for determination of the groundtruth, 50 is the median at default
    RETURN:
        bkg:        (imszy, imszx) float ndarray, the background of the input im
    """
    imszy, imszx = im.shape
    if (not ims_prev is None) and (ims_prev.ndim == 2):
        ims_prev = ims_prev[np.newaxis,...]
    if (not ims_next is None) and (ims_next.ndim == 2):
        ims_next = ims_next[np.newaxis,...]

    nbins = (np.array([imszy, imszx]) - 1) // blocksz + 1 
    
    # Find bin's central coordinates
    raw_xvec = np.arange(imszx)
    raw_yvec = np.arange(imszy)
    bin_xvec = np.nanmean(_bin1d(raw_xvec, nbins[1]), axis=1)
    bin_yvec = np.nanmean(_bin1d(raw_yvec, nbins[0]), axis=1)
    
    # Calc percentile over frames expanded before and after
    binnedData = _bin2d(im, nbins)
    if not ims_prev is None:
        for im_prev in ims_prev:
            binnedData = np.concatenate((binnedData, _bin2d(im_prev, nbins)), axis=1)
    if not ims_next is None:
        for im_next in ims_next:
            binnedData = np.concatenate((binnedData, _bin2d(im_next, nbins)), axis=1)
    percentileMat = np.nanpercentile(binnedData, precentile, axis = (1, 3))
    interp = interpolate.RectBivariateSpline(bin_yvec, bin_xvec, percentileMat)
    interp_bkg = interp.ev(*np.meshgrid(np.arange(imszy), np.arange(imszx), indexing='ij'))

    return interp_bkg



#===========================================================================
#
#                               Image Filter 
#
#===========================================================================
def _get_wvletAtrous(order):
    """get an Atrous kernel with the input order"""
    knl = np.array([1.0/16.0, 1.0/4.0, 3.0/8.0, 1.0/4.0, 1.0/16.0])
    if order <= 1:
        return knl
    wvkernel = np.zeros((len(knl)-1) * order + 1)
    wvkernel[::order] = knl    
    return wvkernel 



def _im_filt_PSF(im_i, kernel_stack):
    """
    convolve the image with the middle plane of the kernel_stack
    INPUTS:
        im_i:           (imszy, imszx) float ndarray, image to filter
        kernel_stack:   (szz, knlszy, knlszx) float ndarray, the normalized kernel stack
    RETURN:
        fim:            (imszy, imszx) float ndarray, image convolved with the Gaussian kernel  
    """
    assert kernel_stack.ndim in {2, 3}, "ndim mismatch: input.PSFkernel.ndim={}, should 2 or 3".format(kernel_stack.ndim)
    if kernel_stack.ndim == 2:
        knl = np.copy(kernel_stack)
    elif kernel_stack.ndim == 3:
        knl = kernel_stack[len(kernel_stack)//2]
    return filter2D(im_i, ddepth=-1, kernel=knl)



def _im_filt_MIP(im_i, kernel_stack):
    """
    convolve the image with each kernel in a kernel_stack, and project the maximum intensity 
    INPUT:
        im_i:           (imszy, imszx) float ndarray, image to filter
        kernel_stack:   (szz, knlszy, knlszx) float ndarray, the normalized kernel stack
    RETURN:
        fim:            (imszy, imszx) float ndarray, image convolved with the Gaussian kernel
    """
    assert kernel_stack.ndim in {2, 3}, "ndim mismatch: input.PSFkernel.ndim={}, should 2 or 3".format(kernel_stack.ndim)
    if kernel_stack.ndim == 2:
        return filter2D(im_i, ddepth=-1, kernel=kernel_stack)
    if len(kernel_stack) == 1:
        return filter2D(im_i, ddepth=-1, kernel=kernel_stack[0])
    
    fim = np.zeros_like(im_i) - np.inf
    for knl in kernel_stack:
        dum = filter2D(im_i, ddepth=-1, kernel=knl)
        fim = np.maximum(fim, dum)
    return fim


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



def _get_pmap_thresh(im, indx, indy, intensities, pvalue, EMexcess):
    """
    get the threshold based on the given p-value
    INPUTS:
        im:             (imszy, imszx) float ndarray, the image for peak detection
        indx:           (ncand,) int ndarray, the indx of the lacal maxima candidates
        indy:           (ncand,) int ndarray, the indy of the local maxima candidates
        intensities:    (ncand,) float ndarray, the intensities of the local maxima candidates
        pvalue:         float, pvalue for thresholding
        EMexcess:       float, excess factor if EM is on
    RETURN:
        ind:            (ncand,) bool ndarray, index of the survival candidates
    """
    imszy, imszx = im.shape
    ncand = len(indx)
    val = _prob2val(pvalue, EMexcess)

    ind = np.zeros(ncand, dtype=bool)
    for i, (ix, iy) in enumerate(zip(indx, indy)):
        if intensities[i] > val:
            dum = im[max(iy-1,0):min(iy+2,imszy), max(ix-1,0):min(ix+2,imszx)]
            dumind = dum > val
            if np.sum(dumind) > 3:
                ind[i] = True
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



def _im_lclmax(im, boxsz, threshmode, threshval, segmethod, EMexcess):
    """
    Localize the local maximum in an input image with given threshold
    INPUT:
        im:             (imszy, imszx) float ndarray, image to find local maximas, usually the filtered image
        boxsz:          int, boxsz for the of the lclMax kernel, must be odd
        threshmode:     str, {'dynamic', 'pvalue', 'std', 'constant'}
        threshval:      float, threshold value corresponding to the threshmode
        EMexcess:       float, excess factor when EM is on     
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
    elif threshmode == 'pvalue':
        ind_survival = _get_pmap_thresh(im, indx, indy, intensity, threshval, EMexcess) 
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



#===========================================================================
#
#                               Peak Detection 
#
#===========================================================================
def im_peakfinder(im, weight, offset, A2D, EMgain, boxsz, sigmax, sigmay, kernel_stack, peak_detection):
    """
    Localize the local maximum in an input image with given threshold
    INPUT:
        im:             (imszy, imszx) float ndarray, image going to be segmented
        weight:         (imszy, imszx) float ndarray or None, the weights of each pixel of the image, None for equal weights
        offset:         (imszy, imszx) float ndarray or float, the offset of each pixel of the image, float for equal offset
        A2D:            (imszy, imszx) float ndarray or float, the Analog-to-Digital converter of each pixel of the image, float for equal A2D
        EMgain:         float, the EMgain of the camera
        boxsz:          int, boxsz for the of the lclMax kernel, must be odd
        sigmax:         float, the sigma (x-axis) of the gauss kernel that is used for filtering if filtmethod==DoG or DoW
        sigmay:         float, the sigma (y-axis) of the gauss kernel that is used for filtering if filtmethod==DoG or DoW
        kernel_stack:   (szz, knlszy, knlszx) float ndarray, kernel stack for convolution 
        peak_detection: dictionary contains corresponding parameters
    RETURN:
        indx:           (nspots,) int ndattay, indices of the local maxima (x-axis)
        indy:           (nspots,) int ndarray, indices of the local maxima (y-axis)
    """

    # parse the input
    EMexcess = 2.0 if EMgain > 1.0 else 1.0
    filtmethod  = peak_detection['filtmethod']
    segmethod   = peak_detection['segmethod']
    threshmode  = peak_detection['threshmode']
    threshval   = peak_detection['threshval']
    if not weight is None:
        assert weight.shape == im.shape, "shape mismatch: weight.shape={}, im.shape={}".format(weight.shape, im.shape)
    assert filtmethod in {'Gauss', 'DoG', 'DoA', 'PSF', 'MIP'}, "{} is unsupported filtmethod, should be 'Gauss', 'DoG', 'DoA', 'PSF', or 'MIP'".format(filtmethod)
    assert threshmode in {'dynamic', 'pvalue', 'std', 'constant'}, "{} is unsupported threshmode, should be 'dynamic', 'pvalue', 'std', 'constant'".format(filtmethod)
    if filtmethod == 'MIP':
        weight = None
    
    # image standarization
    fim0 = (np.array(im, dtype=np.float32) - offset) / (A2D * EMgain)
    imnorm = _im_norm(fim0)
    if filtmethod in {'Gauss', 'PSF', 'MIP'}:
        imnorm_bg = getbkg_median(imnorm, boxsz)
    if filtmethod == 'DoA':
        knl = _get_wvletAtrous(norder=1)
        knl2 = _get_wvletAtrous(norder=2)

    # filter the image
    if not weight is None:
        if filtmethod == 'Gauss':
            fim1 = GaussianBlur((imnorm-imnorm_bg)*weight, ksize=(0, 0), sigmaX=sigmax, sigmaY=sigmay) / GaussianBlur(weight, ksize=(0, 0), sigmaX=sigmax, sigmaY=sigmay)
        elif filtmethod == 'DoG':
            fim1 = GaussianBlur(imnorm*weight, ksize=(0, 0), sigmaX=sigmax, sigmaY=sigmay) / GaussianBlur(weight, ksize=(0, 0), sigmaX=sigmax, sigmaY=sigmay)
            fim2 = GaussianBlur(imnorm*weight, ksize=(0, 0), sigmaX=2.5*sigmax, sigmaY=2.5*sigmay) / GaussianBlur(weight, ksize=(0, 0), sigmaX=2.5*sigmax, sigmaY=2.5*sigmay)
            fim1 -= fim2
        elif filtmethod == 'DoA':
            fim1 = sepFilter2D(imnorm*weight, ddepth=-1, kernelX=knl, kernelY=knl) / sepFilter2D(weight, ddepth=-1, kernelX=knl, kernelY=knl)
            fim2 = sepFilter2D(imnorm*weight, ddepth=-1, kernelX=knl2, kernelY=knl2) / sepFilter2D(weight, ddepth=-1, kernelX=knl2, kernelY=knl2)
            fim1 -= fim2
        elif filtmethod == 'PSF':
            fim1 = _im_filt_PSF((imnorm-imnorm_bg)*weight, kernel_stack) / _im_filt_PSF(weight, kernel_stack)
        elif filtmethod == 'MIP':
            fim1 = _im_filt_MIP((imnorm-imnorm_bg), kernel_stack)
    else:
        if filtmethod == 'Gauss':
            fim1 = GaussianBlur((imnorm-imnorm_bg), (0, 0), sigmaX=sigmax, sigmaY=sigmay)
        elif filtmethod == 'DoG':
            fim1 = GaussianBlur(imnorm, ksize=(0, 0), sigmaX=sigmax, sigmaY=sigmay) - GaussianBlur(imnorm, ksize=(0, 0), sigmaX=2.5*sigmax, sigmaY=2.5*sigmay)
        elif filtmethod == 'DoA':
            fim1 = sepFilter2D(imnorm, ddepth=-1, kernelX=knl, knlnelY=knl) - sepFilter2D(imnorm, ddepth=-1, kernelX=knl2, knlnelY=knl2)
        elif filtmethod == 'PSF':
            fim1 = _im_filt_PSF((imnorm-imnorm_bg), kernel_stack)
        elif filtmethod == 'MIP':
            fim1 = _im_filt_MIP((imnorm-imnorm_bg), kernel_stack)

    # find the local maxima
    indx, indy = _im_lclmax(fim1, boxsz, threshmode, threshval, segmethod, EMexcess)
    return indx, indy



def ims_peakfinder_GPM(ims, weight, offset, A2D, EMgain, boxsz, sigmax, sigmay, kernel_stack, peak_detection):
    """
    Localize the local maximum in an input image with given threshold (for filtmethod in {'Gauss', 'PSF', 'MIP'})
    INPUT:
        ims:            (imszy, imszx) or (nfrm, imszy, imszx) float ndarray, image stack going to be segmented
        weight:         (imszy, imszx) float ndarray or None, the weights of each pixel of the image, None for equal weights
        offset:         (imszy, imszx) float ndarray or float, the offset of each pixel of the image, float for equal offset
        A2D:            (imszy, imszx) float ndarray or float, the Analog-to-Digital conversion factor of each pixel of the image, float for equal A2D
        EMgain:         float, the EMgain for the camera
        boxsz:          int, boxsz for the of the lclMax kernel, must be odd
        sigmax:         float, the sigma (x-axis) of the gauss kernel that is used for filtering if filtmethod==DoG or DoW
        sigmay:         float, the sigma (y-axis) of the gauss kernel that is used for filtering if filtmethod==DoG or DoW
        kernel_stack:   (szz, knlszy, knlszx) float ndarray, kernel stack for convolution
        peak_detection: dictionary contains corresponding parameters
    RETURN:
        indx:           (nspots,) int ndarray, indices of the local maxima (x-axis)
        indy:           (nspots,) int ndarray, indices of the local maxima (y-axis)
        indf:           (nspots,) int ndarray, indices of the local maxima (t-axis)
    """
    # parse the input
    EMexcess = 2.0 if EMgain > 1.0 else 1.0
    filtmethod  = peak_detection['filtmethod']
    segmethod   = peak_detection['segmethod']
    threshmode  = peak_detection['threshmode']
    threshval   = peak_detection['threshval']
    if not weight is None:
        assert weight.shape == ims.shape[-2:], "shape mismatch: weight.shape={}, ims.shape={}".format(weight.shape, ims.shape)
    assert filtmethod in {'Gauss', 'PSF', 'MIP'}, "{} is unsupported filtmethod, should be 'Gauss', 'PSF', or 'MIP'".format(filtmethod)
    assert threshmode in {'dynamic', 'pvalue', 'std', 'constant'}, "{} is unsupported threshmode, should be 'dynamic', 'pvalue', 'std', 'constant'".format(filtmethod)
    if filtmethod == 'MIP':
        weight = None
    
    if ims.ndim == 2:
        indx, indy = im_peakfinder(ims, weight, offset, A2D, EMgain, boxsz, sigmax, sigmay, kernel_stack, peak_detection)
        return indx, indy, np.zeros(len(indx), dtype=np.int32)
    
    # filter the weight
    if not weight is None:
        if filtmethod == 'Gauss':
            fw1 = GaussianBlur(weight, ksize=(0, 0), sigmaX=sigmax, sigmaY=sigmay)
        elif filtmethod == 'PSF':
            fw1 = _im_filt_PSF(weight, kernel_stack)
    
    # peak finding
    dum_indx, dum_indy, dum_indf = [], [], []
    for f, im in enumerate(ims):
        
        # image standarization
        fim0 = (np.array(im, dtype=np.float32) - offset) / (A2D * EMgain)
        fim0_prev = None if f == 0 else (np.array(ims[f-1], dtype=np.float32) - offset) / (A2D * EMgain)
        fim0_next = None if f == len(ims)-1 else (np.array(ims[f+1], dtype=np.float32) - offset) / (A2D * EMgain)
        imnorm = _im_norm(fim0)
        imnorm_prev = None if f == 0 else _im_norm(fim0_prev)
        imnorm_next = None if f == len(ims)-1 else _im_norm(fim0_next)
        imnorm_bg = getbkg_median(imnorm, boxsz, imnorm_prev, imnorm_next)

        # filter the image
        if not weight is None:
            if filtmethod == 'Gauss':
                fim1 = GaussianBlur((imnorm-imnorm_bg)*weight, (0, 0), sigmaX=sigmax, sigmaY=sigmay) / fw1
            elif filtmethod == 'PSF':
                fim1 = _im_filt_PSF((imnorm-imnorm_bg)*weight, kernel_stack) / fw1
            elif filtmethod == 'MIP':
                fim1 = _im_filt_MIP((imnorm-imnorm_bg), kernel_stack)
        else:
            if filtmethod == 'Gauss':
                fim1 = GaussianBlur((imnorm-imnorm_bg), (0, 0), sigmaX=sigmax, sigmaY=sigmay)
            elif filtmethod == 'PSF':
                fim1 = _im_filt_PSF((imnorm-imnorm_bg), kernel_stack)
            elif filtmethod == 'MIP':
                fim1 = _im_filt_MIP((imnorm-imnorm_bg), kernel_stack)
        
        # find the local maxima
        tmp_indx, tmp_indy = _im_lclmax(fim1, boxsz, threshmode, threshval, segmethod, EMexcess)
        tmp_nspots = len(tmp_indx)
        if tmp_nspots > 0:
            dum_indx.append(tmp_indx)
            dum_indy.append(tmp_indy)
            dum_indf.append(np.zeros(tmp_nspots, dtype=np.int32) + f)
    
    indx = [item for sublist in dum_indx for item in sublist]
    indy = [item for sublist in dum_indy for item in sublist]
    indf = [item for sublist in dum_indf for item in sublist]

    return np.asarray(indx), np.asarray(indy), np.asarray(indf)



def ims_peakfinder_DoK(ims, weight, offset, A2D, EMgain, boxsz, sigmax, sigmay, kernel_stack, peak_detection):
    """
    Localize the local maximum in an input image with given threshold (for filtmethod in {'DoG', 'DoA'})
    INPUT:
        ims:            (imszy, imszx) or (nfrm, imszy, imszx) float ndarray, image stack going to be segmented
        weight:         (imszy, imszx) float ndarray or None, the weights of each pixel of the image, None for equal weights
        offset:         (imszy, imszx) float ndarray or float, the offset of each pixel of the image, float for equal offset
        A2D:            (imszy, imszx) float ndarray or float, the Analog-to-Digital conversion factor of each pixel of the image, float for equal A2D
        EMgain:         float, the EMgain for the camera
        boxsz:          int, boxsz for the of the lclMax kernel, must be odd
        sigmax:         float, the sigma (x-axis) of the gauss kernel that is used for filtering if filtmethod==DoG or DoW
        sigmay:         float, the sigma (y-axis) of the gauss kernel that is used for filtering if filtmethod==DoG or DoW
        kernel_stack:   (szz, knlszy, knlszx) float ndarray, kernel stack for convolution
        peak_detection: dictionary contains corresponding parameters
    RETURN:
        indx:           (nspots,) int ndarray, indices of the local maxima (x-axis)
        indy:           (nspots,) int ndarray, indices of the local maxima (y-axis)
        indf:           (nspots,) int ndarray, indices of the local maxima (t-axis)
    """
    # parse the input
    EMexcess = 2.0 if EMgain > 1.0 else 1.0
    filtmethod  = peak_detection['filtmethod']
    segmethod   = peak_detection['segmethod']
    threshmode  = peak_detection['threshmode']
    threshval   = peak_detection['threshval']
    if not weight is None:
        assert weight.shape == ims.shape[-2:], "shape mismatch: weight.shape={}, ims.shape={}".format(weight.shape, ims.shape)
    assert filtmethod in {'DoG', 'DoA'}, "{} is unsupported filtmethod, should be 'DoG', 'DoA'".format(filtmethod)
    assert threshmode in {'dynamic', 'pvalue', 'std', 'constant'}, "{} is unsupported threshmode, should be 'dynamic', 'pvalue', 'std', 'constant'".format(filtmethod)
    
    if ims.ndim == 2:
        indx, indy = im_peakfinder(ims, weight, offset, A2D, EMgain, boxsz, sigmax, sigmay, kernel_stack, peak_detection)
        return indx, indy, np.zeros(len(indx), dtype=np.int32)
    
    # filter the weight
    if filtmethod == 'DoA':
        knl = _get_wvletAtrous(norder=1)
        knl2 = _get_wvletAtrous(norder=2)
    if not weight is None:
        if filtmethod == 'DoG':
            fw1 = GaussianBlur(weight, (0, 0), sigmaX=sigmax, sigmaY=sigmay)
            fw2 = GaussianBlur(weight, (0, 0), sigmaX=2.5*sigmax, sigmaY=2.5*sigmay)
        elif filtmethod == 'DoA':
            fw1 = sepFilter2D(weight, ddepth=-1, kernelX=knl, kernelY=knl)
            fw2 = sepFilter2D(weight, ddepth=-1, kernelX=knl2, kernelY=knl2)
        
    # peak finding
    dum_indx, dum_indy, dum_indf = [], [], []
    for f, im in enumerate(ims):

        # image standarization
        fim0 = (np.array(im, dtype=np.float32) - offset) / (A2D * EMgain)
        imnorm = _im_norm(fim0)
        
        # filter the image
        if not weight is None:
            if filtmethod == 'DoG':
                fim1 = GaussianBlur(imnorm*weight, (0, 0), sigmaX=sigmax, sigmaY=sigmay) / fw1
                fim2 = GaussianBlur(imnorm*weight, (0, 0), sigmaX=2.5*sigmax, sigmaY=2.5*sigmay) / fw2
                fim1 -= fim2
            elif filtmethod == 'DoA':
                fim1 = sepFilter2D(imnorm*weight, ddepth=-1, kernelX=knl, kernelY=knl) / fw1
                fim2 = sepFilter2D(imnorm*weight, ddepth=-1, kernelX=knl2, kernelY=knl2) / fw2
                fim1 -= fim2
        else:
            if filtmethod == 'DoG':
                fim1 = GaussianBlur(imnorm, (0, 0), sigmaX=sigmax, sigmaY=sigmay) - GaussianBlur(imnorm, (0, 0), sigmaX=2.5*sigmax, sigmaY=2.5*sigmay)
            elif filtmethod == 'DoA':
                fim1 = sepFilter2D(imnorm, ddepth=-1, kernelX=knl, kernelY=knl) - sepFilter2D(imnorm, ddepth=-1, kernelX=knl2, kernelY=knl2)
            
        # find the local maxima
        tmp_indx, tmp_indy = _im_lclmax(fim1, boxsz, threshmode, threshval, segmethod, EMexcess)
        
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