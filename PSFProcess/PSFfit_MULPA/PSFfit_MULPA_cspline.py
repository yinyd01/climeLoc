import os
import inspect
import ctypes
import numpy.ctypeslib as ctl 
import numpy as np
from scipy.stats.distributions import chi2


def _perfect_sqrt(n):
    # get the integer square root of the input integer n    
    if not isinstance(n, (int, np.int_)):
        raise TypeError("type mismatch: type(n)={}, should be int ot np.int_".format(type(n)))
    if n <= 1:
        return n
 
    # binary search for the perfect square
    left, right = 1, n
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
 
        if square == n:
            return mid
        elif square < n:
            left = mid + 1
        else:
            right = mid - 1
    
    raise ValueError("not perfect square: sqrt(n)={}".format(np.sqrt(n)))



"""
@brief  batch fit of single-channel box data of PSFs
@param[in]  data:               (NFits, boxsz, boxsz) float ndarray
@param[in]  var:                (NFits, boxsz, boxsz) float ndarray or None
@param[in]  coeff_PSF:          (splineszx * splineszx * 16) float ndarray, the cubic spline coefficients calibrated for the PSF model
@param[in]  nmax:               int, maximum number of emitters for mulpa
@param[in]  p_val:              float, pvalue threshold to consider include one more emitter
@param[in]  optmethod:          str, {'MLE', 'LSQ'} optimization method
@param[in]  iterations:         int, number of iterations for optimization
@param[out] xvec:			    (NFits, vnum) float ndarray
@param[out] CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
@param[out] Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
@param[out] nnum:				(NFits,) int ndarray, number of emitters for the optimization of a PSF fitting.
"""

def _fitter_cspline2d1ch(NFits, boxsz, splinesz, data, var, coeff_PSF, nmax, p_val, optmethod, iterations):
    
    ### WARNING !! NO INPUTS CHECK ###
    ndim = 2
    nchannels = 1
    vmax = nmax * (ndim + nchannels) + nchannels
    opt = 0 if optmethod == 'MLE' else 1

    # allocate
    xvec_o = np.zeros((NFits, vmax), dtype=np.float32)
    crlb_o = np.zeros((NFits, vmax), dtype=np.float32)
    Loss_o = np.zeros(NFits, dtype=np.float32) + np.inf
    nnum_o = np.zeros(NFits, dtype=np.int32)
    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    
    # link the fitter_c
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_cspline2d')
    fname = 'cspline2d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    fitter = lib.cspline2d_1ch
    if var is None:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctypes.c_void_p,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32]
    else:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32]  
    
    BIC_track = np.zeros((NFits, nmax), dtype=np.float32) + np.inf
    ind_test = np.arange(NFits)
    dum_xvec = np.ascontiguousarray(np.zeros(NFits * (ndim + nchannels + nchannels)), dtype=np.float32)
    dum_NFits = NFits
    for n in range(1, nmax + 1):
        
        if dum_NFits > 0:
            vnum = n * (ndim + nchannels) + nchannels
            llr_threshold = chi2.ppf(1.0 - p_val, nchannels * boxsz * boxsz - vnum) if n < nmax else np.inf
            
            dum_data = np.ascontiguousarray(data[ind_test].flatten(), dtype=np.float32)
            dum_var = np.ascontiguousarray(var[ind_test].flatten(), dtype=np.float32) if not var is None else None
            
            dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits * vnum), dtype=np.float32)
            dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
            fitter(dum_NFits, n, boxsz, splinesz, dum_data, dum_var, coeff_PSF, dum_xvec, dum_crlb, dum_loss, opt, iterations)
            dum_xvec = np.reshape(dum_xvec, (dum_NFits, vnum))
            dum_crlb = np.reshape(dum_crlb, (dum_NFits, vnum))
            
            if n == 1:
                xvec_o[ind_test, :vnum] = dum_xvec
                crlb_o[ind_test, :vnum] = dum_crlb
                Loss_o[ind_test] = dum_loss
                nnum_o[ind_test] = n
                BIC_track[ind_test, n - 1] = dum_loss + vnum * 2.0 * np.log(boxsz)
            else:
                dum_ind = (dum_loss + vnum * 2.0 * np.log(boxsz)) < np.min(BIC_track[ind_test, :n], axis=1)
                xvec_o[ind_test[dum_ind], :vnum] = dum_xvec[dum_ind]
                crlb_o[ind_test[dum_ind], :vnum] = dum_crlb[dum_ind]
                Loss_o[ind_test[dum_ind]] = dum_loss[dum_ind]
                nnum_o[ind_test[dum_ind]] = n
                BIC_track[ind_test, n - 1] = dum_loss + vnum * 2.0 * np.log(boxsz)
                
            dum_ind = dum_loss >= llr_threshold
            ind_test = ind_test[dum_ind]
            dum_NFits = len(ind_test)
            dum_xvec = np.hstack((dum_xvec[dum_ind], np.zeros((dum_NFits, ndim + nchannels))))
            dum_xvec = np.ascontiguousarray(dum_xvec.flatten(), dtype=np.float32) 
    
    return xvec_o, crlb_o, Loss_o, nnum_o


def cspline2d_1ch(data, var, coeff_PSF, nmax, p_val, optmethod, iterations, batsz):
    
    ndim = 2
    vmax = nmax * (ndim + 1) + 1
    batsz = np.int32(batsz)
    if (not var is None):
        assert var.shape == data.shape, "shape mismatch: var.shape={}, data.shape={}".format(var.shape, data.shape) 
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        var = var[np.newaxis, ...] if (not var is None) else None
    assert data.ndim == 3, "ndim mismatch: data.shape={}, should be (NFits, boxsz, boxsz)".format(data.shape)
    assert data.shape[-1] == data.shape[-2], "shape mismatch, box.shape={}, shoud be sqaure".format(data.shape[-2:])
    NFits, boxsz = data.shape[:2]
    nbats = (NFits - 1) // batsz + 1
    splineszx = _perfect_sqrt(len(coeff_PSF) // 16)
    
    xvec = np.zeros((NFits, vmax), dtype=np.float32)
    crlb = np.zeros((NFits, vmax), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)
    nnum = np.zeros(NFits, dtype=np.int32)
    
    for i in range(nbats):
        dum_NFits = min(batsz, NFits-i*batsz)
        dum_data = data[i*batsz : i*batsz + dum_NFits]
        dum_var = var[i*batsz : i*batsz + dum_NFits] if not var is None else None
        
        dum_xvec, dum_crlb, dum_loss, dum_nnum = _fitter_cspline2d1ch(dum_NFits, boxsz, splineszx, dum_data, dum_var, coeff_PSF, nmax, p_val, optmethod, iterations)
        
        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss
        nnum[i*batsz : i*batsz + dum_NFits] = dum_nnum

    return xvec, crlb, Loss, nnum




"""
@brief  batch fit of single-channel box data of PSFs
@param[in]  data:               (NFits, boxsz, boxsz) float ndarray
@param[in]  var:                (NFits, boxsz, boxsz) float ndarray or None
@param[in]  coeff_PSF:          (splineszx * splineszx * 16) float ndarray, the cubic spline coefficients calibrated for the PSF model
@param[in]  lc:                 (nchannels, NFits) int ndarray, the left corner of each data square
@param[in]  uc:                 (nchannels, NFits) int ndarray, the upper corner of each data square
@param[in]  coeff_R2T:          (nchannels - 1, 2, warpdeg * warpdeg) float ndarray, the polynomial coefficients calibrated for channel warpping
@param[in]  nmax:               int, maximum number of emitters for mulpa
@param[in]  p_val:              float, pvalue threshold to consider include one more emitter
@param[in]  optmethod:          str, {'MLE', 'LSQ'} optimization method
@param[in]  iterations:         int, number of iterations for optimization
@param[in]  batsz:              int, number of PSFs sent to GPU concurrently
@param[out] xvec:			    (NFits, vnum) float ndarray
@param[out] CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
@param[out] Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
@param[out] nnum:				(NFits,) int ndarray, number of emitters for the optimization of a PSF fitting.
"""

def _fitter_cspline2d2ch(NFits, boxsz, splinesz, data, var, coeff_PSF, warpdeg, lu, coeff_R2T, nmax, p_val, optmethod, iterations):
    
    ### WARNING !! NO INPUTS CHECK ###
    ndim = 2
    nchannels = 2
    vmax = nmax * (ndim + nchannels) + nchannels
    opt = 0 if optmethod == 'MLE' else 1

    # allocate
    xvec_o = np.zeros((NFits, vmax), dtype=np.float32)
    crlb_o = np.zeros((NFits, vmax), dtype=np.float32)
    Loss_o = np.zeros(NFits, dtype=np.float32) + np.inf
    nnum_o = np.zeros(NFits, dtype=np.int32)
    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    coeff_R2T = np.ascontiguousarray(coeff_R2T.flatten(), dtype=np.float32)

    # link the fitter_c
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_cspline2d')
    fname = 'cspline2d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    fitter = lib.cspline2d_2ch
    if var is None:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctypes.c_void_p,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, 
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32]
    else:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, 
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32]  
    
    BIC_track = np.zeros((NFits, nmax), dtype=np.float32) + np.inf
    ind_test = np.arange(NFits)
    dum_xvec = np.ascontiguousarray(np.zeros(NFits * (ndim + nchannels + nchannels)), dtype=np.float32)
    dum_NFits = NFits
    for n in range(1, nmax + 1):
        
        if dum_NFits > 0:
            vnum = n * (ndim + nchannels) + nchannels
            llr_threshold = chi2.ppf(1.0 - p_val, nchannels * boxsz * boxsz - vnum) if n < nmax else np.inf
            
            dum_data = np.ascontiguousarray(data[ind_test].flatten(), dtype=np.float32)
            dum_var = np.ascontiguousarray(var[ind_test].flatten(), dtype=np.float32) if not var is None else None
            dum_lu = np.ascontiguousarray(lu[ind_test].flatten(), dtype=np.int32)

            dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits * vnum), dtype=np.float32)
            dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
            fitter(dum_NFits, n, boxsz, splinesz, dum_data, dum_var, coeff_PSF, warpdeg, dum_lu, coeff_R2T, dum_xvec, dum_crlb, dum_loss, opt, iterations)
            dum_xvec = np.reshape(dum_xvec, (dum_NFits, vnum))
            dum_crlb = np.reshape(dum_crlb, (dum_NFits, vnum))
            
            if n == 1:
                xvec_o[ind_test, :vnum] = dum_xvec
                crlb_o[ind_test, :vnum] = dum_crlb
                Loss_o[ind_test] = dum_loss
                nnum_o[ind_test] = n
                BIC_track[ind_test, n - 1] = dum_loss + vnum * 2.0 * np.log(boxsz)
            else:
                dum_ind = (dum_loss + vnum * 2.0 * np.log(boxsz)) < np.min(BIC_track[ind_test, :n], axis=1)
                xvec_o[ind_test[dum_ind], :vnum] = dum_xvec[dum_ind]
                crlb_o[ind_test[dum_ind], :vnum] = dum_crlb[dum_ind]
                Loss_o[ind_test[dum_ind]] = dum_loss[dum_ind]
                nnum_o[ind_test[dum_ind]] = n
                BIC_track[ind_test, n - 1] = dum_loss + vnum * 2.0 * np.log(boxsz)
                
            dum_ind = dum_loss >= llr_threshold
            ind_test = ind_test[dum_ind]
            dum_NFits = len(ind_test)
            dum_xvec = np.hstack((dum_xvec[dum_ind], np.zeros((dum_NFits, ndim + nchannels))))
            dum_xvec = np.ascontiguousarray(dum_xvec.flatten(), dtype=np.float32)   
            
    return xvec_o, crlb_o, Loss_o, nnum_o


def cspline2d_2ch(data, var, coeff_PSF, lc, uc, coeff_R2T, nmax, p_val, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    ndim = 2
    vmax = nmax * (ndim + 2) + 2
    batsz = np.int32(batsz)
    
    if (not var is None): 
        assert var.shape == data.shape, "shape mismatch: var.shape={}, data.shape={}".format(var.shape, data.shape)
    if data.ndim == 3:
        data = data[:, np.newaxis, ...]
        var = var[:, np.newaxis, ...] if (not var is None) else None
    assert data.ndim == 4, "ndim mismatch: data.shape={}, should be (nchannels, NFits, boxsz, boxsz)".format(data.shape)
    assert data.shape[-1] == data.shape[-2], "shape mismatch, box.shape={}, shoud be sqaure".format(data.shape[-2:])
    nchannels, NFits, boxsz = data.shape[:3]
    nbats = (NFits - 1) // batsz + 1
    assert nchannels == 2, "nchannels mismatch, data.nchannels={}, shoud be 2".format(nchannels)
    data = np.transpose(data, (1, 0, 2, 3))
    var = np.transpose(var, (1, 0, 2, 3)) if not var is None else None

    assert np.any(lc.shape == (nchannels, NFits)), "shape mismatch: lc.(nchannels, NFits)={}, data.(nchannels, NFits)={}".format(lc.shape, (nchannels, NFits))
    assert np.any(uc.shape == (nchannels, NFits)), "shape mismatch: uc.(nchannels, NFits)={}, data.(nchannels, NFits)={}".format(uc.shape, (nchannels, NFits))
    lu = np.transpose(np.array([lc, uc]), (2, 1, 0))

    assert len(coeff_PSF) == nchannels, "nchannel mismatch: coeff_PSF.nchannels={}, data.nchannels={}".format(len(coeff_PSF), nchannels)
    splineszx = _perfect_sqrt(len(coeff_PSF[0]) // 16)
    assert len(coeff_R2T) == nchannels - 1, "nchannel mismatch: coeff_R2T.nchannels={}, data.nchannels={}".format(len(coeff_R2T), nchannels)
    warpdeg = _perfect_sqrt(len(coeff_R2T[0][0]))

    xvec = np.zeros((NFits, vmax), dtype=np.float32)
    crlb = np.zeros((NFits, vmax), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)
    nnum = np.zeros(NFits, dtype=np.int32)
    
    for i in range(nbats):
        dum_NFits = min(batsz, NFits-i*batsz)
        dum_data = data[i*batsz : i*batsz + dum_NFits]
        dum_var = var[i*batsz : i*batsz + dum_NFits] if not var is None else None
        dum_lu = lu[i*batsz : i*batsz + dum_NFits]
        
        dum_xvec, dum_crlb, dum_loss, dum_nnum = _fitter_cspline2d2ch(dum_NFits, boxsz, splineszx, dum_data, dum_var, coeff_PSF, warpdeg, dum_lu, coeff_R2T, 
                                                                      nmax, p_val, optmethod, iterations)
        
        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss
        nnum[i*batsz : i*batsz + dum_NFits] = dum_nnum

    return xvec, crlb, Loss, nnum




"""
@brief  batch fit of single-channel box data of PSFs
@param[in]  data:               (NFits, boxsz, boxsz) float ndarray
@param[in]  var:                (NFits, boxsz, boxsz) float ndarray or None
@param[in]  splineszx:          int, size of the spline cube (lateral-axis)
@param[in]  splineszz:          int, size of the spline cube (axial-axis)
@param[in]  coeff_PSF:          (splineszx * splineszx * splineszz * 64) float ndarray, the cubic spline coefficients calibrated for the PSF model
@param[in]  nmax:               int, maximum number of emitters for mulpa
@param[in]  p_val:              float, pvalue threshold to consider include one more emitter
@param[in]  optmethod:          str, {'MLE', 'LSQ'} optimization method
@param[in]  iterations:         int, number of iterations for optimization
@param[out] xvec:			    (NFits, vnum) float ndarray
@param[out] CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
@param[out] Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
@param[out] nnum:				(NFits,) int ndarray, number of emitters for the optimization of a PSF fitting.
"""

def _fitter_cspline3d1ch(NFits, boxsz, splineszx, splineszz, data, var, coeff_PSF, nmax, p_val, optmethod, iterations):
    
    ### WARNING !! NO INPUTS CHECK ###
    ndim = 3
    nchannels = 1
    vnum0 = ndim + nchannels
    vmax = nmax * vnum0 + nchannels
    opt = 0 if optmethod == 'MLE' else 1

    # allocate
    xvec_o = np.zeros((NFits, vmax), dtype=np.float32)
    crlb_o = np.zeros((NFits, vmax), dtype=np.float32)
    Loss_o = np.zeros(NFits, dtype=np.float32) + np.inf
    nnum_o = np.zeros(NFits, dtype=np.int32)
    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    
    # link the fitter_c
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_cspline3d')
    fname = 'cspline3d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    fitter = lib.cspline3d_1ch
    if var is None:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctypes.c_void_p,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32]
    else:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32]  
    
    BIC_track = np.zeros((NFits, nmax), dtype=np.float32) + np.inf
    ind_test = np.arange(NFits)
    dum_xvec = np.ascontiguousarray(np.zeros(NFits * (1 * vnum0 + nchannels)), dtype=np.float32)
    dum_NFits = NFits
    for n in range(1, nmax + 1):
        
        if dum_NFits > 0:
            vnum = n * vnum0 + nchannels
            llr_threshold = chi2.ppf(1.0 - p_val, nchannels * boxsz * boxsz - vnum) if n < nmax else np.inf
            
            dum_data = np.ascontiguousarray(data[ind_test].flatten(), dtype=np.float32)
            dum_var = np.ascontiguousarray(var[ind_test].flatten(), dtype=np.float32) if not var is None else None
            
            dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits * vnum), dtype=np.float32)
            dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
            fitter(dum_NFits, n, boxsz, splineszx, splineszz, dum_data, dum_var, coeff_PSF, dum_xvec, dum_crlb, dum_loss, opt, iterations)
            dum_xvec = np.reshape(dum_xvec, (dum_NFits, vnum))
            dum_crlb = np.reshape(dum_crlb, (dum_NFits, vnum))
            
            if n == 1:
                xvec_o[ind_test, :vnum] = dum_xvec
                crlb_o[ind_test, :vnum] = dum_crlb
                Loss_o[ind_test] = dum_loss
                nnum_o[ind_test] = n
                BIC_track[ind_test, n - 1] = dum_loss + vnum * 2.0 * np.log(boxsz)
            else:
                dum_ind = (dum_loss + vnum * 2.0 * np.log(boxsz)) < np.min(BIC_track[ind_test, :n], axis=1)
                xvec_o[ind_test[dum_ind], :vnum] = dum_xvec[dum_ind]
                crlb_o[ind_test[dum_ind], :vnum] = dum_crlb[dum_ind]
                Loss_o[ind_test[dum_ind]] = dum_loss[dum_ind]
                nnum_o[ind_test[dum_ind]] = n
                BIC_track[ind_test, n - 1] = dum_loss + vnum * 2.0 * np.log(boxsz)
               
            dum_ind = dum_loss >= llr_threshold
            ind_test = ind_test[dum_ind]
            dum_NFits = len(ind_test)
            dum_xvec = np.hstack((dum_xvec[dum_ind], np.zeros((dum_NFits, vnum0))))
            dum_xvec = np.ascontiguousarray(dum_xvec.flatten(), dtype=np.float32)    
    
    return xvec_o, crlb_o, Loss_o, nnum_o


def cspline3d_1ch(data, var, splineszx, splineszz, coeff_PSF, nmax, p_val, optmethod, iterations, batsz):
    
    ndim = 3
    nchannels = 1
    vnum0 = ndim + nchannels
    vmax = nmax * vnum0 + nchannels
    batsz = np.int32(batsz)
    
    if (not var is None):
        assert var.shape == data.shape, "shape mismatch: var.shape={}, data.shape={}".format(var.shape, data.shape) 
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        var = var[np.newaxis, ...] if (not var is None) else None
    assert data.ndim == 3, "ndim mismatch: data.shape={}, should be (NFits, boxsz, boxsz)".format(data.shape)
    assert data.shape[-1] == data.shape[-2], "shape mismatch, box.shape={}, shoud be sqaure".format(data.shape[-2:])
    NFits, boxsz = data.shape[:2]
    nbats = (NFits - 1) // batsz + 1
    assert len(coeff_PSF) == splineszx * splineszx * splineszz * 64, "size mismatch: input.splineszx={}, input.splineszz={}, len(coeff_PSF)={}".format(splineszx, splineszz, len(coeff_PSF))

    xvec = np.zeros((NFits, vmax), dtype=np.float32)
    crlb = np.zeros((NFits, vmax), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)
    nnum = np.zeros(NFits, dtype=np.int32)

    for i in range(nbats):
        dum_NFits = min(batsz, NFits-i*batsz)
        dum_data = data[i*batsz : i*batsz + dum_NFits]
        dum_var = var[i*batsz : i*batsz + dum_NFits] if not var is None else None
        
        dum_xvec, dum_crlb, dum_loss, dum_nnum = _fitter_cspline3d1ch(dum_NFits, boxsz, splineszx, splineszz, dum_data, dum_var, coeff_PSF, nmax, p_val, optmethod, iterations)
        
        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss
        nnum[i*batsz : i*batsz + dum_NFits] = dum_nnum
    
    return xvec, crlb, Loss, nnum




"""
@brief  batch fit of single-channel box data of PSFs
@param[in]  data:               (NFits, boxsz, boxsz) float ndarray
@param[in]  var:                (NFits, boxsz, boxsz) float ndarray or None
@param[in]  splineszx:          int, size of the spline cube (lateral-axis)
@param[in]  splineszz:          int, size of the spline cube (axial-axis)
@param[in]  coeff_PSF:          (splineszx * splineszx * 16) float ndarray, the cubic spline coefficients calibrated for the PSF model
@param[in]  lc:                 (nchannels, NFits) int ndarray, the left corner of each data square
@param[in]  uc:                 (nchannels, NFits) int ndarray, the upper corner of each data square
@param[in]  coeff_R2T:          (nchannels - 1, 2, warpdeg * warpdeg) float ndarray, the polynomial coefficients calibrated for channel warpping
@param[in]  nmax:               int, maximum number of emitters for mulpa
@param[in]  p_val:              float, pvalue threshold to consider include one more emitter
@param[in]  optmethod:          str, {'MLE', 'LSQ'} optimization method
@param[in]  iterations:         int, number of iterations for optimization
@param[in]  batsz:              int, number of PSFs sent to GPU concurrently
@param[out] xvec:			    (NFits, vnum) float ndarray
@param[out] CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
@param[out] Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
@param[out] nnum:				(NFits,) int ndarray, number of emitters for the optimization of a PSF fitting.
"""

def _fitter_cspline3d2ch(NFits, boxsz, splineszx, splineszz, data, var, coeff_PSF, warpdeg, lu, coeff_R2T, nmax, p_val, isBP, optmethod, iterations):
    
    ### WARNING !! NO INPUTS CHECK ###
    ndim = 3
    nchannels = 2
    vnum0 = ndim + 1 if isBP else ndim + 2
    vmax = nmax * vnum0 + 1 if isBP else nmax * vnum0 + 2
    opt = 0 if optmethod == 'MLE' else 1

    # allocate
    xvec_o = np.zeros((NFits, vmax), dtype=np.float32)
    crlb_o = np.zeros((NFits, vmax), dtype=np.float32)
    Loss_o = np.zeros(NFits, dtype=np.float32) + np.inf
    nnum_o = np.zeros(NFits, dtype=np.int32)
    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    coeff_R2T = np.ascontiguousarray(coeff_R2T.flatten(), dtype=np.float32)

    # link the fitter_c
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_cspline3d')
    fname = 'cspline3d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    fitter = lib.cspline3d_BP if isBP else lib.cspline3d_2ch
    if var is None:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctypes.c_void_p,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, 
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32]
    else:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, 
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32]  
    
    BIC_track = np.zeros((NFits, nmax), dtype=np.float32) + np.inf
    ind_test = np.arange(NFits)
    if isBP:
        dum_xvec = np.ascontiguousarray(np.zeros(NFits * (1 * vnum0 + 1)), dtype=np.float32)
    else:
        dum_xvec = np.ascontiguousarray(np.zeros(NFits * (1 * vnum0 + 2)), dtype=np.float32)
    dum_NFits = NFits
    for n in range(1, nmax + 1):
        
        if dum_NFits > 0:
            vnum = n * vnum0 + 1 if isBP else n * vnum0 + 2
            llr_threshold = chi2.ppf(1.0 - p_val, nchannels * boxsz * boxsz - vnum) if n < nmax else np.inf
            
            dum_data = np.ascontiguousarray(data[ind_test].flatten(), dtype=np.float32)
            dum_var = np.ascontiguousarray(var[ind_test].flatten(), dtype=np.float32) if not var is None else None
            dum_lu = np.ascontiguousarray(lu[ind_test].flatten(), dtype=np.int32)

            dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits * vnum), dtype=np.float32)
            dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
            fitter(dum_NFits, n, boxsz, splineszx, splineszz, dum_data, dum_var, coeff_PSF, warpdeg, dum_lu, coeff_R2T, dum_xvec, dum_crlb, dum_loss, opt, iterations)
            dum_xvec = np.reshape(dum_xvec, (dum_NFits, vnum))
            dum_crlb = np.reshape(dum_crlb, (dum_NFits, vnum))
            
            if n == 1:
                xvec_o[ind_test, :vnum] = dum_xvec
                crlb_o[ind_test, :vnum] = dum_crlb
                Loss_o[ind_test] = dum_loss
                nnum_o[ind_test] = n
                BIC_track[ind_test, n - 1] = dum_loss + vnum * 2.0 * np.log(boxsz)
            else:
                dum_ind = (dum_loss + vnum * 2.0 * np.log(boxsz)) < np.min(BIC_track[ind_test, :n], axis=1)
                xvec_o[ind_test[dum_ind], :vnum] = dum_xvec[dum_ind]
                crlb_o[ind_test[dum_ind], :vnum] = dum_crlb[dum_ind]
                Loss_o[ind_test[dum_ind]] = dum_loss[dum_ind]
                nnum_o[ind_test[dum_ind]] = n
                BIC_track[ind_test, n - 1] = dum_loss + vnum * 2.0 * np.log(boxsz)
                
            dum_ind = dum_loss >= llr_threshold
            ind_test = ind_test[dum_ind]
            dum_NFits = len(ind_test)
            dum_xvec = np.hstack((dum_xvec[dum_ind], np.zeros((dum_NFits, vnum0))))
            dum_xvec = np.ascontiguousarray(dum_xvec.flatten(), dtype=np.float32)   
            
    return xvec_o, crlb_o, Loss_o, nnum_o


def cspline3d_2ch(data, var, splineszx, splineszz, coeff_PSF, lc, uc, coeff_R2T, nmax, p_val, isBP, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    ndim = 3
    vnum0 = ndim + 1 if isBP else ndim + 2
    vmax = nmax * vnum0 + 1 if isBP else nmax * vnum0 + 2
    batsz = np.int32(batsz)
    
    if (not var is None):
        assert var.shape == data.shape, "shape mismatch: var.shape={}, data.shape={}".format(var.shape, data.shape)
    if data.ndim == 3:
        data = data[:, np.newaxis, ...]
        var = var[:, np.newaxis, ...] if (not var is None) else None
    assert data.ndim == 4, "ndim mismatch: data.shape={}, should be (nchannels, NFits, boxsz, boxsz)".format(data.shape)
    assert data.shape[-1] == data.shape[-2], "shape mismatch, box.shape={}, shoud be sqaure".format(data.shape[-2:])
    nchannels, NFits, boxsz = data.shape[:3]
    nbats = (NFits - 1) // batsz + 1
    assert nchannels == 2, "nchannels mismatch, data.nchannels={}, shoud be 2".format(nchannels)
    data = np.transpose(data, (1, 0, 2, 3))
    if not var is None:
        var = np.transpose(var, (1, 0, 2, 3))

    assert np.any(lc.shape == (nchannels, NFits)), "shape mismatch: lc.(nchannels, NFits)={}, data.(nchannels, NFits)={}".format(lc.shape, (nchannels, NFits))
    assert np.any(uc.shape == (nchannels, NFits)), "shape mismatch: uc.(nchannels, NFits)={}, data.(nchannels, NFits)={}".format(uc.shape, (nchannels, NFits))
    lu = np.transpose(np.array([lc, uc]), (2, 1, 0))

    assert len(coeff_PSF) == nchannels, "nchannel mismatch: coeff_PSF.nchannels={}, data.nchannels={}".format(len(coeff_PSF), nchannels)
    assert len(coeff_PSF[0]) == splineszx * splineszx * splineszz * 64, "size mismatch: input.splineszx={}, input.splineszz={}, len(coeff_PSF)={}".format(splineszx, splineszz, len(coeff_PSF[0]))
    assert len(coeff_R2T) == nchannels - 1, "nchannel mismatch: coeff_R2T.nchannels={}, data.nchannels={}".format(len(coeff_R2T), nchannels)
    warpdeg = _perfect_sqrt(len(coeff_R2T[0][0]))

    # INITIALIZATION AND ALLOCATIONS
    xvec = np.zeros((NFits, vmax), dtype=np.float32)
    crlb = np.zeros((NFits, vmax), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)
    nnum = np.zeros(NFits, dtype=np.int32)
    
    for i in range(nbats):
        dum_NFits = min(batsz, NFits-i*batsz)
        dum_data = data[i*batsz : i*batsz + dum_NFits]
        dum_var = var[i*batsz : i*batsz + dum_NFits] if not var is None else None
        dum_lu = lu[i*batsz : i*batsz + dum_NFits]
        
        dum_xvec, dum_crlb, dum_loss, dum_nnum = _fitter_cspline3d2ch(dum_NFits, boxsz, splineszx, splineszz, dum_data, dum_var, coeff_PSF, 
                                                                      warpdeg, dum_lu, coeff_R2T, nmax, p_val, isBP, optmethod, iterations)

        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss
        nnum[i*batsz : i*batsz + dum_NFits] = dum_nnum

    return xvec, crlb, Loss, nnum





"""
wrapper of the fitters above
INPUT:
    data:               (nchannels, NFits, boxsz, boxsz) float ndarray
    var:                (nchannels, NFits, boxsz, boxsz) float ndarray
    PSF:                python dict, calinrated PSF, see README.md
    modality:           str, {'2D', '3D'} modality option for psf fit, see Appendix A for details
    lc:                 (nchannels, NFits) int ndarray, the left corner of each data square
    uc:                 (nchannels, NFits) int ndarray, the upper corner of each data square
    coeff_R2T:          (nchannels-1, 2, warpdeg*warpdeg) float ndarray, the polynomial coefficients calibrated for channel warpping
    nmax:               int, maximum number of emitters for mulpa
    p_val:              float, pvalue threshold to consider include one more emitter
    optmethod:          str, optimization option for psf fit, see Appendix A for details
                        'MLE':  optimization via Maximum Likelihood Estimation, used for raw camera images of PSFs
                        'LSQ':  optimization via Least Square, usually used for aligned / normalized camera images of PSFs
    iterations:         int, number of iterations for optimization
    batsz:              int, number of PSFs sent to GPU concurrently
RETURN:
    xvec:			    (NFits, vnum) float ndarray
    CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
    Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting
    nnum:				(NFits,) int ndarray, number of emitters for the optimization of a PSF fitting.           
"""
def mulpa_cspline_1ch(data, var, PSF, nmax, p_val=0.05, optmethod='MLE', iterations=100, batsz=65535):
    
    modality = PSF['modality']
    coeff_PSF = PSF['coeff'][0]
    if modality == '2Dfrm':
        xvec, crlb, Loss, nnum = cspline2d_1ch(data, var, coeff_PSF, nmax, p_val, optmethod, iterations, batsz)
        
    elif modality in {'2D', 'AS', 'DH'}:
        splineszx = PSF['splineszx']
        splineszz = PSF['splineszz']
        xvec, crlb, Loss, nnum = cspline3d_1ch(data, var, splineszx, splineszz, coeff_PSF, nmax, p_val, optmethod, iterations, batsz)
        
    else:
        raise ValueError("only '2Dfrm', '2D', 'AS', and 'DH' is available for mulpa_cspline_1ch")
    return xvec, crlb, Loss, nnum



def mulpa_cspline_2ch(data, varim, PSF, lc, uc, coeff_R2T, nmax, p_val = 0.05, optmethod='MLE', iterations=100, batsz=65535):
    
    modality = PSF['modality']
    coeff_PSF = PSF['coeff']
    if modality == '2Dfrm':
        xvec, crlb, Loss, nnum = cspline2d_2ch(data, varim, coeff_PSF, lc, uc, coeff_R2T, nmax, p_val, optmethod, iterations, batsz)
    
    elif modality in {'2D', 'AS', 'DH', 'BP'}:
        isBP = modality == 'BP'
        splineszx = PSF['splineszx']
        splineszz = PSF['splineszz']
        xvec, crlb, Loss, nnum = cspline3d_2ch(data, varim, splineszx, splineszz, coeff_PSF, lc, uc, coeff_R2T, nmax, p_val, isBP, optmethod, iterations, batsz)
        
    else:
        raise ValueError("only '2Dfrm', '2D', 'AS', 'DH', and 'BP' is available for mulpa_cspline_2ch")
    return xvec, crlb, Loss, nnum