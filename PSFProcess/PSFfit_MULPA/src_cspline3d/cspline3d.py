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
@param[in]  splineszx:          int, size of the spline cube (lateral-axis)
@param[in]  splineszz:          int, size of the spline cube (axial-axis)
@param[in]	zcoarse:		    (2,) float ndarray, the z can be estimated by z_coarse[0] * (varx - vary) + z_coarse[1] for the 0-th channel
@param[in]  coeff_PSF:          (splineszx * splineszx * splineszz * 64) float ndarray, the cubic spline coefficients calibrated for the PSF model
@param[in]  optmethod:          str, {'MLE', 'LSQ'} optimization method
@param[in]  iterations:         int, number of iterations for optimization
@param[out] xvec:			    (NFits, vnum) float ndarray
@param[out] CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
@param[out] Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
@param[out] nnum:				(NFits,) int ndarray, number of emitters for the optimization of a PSF fitting.
"""

def _fitter_cspline3d1ch(NFits, boxsz, splineszx, splineszz, data, var, zcoarse, coeff_PSF, PSFsigmax, PSFsigmay, optmethod, iterations):
    
    ### WARNING !! NO INPUTS CHECK ###
    ndim = 3
    nchannels = 1
    nmax = 5
    vmax = nmax * (ndim + nchannels) + nchannels
    opt = 0 if optmethod == 'MLE' else 1

    # allocate
    xvec_o = np.zeros((NFits, vmax), dtype=np.float32)
    crlb_o = np.zeros((NFits, vmax), dtype=np.float32)
    Loss_o = np.zeros(NFits, dtype=np.float32) + np.inf
    nnum_o = np.zeros(NFits, dtype=np.int32)
    zcoarse = np.ascontiguousarray(zcoarse, dtype=np.float32)
    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    
    # link the fitter_c
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fname = 'cspline3d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    fitter = lib.cspline3d_1ch
    if var is None:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctypes.c_void_p,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_float, ctypes.c_float,
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
                            ctypes.c_float, ctypes.c_float,
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
            llr_threshold = chi2.ppf(1.0 - 1e-6, boxsz * boxsz - vnum) if n < nmax else np.inf
            
            dum_data = np.ascontiguousarray(data[ind_test, :, :].flatten(), dtype=np.float32)
            dum_var = np.ascontiguousarray(var[ind_test, :, :].flatten(), dtype=np.float32) if not var is None else None
            
            dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits * vnum), dtype=np.float32)
            dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
            fitter(dum_NFits, n, boxsz, splineszx, splineszz, dum_data, dum_var, zcoarse, coeff_PSF, PSFsigmax, PSFsigmay, dum_xvec, dum_crlb, dum_loss, opt, iterations)
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



def cspline3d_1ch(data, var, splineszx, splineszz, zcoarse, coeff_PSF, PSFsigmax, PSFsigmay, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    ndim = 3
    nmax = 5
    vmax = nmax * (ndim + 1) + 1
    batsz = np.int32(batsz)
    
    if (not var is None) and var.shape != data.shape: 
        raise ValueError("shape mismatch: var.shape={}, data.shape={}".format(var.shape, data.shape))
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        var = var[np.newaxis, ...] if (not var is None) else None
    if data.ndim != 3:
        raise ValueError("ndim mismatch: data.shape={}, should be (NFits, boxsz, boxsz)".format(data.shape))
    if data.shape[-1] != data.shape[-2]:
        raise ValueError("shape mismatch, box.shape={}, shoud be sqaure".format(data.shape[-2:]))
    NFits, boxsz = data.shape[:2]
    nbats = (NFits - 1) // batsz + 1
    if len(coeff_PSF) != splineszx * splineszx * splineszz * 64:
        raise ValueError("size mismatch: input.splineszx={}, input.splineszz={}, len(coeff_PSF)={}".format(splineszx, splineszz, len(coeff_PSF)))

    # INITIALIZATION AND ALLOCATIONS
    xvec = np.zeros((NFits, vmax), dtype=np.float32)
    crlb = np.zeros((NFits, vmax), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)
    nnum = np.zeros(NFits, dtype=np.int32)

    for i in range(nbats):
        dum_NFits = min(batsz, NFits-i*batsz)
        dum_data = data[i*batsz : i*batsz + dum_NFits]
        dum_var = var[i*batsz : i*batsz + dum_NFits] if not var is None else None
        
        dum_xvec, dum_crlb, dum_loss, dum_nnum = _fitter_cspline3d1ch(dum_NFits, boxsz, splineszx, splineszz, dum_data, dum_var, zcoarse, coeff_PSF, PSFsigmax, PSFsigmay, optmethod, iterations)
        
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
@param[in]	zcoarse:		    (2,) float ndarray, the z can be estimated by z_coarse[0] * (varx - vary) + z_coarse[1] for the 0-th channel
@param[in]  coeff_PSF:          (splineszx * splineszx * 16) float ndarray, the cubic spline coefficients calibrated for the PSF model
@param[in]  lc:                 (nchannels, NFits) int ndarray, the left corner of each data square
@param[in]  uc:                 (nchannels, NFits) int ndarray, the upper corner of each data square
@param[in]  coeff_R2T:          (nchannels - 1, 2, warpdeg * warpdeg) float ndarray, the polynomial coefficients calibrated for channel warpping
@param[in]  optmethod:          str, {'MLE', 'LSQ'} optimization method
@param[in]  iterations:         int, number of iterations for optimization
@param[in]  batsz:              int, number of PSFs sent to GPU concurrently
@param[out] xvec:			    (NFits, vnum) float ndarray
@param[out] CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
@param[out] Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
@param[out] nnum:				(NFits,) int ndarray, number of emitters for the optimization of a PSF fitting.
"""

def _fitter_cspline3d2ch(NFits, boxsz, splineszx, splineszz, data, var, zcoarse, coeff_PSF, PSFsigmax, PSFsigmay, warpdeg, lu, coeff_R2T, optmethod, iterations):
    
    ### WARNING !! NO INPUTS CHECK ###
    ndim = 3
    nchannels = 2
    nmax = 5
    vmax = nmax * (ndim + nchannels) + nchannels
    opt = 0 if optmethod == 'MLE' else 1

    # allocate
    xvec_o = np.zeros((NFits, vmax), dtype=np.float32)
    crlb_o = np.zeros((NFits, vmax), dtype=np.float32)
    Loss_o = np.zeros(NFits, dtype=np.float32) + np.inf
    nnum_o = np.zeros(NFits, dtype=np.int32)
    zcoarse = np.ascontiguousarray(zcoarse, dtype=np.float32)
    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    coeff_R2T = np.ascontiguousarray(coeff_R2T.flatten(), dtype=np.float32)

    # link the fitter_c
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fname = 'cspline3d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    fitter = lib.cspline3d_2ch
    if var is None:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctypes.c_void_p,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_float, ctypes.c_float, ctypes.c_int32, 
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
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_float, ctypes.c_float, ctypes.c_int32, 
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
            llr_threshold = chi2.ppf(1.0 - 1e-6, boxsz * boxsz - vnum) if n < nmax else np.inf
            
            dum_data = np.ascontiguousarray(data[ind_test, :, :].flatten(), dtype=np.float32)
            dum_var = np.ascontiguousarray(var[ind_test, :, :].flatten(), dtype=np.float32) if not var is None else None
            dum_lu = np.ascontiguousarray(lu[ind_test, :, :].flatten(), dtype=np.int32)

            dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits * vnum), dtype=np.float32)
            dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
            fitter(dum_NFits, n, boxsz, splineszx, splineszz, dum_data, dum_var, zcoarse, coeff_PSF, PSFsigmax, PSFsigmay, warpdeg, dum_lu, coeff_R2T, dum_xvec, dum_crlb, dum_loss, opt, iterations)
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


def cspline3d_2ch(data, var, splineszx, splineszz, zcoarse, coeff_PSF, PSFsigmax, PSFsigmay, lc, uc, coeff_R2T, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    ndim = 3
    nmax = 5
    vmax = nmax * (ndim + 2) + 2
    batsz = np.int32(batsz)
    
    if (not var is None) and var.shape != data.shape: 
        raise ValueError("shape mismatch: var.shape={}, data.shape={}".format(var.shape, data.shape))
    if data.ndim == 3:
        data = data[:, np.newaxis, ...]
        var = var[:, np.newaxis, ...] if (not var is None) else None
    if data.ndim != 4:
        raise ValueError("ndim mismatch: data.shape={}, should be (nchannels, NFits, boxsz, boxsz)".format(data.shape))
    if data.shape[-1] != data.shape[-2]:
        raise ValueError("shape mismatch, box.shape={}, shoud be sqaure".format(data.shape[-2:]))
    nchannels, NFits, boxsz = data.shape[:3]
    nbats = (NFits - 1) // batsz + 1
    if nchannels != 2:
        raise ValueError("nchannels mismatch, data.nchannels={}, shoud be 2".format(nchannels))
    data = np.transpose(data, (1, 0, 2, 3))
    if not var is None:
        var = np.transpose(var, (1, 0, 2, 3))

    if np.any(lc.shape != (nchannels, NFits)):
        raise ValueError("shape mismatch: lc.(nchannels, NFits)={}, data.(nchannels, NFits)={}".format(lc.shape, (nchannels, NFits)))
    if np.any(uc.shape != (nchannels, NFits)):
        raise ValueError("shape mismatch: uc.(nchannels, NFits)={}, data.(nchannels, NFits)={}".format(uc.shape, (nchannels, NFits)))
    lu = np.transpose(np.array([lc, uc]), (2, 1, 0))

    if len(coeff_PSF) != nchannels:
        raise ValueError("nchannel mismatch: coeff_PSF.nchannels={}, data.nchannels={}".format(len(coeff_PSF), nchannels))
    if len(coeff_PSF[0]) != splineszx * splineszx * splineszz * 64:
        raise ValueError("size mismatch: input.splineszx={}, input.splineszz={}, len(coeff_PSF)={}".format(splineszx, splineszz, len(coeff_PSF)))

    if len(coeff_R2T) != nchannels - 1:
        raise ValueError("nchannel mismatch: coeff_R2T.nchannels={}, data.nchannels={}".format(len(coeff_R2T), nchannels))
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
        
        dum_xvec, dum_crlb, dum_loss, dum_nnum = _fitter_cspline3d2ch(dum_NFits, boxsz, splineszx, splineszz, dum_data, dum_var, zcoarse, coeff_PSF, PSFsigmax, PSFsigmay, warpdeg, dum_lu, coeff_R2T, optmethod, iterations)

        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss
        nnum[i*batsz : i*batsz + dum_NFits] = dum_nnum

    return xvec, crlb, Loss, nnum