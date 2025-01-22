import os
import inspect
import ctypes
import numpy.ctypeslib as ctl 
import numpy as np



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
@brief  batch fit of PSF with single cspline2d
@param[in]  data:               (nchannels, NFits, boxsz, boxsz) float ndarray, the nchannel data
@param[in]  var:                (nchannels, NFits, boxsz, boxsz) float ndarray, the nchannel data
@param[in]  coeff_PSF:          (nchannels, splinesz * splinesz * 16) float ndarray, the cubic spline coefficients calibrated for the nchannel-PSF model
@param[in]  lc:                 (nchannels, NFits) int ndarray, the left corner of each data square
@param[in]  uc:                 (nchannels, NFits) int ndarray, the upper corner of each data square
@param[in]  coeff_R2T:          (nchannels - 1, 2, warpdeg * warpdeg) float ndarray, the polynomial coefficients calibrated for channel warpping
@param[in]  optmethod:          str, {'MLE', 'LSQ'} optimization method
@param[in]  iterations:         int, number of iterations for optimization
@param[in]  batsz:              int, number of PSFs sent to GPU concurrently
@param[out] xvec:			    (NFits, vnum) float ndarray
@param[out] CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
@param[out] Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
"""

def _cspline2d_1ch(data, var, coeff_PSF, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    NDIM = 2
    vnum = NDIM + 2
    opt = 0 if optmethod == 'MLE' else 1
    batsz = np.int32(batsz)
    
    if (not var is None):
        assert var.shape == data.shape, "shape mismatch: var.shape={}, data.shape={}".format(var.shape, data.shape) 
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        var = var[np.newaxis, ...] if (not var is None) else None
    assert data.ndim == 3, "ndim mismatch: data.shape={}, should be (NFits, boxsz, boxsz)".format(data.shape)
    assert data.shape[-1] == data.shape[-2], "shape mismatch, box.shape={}, shoud be sqaure".format(data.shape[-2:])
    NFits, boxsz = data.shape[:2]
    splinesz = _perfect_sqrt(len(coeff_PSF)//16)
    nbats = (NFits - 1) // batsz + 1

    # INITIALIZATION AND ALLOCATIONS
    xvec = np.zeros((NFits, vnum), dtype=np.float32)
    crlb = np.zeros((NFits, vnum), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)

    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_cspline2d')
    fname = 'cspline2d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    
    fitter = lib.cspline2d_1ch
    if var is None:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctypes.c_void_p,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32]
    else:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32]
    for i in range(nbats):
        dum_NFits = min(batsz, NFits-i*batsz)
        dum_data = data[i*batsz : i*batsz + dum_NFits]
        dum_data = np.ascontiguousarray(dum_data.flatten(), dtype=np.float32)
        dum_xvec = np.ascontiguousarray(np.zeros(dum_NFits*vnum), dtype=np.float32)
        dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits*vnum), dtype=np.float32)
        dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
        
        if var is None:
            fitter(dum_NFits, boxsz, splinesz, dum_data, None, coeff_PSF, dum_xvec, dum_crlb, dum_loss, opt, iterations)
        else:
            dum_var = var[i*batsz : i*batsz + dum_NFits]
            dum_var = np.ascontiguousarray(dum_var.flatten(), dtype=np.float32)
            fitter(dum_NFits, boxsz, splinesz, dum_data, dum_var, coeff_PSF, dum_xvec, dum_crlb, dum_loss, opt, iterations)
        
        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec.reshape((dum_NFits, vnum))
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb.reshape((dum_NFits, vnum))
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss

    return xvec, crlb, Loss



def _cspline2d_2ch(data, var, coeff_PSF, lc, uc, coeff_R2T, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    NDIM = 2
    vnum = NDIM + 4
    opt = 0 if optmethod == 'MLE' else 1
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
    splinesz = _perfect_sqrt(len(coeff_PSF[0])//16)
    assert len(coeff_R2T) == nchannels - 1, "nchannel mismatch: coeff_R2T.nchannels={}, data.nchannels={}".format(len(coeff_R2T), nchannels)
    warpdeg = _perfect_sqrt(len(coeff_R2T[0][0]))

    # INITIALIZATION AND ALLOCATIONS
    xvec = np.zeros((NFits, vnum), dtype=np.float32)
    crlb = np.zeros((NFits, vnum), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)
    
    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    coeff_R2T = np.ascontiguousarray(coeff_R2T.flatten(), dtype=np.float32)

    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_cspline2d')
    fname = 'cspline2d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    
    fitter = lib.cspline2d_2ch
    if var is None:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
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
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
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
        
    for i in range(nbats):
        dum_NFits = min(batsz, NFits-i*batsz)
        dum_data = data[i*batsz : i*batsz + dum_NFits]
        dum_lu = lu[i*batsz : i*batsz + dum_NFits]
        
        dum_data = np.ascontiguousarray(dum_data.flatten(), dtype=np.float32)
        dum_lu = np.ascontiguousarray(dum_lu.flatten(), dtype=np.int32)
        dum_xvec = np.ascontiguousarray(np.zeros(dum_NFits*vnum), dtype=np.float32)
        dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits*vnum), dtype=np.float32)
        dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
        
        if var is None:
            fitter(dum_NFits, boxsz, splinesz, dum_data, None, coeff_PSF, warpdeg, dum_lu, coeff_R2T,
                    dum_xvec, dum_crlb, dum_loss, opt, iterations)
        else:
            dum_var = var[i*batsz : i*batsz + dum_NFits]
            dum_var = np.ascontiguousarray(dum_var.flatten(), dtype=np.float32)
            fitter(dum_NFits, boxsz, splinesz, dum_data, dum_var, coeff_PSF, warpdeg, dum_lu, coeff_R2T,
                    dum_xvec, dum_crlb, dum_loss, opt, iterations)
        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec.reshape((dum_NFits, vnum))
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb.reshape((dum_NFits, vnum))
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss

    return xvec, crlb, Loss



"""
@brief  batch fit of PSF via single cspline3d model
@param[in]  data:               (nchannels, NFits, boxsz, boxsz) float ndarray, the nchannel data
@param[in]  var:                (nchannels, NFits, boxsz, boxsz) float ndarray, the nchannel data
@param[in]  coeff_PSF:          (nchannels, splineszx * splineszx * splineszz * 64) float ndarray, the cubic spline coefficients calibrated for the nchannel-PSF model
@param[in]  lc:                 (nchannels, NFits) int ndarray, the left corner of each data square
@param[in]  uc:                 (nchannels, NFits) int ndarray, the upper corner of each data square
@param[in]  coeff_R2T:          (nchannels - 1, 2, warpdeg * warpdeg) float ndarray, the polynomial coefficients calibrated for channel warpping
@param[in]  optmethod:          str, {'MLE', 'LSQ'} optimization method
@param[in]  iterations:         int, number of iterations for optimization
@param[in]  batsz:              int, number of PSFs sent to GPU concurrently
@param[out] xvec:			    (NFits, vnum) float ndarray
@param[out] CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
@param[out] Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
"""

def _cspline3d_1ch(data, var, splineszx, splineszz, coeff_PSF, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    NDIM = 3
    vnum = NDIM + 2
    opt = 0 if optmethod == 'MLE' else 1
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

    # INITIALIZATION AND ALLOCATIONS
    xvec = np.zeros((NFits, vnum), dtype=np.float32)
    crlb = np.zeros((NFits, vnum), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)

    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_cspline3d')
    fname = 'cspline3d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    
    fitter = lib.cspline3d_1ch
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
    for i in range(nbats):
        dum_NFits = min(batsz, NFits-i*batsz)
        dum_data = data[i*batsz : i*batsz + dum_NFits]
        dum_data = np.ascontiguousarray(dum_data.flatten(), dtype=np.float32)
        dum_xvec = np.ascontiguousarray(np.zeros(dum_NFits*vnum), dtype=np.float32)
        dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits*vnum), dtype=np.float32)
        dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
        
        if var is None:
            fitter(dum_NFits, boxsz, splineszx, splineszz, dum_data, None, coeff_PSF, dum_xvec, dum_crlb, dum_loss, opt, iterations)
        else:
            dum_var = var[i*batsz : i*batsz + dum_NFits]
            dum_var = np.ascontiguousarray(dum_var.flatten(), dtype=np.float32)
            fitter(dum_NFits, boxsz, splineszx, splineszz, dum_data, dum_var, coeff_PSF, dum_xvec, dum_crlb, dum_loss, opt, iterations)
        
        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec.reshape((dum_NFits, vnum))
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb.reshape((dum_NFits, vnum))
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss

    return xvec, crlb, Loss



def _cspline3d_2ch(data, var, splineszx, splineszz, coeff_PSF, lc, uc, coeff_R2T, isBP, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    NDIM = 3
    vnum = NDIM + 2 if isBP else NDIM + 4
    opt = 0 if optmethod == 'MLE' else 1
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
    assert len(coeff_PSF[0]) == splineszx * splineszx * splineszz * 64, "size mismatch: input.splineszx={}, input.splineszz={}, len(coeff_PSF)={}".format(splineszx, splineszz, len(coeff_PSF))
    assert len(coeff_R2T) == nchannels - 1, "nchannel mismatch: coeff_R2T.nchannels={}, data.nchannels={}".format(len(coeff_R2T), nchannels)
    warpdeg = _perfect_sqrt(len(coeff_R2T[0][0]))

    # INITIALIZATION AND ALLOCATIONS
    xvec = np.zeros((NFits, vnum), dtype=np.float32)
    crlb = np.zeros((NFits, vnum), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)

    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    coeff_R2T = np.ascontiguousarray(coeff_R2T.flatten(), dtype=np.float32)

    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_cspline3d')
    fname = 'cspline3d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    
    fitter = lib.cspline3d_BP if isBP else lib.cspline3d_2ch 
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
        
    for i in range(nbats):
        dum_NFits = min(batsz, NFits-i*batsz)
        dum_data = data[i*batsz : i*batsz + dum_NFits]
        dum_lu = lu[i*batsz : i*batsz + dum_NFits]
        
        dum_data = np.ascontiguousarray(dum_data.flatten(), dtype=np.float32)
        dum_lu = np.ascontiguousarray(dum_lu.flatten(), dtype=np.int32)
        dum_xvec = np.ascontiguousarray(np.zeros(dum_NFits*vnum), dtype=np.float32)
        dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits*vnum), dtype=np.float32)
        dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
        
        if var is None:
            fitter(dum_NFits, boxsz, splineszx, splineszz, dum_data, None, coeff_PSF, warpdeg, dum_lu, coeff_R2T,
                    dum_xvec, dum_crlb, dum_loss, opt, iterations)
        else:
            dum_var = var[i*batsz : i*batsz + dum_NFits]
            dum_var = np.ascontiguousarray(dum_var.flatten(), dtype=np.float32)
            fitter(dum_NFits, boxsz, splineszx, splineszz, dum_data, dum_var, coeff_PSF, warpdeg, dum_lu, coeff_R2T,
                    dum_xvec, dum_crlb, dum_loss, opt, iterations)
        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec.reshape((dum_NFits, vnum))
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb.reshape((dum_NFits, vnum))
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss

    return xvec, crlb, Loss




"""
wrapper of the fitters above
INPUT:
    data:               (nchannels, NFits, boxsz, boxsz) float ndarray
    var:                (nchannels, NFits, boxsz, boxsz) float ndarray
    PSF:                python dict, calinrated PSF, see README.md
    lc:                 (nchannels, NFits) int ndarray, the left corner of each data square
    uc:                 (nchannels, NFits) int ndarray, the upper corner of each data square
    coeff_R2T:          (nchannels-1, 2, warpdeg*warpdeg) float ndarray, the polynomial coefficients calibrated for channel warpping
    optmethod:          str, optimization option for psf fit, see Appendix A for details
                        'MLE':  optimization via Maximum Likelihood Estimation, used for raw camera images of PSFs
                        'LSQ':  optimization via Least Square, usually used for aligned / normalized camera images of PSFs
    iterations:         int, number of iterations for optimization
    batsz:              int, number of PSFs sent to GPU concurrently
RETURN:
    xvec:			    (NFits, vnum) float ndarray
    CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
    Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
"""
def sinpa_cspline_1ch(data, var, PSF, optmethod='MLE', iterations=100, batsz=65535):
    
    modality = PSF['modality']
    coeff_PSF = PSF['coeff'][0]
    if modality == '2Dfrm':
        xvec, crlb, Loss = _cspline2d_1ch(data, var, coeff_PSF, optmethod, iterations, batsz)
        
    elif modality in {'2D', 'AS', 'DH'}:
        splineszx = PSF['splineszx']
        splineszz = PSF['splineszz']
        xvec, crlb, Loss = _cspline3d_1ch(data, var, splineszx, splineszz, coeff_PSF, optmethod, iterations, batsz)
        
    else:
        raise ValueError("only '2Dfrm', '2D', 'AS', and 'DH' is available for sinpa_cspline_1ch")
    return xvec, crlb, Loss



def sinpa_cspline_2ch(data, varim, PSF, lc, uc, coeff_R2T, optmethod='MLE', iterations=100, batsz=65535):
    
    modality = PSF['modality']
    coeff_PSF = PSF['coeff']
    if modality == '2Dfrm':
        xvec, crlb, Loss = _cspline2d_2ch(data, varim, coeff_PSF, lc, uc, coeff_R2T, optmethod, iterations, batsz)
    
    elif modality in {'2D', 'AS', 'DH', 'BP'}:
        isBP = modality == 'BP'
        splineszx = PSF['splineszx']
        splineszz = PSF['splineszz']
        xvec, crlb, Loss = _cspline3d_2ch(data, varim, splineszx, splineszz, coeff_PSF, lc, uc, coeff_R2T, isBP, optmethod, iterations, batsz)
        
    else:
        raise ValueError("only '2Dfrm', '2D', 'AS', 'DH', and 'BP' is available for sinpa_cspline_2ch")
    return xvec, crlb, Loss