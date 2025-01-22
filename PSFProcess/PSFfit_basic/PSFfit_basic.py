import os
import inspect
import ctypes
import numpy.ctypeslib as ctl 
import numpy as np



def _perfect_sqrt(n):
    """get the integer square root of the input integer n"""    
    assert isinstance(n, (int, np.int_)), "type mismatch: type(n)={}, should be int ot np.int_".format(type(n))
    if n <= 1:
        return n
 
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
@brief  batch fit of PSF with single gauss2d
@param[in]  data:               (NFits, boxsz, boxsz) float ndarray, the nchannel data
@param[in]  var:                (NFits, boxsz, boxsz) float ndarray, the nchannel data
@param[in]	PSFsigmax: 		    float, sigma (x-axis) of the Gaussian PSF
@param[in]	PSFsigmay: 		    float, sigma (y-axis) of the Gaussian PSF
@param[in]  astigs:             (9) float ndarray, the astig parameters for gauss3d
@param[in]  coeff_PSF:          (splineszs) float ndarray, the cubic spline coefficients calibrated for the PSF model
@param[in]  fixs:               int, 1 for fixing the PSFsigmax and PSFsigmny during fitting, 0 for free fitting of PSFsigmax and PSFsigmay
@param[in]  optmethod:          str, {'MLE', 'LSQ'} optimization method
@param[in]  iterations:         int, number of iterations for optimization
@param[in]  batsz:              int, number of PSFs sent to GPU concurrently
@param[out] xvec:			    (NFits, vnum) float ndarray
@param[out] CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
@param[out] Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
"""

def gauss2d(data, var, PSFsigmax, PSFsigmay, fixs, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    NDIM = 2
    vnum = NDIM + 2 if fixs == 1 else NDIM + 4
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

    # INITIALIZATION AND ALLOCATIONS
    xvec = np.zeros((NFits, vnum), dtype=np.float32)
    crlb = np.zeros((NFits, vnum), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)

    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_gauss2d')
    fname = 'gauss2d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    
    fitter = lib.gauss2d
    if var is None:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctypes.c_void_p,
                            ctypes.c_float, ctypes.c_float,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    else:
        fitter.argtypes = [ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_float, ctypes.c_float,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    for i in range(nbats):
        dum_NFits = min(batsz, NFits-i*batsz)
        dum_data = data[i*batsz : i*batsz + dum_NFits]
        dum_data = np.ascontiguousarray(dum_data.flatten(), dtype=np.float32)
        dum_xvec = np.ascontiguousarray(np.zeros(dum_NFits*vnum), dtype=np.float32)
        dum_crlb = np.ascontiguousarray(np.zeros(dum_NFits*vnum), dtype=np.float32)
        dum_loss = np.ascontiguousarray(np.zeros(dum_NFits), dtype=np.float32)
        
        if var is None:
            fitter(dum_NFits, boxsz, dum_data, None, PSFsigmax, PSFsigmay, dum_xvec, dum_crlb, dum_loss, fixs, opt, iterations)
        else:
            dum_var = var[i*batsz : i*batsz + dum_NFits]
            dum_var = np.ascontiguousarray(dum_var.flatten(), dtype=np.float32)
            fitter(dum_NFits, boxsz, dum_data, dum_var, PSFsigmax, PSFsigmay, dum_xvec, dum_crlb, dum_loss, fixs, opt, iterations)
        
        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec.reshape((dum_NFits, vnum))
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb.reshape((dum_NFits, vnum))
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss

    return xvec, crlb, Loss



def gauss3d(data, var, zrange, astigs, optmethod, iterations, batsz):
    
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

    # INITIALIZATION AND ALLOCATIONS
    xvec = np.zeros((NFits, vnum), dtype=np.float32)
    crlb = np.zeros((NFits, vnum), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)

    astigs = np.ascontiguousarray(astigs.flatten(), dtype=np.float32)
    
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_gauss3d')
    fname = 'gauss3d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    
    fitter = lib.gauss3d
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
            fitter(dum_NFits, boxsz, zrange, dum_data, None, astigs, dum_xvec, dum_crlb, dum_loss, opt, iterations)
        else:
            dum_var = var[i*batsz : i*batsz + dum_NFits]
            dum_var = np.ascontiguousarray(dum_var.flatten(), dtype=np.float32)
            fitter(dum_NFits, boxsz, zrange, dum_data, dum_var, astigs, dum_xvec, dum_crlb, dum_loss, opt, iterations)
        
        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec.reshape((dum_NFits, vnum))
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb.reshape((dum_NFits, vnum))
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss
        
    return xvec, crlb, Loss



def cspline2d(data, var, coeff_PSF, optmethod, iterations, batsz):
    
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
    
    fitter = lib.cspline2d
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



def cspline3d(data, var, splineszx, splineszz, coeff_PSF, optmethod, iterations, batsz):
    
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
    
    fitter = lib.cspline3d
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