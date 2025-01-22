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
@brief  batch fit of PSF via single cspline3d model
@param[in]  data:               (nchannels, NFits, boxsz, boxsz) float ndarray, the nchannel data
@param[in]  var:                (nchannels, NFits, boxsz, boxsz) float ndarray, the nchannel data
@param[in]  coeff_PSF:          (nchannels, splineszx * splineszx * splineszz * 64) float ndarray, the cubic spline coefficients calibrated for the nchannel-PSF model
@param[in]  lc:                 (nchannels, NFits) int ndarray, the left corner of each data square
@param[in]  uc:                 (nchannels, NFits) int ndarray, the upper corner of each data square
@param[in]  coeff_R2T:          (nchannels - 1, 2, warpdeg * warpdeg) float ndarray, the polynomial coefficients calibrated for channel warpping
@param[in]  fracN:              (nfluorophores,) float ndarray, the frac of photons collected in the reference channel (0-th channel) for each kind of fluorophores
@param[in]	fracb:				float, fraction of background photons collected in the reference channel (0-th channel)
@param[in]  optmethod:          str, {'MLE', 'LSQ'} optimization method
@param[in]  iterations:         int, number of iterations for optimization
@param[in]  batsz:              int, number of PSFs sent to GPU concurrently
@param[out] xvec:			    (NFits, vnum) float ndarray
@param[out] CRLB:			    (NFits, vnum) float ndarray, CRLB variance corresponding to parameters in xvec
@param[out] Loss:				(NFits,) float ndarray, Loss value for the optimization of a PSF fitting.
"""

def cspline3d_1ch(data, var, splineszx, splineszz, coeff_PSF, zcoarse, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    NDIM = 3
    vnum = NDIM + 2
    opt = 0 if optmethod == 'MLE' else 1
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
    xvec = np.zeros((NFits, vnum), dtype=np.float32)
    crlb = np.zeros((NFits, vnum), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)

    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    zcoarse = np.ascontiguousarray(zcoarse, dtype=np.float32)
    
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
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
            fitter(dum_NFits, boxsz, splineszx, splineszz, dum_data, None, coeff_PSF, zcoarse, dum_xvec, dum_crlb, dum_loss, opt, iterations)
        else:
            dum_var = var[i*batsz : i*batsz + dum_NFits]
            dum_var = np.ascontiguousarray(dum_var.flatten(), dtype=np.float32)
            fitter(dum_NFits, boxsz, splineszx, splineszz, dum_data, dum_var, coeff_PSF, zcoarse, dum_xvec, dum_crlb, dum_loss, opt, iterations)
        
        xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec.reshape((dum_NFits, vnum))
        crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb.reshape((dum_NFits, vnum))
        Loss[i*batsz : i*batsz + dum_NFits] = dum_loss

    return xvec, crlb, Loss



def cspline3d_2ch(linkN, data, var, splineszx, splineszz, coeff_PSF, zcoarse, lc, uc, coeff_R2T, fracN, optmethod, iterations, batsz):
    
    # PARSE THE INPUTS
    NDIM = 3
    vnum = NDIM + 4 - linkN
    opt = 0 if optmethod == 'MLE' else 1
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

    if linkN == 1:
        nfluorophores = len(fracN)
        if np.any((fracN <= 0.0) | (fracN >= 1.0)):
            print(fracN[(fracN <= 0.0) | (fracN >= 1.0)])
            raise ValueError("fracN out of range")

    # INITIALIZATION AND ALLOCATIONS
    xvec = np.zeros((NFits, vnum), dtype=np.float32)
    crlb = np.zeros((NFits, vnum), dtype=np.float32)
    Loss = np.zeros(NFits, dtype=np.float32)
    if linkN == 1:
        raw_xvec = np.zeros((NFits, nfluorophores, vnum), dtype=np.float32)
        raw_crlb = np.zeros((NFits, nfluorophores, vnum), dtype=np.float32)
        raw_Loss = np.zeros((NFits, nfluorophores), dtype=np.float32)

    coeff_PSF = np.ascontiguousarray(coeff_PSF.flatten(), dtype=np.float32)
    zcoarse = np.ascontiguousarray(zcoarse, dtype=np.float32)
    coeff_R2T = np.ascontiguousarray(coeff_R2T.flatten(), dtype=np.float32)

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
                            ctypes.c_int32,
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_float,
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
                            ctypes.c_int32,
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_float,
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
        
        if linkN == 1:
            if var is None:
                for j in range(nfluorophores):
                    fitter(linkN, dum_NFits, boxsz, splineszx, splineszz, dum_data, None, coeff_PSF, zcoarse, warpdeg, dum_lu, coeff_R2T, fracN[j],
                            dum_xvec, dum_crlb, dum_loss, opt, iterations)
                    raw_xvec[i*batsz : i*batsz + dum_NFits, j] = dum_xvec.reshape((dum_NFits, vnum))
                    raw_crlb[i*batsz : i*batsz + dum_NFits, j] = dum_crlb.reshape((dum_NFits, vnum))
                    raw_Loss[i*batsz : i*batsz + dum_NFits, j] = dum_loss         
            else:
                dum_var = var[i*batsz : i*batsz + dum_NFits]
                dum_var = np.ascontiguousarray(dum_var.flatten(), dtype=np.float32)
                for j in range(nfluorophores):
                    fitter(linkN, dum_NFits, boxsz, splineszx, splineszz, dum_data, dum_var, coeff_PSF, zcoarse, warpdeg, dum_lu, coeff_R2T, fracN[j],
                            dum_xvec, dum_crlb, dum_loss, opt, iterations)
                    raw_xvec[i*batsz : i*batsz + dum_NFits, j] = dum_xvec.reshape((dum_NFits, vnum))
                    raw_crlb[i*batsz : i*batsz + dum_NFits, j] = dum_crlb.reshape((dum_NFits, vnum))
                    raw_Loss[i*batsz : i*batsz + dum_NFits, j] = dum_loss
        else:
            if var is None:
                fitter(linkN, dum_NFits, boxsz, splineszx, splineszz, dum_data, None, coeff_PSF, zcoarse, warpdeg, dum_lu, coeff_R2T, 0.5, 
                        dum_xvec, dum_crlb, dum_loss, opt, iterations)
            else:
                dum_var = var[i*batsz : i*batsz + dum_NFits]
                dum_var = np.ascontiguousarray(dum_var.flatten(), dtype=np.float32)
                fitter(linkN, dum_NFits, boxsz, splineszx, splineszz, dum_data, dum_var, coeff_PSF, zcoarse, warpdeg, dum_lu, coeff_R2T, 0.5,
                        dum_xvec, dum_crlb, dum_loss, opt, iterations)
            xvec[i*batsz : i*batsz + dum_NFits] = dum_xvec.reshape((dum_NFits, vnum))
            crlb[i*batsz : i*batsz + dum_NFits] = dum_crlb.reshape((dum_NFits, vnum))
            Loss[i*batsz : i*batsz + dum_NFits] = dum_loss
    
    if linkN == 1:
        Loss = np.min(raw_Loss, axis=1)
        ind_fluo = np.argmin(raw_Loss, axis=1)
        for Idx in range(NFits):
            xvec[Idx] = raw_xvec[Idx, ind_fluo[Idx]]
            crlb[Idx] = raw_crlb[Idx, ind_fluo[Idx]]

    return xvec, crlb, Loss