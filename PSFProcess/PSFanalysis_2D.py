from win32api import GetSystemMetrics
import numpy as np
from scipy.optimize import fmin
from scipy.linalg import lstsq

from .array_Utils import upsample, mass_center, translate, ccorr
from .PSFfit_basic.PSFfit_basic import gauss2d, cspline2d


MAX_ITERS = 100
scrWidth, scrHeight = GetSystemMetrics(0), GetSystemMetrics(1)



def _get_lim(sz, center, rng):
    # get the range with a given center and size
    dum_min = max(center - rng//2, 0)
    dum_max = min(center + rng//2 + 1, sz)
    dum_max -= 1 - (dum_max - dum_min) % 2
    return dum_min, dum_max



def _get_cspline2d(psf, nknots):
    """
    Get the cspline coefficients describing each pixel of a nchannel-psf
    INPUT
        psf:        (szy, szx) ndarray, a one-channel psf
        nknots:     int, each pixel of the psf is described by 16 cspline coefficients
    RETURN
        coeffs:     (szy*szx*16,) ndarray, cspline coefficients for the 2D psf
    
    NOTE 1: the i-th coeff for voxel (idy, idx) is indexed as 
            coeff[idy*PSFszx*16 + idx*16 + i]
    
    NOTE 2: a 16 coefficient array for each pixel is organized in a sequence as for
            [dy^0*dx^0, dy^0*dx^1, dy^0*dx^2, ..., dy^1*dx^0, dy^1*dx^1, dy^1*dx^2, ...]
    """
    
    # Parse the inputs
    szy, szx = psf.shape[-2:]

    # Upsample the psf to (szy*nknots, szx*nknots)
    psfup = upsample(psf, np.array([nknots, nknots]), 3)
    
    # Prepare products amongst dz**i, dy**j, and dx**k
    A = np.zeros((nknots*nknots, 16))
    for indy in range(nknots):
        dy = (indy + 0.5) / nknots
        for indx in range(nknots):
            dx = (indx + 0.5) / nknots

            for i in range(4):
                for j in range(4):
                    A[indy*nknots+indx, i*4+j] = dy**i * dx**j

    # Calculate the coefficients
    coeff = np.zeros(szy*szx*16)
    for indy in range(szy):
        for indx in range(szx):
            pxID = indy*szx + indx
            dum_psf = psfup[indy*nknots:(indy+1)*nknots, indx*nknots:(indx+1)*nknots].flatten()
            dumcoeff = lstsq(A, dum_psf)[0]
            coeff[pxID*16 : (pxID+1)*16] = dumcoeff
    
    return coeff



def _corrshift2d(psf_ref, psf_tar, upscalars, svec=np.zeros(2)):
    """
    Calculate the shifts between the svec-shifted psf_tar and the input psf_ref 
    Usually used together with the optimization function fmin to search the best shifts
    INPUT:
        psf_ref:        (nchannels, yrange, xrange) ndarray, the reference psf from one or multiple channels
        psf_tar:        (nchannels, szy, szx) ndarray, the working psf from one or multiple channels
        upscalars:      (upscalery, upscalerx), both psf_ref and psf_tar will be upsampled by upscalars times for correlation calculation
        svec:           [shifty, shiftx] that shifts the psf_tar before calculate the correlation
    RETURN:
        corr_0:         float, the Correlation value at 0 (at span//2 after fftshift)
        corr_max:       float, the maximum of the correlation profile
        shifts:         [shifty, shiftx] ndarray, the shifts from svec-shifted psf_tar to the input psf_ref
        shift_dist:     float, np.sum(shifts**2) the distance of the shifts
    """

    # Parse the inputs
    assert psf_ref.ndim == 3, "shape mismatch: psf_ref.shape={}, should be (nchannels, yrange, xrange)".format(psf_ref.shape)
    assert psf_tar.ndim == 3, "shape mismatch: psf_tar.shape={}, should be (nchannels, szy, szx)".format(psf_tar.shape)
    assert len(psf_ref) == len(psf_tar), "nchannel mismatch: len(psf_ref)={}, len(psf_tar)={}".format(len(psf_ref), len(psf_tar))
    
    nchannels, yrange, xrange = psf_ref.shape
    szy, szx = psf_tar.shape[-2:]
    xmin, xmax = _get_lim(szx, szx//2, xrange)
    ymin, ymax = _get_lim(szy, szy//2, yrange)

    # Translate the psf_tar
    psf_shifted = np.zeros((nchannels, szy, szx))
    for j in range(nchannels):
        psf_shifted[j] = translate(psf_tar[j], svec, method='spline')

    # Concatenate the psf_ref and the effective region of the psf_tar through all the channels
    psf_ref_hstacked = np.zeros((yrange, nchannels*xrange))
    psf_tar_hstacked = np.zeros((yrange, nchannels*xrange))
    for j in range(nchannels):
        psf_ref_hstacked[:, j*xrange:(j+1)*xrange] = psf_ref[j]
        psf_tar_hstacked[:, j*xrange:(j+1)*xrange] = psf_shifted[j, ymin:ymax, xmin:xmax]

    # Correlation between arr3d_ref and tar_arr3d
    corr_0, corr_max, shifts, shift_dist = ccorr(psf_ref_hstacked, psf_tar_hstacked, upscalars)
    return corr_0, corr_max, shifts, shift_dist
        


def _align2ref2d(psf_ref, psfs, precise=True):
    """
    align the array of arr2d to the reference arr2d
    INPUT:
        psf_ref:        (nchannels, yrange, xrange) ndarray, the reference psf for alignment
        psfs:           (nchannels, npsfs, szy, szx) ndarray, npsfs nchannel-psfs
        precise:        bool, to perform a finer search for alignment if True (will be more time-consuming)
    RETURN:
        psfs_shifted:   (psfs.shape) ndarray, the shifted psfs
        shifts:         (npsfs, 2) ndarray, the shifts from svec-shifted psf_tar to the input psf_ref
    """

    # Parse the inputs
    assert psf_ref.ndim == 3, "shape mismatch: psf_ref.shape={}, should be (nchannels, yrange, xrange)".format(psf_ref.shape)
    assert psfs.ndim == 4, "shape mismatch: psfs.shape={}, should be (nchannels, npsfs, szy, szx)".format(psfs.shape)
    assert len(psf_ref) == len(psfs), "nchannel mismatch: len(psf_ref)={}, len(psf_tar)={}".format(len(psf_ref), len(psfs))
    nchannels, npsfs = psfs.shape[:2]
    
    psfs_shifted = np.zeros_like(psfs)
    shifts = np.zeros((npsfs, 2))
    for i in range(npsfs):
        if precise:
            lossfunc = lambda svec: 1e3 * _corrshift2d(psf_ref, psfs[:, i], np.array([4, 4]), svec)[-1]
            dumshift = fmin(lossfunc, np.zeros(2), disp=False)
        else:
            dumshift = _corrshift2d(psf_ref, psfs[:, i], np.array([4, 4]), svec=np.zeros(2))[-2]
        shifts[i] = dumshift
        
        for j in range(nchannels):
            psfs_shifted[j, i] = translate(psfs[j, i], dumshift, method='spline')

    return psfs_shifted, shifts



def _partition2d(psfs, k):
    """
    Perform partition of an array of psfs so that 0:kth psfs have higher quality than that of the (k+1)th psf
    the quality of a psf is roughly determined by their RMSE from their averages.
    INPUT:
        psfs:           (nchannels, npsfs, szy, szx) ndarray, npsfs nchannel-psfs
        k:              int, element index to partition by.
    RETURN:
        Ind:            the index of psf after partition
    """

    # Parse the inputs
    assert psfs.ndim == 4, "shape mismatch: psfs.shape={}, should be (nchannels, npsfs, szy, szx)".format(psfs.shape)
    nchannels, npsfs = psfs.shape[:2]

    # Partition
    devs = np.zeros((nchannels, npsfs))
    for j in range(nchannels):
        dum_psfs = np.copy(psfs[j])
        for i in range(npsfs):
            dum_psfs[i] -= dum_psfs[i].min()
            dum_psfs[i] /= dum_psfs[i].mean()
        avg_psf = np.mean(dum_psfs, axis=0)
        devs[j] = np.array([np.mean((dum_psf-avg_psf)*(dum_psf-avg_psf)) for dum_psf in dum_psfs])
    
    Ind = np.argpartition(np.mean(devs, axis=0), k)
    indgood = np.zeros(npsfs, dtype=bool)
    for i in range(k+1):
        indgood[Ind[i]] = True    
    
    return indgood



def _norm2d(psfs, xyrange, filterf0=False):
    """
    Normalization of the 2d arrays by the average of the central region defined by xyrange
    INPUT:
        psfs:           (szy, szx) or (npsfs, szy, szx) ndarray, npsfs psfs
        xyrange:        int, the effective range around the xycenter (_center-_range//2 : _center+_range//2) for psf analysis
        filterf0:       bool if the psf is pre-filtered (True) or not (False)
    RETURN:
        psfs_norm:      (psfs.shape) ndarray, the array of the normalized psfs 
    """

    # Parse the inputs
    szy, szx = psfs.shape[-2:]
    xmin, xmax = _get_lim(szx, szx//2, xyrange)
    ymin, ymax = _get_lim(szy, szy//2, xyrange)

    if psfs.ndim == 2:
        normF = np.mean(psfs[ymin:ymax, xmin:xmax])
        psfs_norm = psfs / normF if normF > 0 else np.copy(psfs)
        return psfs_norm
    
    assert psfs.ndim == 3, "ndim mismatch: psfs.shape={}, request=(npsfs, szy, szx)".format(psfs.shape)

    # normalization if has been filtered before
    if filterf0:    
        dum_psfs = np.copy(psfs[:, ymin:ymax, xmin:xmax])
        normFs = np.mean(np.mean(dum_psfs, axis=-1), axis=-1)
        psfs_norm = np.array([psf / normF if normF > 0 else psf for psf, normF in zip(psfs, normFs)])
        return psfs_norm
    
    # normalization via iteratively update of the normalization factor
    psfs_norm = np.copy(psfs)
    for _ in range(4):
        dum_psfs = np.copy(psfs_norm[:, ymin:ymax, xmin:xmax])
        psf_mean = np.mean(dum_psfs, axis=0)
        k_thresh = np.int32(0.75 * np.size(psf_mean))
        ind_thresh = psf_mean > np.partition(psf_mean, k_thresh, axis=None)[k_thresh]
        
        ratios = np.array([dum_psf[ind_thresh] / psf_mean[ind_thresh] for dum_psf in dum_psfs])
        normFs = np.median(ratios, axis=1)
        for i in range(len(psfs)):
            if normFs[i] > 0:
                psfs_norm[i] /= normFs[i]

    return psfs_norm



def _qc2d(psfs, xyrange, indgood=None, weights=None):
    """
    Quality control of the array of psf by checking the central region defined by the xyrange
    INPUT:
        psfs:           (npsfs, szy, szx) ndarray, npsfs nchannel-psfs
        xyrange:        int, the effective range around the xycenter (_center-_range//2 : _center+_range//2) for psf analysis
        indgood:        1d boolean ndarray (npsfs,), label of good arr2d
        weights:        2d float ndarray (npsfs,), weights of each arr2d
    RETURN:
        indgood:        1d boolean ndarray (npsfs,), label of good arr2d
        weights:        1d float ndarray (npsfs,), weights of each arr2d
        normAmp:        float, amplitude of the weight-averaged arr2d
        res_thresh:     float, the threshold to exclude psf with high res/cc
        res:            1d float ndarray (npsfs,), rmse from each arr2d to the weight-averaged arr2d
        cc:             1d float ndarray (npsfs,), cross-correlation from each arr2d to the weight-averaged arr2d
    """

    # Parse the inputs
    assert psfs.ndim == 3, "ndim mismatch: psfs.shape={}, request=(npsfs, szy, szx)".format(psfs.shape)
    
    npsfs, szy, szx = psfs.shape    
    xmin, xmax = _get_lim(szx, szx//2, xyrange)
    ymin, ymax = _get_lim(szy, szy//2, xyrange)
    
    if indgood is None:
        indgood = np.ones(npsfs, dtype=bool)
    if weights is None:
        weights = np.ones(npsfs)   
    
    # Weighted-average the good psfs as the reference psf
    dum_weights = weights[indgood]
    dum_psfs = np.copy(psfs[indgood, ymin:ymax, xmin:xmax])
    ref_psf = np.average(dum_psfs, axis=0, weights=dum_weights)
    normAmp = ref_psf.max()

    # calculate the correlation form each individual psf to the reference psf
    ref_psf_avg = ref_psf.mean()
    ref_psf_std = ref_psf.std()
    dum_psfs = np.copy(psfs[:, ymin:ymax, xmin:xmax])
    cc = np.array([(np.mean(dum_psfs[i]*ref_psf)-np.mean(dum_psfs[i])*ref_psf_avg) / np.std(dum_psfs[i]) / ref_psf_std if indgood[i] else 0.0 for i in range(npsfs)])
    
    # calculate the rmse and the cc-normalized rmse
    ref_psf = ref_psf[1:xyrange-1, 1:xyrange-1]
    dum_psfs = np.copy(psfs[:, ymin+1:ymax-1, xmin+1:xmax-1])
    res = np.array([np.sqrt(np.mean((dum_psfs[i]-ref_psf)*(dum_psfs[i]-ref_psf))) if indgood[i] and cc[i]>0 else np.inf for i in range(npsfs)])
    res_norm = np.array([res[i] / cc[i] if indgood[i] and cc[i]>0 else np.inf for i in range(npsfs)])
            
    # label the psf with high rmse as bad psf
    weights = 1.0 / res_norm
    ind_thresh = res_norm < np.inf
    res_thresh = np.mean(res_norm[ind_thresh]) + 2*np.std(res_norm[ind_thresh])
    indgood = indgood & (res_norm<=res_thresh)

    return indgood, weights, normAmp, res_thresh, res, cc 





########## PUBLIC FUNCTIONALS ##########
def PSFregister2D(psfs, reg_xyrange, filterf0=False):
    """
    To align, normalize, and make quality control of the input psf stack
    INPUT:
        psfs:           (npsfs, szy, szx) or (nchannels, npsfs, szy, szx) ndarray, npsfs nchannel-psfs
        reg_xyrange:    int, the xyrange for registration (including alignment, normalization, and quality control)
        filterf0:       label if the bead is pre-filtered (True) or not (False)
    RETURN:
        psf_out:        (szy, szx) or (nchannels, szy, szx) ndarray, the registered psf
        psfs_norm:      (psfs.shape) ndarray, shifted and normalized psfs
        shifts:         (npsfs, 2) ndarray, shift of each bead (the same bead in different channels share the same shift)
        indgood:        (npsfs,) ndarray, label of good bead
    """
    
    ## Parse the Inpus
    sch = True if psfs.ndim == 3 else False
    if psfs.ndim == 3:
        psfs = psfs[np.newaxis, ...]
    assert psfs.ndim == 4, "ndim mismatch: psfs.ndim={}, should be (nchannels, npsfs, szy, szx)".format(psfs.shape)
    assert psfs.shape[-2] == psfs.shape[-1], "the psfs must be squared stack"
    
    nchannels, npsfs, szy, szx = psfs.shape
    xmin, xmax = _get_lim(szx, szx//2, reg_xyrange)
    ymin, ymax = _get_lim(szy, szy//2, reg_xyrange)
    
    # Step 0 --> generate a rough psf_ref by averaging the several 'best' psfs
    numref = max(npsfs//2, min(5, npsfs))
    indgood = _partition2d(psfs, numref-1)
    psfs_core = psfs[:, indgood, ymin:ymax, xmin:xmax]
    psf_ref = np.mean(psfs_core, axis=1)
    
    # Step 1 --> Align all the beads to the psf_ref which is roughly averaged from several the 'best' psfs
    psfs_shifted, shifts0 = _align2ref2d(psf_ref, psfs, precise=True)

    # Step 2 --> Robust normalization of all the shifted beads and label the beads with good-enough qualities
    psfs_norm = np.array([_norm2d(psfs_shifted[j], reg_xyrange, filterf0=filterf0) for j in range(nchannels)])
    indgood = np.ones(npsfs, dtype=bool)
    for j in range(nchannels):
        indgood = indgood & _qc2d(psfs_norm[j], reg_xyrange, indgood=None, weights=None)[0]

    # Step 3 --> Finer align the beads to the psf_ref which is the average of the good shifted psfs
    psfs_core = psfs_norm[:, indgood, ymin:ymax, xmin:xmax]
    psf_ref = np.mean(psfs_core, axis=1)
    psfs_shifted, shifts1 = _align2ref2d(psf_ref, psfs_norm, precise=True)
    shifts = shifts0 + shifts1

    # Step4 --> Robust normalization of the shifted beads and label the beads with good-enough qualities
    psfs_norm = np.array([_norm2d(psfs_shifted[j], reg_xyrange, filterf0=filterf0) for j in range(nchannels)])
    indgood = np.ones(npsfs, dtype=bool)
    for j in range(nchannels):
        dumindgood, dumweights = _qc2d(psfs_norm[j], reg_xyrange, indgood=None, weights=None)[:2]
        dumindgood, dumweights = _qc2d(psfs_norm[j], reg_xyrange, indgood=dumindgood, weights=dumweights)[:2]
        dumindgood, dumweights, normF, res_thresh, res, cc = _qc2d(psfs_norm[j], reg_xyrange, indgood=dumindgood, weights=dumweights)
        psfs_norm[j] /= normF
        indgood = indgood & dumindgood
    psf_out = np.mean(psfs_norm[:, indgood], axis=1)
    
    if sch:
        return psf_out[0], psfs_norm[0], shifts, indgood
    else:
        return psf_out, psfs_norm, shifts, indgood



def PSFCalibration2D(psf, Calibration, ini_sigmax, ini_sigmay, iterations=100, batsz=65535):
    """
    Calibration of a 2D psf from different channels. The psf should usually be priorly aligned, averaged, and normalized
    INPUT:
        psf:                (szy, szx) or (nchannels, szy, szx) ndarray, npsfs nchannel-psfs
        Calibration:        dict, Calibration dictionary to fill in
        ini_sigmax:         float, the estimated psf sigma at focus (x-axis)
        ini_sigmay:         float, the estimated psf sigma at focus (y-axis)
        iterations:         int, number of iterations for PSFfit
        batsz:              int, number of psf sent to GPU concurrently for PSFfit
    RETURN:
        Calibration:        dict, calibration infotmation 
    """  
    
    ##### PARSE THE INPUT #####
    if psf.ndim == 2:
        psf = psf[np.newaxis, ...]
    assert psf.ndim == 3, "ndim mismatch: psf.shape={}, should be (nchannels, splineszx, splineszx)".format(psf.shape)
    
    nchannels, splineszy, splineszx = psf.shape
    assert splineszy == splineszx, "not square: splineszy={}, splineszx={}".format(splineszy, splineszx)
    assert splineszx % 2 != 0, "size not odd: splineszx={}".format(splineszx)
        
    boxsz = splineszx // 2
    ymin, ymax = _get_lim(splineszy, splineszy//2, boxsz)
    xmin, xmax = _get_lim(splineszx, splineszx//2, boxsz)

    ##### MASS CENTER AT FOCUS #####
    ymassc, xmassc = np.array([mass_center(psf[j], np.array([4, 4]))[0] for j in range(nchannels)]).T
    ymassc -= 0.5 * splineszy
    xmassc -= 0.5 * splineszx

    ##### CALIBRATION #####
    psf_cali = np.copy(psf)
    psf_cali_xvec = gauss2d(1e3*psf_cali, None, ini_sigmax, ini_sigmay, 0, 'LSQ', iterations, batsz)[0]
    PSFsigmax = psf_cali_xvec[:, -2]
    PSFsigmay = psf_cali_xvec[:, -1]
    coeffs = np.array([_get_cspline2d(psf_cali[j], 4) for j in range(nchannels)])
    
    ##### VALIDATION #####
    psf_vali = psf_cali[:, ymin:ymax, xmin:xmax]
    gauss_validation = np.zeros((nchannels, 4))
    cspline_validation = np.zeros((nchannels, 4))
    for j in range(nchannels):
        gauss_validation[j] = gauss2d(1e3*psf_vali[j], None, PSFsigmax[j], PSFsigmay[j], 1, 'LSQ', iterations, batsz)[0][0]
        cspline_validation[j] = cspline2d(1e3*psf_vali[j], None, coeffs[j], 'LSQ', iterations, batsz)[0][0]
    gauss_validation[:, :2] -= 0.5 * boxsz
    gauss_validation[:, 2:4] /= 1e3
    cspline_validation[:, :2] -= 0.5 * boxsz
    cspline_validation[:, 2:4] /= 1e3
    
    ##### COLLECT INFO ######
    Calibration['splineszx']    = splineszx                                 # int, the lateral size of the spline cube (splineszx == splineszy)
    Calibration['xmassc']       = xmassc                                    # (nchannels,) float ndarray, the distances from the mass center (x-axis) to the geo center of the input psf in each channel
    Calibration['ymassc']       = ymassc                                    # (nchannels,) float ndarray, the distances from the mass center (y-axis) to the geo center of the input psf in each channel
    Calibration['coeff']        = coeffs                                    # (nchannels, splineszx*splineszx*16) the coefficients of the spline cube for each channel
    Calibration['kernel']       = psf                                       # (nchannels, splineszx, splineszx) ndarray, the psf kernel
    Calibration['cspline_xerr'] = cspline_validation[:, 0]                  # (nchannels,) the error (x-axis) of the Cspline fit for each channel
    Calibration['cspline_yerr'] = cspline_validation[:, 1]                  # (nchannels,) the error (y-axis) of the Cspline fit for each channel
    Calibration['PSFsigmax']    = PSFsigmax                                 # (nchannels,) the sigma of the PSF_at_Focus (x-axis) for each channel 
    Calibration['PSFsigmay']    = PSFsigmay                                 # (nchannels,) the sigma of the PSF_at_Focus (y-axis) for each channel
    Calibration['gauss_xerr']   = gauss_validation[:, 0]                    # (nchannels,) the error (x-axis) of the Gauss fit for each channel
    Calibration['gauss_yerr']   = gauss_validation[:, 1]                    # (nchannels,) the error (y-axis) of the Gauss fit for each channel
    return cspline_validation, gauss_validation 