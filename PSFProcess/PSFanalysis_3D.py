from win32api import GetSystemMetrics
import time
import numpy as np
from scipy.optimize import fmin
from scipy.linalg import lstsq

from .array_Utils import upsample, mass_center, translate, ccorr
from .bspl_Utils import BSplineSmooth1D, _spline1d
from .astig_Utils import astigmatic, get_astig_parameters, get_astig_z
from .PSFfit_basic.PSFfit_basic import gauss2d, gauss3d, cspline3d


MAX_ITERS = 100
scrWidth, scrHeight = GetSystemMetrics(0), GetSystemMetrics(1)



def _get_lim(sz, center, rng):
    # get the range with a given center and size
    dum_min = max(center - rng//2, 0)
    dum_max = min(center + rng//2 + 1, sz)
    dum_max -= 1 - (dum_max - dum_min) % 2
    return dum_min, dum_max


def _get_lims(szs, centers, rngs, samerng=True):
    # get the ranges with an array of szs, centers, and rngs
    # the ranges will be adjusted to be uniqe if uniqrng is set True
    dum_lims = [_get_lim(sz, center, rng) for (sz, center, rng) in zip(szs, centers, rngs)]
    if samerng:
        dum_rngs = np.array([dum_lim[1] - dum_lim[0] for dum_lim in dum_lims], dtype=np.int32)
        if dum_rngs.ptp() > 0:
            dum_rng_min = dum_rngs.min()
            for i, dum_rng in enumerate(dum_rngs):
                if dum_rng > dum_rng_min:
                    dum_margin = (dum_rng - dum_rng_min) // 2   # dum_rng and dum_rng_min are both assured to be odd in _get_lim()
                    dum_lims[i][0] += dum_margin
                    dum_lims[i][1] -= dum_margin
    return dum_lims



def _get_cspline3d(psf, nknots):
    """
    Get the cspline coefficients describing each voxel of a nchannel-psf
    INPUT
        psf:        (szz, szy, szx) ndarray, an nchannel-psf
        nknots:     int, each voxel of the psf is described by 64 cspline coefficients
    RETURN
        coeffs:     (szz*szy*szx*64,) ndarray, cspline coefficients for a nchannel-psf
    
    NOTE 1: the i-th coeff for voxel (idz, idy, idx) is indexed as 
            coeff[idz*szy*szx*64 + idy*szx*64 + idx*64 + i]
    
    NOTE 2: a 64 coefficient array for each voxel is organized in a sequence as for
            [   dz^0*dy^0*dx^0, dz^0*dy^0*dx^1, dz^0*dy^0*dx^2, ..., dz^0*dy^1*dx^0, dz^0*dy^1*dx^1, dz^0*dy^1*dx^2, ...
                dz^1*dy^0*dx^0, dz^1*dy^0*dx^1, dz^1*dy^0*dx^2, ..., dz^1*dy^1*dx^0, dz^1*dy^1*dx^1, dz^1*dy^1*dx^2, ...
                dz^2*dy^0*dx^0, dz^2*dy^0*dx^1, dz^2*dy^0*dx^2, ..., dz^2*dy^1*dx^0, dz^2*dy^1*dx^1, dz^2*dy^1*dx^2, ...    ]
    """
    
    szz, szy, szx = psf.shape

    # Perpare products amongst dz**i, dy**j, and dx**k
    A = np.zeros((nknots*nknots*nknots, 64))
    for indz in range(nknots):
        dz = (indz + 0.5) / nknots
        for indy in range(nknots):
            dy = (indy + 0.5) / nknots
            for indx in range(nknots):
                dx = (indx + 0.5) / nknots

                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            A[indz*nknots*nknots + indy*nknots + indx, i*16 + j*4 + k] = dz**i * dy**j * dx**k

    # Calculate the coefficients
    psfup = upsample(psf, np.repeat([nknots], 3), 3)
    coeff = np.zeros(szz*szy*szx*64)
    for indz in range(szz):
        for indy in range(szy):
            for indx in range(szx):
                vxID = indz*szy*szx + indy*szx + indx
                dum_psf = psfup[indz*nknots:(indz+1)*nknots, indy*nknots:(indy+1)*nknots, indx*nknots:(indx+1)*nknots].flatten()
                dumcoeff = lstsq(A, dum_psf)[0]
                coeff[vxID*64:(vxID+1)*64] = dumcoeff

    return coeff
        


def _corrshift_3d(psf_ref, psf_tar, upscalars, z0, svec=np.zeros(3)):
    """
    Calculate the shifts between the svec-shifted psf_tar and the input psf_ref 
    Usually used together with the optimization function fmin to search the best shifts
    INPUT:
        psf_ref:        (nchannels, zrange, yrange, xrange) float ndarray, the reference psf
        psf_tar:        (nchannels, szz, szy, szx) float ndarray, the working psf
        upscalars:      (3,) int ndarray, [upscalerz, upscalery, upscalerx] both psf_ref and psf_tar will be upsampled by upscalars times for correlation calculation
        z0:             int, the zcenter of the psf_tar
        svec:           (3,) float ndarray, [shiftz, shifty, shiftx] the shifts of the psf_tar before calculate the correlation
    RETURN:
        corr_0:         float, the correlation value at 0 (at span//2 after fftshift)
        corr_max:       float, the maximum of the correlation profile
        shifts:         (3,) float ndarray, [shiftz, shifty, shiftx] the shifts from the svec-shifted psf_tar to the input psf_ref
        shift_dist:     float, np.sum(shifts**2) square distance of the shifts
    """

    # Parse the inputs
    assert psf_ref.ndim == 4, "shape mismatch: psf_ref.shape={}, should be (nchannels, zrange, yrange, xrange)".format(psf_ref.shape)
    assert psf_tar.ndim == 4, "shape mismatch: psf_tar.shape={}, should be (nchannels, szz, Szy, szx)".format(psf_tar.shape)
    assert len(psf_ref) == len(psf_tar), "nchannel mismatch: len(psf_ref)={}, len(psf_tar)={}".format(len(psf_ref), len(psf_tar))
    
    nchannels, zrange, yrange, xrange = psf_ref.shape
    szz, szy, szx = psf_tar.shape[-3:]
    zmin, zmax = z0-zrange//2, z0+zrange//2
    ymin, ymax = szy//2-yrange//2, szy//2+yrange//2
    xmin, xmax = szx//2-xrange//2, szx//2+xrange//2

    # Translate the psf_tar
    psf_shifted = np.zeros((nchannels, szz, szy, szx))
    for j in range(nchannels):
        psf_shifted[j] = translate(psf_tar[j], svec, method='spline')

    # Concatenate the psf_ref and the effective region of the psf_tar through all the channels
    psf_ref_hstacked = np.zeros((zrange, yrange, nchannels*xrange))
    psf_tar_hstacked = np.zeros((zrange, yrange, nchannels*xrange))
    for j in range(nchannels):
        psf_ref_hstacked[:, :, j*xrange:(j+1)*xrange] = psf_ref[j]
        psf_tar_hstacked[:, :, j*xrange:(j+1)*xrange] = psf_shifted[j, zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]

    # Correlation between psf_ref and psf_tar
    corr_0, corr_max, shifts, shift_dist = ccorr(psf_ref_hstacked, psf_tar_hstacked, upscalars)
    return corr_0, corr_max, shifts, shift_dist
        


def _align2ref_3d(psf_ref, psfs, z0, precise=True):
    """
    Align the array of psf to the reference psf
    INPUT:
        psf_ref:        (nchannels, zrange, yrange, xrange) float ndarray, the reference psf for alignment
        psfs:           (nchannels, npsfs, szz, szy, szx) float ndarray, npsfs nchannel-psfs
        z0:             int, the zcenter of each of the psf
        precise:        bool, to perform a finer search for alignment if True (will be more time-consuming)
    RETURN:
        psfs_shifted:   (nchannels, npsfs, szz, szy, szx) float ndarray, shifted psfs
        shifts:         (npsfs, 3) float ndarray, [shiftz, shifty, shiftx] for each psf
    """
    
    # Parse the inputs
    assert psf_ref.ndim == 4, "shape mismatch: psf_ref.shape={}, should be (nchannels, zrange, yrange, xrange)".format(psf_ref.shape)
    assert psfs.ndim == 5, "shape mismatch: psfs.shape={}, should be (nchannels, npsfs, szz, szy, szx)".format(psfs.shape)
    assert len(psf_ref) == len(psfs), "nchannel mismatch: len(psf_ref)={}, len(psf_tar)={}".format(len(psf_ref), len(psfs))
    
    # Aligne each psf to the reference psf
    nchannels, npsfs = psfs.shape[:2]
    psfs_shifted = np.zeros_like(psfs)
    shifts = np.zeros((npsfs, 3))
    for i in range(npsfs):
        tas = time.time()
        if precise:
            lossfunc = lambda svec: 1e3 * _corrshift_3d(psf_ref, psfs[:, i], np.array([4, 4, 4]), z0, svec)[-1]
            dumshift = fmin(lossfunc, np.zeros(3), disp=False)
        else:
            dumshift = _corrshift_3d(psf_ref, psfs[:, i], np.array([4, 4, 4]), z0, svec=np.zeros(3))[-2]
        shifts[i] = dumshift

        for j in range(nchannels):
            psfs_shifted[j, i] = translate(psfs[j, i], dumshift, method='spline')
        print("Aligning psf-#{n:02} [z_f0={z:d}], {t:.2f} secs elapsed".format(n=i, z=z0, t=time.time()-tas))

    return psfs_shifted, shifts



def _partition3d(psfs, k, z0, zrange):
    """
    Perform partition of an array of psf so that 0:k-th psfs have higher quality than that of the (k+1)-th psf
    the quality of a psf is roughly determined by their RMSE from the averages of the psfs.
    INPUT:
        psfs:           (nchannels, npsfs, szz, szy, szx) float ndarray, npsfs nchannel-psfs
        k:              int, element index to partition by.
        z0:             int, the zcenter of each of the psf
        zrange:         int, the effective range around the zcenter (zcenter-zrange//2 : zcenter+zrange//2+1) for psf analysis
    RETURN:
        Ind:            int, the index of psfs after partition
    """

    # Parse the inputs
    assert psfs.ndim == 5, "shape mismatch: psfs.shape={}, should be (nchannels, npsfs, szz, szy, szx)".format(psfs.shape)
    nchannels, npsfs = psfs.shape[:2]
    zmin, zmax = z0-zrange//2, z0+zrange//2

    # Partition according to the sum of the rmse in each channel
    devs = np.zeros((nchannels, npsfs))
    for j in range(nchannels):
        dum_psfs = np.array([psfs[j, i, zmin:zmax+1, :, :] for i in range(npsfs)])
        for i in range(npsfs):
            dum_psfs[i] -= dum_psfs[i].min()
            dum_psfs[i] /= dum_psfs[i].mean()
        avg_psf = np.mean(dum_psfs, axis=0)
        devs[j] = np.array([np.mean((dum_psf-avg_psf)*(dum_psf-avg_psf)) for dum_psf in dum_psfs])

    Ind = np.argpartition(np.mean(devs, axis=0), k)
    return Ind



def _norm3d(psfs, z0, zrange, xyrange, filterf0=False):
    """
    Normalization of the 3d arrays by the average of the central region defined by zrange and xyrange
    INPUT:
        psfs:           (npsfs, szz, szy, szx) float ndarray, npsfs psfs
        z0:             int, the zcenter of each of the psf
        zrange:         int, the effective range around the zcenter
        xyrange:        int, the effective range around the xycenter
        filterf0:       bool if the psf is pre-filtered (True) or not (False)
    RETURN:
        psfs_norm:      (npsfs, szz, szy, szx) ndarray, the normalized psfs 
    """
    
    # Parse the inputs
    npsfs, szz, szy, szx = psfs.shape
    zmin, zmax = z0-zrange//2, z0+zrange//2
    ymin, ymax = szy//2-xyrange//2, szy//2+xyrange//2
    xmin, xmax = szx//2-xyrange//2, szx//2+xyrange//2
    
    if npsfs == 1:    
        normF = np.mean(psfs[0, zmin:zmax+1, ymin:ymax+1, xmin:xmax+1])
        psfs_norm = psfs / normF if normF > 0 else np.copy(psfs)
        return psfs_norm

    # normalization if has been filtered before
    if filterf0:    
        dum_psfs = np.array([psfs[i, zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] for i in range(npsfs)])
        normFs = np.mean(np.mean(np.mean(dum_psfs, axis=-1), axis=-1), axis=-1)
        psfs_norm = np.array([psf / normF if normF > 0 else psf for psf, normF in zip(psfs, normFs)])
        return psfs_norm
    
    # normalization via iteratively update of the normalization factor
    psfs_norm = np.copy(psfs)
    for _ in range(4):
        dum_psfs = np.array([psfs_norm[i, zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] for i in range(npsfs)])
        psf_mean = np.mean(dum_psfs, axis=0)
        k_thresh = np.int32(0.75 * np.size(psf_mean))
        ind_thresh = psf_mean > np.partition(psf_mean, k_thresh, axis=None)[k_thresh]
        ratios = np.array([dum_psf[ind_thresh] / psf_mean[ind_thresh] for dum_psf in dum_psfs])
        normFs = np.median(ratios, axis=1)
        for i in range(len(psfs)):
            if normFs[i] > 0:
                psfs_norm[i] /= normFs[i]
    return psfs_norm



def _qc3d(psfs, z0, zrange, xyrange, indgood=None, weights=None):
    """
    Quality control of the array of 3d array by checking the central region defined by zrange and xyrange
    INPUT:
        psfs:           (npsfs, szz, szy, szx) float ndarray, npsfs psfs
        z0:             int, the zcenter of each of the psf
        zrange:         int, the effective range around the zcenter
        xyrange:        int, the effective range around the xycenter
        indgood:        (npsfs,) bool ndarray, label of good psf
        weights:        (npsfs,) float ndarray, weights of each psf
    RETURN:
        indgood:        (npsfs,) bool ndarray, label of good psf
        weights:        (npsfs,) float ndarray, weights of each psf
        normAmp:        float, amplitude of the weight-averaged psf
        res_thresh:     float, the threshold to exclude psfs with high res/cc
        res:            (npsfs,) float ndarray, rmse from each psf to the weight-averaged psf
        cc:             (npsfs,) float ndarray, cross-correlation from each psf to the weight-averaged psf
    """
    
    # Parse the inputs
    assert psfs.ndim == 4, "ndim mismatch: psfs.shape={}, request=(npsfs, szz, szy, szx)".format(psfs.shape)
    
    npsfs, szz, szy, szx = psfs.shape
    xmin, xmax = szx//2-xyrange//2, szx//2+xyrange//2
    ymin, ymax = szy//2-xyrange//2, szy//2+xyrange//2
    zmin, zmax = z0-zrange//2, z0+zrange//2
    
    if indgood is None:
        indgood = np.ones(npsfs, dtype=bool)
    if weights is None:
        weights = np.ones(npsfs)   
        
    # Weighted-average the good psfs as the reference psf
    ref_psf = np.zeros((zrange, xyrange, xyrange))
    for i in range(npsfs):
        if indgood[i]:
            ref_psf += psfs[i, zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] * weights[i]
    ref_psf /= np.sum(weights[indgood])
    normAmp = ref_psf.max()

    # calculate the correlation form each individual psf to the reference psf
    ref_psf_avg = ref_psf.mean()
    ref_psf_std = ref_psf.std()
    cc = np.zeros(npsfs)
    for i in range(npsfs):
        if indgood[i]:
            dum_psf = np.copy(psfs[i, zmin:zmax+1, ymin:ymax+1, xmin:xmax+1])
            cc[i] = (np.mean(dum_psf*ref_psf)-np.mean(dum_psf)*ref_psf_avg) / np.std(dum_psf) / ref_psf_std
    
    # calculate the rmse and the cc-normalized rmse
    ref_psf = ref_psf[1:zrange-1, 1:xyrange-1, 1:xyrange-1]
    res = np.zeros(npsfs) + np.inf
    res_norm = np.zeros(npsfs) + np.inf
    for i in range(npsfs):
        if indgood[i] and cc[i] > 0:
            dum_psf = np.copy(psfs[i, zmin+1:zmax, ymin+1:ymax, xmin+1:xmax])
            res[i] = np.sqrt(np.mean((dum_psf-ref_psf)*(dum_psf-ref_psf)))
            res_norm[i] = res[i] / cc[i]
    
    # label the psf with high rmse as bad psf
    weights = 1.0 / res_norm
    ind_thresh = res_norm < np.inf
    res_thresh = np.mean(res_norm[ind_thresh]) + 2*np.std(res_norm[ind_thresh])
    indgood = indgood & (res_norm<=res_thresh)

    return indgood, weights, normAmp, res_thresh, res, cc 





########## PUBLIC FUNCTIONALS ##########
def PSFGaussProfile3D(psf, varim, ini_sigmax, ini_sigmay, gauss_zrange, optmethod, lmbdaZ=10.0, iterations=100, batsz=65535):
    """
    To profile the input 3d psf via gauss fit
    INPUT:
        psf:            (szz, szy, szx) ndarray, single 1-channel psf
        varim:          (szy, szx) ndarray or None, readout noise of the camera corresponding to the psf
        ini_sigmax:     float, the estimated psf sigma at focus (x-axis)
        ini_sigmay:     float, the estimated psf sigma at focus (y-axis)
        gauss_zrange:   int, the zrange for gauss calibration
        optmethod:      'MLE' or 'LSQ' for maximum likelihood estimation or least square for PSFfit
        lambdaZ:        float, smooth factor for z-axis, default=10, visiable but not too much smoothing
        iterations:     int, number of iterations for PSFfit
        batsz:          int, number of psf sent to GPU concurrently for PSFfit
    RETURN: 
        Astigs:         dict, Astigmatic profiling of the input psf          
    """
    
    # initialize the zcenter at the axial-maximum and the available range around
    szz, szy, szx = psf.shape
    dumImax = np.max(np.max(psf, axis=-1), axis=-1)
    zcenter = np.argmax(BSplineSmooth1D(dumImax, 3, lmbdaZ))
    zmin = max(zcenter - gauss_zrange // 2, 0)
    zmax = min(zcenter + gauss_zrange // 2, szz)
    zdata = np.arange(zmin, zmax, dtype=np.float64) + 0.5
    
    # Gauss fit for astigmatism
    var_stack = np.tile(varim[np.newaxis, :, :], (zmax - zmin, 1, 1)) if not varim is None else None
    xvec, CRLB, LLR = gauss2d(psf[zmin : zmax], var_stack, ini_sigmax, ini_sigmay, 0, optmethod, iterations, batsz)
    locx, locy, Int, bg, sigmax, sigmay= xvec.T
    CRLBx, CRLBy, CRLBI, CRLBbg, CRLBsx, CRLBsy = CRLB.T
    
    # the zs parofile and astigmatic fit of the sigmax and sigmay
    sigma2 = sigmax*sigmax - sigmay*sigmay
    sigma2_fit = BSplineSmooth1D(sigma2, 3, lmbdaZ)
    astig_parameters = get_astig_parameters(zdata, sigmax, sigmay)
    sigmax_fit, sigmay_fit = astigmatic(zdata, *astig_parameters).reshape((2, -1))
    zFocus = get_astig_z(zdata, astig_parameters)

    # spline smooth for apparent wobbles
    locx -= 0.5 * szx
    locy -= 0.5 * szy
    wobblex = BSplineSmooth1D(locx, 3, lmbdaZ)
    wobbley = BSplineSmooth1D(locy, 3, lmbdaZ)    

    # Collect the information
    Astigs = {  'locz':         zdata,                          # (gauss_zrange,) the z positions defined for 2dGauss analysis at each z position 
                'zFocus':       zFocus,                         # int, index of the focus slice (sigmax == sigmay)
                'locx':         (locx, wobblex, CRLBx),         # (3, gauss_zrange) ndarray, locx, locx's spline1d smooth, and crlb at each z position
                'locy':         (locy, wobbley, CRLBy),         # (3, gauss_zrange) ndarray, locy, locy's spline1d smppth, and crlb at each z position
                'sigmax':       (sigmax, sigmax_fit, CRLBsx),   # (3, gauss_zrange) ndarray, sigmax, sigmax's astigmatic fit, and crlb at each z position
                'sigmay':       (sigmay, sigmay_fit, CRLBsy),   # (3, gauss_zrange) ndarray, sigmay, sigmay's astigmatic fit, and crlb at each z position
                'sigma2':       (sigma2, sigma2_fit),           # (2, gauss_zrange,) ndarray, spline fitted sigmax**2-sigmay**2 along dumz
                'photon':       (Int, CRLBI),                   # (2, gauss_zrange) ndarray, photon and its crlb at each z position
                'bkg':          (bg, CRLBbg),                   # (2, gauss_zrange) ndarray, bkg and its crlb at each z position
                'llr':          LLR                             # (gauss_zrange,) ndarray, LLR at each z position
    }       
    return Astigs



def PSFregister3D(psfs, z_focus, reg_zrange, reg_xyrange, filterf0=False):
    """
    To align, normalize, and make quality control of the input psf stack
    INPUT:
        psfs:           (npsfs, szz, szy, szx) or (nchannels, npsfs, szz, szy, szx) ndarray, npsfs nchannel-psfs
        z_focus:        int, user-defined z-focus, None for automatic search of the z_focus via Astigmatism
        reg_zrange:     int, the zrange for registration (including alignment, normalization, and quality control)
        reg_xyrange:    int, the xyrange for registration (including alignment, normalization, and quality control)
        lambdaZ:        float, smooth factor for z-axis, default=10, visiable but not too much smoothing
        filterf0:       bool, True if the beads had been filtered before
    RETURN:
        psf_out:        (szz, szy, szx) or (nchannels, szz, szy, szx) ndarray, the registered psf
        psfs_norm:      (psfs.shape) ndarray, the shifted and normalized psfs
        shifts:         (npsfs, 3) ndarray, the shifts of each bead during alignment (the same psf in different channels share the same shift)
        indgood:        (npsfs,) bool ndarray, label of good psf
    """
    
    ## Parse the inpus
    if psfs.ndim == 4:
        psfs = psfs[np.newaxis, ...]
    assert psfs.ndim == 5, "ndim mismatch: psfs.ndim={}, should be (nchannels, npsfs, szz, szy, szx)".format(psfs.shape)
    assert reg_xyrange % 2 != 0, "reg_xyrange must be odd"
    assert reg_zrange % 2 != 0, "reg_zrange must be odd" 
    nchannels, npsfs, szz, szy, szx = psfs.shape
    assert szx >= reg_xyrange, "reg_xyrange={} out of shapex={}".format(reg_xyrange, szx)
    assert szy >= reg_xyrange, "reg_xyrange={} out of shapey={}".format(reg_xyrange, szy)
    assert z_focus - reg_zrange//2 >= 0 and z_focus + reg_zrange//2 <= szz - 1, "reg_zrange={}, out of shapez={}".format(reg_zrange, (z_focus, szz))
    xmin, xmax = szx//2 - reg_xyrange//2, szx//2 + reg_xyrange//2
    ymin, ymax = szy//2 - reg_xyrange//2, szx//2 + reg_xyrange//2
    zmin, zmax = z_focus - reg_zrange//2, z_focus + reg_zrange//2
    
    # Step 0.1 --> generate a rough psf_ref by averaging the several 'best' psfs
    numref = max(npsfs//2, min(5, npsfs))
    IndSort = _partition3d(psfs, numref-1, z_focus, reg_zrange)
    psf_ref = np.zeros((nchannels, reg_zrange, reg_xyrange, reg_xyrange))
    for i in range(numref):
        psf_ref += psfs[:, IndSort[i], zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    psf_ref = psf_ref / numref

    # Step 0.2 --> Align all the beads to the psf_ref which is roughly averaged from several the 'best' psfs
    psfs_shifted, shifts0 = _align2ref_3d(psf_ref, psfs, z_focus, precise=True)

    # Step 0.3 --> Robust normalization of all the shifted beads and label the beads with good-enough qualities
    psfs_norm = np.zeros_like(psfs_shifted)
    indgood = np.ones(npsfs, dtype=bool)
    for j in range(nchannels):
        psfs_norm[j] = _norm3d(psfs_shifted[j], z_focus, reg_zrange, reg_xyrange, filterf0=filterf0)
        indgood = indgood & _qc3d(psfs_norm[j], z_focus, reg_zrange, reg_xyrange, indgood=None, weights=None)[0]

    # step 1.1 --> generate new reference psf
    psf_ref = np.zeros((nchannels, reg_zrange, reg_xyrange, reg_xyrange))
    for i in range(npsfs):
        if indgood[i]:
            psf_ref += psfs_norm[:, i, zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    psf_ref = psf_ref / indgood.sum()

    # Step 1.2 --> Align again all the beads to the psf_ref which is the average of the good shifted psfs
    psfs_shifted, shifts1 = _align2ref_3d(psf_ref, psfs_norm, z_focus, precise=True)
    shifts = shifts0 + shifts1

    # Step 1.3 --> Robust normalization of the shifted beads and label the beads with good-enough qualities
    psfs_norm = np.zeros_like(psfs_shifted)
    indgood = np.ones(npsfs, dtype=bool)
    for j in range(nchannels):
        psfs_norm[j] = _norm3d(psfs_shifted[j], z_focus, reg_zrange, reg_xyrange, filterf0=filterf0)
        dumindgood, dumweights = _qc3d(psfs_norm[j], z_focus, reg_zrange, reg_xyrange, indgood=None, weights=None)[:2]
        dumindgood, dumweights = _qc3d(psfs_norm[j], z_focus, reg_zrange, reg_xyrange, indgood=dumindgood, weights=dumweights)[:2]
        dumindgood, dumweights, normF, res_thresh, res, cc = _qc3d(psfs_norm[j], z_focus, reg_zrange, reg_xyrange, indgood=dumindgood, weights=dumweights)
        psfs_norm[j] /= normF
        indgood = indgood & dumindgood
    psf_out = np.mean(psfs_norm[:, indgood], axis=1)
    
    if nchannels == 1:
        return psf_out[0], psfs_norm[0], shifts, indgood
    else:
        return psf_out, psfs_norm, shifts, indgood
    


def PSFCalibration3D_cspline(psf, Calibration, iterations=100, batsz=65535):
    """
    Calibration of a 3D psf stack. The psf should usually be priorly aligned, averaged, and normalized
    INPUT:
        psf:                    (nchannels, splineszz, splineszx, splineszx) ndarray, single nchannel-psfs
        Calibration:            dict, Calibration dictionary to fill in
        iterations:             int, number of iterations for PSFfit
        batsz:                  int, number of psf sent to GPU concurrently for PSFfit
    RETURN:
        Calibration:            dict, calibration information
        cspline_validation:     (nchannels, 6, splineszz) the cspline fit result using the calibrated cspline parameters 
    """  
    
    ##### PARSE THE INPUTS #####
    assert psf.ndim == 4, "ndim mismatch: psf.shape={}, should be (nchannels, szz, szy, szx)".format(psf.shape)
    
    nchannels, splineszz, splineszy, splineszx = psf.shape
    assert splineszy == splineszx, "not square: splineszy={}, splineszx={}".format(splineszy, splineszx)
    assert splineszx % 2 != 0, "size not odd: splineszx={}".format(splineszx)
    assert splineszz % 2 != 0, "size not odd: splineszz={}".format(splineszz)
    boxsz = splineszx // 2

    ##### CALIBRATION #####
    ymassc, xmassc = np.array([mass_center(psf[j, splineszz//2], np.array([4, 4]))[0] for j in range(nchannels)]).T
    ymassc -= 0.5 * splineszy
    xmassc -= 0.5 * splineszx
    coeffs = np.zeros((nchannels, splineszz*splineszx*splineszx*64))
    for j in range(nchannels):
        coeffs[j] = _get_cspline3d(psf[j], 4)
    
    ##### VALIDATION #####
    ymin, ymax = splineszx//2 - boxsz//2, splineszx//2 + boxsz//2
    xmin, xmax = splineszx//2 - boxsz//2, splineszx//2 + boxsz//2
    psf_vali = np.copy(psf[:, :, ymin:ymax+1, xmin:xmax+1])
    
    cspline_validation = np.zeros((nchannels, splineszz, 6))
    for j in range(nchannels):
        cspline_xvec, cspline_CRLB, cspline_LS = cspline3d(psf_vali[j], None, splineszx, splineszz, coeffs[j], 'LSQ', iterations, batsz)
        cspline_xvec[:, 0] -= 0.5 * boxsz
        cspline_xvec[:, 1] -= 0.5 * boxsz
        cspline_validation[j] = np.hstack((cspline_xvec, cspline_LS[...,np.newaxis]))

    ##### COLLECT INFO #####
    Calibration['splineszx']    = splineszx # int, the lateral size of the spline cube (splineszx == splineszy)
    Calibration['splineszz']    = splineszz # int, the axial size of the spline cube 
    Calibration['xmassc']       = xmassc    # (nchannels,) float ndarray, the distances from the mass center (x-axis) to the geo center of the input psf in each channel
    Calibration['ymassc']       = ymassc    # (nchannels,) float ndarray, the distances from the mass center (y-axis) to the geo center of the input psf in each channel
    Calibration['coeff']        = coeffs    # (nchannels, splineszz*splineszx*splineszx*64) float, the coefficients of the spline cube in each channel
    Calibration['kernel']       = psf       # (nchannels, splineszz, splineszx, splineszx) ndarray, the registered psf z stack   
    return cspline_validation



def PSFCalibration3D_gauss(psf, Calibration, ini_sigmax, ini_sigmay, nbreaks=20, iterations=100, batsz=65535):
    """
    Calibration of a 3D psf stack. The psf should usually be priorly aligned, averaged, and normalized
    INPUT:
        psfs:                   (nchannels, szz, szy, szx) ndarray, single nchannel-psfs
        Calibration:            dict, Calibration dictionary to fill in
        ini_sigmax:             float, the estimated psf sigma at focus (x-axis)
        ini_sigmay:             float, the estimated psf sigma at focus (y-axis)
        nbreaks:                int, number of breaks for spline fit of the wobbles
        iterations:             int, number of iterations for PSFfit
        batsz:                  int, number of psf sent to GPU concurrently for PSFfit
    RETURN:
        Calibration:            dict, calibration information
        astigmatic2d_result:    (nchannels, szz, 7) the initial 2D Gauss fit of each slice of the psf stack in each channel
        gauss_validation:       (nchannels, szz, 6) the gauss fit result using the calibrated gauss parameters
    """  
    
    ##### PARSE THE INPUTS #####
    assert psf.ndim == 4, "ndim mismatch: psf.shape={}, should be (nchannels, szz, szy, szx)".format(psf.shape)
    nchannels, zrange, szy, szx = psf.shape
    assert szy == szx, "not square: splineszy={}, splineszx={}".format(szy, szx)
    assert szx % 2 != 0, "size not odd: splineszx={}".format(szx)
    boxsz = szx // 2
    zdata = np.arange(zrange) + 0.5
    
    ##### CALIBRATION #####
    astigmatic2d_result = np.zeros((nchannels, zrange, 7))
    astigs = np.zeros((nchannels, 9))
    sigmax = np.zeros((nchannels, zrange))
    sigmay = np.zeros((nchannels, zrange))  
    fspl_sxy2z = []
    for j in range(nchannels):
        
        astigmatic2d_xvec, astigmatic2d_crlb, astigmatic2d_LS = gauss2d(psf[j], None, ini_sigmax, ini_sigmay, 0, 'LSQ', iterations, batsz)
        astigmatic2d_result[j] = np.hstack((astigmatic2d_xvec, astigmatic2d_LS[...,np.newaxis]))
        
        ini_sx, ini_sy = astigmatic2d_xvec[:, -2], astigmatic2d_xvec[:, -1]
        popt = get_astig_parameters(zdata, ini_sx, ini_sy)
        astigs[j] = popt    #[PSFsigmax, PSFsigmay, shiftx, shifty, dof, Ax, Bx, Ay, By]
        sigmax[j], sigmay[j] = astigmatic(zdata, *popt).reshape((2, -1))

        dum_spl_sxy2z = _spline1d(ini_sx*ini_sx-ini_sy*ini_sy, zdata, nbreaks, sorted=False)
        fspl_sxy2z.append(dum_spl_sxy2z)

    ##### VALIDATION #####
    ymin, ymax = szx//2 - boxsz//2, szx//2 + boxsz//2
    xmin, xmax = szx//2 - boxsz//2, szx//2 + boxsz//2
    psf_vali = np.copy(psf[:, :, ymin:ymax+1, xmin:xmax+1])
    
    gauss_validation = np.zeros((nchannels, zrange, 6))
    fspl_wobblez = []
    fspl_wobbley = []
    fspl_wobblex = []
    for j in range(nchannels):
        gauss_xvec, gauss_CRLB, gauss_LS = gauss3d(psf_vali[j], None, zrange, astigs[j], 'LSQ', iterations, batsz)
        gauss_xvec[:, 0] -= 0.5 * boxsz
        gauss_xvec[:, 1] -= 0.5 * boxsz
        gauss_validation[j] = np.hstack((gauss_xvec, gauss_LS[...,np.newaxis]))

        dum_spl_wobblez = _spline1d(zdata, gauss_xvec[:, 2] - zdata, nbreaks)
        dum_spl_wobbley = _spline1d(zdata, gauss_xvec[:, 1], nbreaks)
        dum_spl_wobblex = _spline1d(zdata, gauss_xvec[:, 0], nbreaks)
        fspl_wobblez.append(dum_spl_wobblez)
        fspl_wobbley.append(dum_spl_wobbley)
        fspl_wobblex.append(dum_spl_wobblex)
    
    ##### COLLECT INFO #####
    Calibration['formular_sx']  = 'sigmazx = PSFsigmax * sqrt(1 + ((z-shiftx)/dof)^2 + Ax((z-shiftx)/dof)^3 + Bx((z-shiftx)/dof)^4)'
    Calibration['formular_sy']  = 'sigmazy = PSFsigmay * sqrt(1 + ((z-shifty)/dof)^2 + Ay((z-shifty)/dof)^3 + By((z-shifty)/dof)^4)'
    Calibration['zrange']       = zrange        # int, the zrange that will be used in PSFfit 
    Calibration['sigmax']       = sigmax        # (nchannels, zrange) float ndarray, the sigmax at different z positions
    Calibration['sigmay']       = sigmay        # (nchannels, zrange) float ndarray, the sigmay at different z positions 
    Calibration['astigs']       = astigs        # (nchannels, 9) ndarry, the astig parameters in each channel
    Calibration['spl_wobblez']  = fspl_wobblez  # (nchannels,) func obj list, 1d spline function calculate the error (z-axis) at a z-position for each channel
    Calibration['spl_wobbley']  = fspl_wobbley  # (nchannels,) func obj list, 1d spline function calculate the error (y-axis) at a z-position for each channel
    Calibration['spl_wobblex']  = fspl_wobblex  # (nchannels,) func obj list, 1d spline function calculate the error (x-axis) at a z-position for each channel
    Calibration['spl_sxy2z']    = fspl_sxy2z    # (nchannels,) func obj list, 1d spline function translate the sigmas to z-positions
    return astigmatic2d_result, gauss_validation