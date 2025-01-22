import sys
import os
import inspect
import ctypes
from math import prod
import numpy as np
import numpy.ctypeslib as ctl 
from scipy import stats
from scipy.ndimage import convolve1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import lstsq


"""
drift correction via redundant cross-correlation (rcc)
General Arguments:
    ndim:               int, number of dimention
    frmbinsz:           int, number of frames to bin
    frm:                (nspots,) int ndarray, the frm index of each detection
    locs:               (nspots, ndim) float ndarray, the [[x, y, z],...] localization of each detection
    crlb:               (nspots, ndim) float ndarray, the [[crlbx, crlby, crlbz],...] crlb of each detection
    photons:            (nspots,) float ndarray, the photons of each detection
NOTE:
    the locs, crlbs, and photons should be sorted by index = np.argsort(frm)
"""



def _loc_render(locs, crlb, photon, Imsz, option='gauss'):
    """
    Render the localizations into super-resolution image
    INPUT:  
        Imsz:       (Imszy, Imszx) the size of the output image
        option:     'gauss':    each localization is blurred with its CRLBx and CRLBy
                    'hist':     each localization is directly rendered w/o CRLB bluring
    RETURN: 
        Im:         rendered image  
    """

    nspots = len(locs)
    b_opt = option.encode('utf-8')
    Imszy, Imszx = Imsz

    locx = np.ascontiguousarray(locs[:, 0], dtype=np.float32)
    locy = np.ascontiguousarray(locs[:, 1], dtype=np.float32)
    crlbx = np.ascontiguousarray(crlb[:, 0], dtype=np.float32)
    crlby = np.ascontiguousarray(crlb[:, 1], dtype=np.float32)
    photon = np.ascontiguousarray(photon, dtype=np.float32)
    Im = np.ascontiguousarray(np.zeros((Imszy, Imszx)), dtype=np.float32)

    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_locrender_lite')
    fname = 'locrender.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))

    render_kernel = lib.loc_render
    render_kernel.argtypes = [ctypes.c_int32,                                              
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                ctypes.c_int32, ctypes.c_int32,
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                ctypes.c_char_p]
    render_kernel(nspots, locx, locy, crlbx, crlby, photon, Imszy, Imszx, Im, b_opt)
    
    return Im



def _ccorr(ndim, ccorrsz, locsA, photonsA, locsB, photonsB, leafLim=32):
    """
    Calculate the cross-correlation between two sets of localizations
    dependent on ccorr.dll in the same pacakge
    INPUT:  ndim:       int, number of dimention of the localizations (upto 3)
            ccorrsz:    (ndim,) int ndarray, (ccorrszx, ccorrszy, corrszz) windowsz for correlation computation
            locsA:      (nspotsA, ndim) float ndarray, [[locx, locy, locz],...] coordinates of detections in channel A
            photonsA:   (nspotsA,) float ndarray, the photon number of detections in channel A
            locsB:      (nspotsB, ndim) float ndarray, [[locx, locy, locz],...] coordinates of detections in channel B
            photonsB:   (nspotsB,) float ndarray, the photon number of detections in channel B
            leafLim:    int maximum number of points stored in a leaf node of a kd-tree
    RETURN: cc:         (prod(ccorrsz)) float ndarray, the cross-correlation profile
    """

    nspotsA = len(locsA)
    nspotsB = len(locsB)
    if nspotsA == 0 or nspotsB == 0:
        cc = np.zeros(prod(ccorrsz[:ndim]))
        cnt = 0
        for i in range(ndim-1, -1, -1):
            cnt = cnt*ccorrsz[i] + ccorrsz[i]//2
        cc[cnt] = 1.0
        return cc.reshape(tuple(ccorrsz[ndim-1::-1]))
    
    ccorrsz = np.ascontiguousarray(ccorrsz, dtype=np.int32)
    locsA = np.ascontiguousarray(np.array(locsA).flatten(), dtype=np.float32)
    photonsA = np.ascontiguousarray(photonsA, dtype=np.float32)
    locsB = np.ascontiguousarray(np.array(locsB).flatten(), dtype=np.float32)
    photonsB = np.ascontiguousarray(photonsB, dtype=np.float32)
    cc = np.ascontiguousarray(np.zeros(np.prod(ccorrsz)), dtype=np.float32)

    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_ccorr')
    fname = 'ccorr.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    ccorr_kernel = lib.kernel
    ccorr_kernel.argtypes = [ctypes.c_int32, 
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                            ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
    ccorr_kernel(ndim, ccorrsz, nspotsA, locsA, photonsA, nspotsB, locsB, photonsB, leafLim, cc)
    cc *= np.prod(ccorrsz) / np.sum(photonsA) / np.sum(photonsB)
    
    return cc.reshape(tuple(ccorrsz[ndim-1::-1]))



def _peakfit(img, MAX_ITERS=100):
    """
    fit the input img with a 2D Gauss kernel
    """
    vnum = 7 # constant
    
    h_data = np.asarray(img, dtype=np.float32)
    if h_data.ndim < 2:
        raise TypeError('PSF should be cropped into a 2D square, but the input is an 1D array\n')
    if h_data.ndim == 2:
        NFits = 1 
        imHeight, imWidth = h_data.shape
    elif h_data.ndim > 2:
        NFits, imHeight, imWidth = h_data.shape
    
    tinv095 = stats.t.ppf(0.975, imHeight * imWidth - vnum)
    
    h_data = np.ascontiguousarray(h_data.flatten(), dtype=np.float32)
    h_xvec = np.ascontiguousarray(np.zeros(NFits * vnum), dtype=np.float32)
    h_Loss = np.ascontiguousarray(np.zeros(NFits), dtype=np.float32)
    h_crlb = np.ascontiguousarray(np.zeros(NFits * vnum), dtype=np.float32)
    h_ci = np.ascontiguousarray(np.zeros(NFits * vnum), dtype=np.float32)

    ## Load cuda module file
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_peakfit')
    fname = 'peakfit.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    peakfit_kernel = lib.kernel
    peakfit_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctypes.c_float, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
    peakfit_kernel(NFits, imHeight, imWidth, h_data, tinv095, MAX_ITERS, h_xvec, h_Loss, h_crlb, h_ci)
    return h_xvec.reshape((NFits, vnum)), h_Loss, h_crlb.reshape((NFits, vnum)), h_ci.reshape((NFits, vnum))



def _frmbin(binsz, indf, frm_st, frm_ed):
    """
    rebin the frm index by binsz
    INPUT:  
        binsz:              int, number of frames to bin
        indf:               (nspots,) int ndarray, the frame index for each spot, should be accendingly pre-sorted
        frm_st:             int, starting frame
        frm_ed:             int, ending frame
    RETURN: 
        binfrm_start:       (nbins + 1,) int ndarray, the frame starting index wrt the indf
    """
    nspots = len(indf)
    nfrm = frm_ed - frm_st
    nbins = (nfrm - 1) // binsz + 1     # np.int32(np.ceil(nfrm / binsz))

    bin_starts = np.zeros(nbins + 1, dtype=np.int32)
    bin_starts[-1] = nspots

    j = 0
    for i in range(nbins):
        bin_starts[i] = j
        while j < nspots and indf[j] < frm_st + (i+1)*binsz:
            j += 1

    return bin_starts



def _findshift_fft(images, pairs, windowsz, fitsz):
    """
    Calculate the shifts between image pairs
    INPUT:  
        images:     (nbins, imszy, imszx) float ndarray, 3D image stack
        pairs:      (npairs, 2) int ndarray, [[a1, b1], [a2, b2],...] frm index pairs for cross-correlation computation
        windowsz:   (2), [windowszx, windowszy] windowsz for peak fit
        fitsz:      (2), [fitszx, fitszy] fitsz for peak fit
    RETURN: 
        shifts:     (npairs, 2) float ndarray, [[shiftx, shifty],...] shifts between each pair
    """
    
    windowszx, windowszy = windowsz
    if windowszx % 2 == 0:
        windowszx += 1
    if windowszy % 2 == 0:
        windowszy += 1    
    hwindowszx = windowszx // 2
    hwindowszy = windowszy // 2

    fitszx, fitszy = fitsz
    if fitszx % 2 == 0:
        fitszx += 1
    if fitszy % 2 == 0:
        fitszy += 1    
    hfitszx = fitszx // 2
    hfitszy = fitszy // 2

    nframes, imHeight, imWidth = images.shape
    print("RCC: Computing image cross correlations. Image stack shape: ({szz:d}, {szy:d}, {szx:d}). Size: {sz:5.2f} MB"
            .format(szz=nframes, szy=imHeight, szx=imWidth, sz=images.size*4.0/1024.0/1024.0))
    
    npairs = len(pairs)
    cc_ims = np.zeros((npairs, fitszy, fitszx), dtype=np.float32)
    cc_lefts = np.zeros(npairs)
    cc_tops = np.zeros(npairs)
    
    for i, (a, b) in enumerate(pairs):

        fft_A = np.fft.fft2(images[a])
        fft_B = np.fft.fft2(images[b])
        cc = np.fft.fftshift(np.abs(np.fft.ifft2(fft_A * np.conj(fft_B))))
        
        cc_win_left = max(imWidth//2-hwindowszx, 0)
        cc_win_right = min(imWidth//2+hwindowszx, imWidth)
        cc_win_top = max(imHeight//2-hwindowszy, 0)
        cc_win_bottom = min(imHeight//2+hwindowszy, imHeight)
        cc_win = cc[cc_win_top:cc_win_bottom, cc_win_left:cc_win_right]

        indy, indx = np.array(np.unravel_index(np.argmax(cc_win), cc_win.shape))
        indy += cc_win_top
        indx += cc_win_left
        indy = max(min(indy, imHeight-hfitszy-1), hfitszy)
        indx = max(min(indx, imWidth-hfitszx-1), hfitszx)
        
        cc_lefts[i] = indx - hfitszx
        cc_tops[i] = indy - hfitszy
        roi = cc[indy-hfitszy : indy+hfitszy+1, indx-hfitszx : indx+hfitszx+1]
        cc_ims[i] = (roi - roi.min()) / (roi.max() - roi.min())

    xvec = _peakfit(cc_ims)[0]
    
    shifts = np.copy(xvec[:, :2])
    shifts[:, 0] += cc_lefts - 0.5*imWidth
    shifts[:, 1] += cc_tops - 0.5*imHeight 
    if imWidth%2 == 0:
        shifts[:, 0] -= 0.5
    if imHeight%2 == 0:
        shifts[:, 1] -= 0.5

    return shifts



def _findshifts2d_cc(ccorrsz, locs, photons, frm_starts, pairs, fitsz):
    """
    Calculate the shifts between image pairs
    INPUT:  
        ccorrsz:            (2) int ndarray, [ccorrszx, ccorrszy] windowsz for correlation computation
        locs:               (nspots, ndim) float ndarray, the [[x, y, z],...] localization of each detection
        photons:            (nspots,) float ndarray, the photons of each detection
        frm_starts:         (nfrm + 1,) int ndarray, the starting index (wrt the photons) for detections within a frame, frm_starts[-1] = nspots 
        pairs:              (npairs, 2) int ndarray, [[a1, b1], [a2, b2],...] frm index pairs for cross-correlation computation
        fitsz:              (2) int ndarray, [fitszx, fitszy] small windowsz for peak fit
    RETURN: 
        shifts:             (npairs, 2) float ndarray, [[shiftx, shifty],...] shifts between each pair
    """
    
    fitszx, fitszy = fitsz
    if fitszx % 2 == 0:
        fitszx += 1
    if fitszy % 2 == 0:
        fitszy += 1    
    hfitszx = fitszx // 2
    hfitszy = fitszy // 2

    npairs = len(pairs)   
    shifts = np.zeros((npairs, 2))
    rois = np.zeros((npairs, fitszy, fitszx))
    for i, (a, b) in enumerate(pairs):

        locA = locs[frm_starts[a] : frm_starts[a+1]]
        photonA = photons[frm_starts[a] : frm_starts[a+1]]

        locB = locs[frm_starts[b] : frm_starts[b+1]]
        photonB = photons[frm_starts[b] : frm_starts[b+1]]

        if len(photonA) == 0 or len(photonB) == 0:
            shifts[i] = 0.0
            continue
        
        cc = _ccorr(2, ccorrsz, locA, photonA, locB, photonB)
        cc = convolve1d(cc, np.ones(7), axis=1)
        cc = convolve1d(cc, np.ones(7), axis=0)
        
        indy, indx = np.array(np.unravel_index(np.argmax(cc), (ccorrsz[1], ccorrsz[0])))
        indy = max(min(indy, ccorrsz[1]-hfitszy-1), hfitszy)
        indx = max(min(indx, ccorrsz[0]-hfitszx-1), hfitszx)
        rois[i] = cc[indy-hfitszy : indy+hfitszy+1, indx-hfitszx : indx+hfitszx+1]
        shifts[i] = np.array([indx - hfitszx - 0.5*ccorrsz[0], indy - hfitszy - 0.5*ccorrsz[1]])
    
    xvec = _peakfit(rois)[0]
    shifts += xvec[:, :2]
    
    return shifts



def rcc2d(frmbinsz, ccorrsz, indf, locs, crlb, photon, fitsz, frm_st, frm_ed, method='cc', sorted=False):
    """
    Calculate the x-y-shift via Redundant Cross Correlation (RCC)
    INPUT:  
        frmbinsz:           int, number of frames to bin
        ccorrsz:            (2) int ndarray, [ccorrszx, ccorrszy] windowsz for correlation computation
        indf:               (nspots,) int ndarray, the frm index of each detection
        locs:               (nspots, ndim) float ndarray, the [[x, y, z],...] localization of each detection
        crlb:               (nspots, ndim) float ndarray, the [[crlbx, crlby, crlbz],...] crlb of each detection
        photons:            (nspots,) float ndarray, the photons of each detection
        fitsz:              (fitszx, fitszy) small windowsz for peak fit
        frm_st:             int, starting frame
        frm_ed:             int, ending frame
        method:             str, {'cc', 'fft'} use 'cc' for cases with a few localizations, 'fft' for cases with a huge number of localizations (high frmbinsz) 
        sorted:             True if the input localizations are sorted along the frm
                            False to sort the input localizations along the frm IN-PLACE
    RETURN: 
        interp_shifts:      (nfrm, 2) float ndarry, [[shift_x, shift_y],...] the interpolated shifts to all the frames
        raw_shifts:         (nbins, 3) float ndarry, [[t, shift_x, shift_y],...] the RCC calculated shifts at each timebin
    """
    
    # Sort the localizations according to time (the frame) IN-PLACE
    if not sorted:
        ind = np.argsort(indf)
        indf[:] = indf[ind]
        locs[:] = locs[ind]
        crlb[:] = crlb[ind]
        photon[:] = photon[ind]

    # frm bin and get the pairs
    nfrm = frm_ed - frm_st
    bin_edges = np.arange(frm_st, frm_ed + 1, frmbinsz)
    bin_cs = 0.5 * (bin_edges[:-1] + bin_edges[1:])                        # center of each bin 
    nbins = len(bin_cs)
    frmbin_starts = _frmbin(frmbinsz, indf, frm_st, frm_ed)
    
    pairs = np.array(np.triu_indices(nbins, 1)).T
    if len(pairs) > 10 * nbins:
        pairs = pairs[np.random.choice(len(pairs), 10 * nbins)]  
    npairs = len(pairs)

    # get the pairwise shifts
    if method == 'cc':
        pair_shifts = _findshifts2d_cc(ccorrsz, locs, photon, frmbin_starts, pairs, fitsz)
    else:
        Imszy = np.int32(np.ceil(locs[:, 1].max()))
        Imszx = np.int32(np.ceil(locs[:, 0].max()))
        images = np.zeros((nbins, Imszy, Imszx), dtype=np.float32)
        for k in range(nbins):
            
            dum_locs = locs[frmbin_starts[k] : frmbin_starts[k+1]]
            dum_crlb = crlb[frmbin_starts[k] : frmbin_starts[k+1]]
            dum_photon = photon[frmbin_starts[k] : frmbin_starts[k+1]]
            
            assert len(dum_locs) > 0, "NO localizations between frame {f1:d} -- {f2:d}".format(f1=k*frmbinsz, f2=min(nfrm, (k+1)*frmbinsz))
            im = _loc_render(dum_locs, dum_crlb, dum_photon, (Imszy, Imszx))
            images[k] = im
        pair_shifts = _findshift_fft(images, pairs, ccorrsz, fitsz)
    
    # Solve the shifts from the pairwise shifts
    A = np.zeros((npairs, nbins))
    A[np.arange(npairs), pairs[:, 0]] = -1.0
    A[np.arange(npairs), pairs[:, 1]] = 1.0
    
    shifts = lstsq(A, pair_shifts)[0]
    shifts = shifts - np.mean(shifts, axis=0)

    raw_shifts = np.zeros((nbins, 3))
    raw_shifts[:, 0] = bin_cs
    raw_shifts[:, 1:3] = shifts

    # spline interpolate the shifts
    interp_shifts = np.zeros((nfrm, 2))
    if frmbinsz != 1:
        spl_x = InterpolatedUnivariateSpline(bin_cs, shifts[:, 0], k=2)
        spl_y = InterpolatedUnivariateSpline(bin_cs, shifts[:, 1], k=2)
        interp_shifts[:, 0] = spl_x(np.arange(frm_st, frm_ed))
        interp_shifts[:, 1] = spl_y(np.arange(frm_st, frm_ed))
    else:
        interp_shifts = shifts
            
    return interp_shifts, raw_shifts



def rcc3d(frmbinsz, ccorrsz, indf, locs, crlb, photon, fitsz, frm_st, frm_ed, method='cc', sorted=False):
    """
    Calculate the x-y-shift via Redundant Cross Correlation (RCC)
    INPUT:  
        frmbinsz:           int, number of frames to bin
        ccorrsz:            (2) int ndarray, [ccorrszx, ccorrszy] windowsz for correlation computation
        indf:               (nspots,) int ndarray, the frm index of each detection
        locs:               (nspots, ndim) float ndarray, the [[x, y, z],...] localization of each detection
        crlb:               (nspots, ndim) float ndarray, the [[crlbx, crlby, crlbz],...] crlb of each detection
        photons:            (nspots,) float ndarray, the photons of each detection
        fitsz:              (fitszx, fitszy) small windowsz for peak fit
        frm_st:             int, starting frame
        frm_ed:             int, ending frame
        sorted:             True if the input localizations are sorted along the frm
                            False to sort the input localizations along the frm IN-PLACE
        method:             str, {'cc', 'fft'} use 'cc' for cases with a few localizations, 'fft' for cases with a huge number of localizations (high frmbinsz)
    RETURN: 
        interp_shifts:      (nfrm, 2) float ndarry, [[shift_x, shift_y],...] the interpolated shifts to all the frames
        raw_shifts:         (nbins, 3) float ndarry, [[t, shift_x, shift_y],...] the RCC calculated shifts at each timebin
    """
    
    # Sort the localizations according to time (the frame) IN-PLACE
    if not sorted:
        ind = np.argsort(indf)
        indf[:] = indf[ind]
        locs[:] = locs[ind]
        crlb[:] = crlb[ind]
        photon[:] = photon[ind]
    
    ccorrszx, ccorrszy, ccorrszz = ccorrsz
    fitszx, fitszy, fitszz = fitsz

    interp_shift_xy, raw_shift_xy = rcc2d(frmbinsz, (ccorrszx, ccorrszy), indf, locs[:, [0, 1]], crlb[:, [0, 1]], photon, (fitszx, fitszy), frm_st, frm_ed, method, sorted=True)
    dumlocs = np.copy(locs[:, [0, 2]])
    for i in range(len(interp_shift_xy)):
        dumind = (indf == i)
        dumlocs[dumind, 0] -= interp_shift_xy[i, 0]
    interp_shift_xz, raw_shift_xz = rcc2d(frmbinsz, (ccorrszx, ccorrszz), indf, dumlocs, crlb[:, [0, 2]], photon, (fitszx, fitszz), frm_st, frm_ed, method, sorted=True)

    interp_shifts = np.hstack((interp_shift_xy, interp_shift_xz[:, -1:]))
    raw_shifts = np.hstack((raw_shift_xy, raw_shift_xz[:, -1:]))
    
    return interp_shifts, raw_shifts






def smlm_simulation(fov_sz, numMols, numFrms, on_prob, drift_mean, drift_std, loc_error): 
    """
    Simulate an SMLM dataset in 3D with blinking molecules
    INPUT:  fov_sz:         (fov_szx, fov_szy, fov_szz) field of view size in pixels
            numMols:        number of molecules blinking on and off
            numFrms:        number of frames
            on_prob:        probability of a binding site generating a localization in a frame
            drift_mean:     (driftx_mean, drifty_mean, driftz_mean) average drift along each dimention
            drift_std:      (driftx_std, drifty_std, driftz_std), the std of the drift along each dimention 
            loc_error:      (loc_errx, loc_erry, loc_errz) localization error in x-, y-, and z-axis in pixels
    RETURN: frm:            frame number of accquired localizations
            locx:           localizations (x-axis)
            locy:           localizations (y-axis)
            locz:           localizations (z-axis)
            driftx:         drifts (x-axis)
            drifty:         drifts (y-axis)
            driftz:         drifts (z-axis)
    """

    fov_szx, fov_szy, fov_szz = fov_sz
    driftx_mean, drifty_mean, driftz_mean = drift_mean
    driftx_std, drifty_std, driftz_std = drift_std
    loc_errx, loc_erry, loc_errz = loc_error
    
    # randomly distributes the positions of the emitters    
    rng = np.random.default_rng()
    molPos_x = rng.uniform(low=0.0, high=fov_szx, size=numMols)      
    molPos_y = rng.uniform(low=0.0, high=fov_szy, size=numMols)      
    molPos_z = rng.uniform(low=0.0, high=fov_szz, size=numMols)

    # simulate the drifts
    driftx = np.cumsum(rng.normal(loc=driftx_mean, scale=driftx_std, size=numFrms))
    drifty = np.cumsum(rng.normal(loc=drifty_mean, scale=drifty_std, size=numFrms))
    driftz = np.cumsum(rng.normal(loc=driftz_mean, scale=driftz_std, size=numFrms))
    driftx -= np.mean(driftx)
    drifty -= np.mean(drifty)
    driftz -= np.mean(driftz)  

    # Accquire all emitters with localizations with drifts and errs
    dumfrm = np.repeat(np.arange(numFrms), numMols)
    dumlocx = np.tile(molPos_x, numFrms) + np.repeat(driftx, numMols) + rng.normal(loc=0.0, scale=loc_errx, size=numMols*numFrms)
    dumlocy = np.tile(molPos_y, numFrms) + np.repeat(drifty, numMols) + rng.normal(loc=0.0, scale=loc_erry, size=numMols*numFrms)
    dumlocz = np.tile(molPos_z, numFrms) + np.repeat(driftz, numMols) + rng.normal(loc=0.0, scale=loc_errz, size=numMols*numFrms) 
    
    # Simulate blinking, collect on molecules only
    ind_on = rng.binomial(1, on_prob, size=numMols*numFrms).astype(bool)
    indf = dumfrm[ind_on]
    locx = dumlocx[ind_on]
    locy = dumlocy[ind_on]
    locz = dumlocz[ind_on]
        
    return indf, locx, locy, locz, driftx, drifty, driftz



if __name__ == '__main__':
    
    import os
    from matplotlib import pyplot as plt
    
    # Camera parameters (unit = nm)
    CamPxszx, CamPxszy, Camzstepsz = 100, 100, 10
    TarPxszx, TarPxszy, Tarzstepsz = 10, 10, 5

    # RCC parameters (unit = nm)
    frmbinsz = 10
    fitsz_lateral = 100
    fitsz_axial = 100
    ccorrsz_lateral = 2000
    ccorrsz_axial = 500
    ccorrsz = np.int32((np.ceil(ccorrsz_lateral/TarPxszx), np.ceil(ccorrsz_lateral/TarPxszy), np.ceil(ccorrsz_axial/Tarzstepsz)))
    fitsz = np.int32((np.ceil(fitsz_lateral/TarPxszx), np.ceil(fitsz_lateral/TarPxszy), np.ceil(fitsz_axial/Tarzstepsz)))

    # simulation parameters
    frame_offset = 1000
    numMols = 200
    numFrms = 8192
    photon = 1000
    photon_sig = 10
    on_prob = 0.1
    
    # simulation parameters (unit = nm)
    fov_lateral = 20000
    fov_axial = 2000
    driftx_mean, drifty_mean, driftz_mean = 0.1, 0.1, 0.0
    driftx_std, drifty_std, driftz_std = 2.0, 2.0, 0.2
    loc_errx, loc_erry, loc_errz = 10.0, 10.0, 10.0 # Note: loc_errz is usually 3 times higher

    # translate to analysis units (TarPxsz etc)
    fov_sz = (fov_lateral/TarPxszx, fov_lateral/TarPxszy, fov_axial/Tarzstepsz)
    drift_mean = (driftx_mean/TarPxszx, drifty_mean/TarPxszy, driftz_mean/Tarzstepsz)
    drift_std = (driftx_std/TarPxszx, drifty_std/TarPxszy, driftz_std/Tarzstepsz)
    loc_err = (loc_errx/TarPxszx, loc_erry/TarPxszy, loc_errz/Tarzstepsz)

    # simulation
    indf, locx, locy, locz, driftx, drifty, driftz = smlm_simulation(fov_sz, numMols, numFrms, on_prob, drift_mean, drift_std, loc_err)
    numLocs = len(indf)
    indf += frame_offset
    print("Total localizations: {n:d}".format(n=numLocs))
    print("frames: {}, {}, nfrm={}".format(indf[0], indf[-1], indf[-1]+1-indf[0]))

    crlbx = loc_errx*loc_errx * np.ones(numLocs)
    crlby = loc_erry*loc_erry * np.ones(numLocs)
    crlbz = loc_errz*loc_errz * np.ones(numLocs)
    rng = np.random.default_rng()
    photons = rng.normal(loc=photon, scale=photon_sig, size=numLocs)

    # RCC
    if sys.argv[1] == 'cc':
        interp_shifts, raw_shifts = rcc3d(frmbinsz, ccorrsz, indf, np.vstack((locx, locy, locz)).T, photons, fitsz, frame_offset, frame_offset+numFrms, sorted=False, method='cc')
    else:
        interp_shifts, raw_shifts = rcc3d(frmbinsz, ccorrsz, indf, np.vstack((locx, locy, locz)).T, np.vstack((crlbx, crlby, crlbz)).T, photons, fitsz, frame_offset, frame_offset+numFrms, sorted=False, method='fft')
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(np.arange(numFrms), driftx*TarPxszx)
    axs[0, 0].plot(np.arange(numFrms), interp_shifts[:,0]*TarPxszx)
    axs[0, 0].set_ylabel('unit = nm')
    axs[1, 0].plot(np.arange(numFrms), (interp_shifts[:,0]-driftx)*TarPxszx)
    axs[1, 0].set_ylabel('unit = nm')
    axs[0, 1].plot(np.arange(numFrms), drifty*TarPxszy)
    axs[0, 1].plot(np.arange(numFrms), interp_shifts[:,1]*TarPxszy)
    axs[0, 1].set_ylabel('unit = nm')
    axs[1, 1].plot(np.arange(numFrms), (interp_shifts[:,1]-drifty)*TarPxszy)
    axs[1, 1].set_ylabel('unit = nm')
    axs[0, 2].plot(np.arange(numFrms), driftz*Tarzstepsz)
    axs[0, 2].plot(np.arange(numFrms), interp_shifts[:,2]*Tarzstepsz)
    axs[0, 2].set_ylabel('unit = nm')
    axs[1, 2].plot(np.arange(numFrms), (interp_shifts[:,2]-driftz)*Tarzstepsz)
    axs[1, 2].set_ylabel('unit = nm')
    plt.show()