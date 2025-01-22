import os
import inspect
import ctypes
from numbers import Number
import numpy as np
from scipy.spatial import KDTree
import numpy.ctypeslib as ctl
import matplotlib as mpl
import matplotlib.cm as mplcm



def _get_nonsinglet(xvec, nneighbours, rlim):
    """
    find out the localizations that have more than nneighbours neighbours within a given radiusin 2d
    INPUTS:
        xvec:           (nspots, vnum) float ndarray, [[locx, locy(, locz), ...]]
        nneighbours:    int, number of neighbours
        rlim:           float, radius to search neighbours
    RETURN:
        ind_nonsinglet: (nspots) boolean array, true if a localization has more than nneighbours neighbours with in rlim
    """
    locs = xvec[:, :2].copy()
    kdt = KDTree(locs)
    ind_nonsinglet = np.ones(len(locs), dtype=bool)
    for i, loc in enumerate(locs):
        dists = kdt.query(loc, k=nneighbours+1)[0]
        if np.max(dists) > rlim:
            ind_nonsinglet[i] = False
    return ind_nonsinglet



def _get_luts(colormaps, colorbits):
    """
    get the luts with given colormaps
    INPUT:
        colormaps:      (nfluorophores,) of str, list or single string of the matplotlib colormap
        colorbits:      int, number of entries for each colormap
    RETURN:
        luts:           (nfluorophores, 3, colorbits) float, each (3, colorbits) is the RGB channel value for each given colormap
    """
    
    if isinstance(colormaps, str):
        colormaps = [colormaps]
    nfluorophores = len(colormaps)

    luts = np.zeros((nfluorophores, 3, colorbits))
    for (i, cname) in enumerate(colormaps):
        if cname in ('Red', 'Green', 'Blue'):
            luts[i][('Red', 'Green', 'Blue').index(cname)] = np.linspace(0.0, 1.0, num=colorbits)
            continue
        if cname in ('Cyan', 'Magenta', 'Yellow'):
            k = ('Cyan', 'Magenta', 'Yellow').index(cname)
            for j in range(3):
                if j != k:
                    luts[i][j] = np.linspace(0.0, 1.0, num=colorbits)
            continue
        if cname in {'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'}:
            cname = cname + '_r'
        cmap = mplcm.get_cmap(cname)
        lut = cmap(np.linspace(0.0, 1.0, num=colorbits)).T
        luts[i] = lut[:-1]

    return luts[0] if nfluorophores == 1 else luts



def _get_ccoder(ndim, feature, flIDs, clims, creg, xvec, crlb, indf, loss, zmin_nm, zmax_nm, frm_st, frm_ed):
    """
    get the color coder with the given feature and source data
    INPUTS:
        ndim:       int, number of dimensions of the xvec and crlb
        feature:    str, the feature to generate the color code
        flIDs:      int or list, specific fluorophore(s) to render
        clims:      (nfluorophore, 2) float, the quantile (min, max) for normalization of the color code for each fluorophore
        creg:       (nspots,) int, the registration to one of the fluorophores
        xvec:       (nspots, ndim + nchannels + nchannels) float, the source data xvec for the color code
        crlb:       (nspots, ndim + nchannels + nchannels) float, crlb correspodonding to the xvec
        indf:       (nspots,) int, the source data indf for the color code
        loss:       (nspots,) float, the source data loss for the color code
        zmin_nm:    float, the minimum z in nm if feature == locz
        zmax_nm:    float, the maximum z in nm if feature == locz
        frm_st:     int, the starting frame if feature == indf
        frm_ed:     int, the ending frame if feature == indf
    RETURN:
        ccoder:     (nspots,) float, the numerical coder for each spot
    """
    # parse the input
    assert feature in {'photon', 'locz', 'indf', 'precision_x', 'precision_y', 'precision_z', 'loss'}, "only 'photon', 'locz', 'indf', 'precision_x', 'precision_y', 'precision_z', and 'loss' are supported as the feature to be color-encoded"
    if ndim != 3: 
        assert feature in {'locz', 'z', 'precision_z'}, "ndim mismatch: ndim={}, not support the input.feature={}".format(ndim, feature)
    if isinstance(flIDs, Number):
        flIDs = [flIDs]
    if clims.ndim == 1:
        clims = clims[np.newaxis, :]
    assert len(flIDs) == len(clims), "nfluorophore mismatch: flIDs.len={}, clims.len={}".format(len(flIDs), len(clims))

    # initialize the ccoder
    if feature in ('photon', 'I'):
        ccoder = np.copy(xvec[:, ndim])
    elif feature in ('locz', 'z'):
        ccoder = np.copy(xvec[:, 2])
    elif feature in {'precision_x', 'precision_y', 'precision_z'}:
        ccoder = np.sqrt(1.0 / crlb[:, ('precision_x', 'precision_y', 'precision_z').index(feature)].copy())
    elif feature == 'indf':
        ccoder = np.float64(indf.copy())
    elif feature == 'loss':
        ccoder = loss.copy()
    
    # normalize the ccoder
    for flID, clim in zip(flIDs, clims):
        if len(ccoder[creg == flID]) > 0:
            if feature in ('locz', 'z'):
                ccoder[creg == flID] = np.clip(ccoder[creg == flID], zmin_nm, zmax_nm)
            elif feature == 'indf':
                ccoder[creg == flID] = np.clip(ccoder[creg == flID], frm_st, frm_ed)
            else:
                cmin_i = max(np.min(clim), 0.001)
                cmax_i = min(np.max(clim), 0.999)
                cmin_abs, cmax_abs = np.quantile(ccoder[creg == flID], (cmin_i, cmax_i))
                if cmin_abs == cmax_abs:
                    raise ValueError("fails to encode the colors from the feature (the ccoders are the same)")
                ccoder[creg == flID] = np.clip(ccoder[creg == flID], cmin_abs, cmax_abs)
            ccoder[creg == flID] = (ccoder[creg == flID] - ccoder[creg == flID].min()) / ccoder[creg == flID].ptp()

    return ccoder



def loc_render_mulfl(smlm_data, tar_pxsz, isblur, intensity, clim, colormap, zmin_nm, zmax_nm,
                    nosinglet=True, rlim_nm=50.0, sig=-1.0, frmrange=(0, -1), feature='photon', alpha=-1, colorbits=256, sigmin=5.0, sigmax=20.0, norm=0.0, _bits=32):
    """
    Render the smlm_data (with multipule fluorophore registered) into super-resolution image
    dependent on locrender.dll in the src_locrender sub-directory
    INPUT:
        smlm_data:      'roi_nm':       (4,) float ndarray, [top, left, bottom, right] in nm
                        'ndim':         int, number of dimention
                        'nchannels':    int, number of channels
                        'indf':         (nspots,) int ndarray, the frame index of each localization (extended from (NFits,))
                        'xvec':         (nspots, ndim + nchannels + nchannels) float ndarray, the localization information
                        'crlb':         (nspots, ndim + nchannels + nchannels) float ndarray, the crlb for the corresponding to the xvec
                        'loss':         (nspots,) float ndarray, the loss for each localization (extended from (NFits,))
                        'creg':         (nspots,) int ndarray, the registration of each fit to different channels
        tar_pxsz:       int, size of the rendered pixel
        isblur:         bool, whether the localization is blurred with its crlb
        intensity:      str, (photon, blink) intensity method for the rendered image
        clim:           (nfluorophores, 2) float ndarray, [cmin, cmax] percentile of the color code limits for each fluorophore
        colormap:       (nfluorophores,) str list, colormap for each fluorophore
        zmin_nm:        float, the minimum z in nm if feature == locz
        zmax_nm:        float, the maximum z in nm if feature == locz
        nosinglet:      bool, set it to True to ignore the localizations that has no neighbours with in rlim_nm
        rlim_nm:        float, the radius (in nm) to search neighbours in 2d, ignored if nosinglet is False
        sig:            float, constant sigma to render if sigma > 0
        frmrange:       (2,) int sequence-like, [frm_min, frm_max] frame range for rendering
        alpha:          float, the transparence of each localization, -1 for summing up
        colorbits:      int, number of bits for the colormap  
        sigmin:         float, minimum sigma to blur
        sigmax:         float, maximum sigma to blur
        norm:           float, the image value for image normalization, 0 for (0, 1) normalization
    RETURN: 
        Im:             rendered image
    NOTE:
        for xvec, crlb from mulpa_unpack, len(xvec)=ndim+nchannels*(mod+1) without bkg information, which is seperated out and will not be used in here
        for mulpa fit, NFits != nspots, and loss wont be used in rendering  
    """

    # parse the input
    _dtype = np.float32 if _bits == 32 else np.float64
    assert intensity in ('photon', 'blink'), "The intensity should be 'photon' or 'blink'."
    if intensity == 'photon':
        alpha = -1.0
    alpha = min(alpha, 1.0)
    sigmin = max(0.5 * tar_pxsz, sigmin)
    b_intensity = intensity.encode('utf-8')
    alpha = np.float32(alpha)
    tar_pxsz = np.float32(tar_pxsz)
    
    # generate the luts
    if len(colormap) != len(clim):
        raise ValueError("nfluorophores mismatch: len(colormap)={}, len(clim)={}".format(len(colormap), len(clim)))
    luts = _get_luts(colormap, colorbits)
    luts = np.ascontiguousarray(luts.flatten(), dtype=_dtype)
    nfluorophores = len(colormap)

    # link to the renderer
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_locrender')
    fname = 'locrender{}.dll'.format(_bits)
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    render_kernel = lib.loc_render_rgb
    render_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),                                              
                                ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),  
                                ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),  
                                ctl.ndpointer(_dtype, flags="aligned, c_contiguous"), 
                                ctl.ndpointer(_dtype, flags="aligned, c_contiguous"), 
                                ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),
                                ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),
                                ctypes.c_float, ctypes.c_int32, ctypes.c_int32,
                                ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),
                                ctypes.c_int32, ctypes.c_char_p]
    
    # images size
    roi = smlm_data['roi_nm']
    Imszy = np.int32(np.ceil((roi[2] - roi[0]) / tar_pxsz))
    Imszx = np.int32(np.ceil((roi[3] - roi[1]) / tar_pxsz))

    # generate the ccoder
    ndim = smlm_data['ndim']
    indf = smlm_data['indf']
    xvec = smlm_data['xvec']
    crlb = smlm_data['crlb']
    loss = smlm_data['loss']
    creg = smlm_data['creg']
    if frmrange[1] == -1:
        frmrange[1] = indf.max()
    flIDs = np.arange(nfluorophores)
    ccoder = _get_ccoder(ndim, feature, flIDs, clim, creg, xvec, crlb, indf, loss, zmin_nm, zmax_nm, frmrange[0], frmrange[1])

    # load data within the frame range
    frm_start = min(max(indf.min(), frmrange[0]), indf.max())
    frm_end = min(max(indf.min(), frmrange[1]), indf.max()) if frmrange[1] != -1 else indf.max()
    ind = (indf >= frm_start) & (indf <= frm_end)
    indf = indf[ind]
    xvec = xvec[ind]
    crlb = crlb[ind]
    loss = loss[ind]
    creg = creg[ind]
    ccoder = ccoder[ind]

    # exclude singlet data
    if nosinglet:
        ind_nonsinglet = _get_nonsinglet(xvec, 2, rlim_nm)
        indf = indf[ind_nonsinglet]
        xvec = xvec[ind_nonsinglet]
        crlb = crlb[ind_nonsinglet]
        loss = loss[ind_nonsinglet]
        creg = creg[ind_nonsinglet]
        ccoder = ccoder[ind_nonsinglet]
    sigs = np.zeros(len(xvec)) + sig if sig > 0 else np.clip(np.sqrt(0.5 * (crlb[:, 0] + crlb[:, 1])), sigmin, sigmax)
    
    # render
    nspots = len(xvec)
    creg = np.ascontiguousarray(creg, dtype=np.int32)
    locx = np.ascontiguousarray(xvec[:, 0] / tar_pxsz, dtype=_dtype)
    locy = np.ascontiguousarray(xvec[:, 1] / tar_pxsz, dtype=_dtype)
    sigs = np.ascontiguousarray(sigs / tar_pxsz, dtype=_dtype)
    photon = np.sqrt(xvec[:, ndim])
    photon_norm = np.median(photon)
    photon = np.clip(photon, 0.0, photon_norm)
    photon = np.ascontiguousarray(photon / photon_norm, dtype=_dtype)
    ccoder = np.ascontiguousarray(ccoder, dtype=_dtype)

    Im = np.ascontiguousarray(np.zeros(Imszy * Imszx * 3, dtype=_dtype), dtype=_dtype)
    render_kernel(nfluorophores, nspots, colorbits, creg, locx, locy, sigs, photon, ccoder, luts, alpha, Imszy, Imszx, Im, isblur, b_intensity)
    Im_norm = np.quantile(Im[Im > 0.0], 0.95)
    if Im_norm > 0:
        Im = Im / Im_norm if norm <= 0 else Im / norm
    Im = np.uint8(np.clip(Im.reshape((Imszy, Imszx, 3)) * 255, 0.0, 255.0))
    return Im, Im_norm



def loc_render_sinfl(smlm_data, flID, tar_pxsz, isblur, intensity, isRGB, clim, colormap, zmin_nm, zmax_nm,
                    nosinglet=False, rlim_nm=50.0, sig=-1.0, frmrange=(0, -1), feature='photon', alpha=-1, colorbits=256, sigmin=5.0, sigmax=20.0, norm=0.0, _bits=32):
    """
    Render the localizations into super-resolution image
    dependent on locrender.dll in the src_locrender sub-directory
    INPUT:
        smlm_data:      'roi_nm':       (4,) float ndarray, [top, left, bottom, right] in nm
                        'ndim':         int, number of dimention
                        'nchannels':    int, number of channels
                        'indf':         (nspots,) int ndarray, the frame index of each localization (extended from (NFits,))
                        'xvec':         (nspots, ndim + nchannels + nchannels) float ndarray, the localization information
                        'crlb':         (nspots, ndim + nchannels + nchannels) float ndarray, the crlb for the corresponding to the xvec
                        'loss':         (nspots,) float ndarray, the loss for each localization (extended from (NFits,))
                        'creg':         (nspots,) int ndarray, the registration of each fit to different channels
        flID:           int, the localizations registered to the flID-th fluorophore (ch_reg == flID) will be rendered
        tar_pxsz:       int, size of the rendered pixel
        isblur:         bool, whether the localization is blurred with its crlb
        intensity:      str, (photon, blink) intensity method for the rendered image
        isRGB:          bool, whether the localization is rendered into RGB or gray-scale
        clim:           (nfluorophores, 2) float ndarray, [cmin, cmax] percentile of the color code limits for each fluorophore
        colormap:       (nfluorophores,) str list, colormap for each fluorophore
        zmin_nm:        float, the minimum z in nm if feature == locz
        zmax_nm:        float, the maximum z in nm if feature == locz
        nosinglet:      bool, set it to True to ignore the localizations that has no neighbours with in rlim_nm
        rlim_nm:        float, the radius (in nm) to search neighbours in 2d, ignored if nosinglet is False
        sig:            float, specified bluring sigma
        frmrange:       (2,) int sequence-like, [frm_min, frm_max] frame range for rendering
        alpha:          float, the transparence of each localization, -1 for summing up
        colorbits:      int, number of bits for the colormap    
        sigmin:         float, minimum sigma to blur
        sigmax:         float, maximum sigma to blur
        norm:           float, the image value for image normalization, 0 for (0, 1) normalization
    RETURN: 
        Im:             rendered image

    NOTE:
        for xvec, crlb from mulpa_unpack, len(xvec)=ndim+nchannels*(mod+1) without bkg information, which is seperated out and will not be used in here
        for mulpa fit, NFits != nspots, and loss wont be used in rendering 
    """

    # parse the inputs
    _dtype = np.float32 if _bits == 32 else np.float64
    assert intensity in ('photon', 'blink'), "The intensity should be 'photon' or 'blink'."
    if intensity == 'photon':
        alpha = -1.0
    alpha = min(alpha, 1.0)
    sigmin = max(0.5 * tar_pxsz, sigmin)
    b_intensity = intensity.encode('utf-8')
    alpha = np.float32(alpha)
    tar_pxsz = np.float32(tar_pxsz)

    # generate the luts
    if isRGB:
        lut = _get_luts(colormap, colorbits)
        lut = np.ascontiguousarray(lut.flatten(), dtype=_dtype)

    # link to the renderer
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_locrender')
    fname = 'locrender{}.dll'.format(_bits)
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    if isRGB:
        render_kernel = lib.loc_render_rgb
        render_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                    ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),                                              
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),  
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"), 
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"), 
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),
                                    ctypes.c_float, ctypes.c_int32, ctypes.c_int32,
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),
                                    ctypes.c_int32, ctypes.c_char_p]
    else:
        render_kernel = lib.loc_render_gray
        render_kernel.argtypes = [ctypes.c_int32,                                          
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"), 
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"), 
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),
                                    ctypes.c_float, ctypes.c_int32, ctypes.c_int32,
                                    ctl.ndpointer(_dtype, flags="aligned, c_contiguous"),
                                    ctypes.c_int32, ctypes.c_char_p]
    
    # image size
    roi = smlm_data['roi_nm']
    Imszy = np.int32(np.ceil((roi[2] - roi[0]) / tar_pxsz))
    Imszx = np.int32(np.ceil((roi[3] - roi[1]) / tar_pxsz))

    # generate the ccoder
    if isRGB:
        ndim = smlm_data['ndim']
        indf = smlm_data['indf']
        xvec = smlm_data['xvec']
        crlb = smlm_data['crlb']
        loss = smlm_data['loss']
        creg = smlm_data['creg']
        if frmrange[1] < 0:
            frmrange[1] = indf.max()
        ccoder = _get_ccoder(ndim, feature, flID, clim, creg, xvec, crlb, indf, loss, zmin_nm, zmax_nm, frmrange[0], frmrange[1])
    
    # load data-in-need
    frm_start = min(max(indf.min(), frmrange[0]), indf.max())
    frm_end = min(max(indf.min(), frmrange[1]), indf.max()) if frmrange[1] != -1 else indf.max()
    ind = (indf >= frm_start) & (indf <= frm_end) & (creg == flID)
    indf = indf[ind]
    xvec = xvec[ind]
    crlb = crlb[ind]
    loss = loss[ind]
    creg = creg[ind]
    ccoder = ccoder[ind]
    
    # exclude the singlet data
    if nosinglet:
        ind_nonsinglet = _get_nonsinglet(xvec, 2, rlim_nm)
        indf = indf[ind_nonsinglet]
        xvec = xvec[ind_nonsinglet]
        crlb = crlb[ind_nonsinglet]
        loss = loss[ind_nonsinglet]
        creg = creg[ind_nonsinglet]
        ccoder = ccoder[ind_nonsinglet]
    sigs = np.zeros(len(xvec)) + sig if sig > 0 else np.clip(np.sqrt(0.5 * (crlb[:, 0] + crlb[:, 1])), sigmin, sigmax)

    nspots = len(xvec)
    if isRGB:
        creg = np.ascontiguousarray(np.zeros(nspots, dtype=np.int32), dtype=np.int32)
        locx = np.ascontiguousarray(xvec[:, 0] / tar_pxsz, dtype=_dtype)
        locy = np.ascontiguousarray(xvec[:, 1] / tar_pxsz, dtype=_dtype)
        sigs = np.ascontiguousarray(sigs / tar_pxsz, dtype=_dtype)
        photon = np.sqrt(xvec[:, ndim])
        photon_norm = np.median(photon)
        photon = np.clip(photon, 0.0, photon_norm)
        photon = np.ascontiguousarray(photon / photon_norm, dtype=_dtype)
        ccoder = np.ascontiguousarray(ccoder, dtype=_dtype)

        Im = np.ascontiguousarray(np.zeros(Imszy * Imszx * 3, dtype=_dtype), dtype=_dtype)
        render_kernel(1, nspots, colorbits, creg, locx, locy, sigs, photon, ccoder, lut, alpha, Imszy, Imszx, Im, isblur, b_intensity)
        Im_norm = np.quantile(Im[Im > 0.0], 0.95)
        if Im_norm > 0:
            Im = Im / Im_norm if norm <= 0 else Im / norm
        Im = np.uint8(np.clip(Im.reshape((Imszy, Imszx, 3)) * 255, 0.0, 255.0))
    else:
        locx = np.ascontiguousarray(xvec[:, 0] / tar_pxsz, dtype=_dtype)
        locy = np.ascontiguousarray(xvec[:, 1] / tar_pxsz, dtype=_dtype)
        sigs = np.ascontiguousarray(sigs / tar_pxsz, dtype=_dtype)
        photon = np.sqrt(xvec[:, ndim])
        photon_norm = np.median(photon)
        photon = np.clip(photon, 0.0, photon_norm)
        photon = np.ascontiguousarray(photon / photon_norm, dtype=_dtype)
        
        Im = np.ascontiguousarray(np.zeros(Imszy * Imszx), dtype=_dtype)
        render_kernel(nspots, locx, locy, sigs, photon, alpha, Imszy, Imszx, Im, isblur, b_intensity)
        Im_norm = np.quantile(Im[Im > 0.0], 0.95)
        if Im_norm > 0:
            Im = Im / Im_norm if norm <= 0 else Im / norm
        Im = Im.reshape((Imszy, Imszx))
        
    return Im, Im_norm