import os
import inspect
from numbers import Number
import ctypes
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



def _cart2pol(x, y):
    """transform the cartesian (x, y) into (rho, theta)"""
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return rho, theta



def _rot_locs(locs_i, coord_0, theta):
    """
    counter-clock rotate the input locs by theta (in rad) around coord_0
    INPUTS:
        locs_i:     (nspots, 2) float ndarray, [[locx_i, locy_i],...] the input localizations
        coord_0:    (2,) float ndarray, [coordx, coordy] the first coordinate of the line
        theta:      float, the rad to rotate
    RETURN:
        locs_o:     (nspots, 2) float ndarray, [[locx_o, locy_o],...] the rotated localizations
    """ 
    rotMat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    locs_o = (locs_i - coord_0).dot(rotMat)
    return locs_o + coord_0



def _get_verts(coord_0, coord_1, thickness):
    """
    get the 4 vertex of the roi defined by a line (coord_0, coord_1) and its thickness
    INPUTS:
        coord_0:    (2,) float ndarray, [coordx, coordy] the first coordinate of the line
        coord_1:    (2,) float ndarray, [coordx, coordy] the second coordinate of the line
        thickness:  int, the thickness of the line
    RETURN:
        verts:      (4, 2) float ndarray, [[vertx, verty],...] the 4 vertex
    """
    rho, theta = _cart2pol(coord_1[0] - coord_0[0], coord_1[1] - coord_0[1])
    verts = np.array([[coord_0[0], coord_0[1] + 0.5 * thickness],
                      [coord_0[0] + rho, coord_0[1] + 0.5 * thickness],
                      [coord_0[0] + rho, coord_0[1] - 0.5 * thickness],
                      [coord_0[0], coord_0[1] - 0.5 * thickness]])
    return _rot_locs(verts, coord_0, theta)



def _isinline(locs, coord_0, coord_1, thickness):
    """
    return the index for locx_i and locy_i that within a line defined by coord_0, coord_1, and thickness
    INPUTS:
        locs:       (nspots, 2) float ndarray, [[locx, locy],...] the input localizations
        coord_0:    (2,) float ndarray, [coordx, coordy] the first coordinate of the line
        coord_1:    (2,) float ndarray, [coordx, coordy] the second coordinate of the line
        thickness:  int, the thickness of the line
    RETURN:
        idx:        (nspots,) bool ndarray, the index of the locs within the line-region
    """
    nspots = len(locs)
    rho, theta = _cart2pol(*(coord_1 - coord_0))

    verts = _get_verts(coord_0, coord_1, thickness)
    dum_ind = np.where((locs[:, 0] > verts[:, 0].min()) & (locs[:, 0] < verts[:, 0].max()) & (locs[:, 1] > verts[:, 1].min()) & (locs[:, 1] < verts[:, 1].max()))[0]
    
    dum_locs = _rot_locs(locs[dum_ind], coord_0, -theta)
    idx_1 = dum_ind[(dum_locs[:, 0] > coord_0[0]) & (dum_locs[:, 0] < coord_0[0] + rho) & (dum_locs[:, 1] > coord_0[1] - 0.5 * thickness) & (dum_locs[:, 1] < coord_0[1] + 0.5 * thickness)]

    idx = np.zeros(nspots, dtype=bool)
    idx[idx_1] = True
    return idx



def _isinrect(locs, roi):
    """return the index for locx_i and locy_i that within a rectangle defined by roi = [top, left, bottom, right]"""
    return (locs[:, 0] > roi[1]) & (locs[:, 0] < roi[3]) & (locs[:, 1] > roi[0]) & (locs[:, 1] < roi[2])
    


def _get_DataInNeed(ndim, roi_type, roi_orient, roi, thickness, frmrange, flIDs, xvec_i, crlb_i, indf_i, loss_i, creg_i, ccoder_i):
    """
    get data within the roi and frmrange
    INPUTS:
        ndim:           int, dimension of the xvec_i
        roi_type:       str, 'rect' or 'line' types of the roi
        roi_orient:     str, 'xy' or 'xz' orientation of the roi
        roi:            [top, left, bottom, right] if roi_type == 'rect' else [[x_start, y_start], [x_end, y_end]] (unit = nm)
        thickness:      int, the thickness (in nm) of the roi if the roi is a line
        frmrange:       (2,) int, the start and end of the frame
        flIDs:          int or list, specific fluorophore(s) to render
        xvec_i:         (nspots_i, ndim + nchannels + nchannels) float, the input xvec data (in nm)
        crlb_i:         (nspots_i, ndim + nchannels + nchannels) float, the crlb corresponding to xvec_i
        indf_i:         (nspots_i,) int, the input indf
        loss_i:         (nspots_i,) flaot, the input loss
        creg_i:         (nspots_i,) int, the registration to one of the fluorphore of each spot
        ccoder_i:       (nspots_i,) float, the color coder for each spot
    RETURN:
        locs_o:         (nspots_o, 2) float, the output locx in nm to render (will be modified from locx and locy accordingly if the roi type is line)
        xvec_o:         (nspots_o, ndim + nchannels + nchannels) float, the output xvec accordingly
        crlb_o:         (nspots_o, ndim + nchannels + nchannels) float, the output crlb accordingly
        indf_o:         (nspots_o,) int, the output indf
        loss_o:         (nspots_o,) flaot, the output loss
        creg_o:         (nspots_o,) int, the output creg accordingly
        ccoder_o:       (nspots_o,) float, the output ccoder accordingly
    """
    if ndim != 3 and roi_type == 'line':
        raise ValueError("ndim mismatch: line roi is only accepted on 3d data, input.ndim={}".format(ndim))
    
    locs_i = xvec_i[:, :2]
    Idx_inroi = _isinrect(locs_i, roi) if roi_type == 'rect' else _isinline(locs_i, roi[0], roi[1], thickness)
    Idx_infrm = (indf_i >= frmrange[0]) & (indf_i <= frmrange[1]) if frmrange[1] != -1 else indf_i >= frmrange[0]
    Idx_flreg = creg_i == flIDs if isinstance(flIDs, Number) else np.isin(creg_i, flIDs)
    Idx = Idx_inroi & Idx_infrm & Idx_flreg 

    if roi_type == 'rect':
        locx_o = xvec_i[Idx, 0] - roi[1]
        locy_o = xvec_i[Idx, 2] * 0.81 if roi_orient == 'xz' else xvec_i[Idx, 1] - roi[0]
    else:
        rho, theta = _cart2pol(*(roi[1] - roi[0]))
        dum_locs = _rot_locs(xvec_i[Idx, :2], roi[0], -theta)
        locx_o = dum_locs[:, 0] - roi[0][0]
        locy_o = xvec_i[Idx, 2] * 0.81 if roi_orient == 'xz' else dum_locs[:, 1] - roi[0][1] + 0.5 * thickness
    
    locs_o = np.hstack((locx_o[...,np.newaxis], locy_o[...,np.newaxis]))
    xvec_o = xvec_i[Idx]
    crlb_o = crlb_i[Idx]
    indf_o = indf_i[Idx]
    loss_o = loss_i[Idx]
    creg_o = creg_i[Idx]
    ccoder_o = ccoder_i[Idx]
    return locs_o, xvec_o, crlb_o, indf_o, loss_o, creg_o, ccoder_o



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
        ccoder = np.float32(indf.copy())
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



def finer_render_mulfl(smlm_data, roi_type, roi_orient, roi, thickness, tar_pxsz, isblur, intensity, flID, clim, colormap, zmin_nm, zmax_nm,
                    nosinglet=False, rlim_nm=50.0, sig=-1.0, frmrange=(0, -1), feature='photon', alpha=-1, colorbits=256, zrange=1500, sigmin=2.5, sigmax=12.5, norm=0.0, _bits=32):
    """
    Render the smlm_data (with multipule fluorophore registered) into super-resolution image
    dependent on locrender.dll in the src_locrender sub-directory
    INPUT:
        smlm_data:      'ndim':         int, number of dimention
                        'nchannels':    int, number of channels
                        'indf':         (nspots,) int ndarray, the frame index of each localization (extended from (NFits,))
                        'xvec':         (nspots, ndim + nchannels + nchannels) float ndarray, the localization information in nm
                        'crlb':         (nspots, ndim + nchannels + nchannels) float ndarray, the crlb for the corresponding to the xvec
                        'loss':         (nspots,) float ndarray, the loss for each localization (extended from (NFits,))
                        'creg':         (nspots,) int ndarray, the registration of each fit to different channels
        roi_type:       str, 'rect' or 'line' types of the roi
        roi_orient:     str, 'xy' or 'xz' orientation of the roi
        roi:            [top, left, bottom, right] if roi_type == 'rect' else [[x_start, y_start], [x_end, y_end]] (unit = nm)
        thickness:      int, the thickness (in nm) of the roi if it is a line roi
        tar_pxsz:       int, size of the rendered pixel
        isblur:         bool, whether the localization is blurred with its crlb
        intensity:      str, (photon, blink) intensity method for the rendered image
        flID:           int or list, specific fluorophore(s) to render
        clim:           (nfluorophores, 2) float ndarray, [cmin, cmax] percentile of the color code limits for each fluorophore
        colormap:       (nfluorophores,) str list, colormap for each fluorophore
        zmin_nm:        float, the minimum z in nm if feature == locz
        zmax_nm:        float, the maximum z in nm if feature == locz
        nosinglet:      bool, set it to True to ignore the localizations that has no neighbours with in rlim_nm
        rlim_nm:        float, the radius (in nm) to search neighbours in 2d, ignored if nosinglet is False
        sig:            float, constant sigma to render if > 0
        frmrange:       (2,) int sequence-like, [frm_min, frm_max] frame range for rendering
        alpha:          float, the transparence of each localization, -1 for summing up
        colorbits:      int, number of bits for the colormap  
        zmargin:        int, the margin (nm) added to the z-axis if the roi_type is 'line'
        sigmin:         float, minimum sigma to blur
        sigmax:         float, maximum sigma to blur
        norm:           float, the image value for image normalization, 0 for (0, 1) normalization
    RETURN: 
        Im:             rendered image
    """
    
    # PARSE THE INPUTS
    _dtype = np.float32 if _bits == 32 else np.float64
    ndim = smlm_data['ndim']
    assert intensity in ('photon', 'blink'), "The intensity should be 'photon' or 'blink'."
    if intensity == 'photon':
        alpha = -1.0
    b_intensity = intensity.encode('utf-8')
    if isinstance(flID, Number):
        flID = [flID]
    alpha = min(alpha, 1.0)
    sigmin = max(0.5 * tar_pxsz, sigmin)
    alpha = np.float32(alpha)
    tar_pxsz = np.float32(tar_pxsz)

    # GENERATES THE LUTS
    if len(colormap) != len(flID):
        raise ValueError("nfluorophores mismatch: len(colormap)={}, len(flID)={}".format(len(colormap), len(flID)))
    if len(colormap) != len(clim):
        raise ValueError("nfluorophores mismatch: len(colormap)={}, len(clim)={}".format(len(colormap), len(clim)))
    nfluorophores = len(colormap)
    luts = _get_luts(colormap, colorbits)
    luts = np.ascontiguousarray(luts.flatten(), dtype=_dtype)

    # LINK TO THE RENDERER
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
    
    # GENERATE THE CCODER
    ndim = smlm_data['ndim']
    indf = smlm_data['indf']
    xvec = smlm_data['xvec']
    crlb = smlm_data['crlb']
    loss = smlm_data['loss']
    creg = smlm_data['creg']
    if frmrange[1] == -1:
        frmrange[1] = indf.max()
    ccoder = _get_ccoder(ndim, feature, flID, clim, creg, xvec, crlb, indf, loss, zmin_nm, zmax_nm, frmrange[0], frmrange[1])

    # GET THE SMLM_DATA IN ROI
    source_data = [xvec, crlb, indf, loss, creg, ccoder]
    locs, xvec, crlb, indf, loss, creg, ccoder = _get_DataInNeed(ndim, roi_type, roi_orient, roi, thickness, frmrange, flID, *source_data)
    
    # EXCLUDE THE SINGLET DATA
    if nosinglet:
        ind_nonsinglet = _get_nonsinglet(locs, 2, rlim_nm)
        indf = indf[ind_nonsinglet]
        locs = locs[ind_nonsinglet]
        xvec = xvec[ind_nonsinglet]
        crlb = crlb[ind_nonsinglet]
        loss = loss[ind_nonsinglet]
        creg = creg[ind_nonsinglet]
        ccoder = ccoder[ind_nonsinglet]
    if sig > 0:
        sigs = np.zeros(len(xvec)) + sig
    else:
        sigs = np.sqrt(0.5 * (crlb[:, 0] + 0.25*crlb[:, 2])) if roi_orient == 'xz' else np.sqrt(0.5 * (crlb[:, 0] + crlb[:, 1]))
        #sigs = np.sqrt(np.minimum(crlb[:, 0], crlb[:, 2])) if roi_orient == 'xz' else np.sqrt(0.5 * (crlb[:, 0] + crlb[:, 1]))
        sigs = np.clip(sigs, sigmin, sigmax) 
    
    nspots = len(xvec)
    if nspots == 0:
        if roi_type == 'rect':
            Imszx = np.int32(np.ceil((roi[3] - roi[1]) / tar_pxsz))
            Imszy = np.int32(np.ceil(zrange / tar_pxsz)) if roi_orient == 'xz' else np.int32(np.ceil((roi[2] - roi[0]) / tar_pxsz))
        else:
            Imszx = np.int32(np.ceil(np.sqrt(np.sum((roi[1] - roi[0]) * (roi[1] - roi[0]))) / tar_pxsz))
            Imszy = np.int32(np.ceil(zrange / tar_pxsz)) if roi_orient == 'xz' else np.int32(thickness / tar_pxsz)            
        return np.zeros((Imszy, Imszx, 3), dtype=np.uint8), 0  
    
    # CONVERSION
    creg_c = np.copy(creg)
    for i, f in enumerate(flID):
        creg_c[creg == f] = i
    
    if roi_type == 'rect':
        Imszx = np.int32(np.ceil((roi[3] - roi[1]) / tar_pxsz))
        Imszy = np.int32(np.ceil(zrange / tar_pxsz)) if roi_orient == 'xz' else np.int32(np.ceil((roi[2] - roi[0]) / tar_pxsz))
    else:
        Imszx = np.int32(np.ceil(np.sqrt(np.sum((roi[1] - roi[0]) * (roi[1] - roi[0]))) / tar_pxsz))
        Imszy = np.int32(np.ceil(zrange / tar_pxsz)) if roi_orient == 'xz' else np.int32(thickness / tar_pxsz)

    creg_c      = np.ascontiguousarray(creg_c, dtype=np.int32)
    locx        = np.ascontiguousarray(locs[:, 0] / tar_pxsz, dtype=_dtype)
    locy        = np.ascontiguousarray(locs[:, 1] / tar_pxsz, dtype=_dtype)
    sigs        = np.ascontiguousarray(sigs / tar_pxsz, dtype=_dtype)
    photon      = np.sqrt(xvec[:, ndim])
    photon_norm = np.median(photon)
    photon      = np.clip(photon, 0.0, photon_norm)
    photon      = np.ascontiguousarray(photon / photon_norm, dtype=_dtype)
    ccoder      = np.ascontiguousarray(ccoder, dtype=_dtype)
    
    # RENDER
    Im = np.ascontiguousarray(np.zeros(Imszy * Imszx * 3, dtype=_dtype), dtype=_dtype)
    render_kernel(nfluorophores, nspots, colorbits, creg_c, locx, locy, sigs, photon, ccoder, luts, alpha, Imszy, Imszx, Im, isblur, b_intensity)
    Im_norm = np.quantile(Im[Im > 0.0], 0.98) if np.any(Im > 0) else 0.0
    if Im_norm > 0:
        Im = Im / Im_norm if norm <= 0 else Im / norm
    Im = np.uint8(np.clip(Im.reshape((Imszy, Imszx, 3)) * 255.0, 0, 255.0))
    return Im, Im_norm