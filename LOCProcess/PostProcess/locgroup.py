import os
import inspect
import ctypes
import numpy as np
import numpy.ctypeslib as ctl



def _dbscan(locs, epsilon, minpts):
    """
    dbscan cluster analysis of the input locs
    dependent on dbscan.dll in the same pacakge
    INPUT:
        locs:               (nspots, ndim) float ndarray, [[locx, locy, locz],...] localizations of the detections 
        epsilon:            (ndim,) the epsilon radius that defines the cluster
        minpts:             the number of minimal points that defines the cluster                      
    RETURN:
        classification:     (nspots,) int, the labels of the cluster_id of each localization in locs, -1 if not clusterd
        iscore:             (nspots,) bool, True for the core points of the locs
    """
    nspots, ndim = locs.shape

    locs = np.ascontiguousarray(locs.flatten(), dtype=np.float32)
    epsilon = np.ascontiguousarray(epsilon.flatten(), dtype=np.float32)
    classification = np.ascontiguousarray(np.zeros(nspots), dtype=np.int32)
    iscore = np.ascontiguousarray(np.zeros(nspots), dtype=bool)
    
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_dbscan')
    fname = 'dbscan.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    dbscan_kernel = lib.dbscan
    dbscan_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32,
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                ctypes.c_int32,
                                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                                ctl.ndpointer(bool, flags="aligned, c_contiguous")]
    dbscan_kernel(ndim, nspots, locs, epsilon, minpts, classification, iscore)
    return classification, iscore



def _loc_inframe_connect(locs, crlb, sigma_factor, dist_min, dist_max):
    """
    Connect the localizations appear within a given range in concsecutive frames (similar to single-particle tracking)    
    INPUT:
        locs:           (nspots, ndim) float ndarray, [[locx, locy(, locz)], ...]
        crlb:           (nspots, ndim) float ndarray, corresponding crlb for xvec
        sigma_factor:   float, scaler to define the searching radius
        dist_min:       (ndim,) float ndarray, minimum distance for neighbor search
        dist_max:       (ndim,) float ndarray, maximum distance for neighbor search
    RETURN:
        labels:         (nspots,) int array, the labels of the cluster_id of each localization in locs, -1 if not clusterd
    NOTE:
        !!!!! xvec and crlb should be sorted by xvec[:, 0]
    """
    
    assert locs.shape == crlb.shape, "shape mismatch, locs.shape={}, crlb.shape={}".format(locs.shape, crlb.shape)
    nspots, ndim = locs.shape
    if sigma_factor > 0:
        locvar = np.zeros((nspots, ndim))
        for dimID in range(ndim):
            locvar[:, dimID] = np.maximum(sigma_factor * crlb[:, dimID], dist_min[dimID] * dist_min[dimID])

    # Link to the connector
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_group')
    fname = 'loc_group.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    if sigma_factor <= 0:
        connect_kernel = lib.loc_merge_const
        connect_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32,                                              
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),   
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                    ctl.ndpointer(np.int32, flags="aligned, c_contiguous")]
    else:
        connect_kernel = lib.loc_merge_var
        connect_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32,                                              
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                    ctl.ndpointer(np.int32, flags="aligned, c_contiguous")]
    
    locs = np.ascontiguousarray(locs.flatten(), dtype=np.float32)
    locvar = np.ascontiguousarray(locvar.flatten(), dtype=np.float32) if sigma_factor > 0 else None 
    dist_max = np.ascontiguousarray(dist_max, dtype=np.float32)
    labels = np.ascontiguousarray(np.zeros(nspots), dtype=np.int32)
    
    if sigma_factor > 0:
        connect_kernel(nspots, ndim, locs, locvar, dist_max, labels)
    else:
        connect_kernel(nspots, ndim, locs, dist_max, labels)
    
    return labels - 1 # -1 is to keep consistant with the _dbscan labels



def _loc_consecutive_connect(indf, locs, crlb, sigma_factor, dist_min, dist_max, gapf_tol):
    """
    Connect the localizations appear within a given range in concsecutive frames (similar to single-particle tracking)    
    INPUT:
        indf:           (nspots,) int ndarray, the frame of each localization
        xvec:           (nspots, ndim) float ndarray, [[locx, locy(, locz)], ...]
        crlb:           (nspots, ndim) float ndarray, corresponding crlb for xvec
        sigma_factor:   float, scaler to define the searching radius
        dist_min:       (ndim,) float ndarray, minimum distance for neighbor search
        dist_max:       (ndim,) float ndarray, maximum distance for neighbor search
        gap_tol:        int, maximum frame gap allowed to consider consecutivity
    RETURN:
        labels:         (nspots,) int array, the labels of the cluster_id of each localization in locs, -1 if not clusterd
    NOTE:
        !!!!! indf, xvec, crlb should be primarily sorted by indf and then xvec[:, 0]
    """

    assert locs.shape == crlb.shape, "shape mismatch, locs.shape={}, crlb.shape={}".format(locs.shape, crlb.shape)
    nspots, ndim = locs.shape
    if sigma_factor > 0:
        locvar = np.zeros((nspots, ndim))
        for dimID in range(ndim):
            locvar[:, dimID] = np.maximum(sigma_factor * crlb[:, dimID], dist_min[dimID] * dist_min[dimID])

    # Link to the connector
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_group')
    fname = 'loc_group.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    if sigma_factor <= 0:
        connect_kernel = lib.loc_connect_const
        connect_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32,                                             
                                    ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),     
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),   
                                    ctypes.c_int32,
                                    ctl.ndpointer(np.int32, flags="aligned, c_contiguous")]
    else:
        connect_kernel = lib.loc_connect_var
        connect_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32,                                             
                                    ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),     
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                    ctypes.c_int32,
                                    ctl.ndpointer(np.int32, flags="aligned, c_contiguous")]
    
    frm = np.ascontiguousarray(indf, dtype=np.int32)
    locs = np.ascontiguousarray(locs.flatten(), dtype=np.float32)
    locvar = np.ascontiguousarray(locvar.flatten(), dtype=np.float32) if sigma_factor > 0 else None 
    dist_max = np.ascontiguousarray(dist_max, dtype=np.float32)
    labels = np.ascontiguousarray(np.zeros(nspots), dtype=np.int32)
    
    if sigma_factor > 0:
        connect_kernel(nspots, ndim, frm, locs, locvar, dist_max, gapf_tol, labels)
    else:
        connect_kernel(nspots, ndim, frm, locs, dist_max, gapf_tol, labels)
    
    return labels - 1 # -1 is to keep consistant with the _dbscan labels
    


def inframe_merge_c(ndim, indf, xvec, crlb, loss, sigma_factor, dist_min, dist_max, w='crlb'):
    """
    merge the closely distributed localizations from the same frame, cluster with _dbscan
    INPUT:
        ndim:           int, number of dimensions
        indf:           (nspots_i,) int ndarray, the frame index corresponding to the xvec (extended from (NFits,))
        xvec:           (nspots_i, vnum) float ndarray, [[locx, locy(, locz), ...],...]
        crlb:           (nspots_i, vnum) float ndarray, crlb corresponding to xvec
        loss:           (nspots_i,) float ndarray, the loss of each localization (extended from (NFits,))
        sigma_factor:   float, scaler to define the searching radius
        dist_min:       (ndim,) float ndarray, minimum distance for neighbor search
        dist_max:       (ndim,) float ndarray, maximum distance for neighbor search
        w:              str, {'photon', 'crlb'} option for weighting
    RETURN:
        mask:           (nspots_i,) bool array, True for unmerged entities
    NOTE:
        !!!!! indf, xvec, crlb should be sorted primarily by indf and then by xvec[:, 0]
        xvec, crlb, loss are modifyed in-place
    """
    if xvec.ndim == 1 or len(xvec) == 1:
        return [True]
    
    nspots = len(indf)
    mask = np.zeros(nspots, dtype=bool)  
    
    nperfrm = np.unique(indf, return_counts=True)
    nsegs = len(nperfrm[0])
    frm_segs = np.concatenate(([0], np.cumsum(nperfrm[1])))
    
    for i in range(nsegs):
        dum_xvec = xvec[frm_segs[i] : frm_segs[i+1]]
        dum_crlb = crlb[frm_segs[i] : frm_segs[i+1]]
        dum_loss = loss[frm_segs[i] : frm_segs[i+1]]
        dum_labels = _loc_inframe_connect(dum_xvec[:, :ndim], dum_crlb[:, :ndim], sigma_factor, dist_min, dist_max)
        dum_mask = dum_labels == -1
        
        clusters = set(dum_labels)
        if -1 in clusters:
            clusters.remove(-1)
        
        for cluster_ID in clusters:
            dum_Idx = np.where(dum_labels == cluster_ID)[0]
            ndum = len(dum_Idx)
            dum_weit = 1.0 / dum_crlb[dum_Idx] if w == 'crlb' else dum_xvec[dum_Idx, ndim]
            dum_weit_sum = np.sum(dum_weit, axis=0) if w == 'crlb' else np.sum(dum_weit)
            if w == 'crlb':
                dum_xvec[dum_Idx[0]] = np.sum(dum_xvec[dum_Idx] * dum_weit, axis=0) / dum_weit_sum
                dum_crlb[dum_Idx[0]] = ndum / dum_weit_sum
            else:
                dum_xvec[dum_Idx[0]] = np.sum(dum_xvec[dum_Idx].T * dum_weit, axis=1) / dum_weit_sum
                dum_crlb[dum_Idx[0]] = np.sum(dum_crlb[dum_Idx].T * dum_weit, axis=1) / dum_weit_sum
            dum_loss[dum_Idx[0]] = np.mean(dum_loss[dum_Idx])
            dum_mask[dum_Idx[0]] = True
        
        xvec[frm_segs[i] : frm_segs[i+1]] = dum_xvec
        crlb[frm_segs[i] : frm_segs[i+1]] = dum_crlb
        loss[frm_segs[i] : frm_segs[i+1]] = dum_loss
        mask[frm_segs[i] : frm_segs[i+1]] = dum_mask

    return mask



def inframe_merge_dbscan(ndim, indf, xvec, crlb, loss, maxR, w='crlb'):
    """
    merge the closely distributed localizations from the same frame, cluster with _dbscan
    INPUT:
        ndim:       int, number of dimensions
        indf:       (nspots_i,) int ndarray, the frame index corresponding to the xvec (extended from (NFits,))
        xvec:       (nspots_i, vnum) float ndarray, [[locx, locy(, locz), ...],...]
        crlb:       (nspots_i, vnum) float ndarray, crlb corresponding to xvec
        loss:       (nspots_i,) float ndarray, the loss of each localization (extended from (NFits,))
        maxR:       (ndim) float ndarray, maximum radius allowed to be considered to merge
        w:          str, {'photon', 'crlb'} option for weighting
    RETURN:
        mask:       (nspots_i,) bool array, True for unmerged entities
    NOTE:
        xvec, crlb, loss are modifyed in-place
    """
    if xvec.ndim == 1 or len(xvec) == 1:
        return [True]
    
    nspots = len(indf)
    mask = np.zeros(nspots, dtype=bool)  
    
    nperfrm = np.unique(indf, return_counts=True)
    nsegs = len(nperfrm[0])
    frm_segs = np.concatenate(([0], np.cumsum(nperfrm[1])))

    for i in range(nsegs):
        dum_xvec = xvec[frm_segs[i] : frm_segs[i+1]]
        dum_crlb = crlb[frm_segs[i] : frm_segs[i+1]]
        dum_loss = loss[frm_segs[i] : frm_segs[i+1]]
        dum_labels = _dbscan(dum_xvec[:, :ndim], np.array(maxR[:ndim]), minpts=2)[0]
        dum_mask = dum_labels == -1

        clusters = set(dum_labels)
        if -1 in clusters:
            clusters.remove(-1)
        
        for cluster_ID in clusters:
            dum_Idx = np.where(dum_labels == cluster_ID)[0]
            ndum = len(dum_Idx)
            dum_weit = 1.0 / dum_crlb[dum_Idx] if w == 'crlb' else dum_xvec[dum_Idx, ndim]
            dum_weit_sum = np.sum(dum_weit, axis=0) if w == 'crlb' else np.sum(dum_weit)
            if w == 'crlb':
                dum_xvec[dum_Idx[0]] = np.sum(dum_xvec[dum_Idx] * dum_weit, axis=0) / dum_weit_sum
                dum_crlb[dum_Idx[0]] = ndum / dum_weit_sum
            else:
                dum_xvec[dum_Idx[0]] = np.sum(dum_xvec[dum_Idx].T * dum_weit, axis=1) / dum_weit_sum
                dum_crlb[dum_Idx[0]] = np.sum(dum_crlb[dum_Idx].T * dum_weit, axis=1) / dum_weit_sum
            dum_loss[dum_Idx[0]] = np.mean(dum_loss[dum_Idx])
            dum_mask[dum_Idx[0]] = True
        
        xvec[frm_segs[i] : frm_segs[i+1]] = dum_xvec
        crlb[frm_segs[i] : frm_segs[i+1]] = dum_crlb
        loss[frm_segs[i] : frm_segs[i+1]] = dum_loss
        mask[frm_segs[i] : frm_segs[i+1]] = dum_mask
        
    return mask



def consecutive_align_c(ndim, indf, xvec, crlb, sigma_factor, dist_min, dist_max, gapf_tol, w='crlb'):
    """
    Align the localizations appear within a given range in concsecutive frames (similar to single-particle tracking)    
    xvec and crlb are modified in-place 
    INPUT:
        ndim:           int, number of dimensions
        indf:           (nspots_i,) int ndarray, the frame of each localization (extended from (NFits,))
        xvec:           (nspots_i, vnum) float ndarray, [locx, locy(, locz), ...]
        crlb:           (nspots_i, vnum) float ndarray, corresponding crlb for xvec
        sigma_factor:   float, scaler to define the searching radius
        dist_min:       (ndim,) float ndarray, minimum distance for neighbor search
        dist_max:       (ndim,) float ndarray, maximum distance for neighbor search
        gap_tol:        int, maximum frame gap allowed to consider consecutivity
        w:              str, {'photon', 'crlb'} option for weighting
    NOTE:
        !!!!! indf, xvec, crlb should be sorted primarily by indf and then by xvec[:, 0]
        xvec and crlb are modified in-place
    """
    if xvec.ndim == 1 or len(xvec) == 1:
        return xvec, crlb

    labels = _loc_consecutive_connect(indf, xvec[:, :ndim], crlb[:, :ndim], sigma_factor, dist_min, dist_max, gapf_tol)

    clusters = set(labels)
    if -1 in clusters:
        clusters.remove(-1)

    for cluster_ID in clusters:
        dum_Idx = np.where([labels == cluster_ID])[0]
        ndum = len(dum_Idx)
        dum_locs = xvec[dum_Idx, :ndim]
        dum_crlb = crlb[dum_Idx, :ndim]
        dum_weit = 1.0 / dum_crlb if w == 'crlb' else xvec[dum_Idx, ndim]
        dum_weit_sum = np.sum(dum_crlb, axis=0) if w == 'crlb' else np.sum(dum_weit)
        if w == 'crlb':
            xvec[dum_Idx, :ndim] = np.tile(np.sum(dum_locs * dum_weit, axis=0) / dum_weit_sum, ndum).reshape(ndum, ndim)
            crlb[dum_Idx, :ndim] = np.tile(ndum / dum_weit_sum, ndum).reshape(ndum, ndim)
        else:
            xvec[dum_Idx, :ndim] = np.tile(np.sum(dum_locs.T * dum_weit, axis=1) / dum_weit_sum, ndum).reshape(ndum, ndim)
            crlb[dum_Idx, :ndim] = np.tile(np.sum(dum_crlb.T * dum_weit, axis=1) / dum_weit_sum, ndum).reshape(ndum, ndim)
    return



def consecutive_merge_c(ndim, indf, xvec, crlb, loss, sigma_factor, dist_min, dist_max, gapf_tol, w='crlb'):
    """
    Group the localizations appear within a given range in concsecutive frames (similar to single-particle tracking)    
    INPUT:
        ndim:           int, number of dimensions
        indf:           (nspots_i,) int ndarray, the frame of each localization (extended from (NFits,))
        xvec:           (nspots_i, vnum) float ndarray, [[locx, locy(, locz)], ...]
        crlb:           (nspots_i, vnum) float ndarray, corresponding crlb for xvec
        loss:           (nspots_i,) float ndarray, the loss of each localization (extended from (NFits,))
        sigma_factor:   float, scaler to define the searching radius
        dist_min:       (ndim,) float ndarray, minimum distance for neighbor search
        dist_max:       (ndim,) float ndarray, maximum distance for neighbor search
        gap_tol:        int, maximum frame gap allowed to consider consecutivity
        w:              str, {'photon', 'crlb'} option for weighting
    RETURN:
        mask:           (nspots_i,) bool array, True for unmerged entities
    NOTE:
        !!!!! indf, xvec, crlb should be sorted primarily by indf and then by xvec[:, 0]
        indf, xvec, crlb, loss are modified in-place
    """
    if xvec.ndim == 1 or len(xvec[0]) == 1:
        return indf, indf.copy(), xvec, crlb, loss

    mask = np.zeros(len(indf), dtype=bool)
    labels = _loc_consecutive_connect(indf, xvec[:, :ndim], crlb[:, :ndim], sigma_factor, dist_min, dist_max, gapf_tol)
    mask[labels == -1] = True
    
    clusters = set(labels)
    if -1 in clusters:
        clusters.remove(-1)

    for cluster_ID in clusters:
        
        dum_Idx = np.where(labels == cluster_ID)[0]
        ndum = len(dum_Idx)

        dum_xvec = xvec[dum_Idx]
        dum_crlb = crlb[dum_Idx]
        dum_loss = loss[dum_Idx]
        dum_weit = 1.0 / dum_crlb if w == 'crlb' else dum_xvec[:, ndim]
        dum_weit_sum = np.sum(dum_weit, axis=0) if w == 'crlb' else np.sum(dum_weit)
        if w == 'crlb':
            xvec[dum_Idx[0]] = np.sum(dum_xvec * dum_weit, axis=0) / dum_weit_sum
            crlb[dum_Idx[0]] = ndum / dum_weit_sum
        else:
            xvec[dum_Idx[0]] = np.sum(dum_xvec.T * dum_weit, axis=1) / dum_weit_sum
            crlb[dum_Idx[0]] = np.sum(dum_crlb.T * dum_weit, axis=1) / dum_weit_sum
        indf[dum_Idx[0]] = indf[dum_Idx].min()
        loss[dum_Idx[0]] = np.mean(dum_loss)
        mask[dum_Idx[0]] = True

    return mask



def consecutive_align_dbscan(ndim, indf, xvec, crlb, maxR, gapf_tol, w='crlb'):
    """
    Align the localizations appear within a given range in concsecutive frames (similar to single-particle tracking)    
    xvec and crlb are modified in-place 
    INPUT:
        ndim:       int, number of dimensions
        indf:       (nspots_i,) int ndarray, the frame of each localization (extended from (NFits,))
        xvec:       (nspots_i, vnum) float ndarray, [locx, locy(, locz), ...]
        crlb:       (nspots_i, vnum) float ndarray, corresponding crlb for xvec
        maxR:       (ndim,) float, maximum radius allowed to be considered as the same-place
        gap_tol:    int, maximum frame gap allowed to consider consecutivity
        w:          str, {'photon', 'crlb'} option for weighting
    NOTE:
        xvec and crlb are modified in-place
    """
    if xvec.ndim == 1 or len(xvec) == 1:
        return xvec, crlb

    epsilon = np.append(maxR[:ndim], 1.0 + gapf_tol)
    locs = np.hstack((xvec[:, :ndim], np.float32(indf[..., np.newaxis])))
    labels = _dbscan(locs, epsilon, minpts=2)[0]

    clusters = set(labels)
    if -1 in clusters:
        clusters.remove(-1)
    
    for cluster_ID in clusters:
        dum_Idx = np.where(labels == cluster_ID)[0]
        ndum = len(dum_Idx)
        dum_locs = xvec[dum_Idx, :ndim]
        dum_crlb = crlb[dum_Idx, :ndim]
        dum_weit = 1.0 / dum_crlb if w == 'crlb' else xvec[dum_Idx, ndim]
        dum_weit_sum = np.sum(dum_weit, axis=0) if w == 'crlb' else np.sum(dum_weit)
        if w == 'crlb':
            xvec[dum_Idx, :ndim] = np.tile(np.sum(dum_locs * dum_weit, axis=0) / dum_weit_sum, ndum).reshape(ndum, ndim)
            crlb[dum_Idx, :ndim] = np.tile(ndum / dum_weit_sum, ndum).reshape(ndum, ndim)
        else:
            xvec[dum_Idx, :ndim] = np.tile(np.sum(dum_locs.T * dum_weit, axis=1) / dum_weit_sum, ndum).reshape(ndum, ndim)
            crlb[dum_Idx, :ndim] = np.tile(np.sum(dum_crlb.T * dum_weit, axis=1) / dum_weit_sum, ndum).reshape(ndum, ndim)
    return



def consecutive_merge_dbscan(ndim, indf, xvec, crlb, loss, maxR, gapf_tol, w='crlb'):
    """
    Group the localizations appear within a given range in concsecutive frames (similar to single-particle tracking)    
    INPUT:
        ndim:       int, number of dimensions
        indf:       (nspots_i,) int ndarray, the frame of each localization (extended from (NFits,))
        xvec:       (nspots_i, vnum) float ndarray, [[locx, locy(, locz)], ...]
        crlb:       (nspots_i, vnum) float ndarray, corresponding crlb for xvec
        loss:       (nspots_i,) float ndarray, the loss of each localization (extended from (NFits,))
        maxR:       (ndim,) float, maximum radius allowed to be considered as the same-place
        gap_tol:    int, maximum frame gap allowed to consider consecutivity
        w:          str, {'photon', 'crlb'} option for weighting
    OUTPUT:
        mask:       (nspots_i,) bool array, True for unmerged entities
    NOTE:
        indf, xvec, crlb, loss are modified in-place   
    """
    if xvec.ndim == 1 or len(xvec[0]) == 1:
        return indf, indf.copy(), xvec, crlb, loss

    epsilon = np.append(maxR[:ndim], 1.0 + gapf_tol)
    mask = np.zeros(len(indf), dtype=bool)
    locs = np.hstack((xvec[:, :ndim], np.float32(indf[..., np.newaxis])))
    labels = _dbscan(locs, epsilon, minpts=2)[0]
    mask[labels == -1] = True
    
    clusters = set(labels)
    if -1 in clusters:
        clusters.remove(-1)
    
    for cluster_ID in clusters:
        dum_Idx = np.where(labels == cluster_ID)[0]
        ndum = len(dum_Idx)
        dum_xvec = xvec[dum_Idx]
        dum_crlb = crlb[dum_Idx]
        dum_loss = loss[dum_Idx]
        dum_weit = 1.0 / dum_crlb if w == 'crlb' else dum_xvec[:, ndim]
        dum_weit_sum = np.sum(dum_weit, axis=0) if w == 'crlb' else np.sum(dum_weit)
        if w == 'crlb':
            xvec[dum_Idx[0]] = np.sum(dum_xvec * dum_weit, axis=0) / dum_weit_sum
            crlb[dum_Idx[0]] = ndum / dum_weit_sum
        else:
            xvec[dum_Idx[0]] = np.sum(dum_xvec.T * dum_weit, axis=1) / dum_weit_sum
            crlb[dum_Idx[0]] = np.sum(dum_crlb.T * dum_weit, axis=1) / dum_weit_sum
        indf[dum_Idx[0]] = indf[dum_Idx].min()
        loss[dum_Idx[0]] = np.mean(dum_loss)
        mask[dum_Idx[0]] = True

    return mask