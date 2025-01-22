from win32api import GetSystemMetrics
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from IMGProcess.ImageLoader import im_loader
from IMGProcess.ImageSegmentation.fim_pfinder_basic import im_peakfinder
from IMGProcess.ImageSegmentation.fim_ppicker import im_peakpicker2d
from IMGProcess.ImageSegmentation.fim_pseg import im_seg2d
from PSFProcess.PSFfit_basic.PSFfit_basic import cspline3d
from CHNProcess.Polywarp_Utils import get_coeff_B2A, coor_polywarpB2A, nnd_match

scrWidth, scrHeight = GetSystemMetrics(0), GetSystemMetrics(1)


def _textPos(ax, relativePos):
    """
    Return the absolute position of putting the text into a matplotlib figure
    INPUT:
        ax:                 the aixs handle
        relativePos:        (float, float) the relative position where to put the text for x- and y- axis
                            (i.e. 0.0 on the left(bottom), 0.5 in the middle, 1.0 on the right(top))
    RETURN:
        posx, posy:         the absolut coordinate in the axis
    """
    posx = relativePos[0]*(ax.get_xlim()[1]-ax.get_xlim()[0]) + ax.get_xlim()[0]
    posy = relativePos[1]*(ax.get_ylim()[1]-ax.get_ylim()[0]) + ax.get_ylim()[0]
    return posx, posy



def _validate(boxsz, coords_ref, coords_tar, matched_ind_ref, matched_ind_tar, coeff_R2T, coeff_T2R, scalex, scaley, tolr_nm):
    """
    validate the warpping coefficients
    INPUT:
        coords_ref:         (2, nref) float ndarray, [[locx_ref], [locy_ref]] the coordinates in the reference channel
        coords_tar:         (2, ntar) float ndarray, [[locx_tar], [locy_tar]] the coordinates in the target channel
        matched_ind_ref:    (nmatch,) int ndarray, the index matching the coords_ref with corresponding coords_tar for generating the coeff_R2T/T2R 
        matched_ind_tar:    (nmatch,) int ndarray, the index matching the coords_tar with corresponding coords_ref for generating the coeff_R2T/T2R
        coeff_R2T:          (2, warpdeg*warpdeg) float ndarray, [[R2T_x], [R2T_y]] the polynomial coefficients warping the coords_ref to the target channel
        coeff_T2R:          (2, warpdeg*warpdeg) float ndarray, [[T2R_x], [T2R_y]] the polynomial coefficients warping the coords_tar to the reference channel
        scalex:             float, scaler for coordx translated to real-world dimention (e.g. cam_pxszx)
        scaley:             float, scaler for coordy translated to real-world dimention (e.g. cam_pxszy)
        tolr_nm:            float, the tolerance distance (real-world dimention) for nnd analysis
    RETURN:
        fig:                matplotlib figure handler displaying the validation results
    """
    
    coordx_ref, coordy_ref = coords_ref
    matched_coordx_ref = coordx_ref[matched_ind_ref]
    matched_coordy_ref = coordy_ref[matched_ind_ref]

    coordx_tar, coordy_tar = coords_tar
    matched_coordx_tar = coordx_tar[matched_ind_tar]
    matched_coordy_tar = coordy_tar[matched_ind_tar]

    fig = plt.figure(figsize=(scrWidth/72, scrHeight/72))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])
    ax2 = fig.add_subplot(gs[:, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, 3])

    # validate the initial matching that generates the coeff_R2T and coeff_T2R 
    for i in range(len(coordx_ref)):
        dumxbnd = np.asarray([coordx_ref[i]-0.5*boxsz, coordx_ref[i]+0.5*boxsz, coordx_ref[i]+0.5*boxsz, coordx_ref[i]-0.5*boxsz, coordx_ref[i]-0.5*boxsz])
        dumybnd = np.asarray([coordy_ref[i]-0.5*boxsz, coordy_ref[i]-0.5*boxsz, coordy_ref[i]+0.5*boxsz, coordy_ref[i]+0.5*boxsz, coordy_ref[i]-0.5*boxsz])
        ax0.plot(dumxbnd-0.5, dumybnd-0.5, color='tab:blue', lw=0.5)
    for i in range(len(coordx_tar)):
        dumxbnd = np.asarray([coordx_tar[i]-0.5*boxsz, coordx_tar[i]+0.5*boxsz, coordx_tar[i]+0.5*boxsz, coordx_tar[i]-0.5*boxsz, coordx_tar[i]-0.5*boxsz])
        dumybnd = np.asarray([coordy_tar[i]-0.5*boxsz, coordy_tar[i]-0.5*boxsz, coordy_tar[i]+0.5*boxsz, coordy_tar[i]+0.5*boxsz, coordy_tar[i]-0.5*boxsz])
        ax0.plot(dumxbnd-0.5, dumybnd-0.5, color='tab:orange', lw=0.5)
    for i in range(len(matched_ind_ref)):
        dumx = np.array([matched_coordx_ref[i]-0.5, matched_coordx_tar[i]-0.5])
        dumy = np.array([matched_coordy_ref[i]-0.5, matched_coordy_tar[i]-0.5])
        ax0.plot(dumx, dumy, color='tab:red', lw=0.5)
    ax0.invert_yaxis()
    ax0.set_aspect('equal', 'box')
    ax0.grid(axis='both')
    ax0.set_title('nnd matches channels') 

    # visualize the matching with the warping coefficients
    coordx_rint, coordy_rint = coor_polywarpB2A(2, coords_ref, coeff_R2T)
    ax1.plot(coordx_rint, coordy_rint, 'x', color='tab:blue', label='ref-in-tar')
    ax1.plot(coordx_tar, coordy_tar, 'o', color='tab:orange', mfc='none', label='tar')
    ax1.invert_yaxis()
    ax1.set_aspect('equal', 'box')
    ax1.grid(axis='both')
    ax1.set_title('pairing in the target channel')
    ax1.legend()
    
    coordx_tinr, coordy_tinr = coor_polywarpB2A(2, coords_tar, coeff_T2R)
    ax2.plot(coordx_tinr, coordy_tinr, 'x', color='tab:orange', label='tar-in-ref')
    ax2.plot(coordx_ref, coordy_ref, 'o', color='tab:blue', mfc='none', label='ref')
    ax2.invert_yaxis()
    ax2.set_aspect('equal', 'box')
    ax2.grid(axis='both')
    ax2.set_title('pairing in the reference channel')
    ax2.legend()
     
    # evaluate the matching distances
    dum_ind_tar, dum_ind_ref = nnd_match(np.vstack((coordx_tar*scalex, coordy_tar*scaley)), np.vstack((coordx_rint*scalex, coordy_rint*scaley)), tolr_nm)[:-1]
    deltax = (coordx_tar[dum_ind_tar]-coordx_rint[dum_ind_ref]) * scalex
    deltay = (coordy_tar[dum_ind_tar]-coordy_rint[dum_ind_ref]) * scaley
    ax3.plot(deltax, deltay, 'x')
    ax3.set_aspect('equal', 'box')
    ax3.grid(axis='both')
    ax3.set_title('ref-2-tar pairing errors')
    RMSEx = np.sqrt(np.mean(deltax*deltax))
    RMSEy = np.sqrt(np.mean(deltay*deltay))
    txtPosx, txtPosy = _textPos(ax3, (0.75, 0.9))
    ax3.text(txtPosx, txtPosy, 'RMSEx={accx:.4f} nm\nRMSEy={accy:.4f} nm'.format(accx=RMSEx, accy=RMSEy))

    dum_ind_ref, dum_ind_tar = nnd_match(np.vstack((coordx_ref*scalex, coordy_ref*scaley)), np.vstack((coordx_tinr*scalex, coordy_tinr*scaley)), tolr_nm)[:-1]
    deltax = (coordx_ref[dum_ind_ref]-coordx_tinr[dum_ind_tar]) * scalex
    deltay = (coordy_ref[dum_ind_ref]-coordy_tinr[dum_ind_tar]) * scaley
    ax4.plot(deltax, deltay, 'x')
    ax4.set_aspect('equal', 'box')
    ax4.grid(axis='both')
    ax4.set_title('tar-2-ref pairing errors')
    RMSEx = np.sqrt(np.mean(deltax*deltax))
    RMSEy = np.sqrt(np.mean(deltay*deltay))
    txtPosx, txtPosy = _textPos(ax4, (0.75, 0.9))
    ax4.text(txtPosx, txtPosy, 'RMSEx={accx:.4f} nm\nRMSEy={accy:.4f} nm'.format(accx=RMSEx, accy=RMSEy))

    return fig

    

def chnlwarp(bead_fpath, Camsetting, ims, nbeads, PSF, tolr_nm, warpdeg, Segmentation, iterations=100, batsz=65535):
    """
    main function to warp the left-and-right channels in a single-frame bead image
    INPUT:
        bead_fpath:         str, the absolute path to save the calibration results
        Camsetting:         python dictionary containing camera informations, see Appendix B in parameter_list.md
        ims:                python dictionary generated by IMGProcess.ImageLoader.im_loader:
                            'roi':      [top, left, bottom, right] bounds for each roi (should be the full canvas)
                            'cam_offset':   (nchannels, roiszy, roiszx) or (nchannels,) float ndarray, cam_offset from each channel within the roi
                            'cam_var':      (nchannels, roiszy, roiszx) or (nchannels,) float ndarray, cam_var from each channel within the roi, None if Camsetting['scmos_var'] == False
                            'cam_a2d':      (nchannels, roiszy, roiszx) or (nchannels,) float ndarray, cam_a2d from each channel within the roi
                            'cam_gain':     float, EMgain of the camera if avaiable
                            'imstack':      (nchannels, roiszy, roiszx) uint16 ndarray, image from each channel within the roi
        nbeads:             int, user-defined number of beads in each channel for calibration
        PSF:                python dictionary, see PSFCalibration_focus_unlinked in Appendix B in parameter_list.md
        tolr_nm:            float, tolerance in nm for bead pairing between two channels
        warpdeg:            int, the degree for polynomial warping
        Segmentation:       dict, segmentation parameters
            boxsz:          int, size of the box for segmentation
            filtmethod:     
                Gauss:      convolve the raw image with a Gaussian kernel
                DoG:        convolve the raw image with a Gaussian kernel (sig=PSFsigma) minus a larger Gaussian kernel (sig=2.5*PSFsigma), respectively.
                DoA:        convolve the raw image with a 1-st wavelet kernel minus a wavelet kernel with 0 insertions (a-trous), respectively.
                PSF:        convolve the raw image with the (moddle plane of the) PSF kernel
                MIP:        convolve the raw image with each kernel of a kernel stack and projects the maximum intensity (weights are ignored)        
            segmethod:     
                nms:        the local maxima identified via connectivity are further filtered if they fall into the same box
                grid:       the local maxima identified via connectivity are gathered into box-grids from the image
                None:       the local maxima are identified via connectivity without further filtering            
            threshmode and threshval:     
                'dynamic':  thresh = median + (quantile_0.8 - quantile_0.2)/(0.8-0.2) * 0.5 * 2.0 * threshval (threshval = 1.7 in SMAP) 
                            (quantile is that of the local maxima identified by connection prior to the thresholding, with or without maximum suppression)
                'pvalue':   thresh = threshval if filtmethod in {'Gauss', 'DoG', 'DoA'} on probability map
                            thresh = sqrt(2) * erfinv(1 - 2 * pval) * sqrt(0.5 * excess) (excess = 2 if EMGain > 1) on standarized image with background subtracted
        """
    
    ##### PARSE INPUTS #####
    chipszy         = Camsetting['chipszh']
    chipszx         = Camsetting['chipszw']
    cam_pxszx       = Camsetting['cam_pxszx']
    cam_pxszy       = Camsetting['cam_pxszy']

    roi             = ims['roi']
    cam_var         = ims['cam_var']
    cam_offset      = ims['cam_offset']
    cam_a2d         = ims['cam_a2d']
    cam_gain        = ims['cam_gain']
    imbead_focus    = ims['imstack'].copy()

    modality        = PSF['modality']
    nchannels       = PSF['nchannels']
    splineszx       = PSF['splineszx']
    splineszz       = PSF['splineszz']
    coeff_PSF       = PSF['coeff']
    kernel          = PSF['kernel']
    
    boxsz           = Segmentation['boxsz']
    margin          = (boxsz-1) // 2 + 1 
    if modality == '2Dfrm':
        PSFsigmax = PSF['PSFsigmax']
        PSFsigmay = PSF['PSfsigmay']
    elif modality == 'DH':
        PSFsigmax = [None] * nchannels
        PSFsigmay = [None] * nchannels
        Segmentation['filtmethod'] = 'MIP'
    else:
        PSFsigmax = PSF['astigs'][:, 0]
        PSFsigmay = PSF['astigs'][:, 1]
    
    assert np.all(roi[:2] == 0), "roi mismatch: (top,left)={}, should be 0s".format(roi[:2])
    if nchannels == 2:
        assert roi[2] == chipszy and roi[3] == chipszx//2, "roi mismatch (dualview): (roiszy, roiszx)={}, should be (chipszy, chipszx//2)={}".format((roi[2], roi[3]), (chipszy, chipszx//2))
    elif nchannels == 4:
        assert roi[2] == chipszy//2 and roi[3] == chipszx//2, "roi mismatch (quadralview): (roiszy, roiszx)={}, should be (chipszy//2, chipszx//2)={}".format((roi[2], roi[3]), (chipszy//2, chipszx//2))
    else:
        raise ValueError("unsupported number of channels")
    

    ### Manually Choose the bead pairs and generate a set of preliminary warping coefficients 
    print("Pick the beads bead#0: chnl0, chnl1, chnl..., bead#1: chnl0, chnl1, chnl...")
    dum_indx, dum_indy = im_peakpicker2d(imbead_focus, boxsz, nbeads)
    tmp_indx = np.array([dum_indx[j::nchannels] for j in range(nchannels)])
    tmp_indy = np.array([dum_indy[j::nchannels] for j in range(nchannels)])
    
    tmp_raw_stack = np.array([im_seg2d(imbead_focus[j], boxsz, tmp_indx[j], tmp_indy[j], offset=cam_offset[j], a2d=cam_a2d[j]*cam_gain) for j in range(nchannels)])
    tmp_var_stack = np.array([im_seg2d(cam_var[j], boxsz, tmp_indx[j], tmp_indy[j]) for j in range(nchannels)]) if not cam_var is None else [None] * nchannels
    tmp_coordx = np.zeros((nchannels, nbeads))
    tmp_coordy = np.zeros((nchannels, nbeads))
    for j in range(nchannels):
        dum_xvec = cspline3d(tmp_raw_stack[j], tmp_var_stack[j], splineszx, splineszz, coeff_PSF[j], 'MLE', iterations, batsz)[0]
        tmp_coordx[j] = dum_xvec[:, 0] + tmp_indx[j] - boxsz//2
        tmp_coordy[j] = dum_xvec[:, 1] + tmp_indy[j] - boxsz//2
        
    tmp_coeff_R2Ts = np.zeros((nchannels-1, 2, warpdeg*warpdeg))
    tmp_coord_ref = np.vstack([tmp_coordx[0], tmp_coordy[0]])
    for j in range(1, nchannels):
        tmp_coord_tar = np.vstack([tmp_coordx[j], tmp_coordy[j]])
        tmp_coeff_R2Ts[j-1] = get_coeff_B2A(2, warpdeg, tmp_coord_ref, tmp_coord_tar)
    
    
    ### NND match the coordinates in the reference channel and each of the target channel
    weights = 1.0 / cam_var if not cam_var is None else [None] * nchannels
    indx_ref, indy_ref = im_peakfinder(imbead_focus[0], weights[0], cam_offset[0], cam_a2d[0], cam_gain, boxsz, PSFsigmax[0], PSFsigmay[0], kernel[0], Segmentation)
    isinside = (indx_ref > margin) & (indx_ref < chipszx//2-margin) & (indy_ref > margin) & (indy_ref < chipszy-margin)
    indx_ref = indx_ref[isinside]
    indy_ref = indy_ref[isinside]
    raw_stack_ref = im_seg2d(imbead_focus[0], boxsz, indx_ref, indy_ref, offset=cam_offset[0], a2d=cam_a2d[0]*cam_gain)
    var_stack_ref = im_seg2d(cam_var[0], boxsz, indx_ref, indy_ref) if not cam_var is None else None
    xvec_ref = cspline3d(raw_stack_ref, var_stack_ref, splineszx, splineszz, coeff_PSF[0], 'MLE', iterations, batsz)[0]
    coordx_ref = xvec_ref[:, 0] + np.float32(indx_ref - boxsz//2)
    coordy_ref = xvec_ref[:, 1] + np.float32(indy_ref - boxsz//2)
    coords_ref = np.vstack([coordx_ref, coordy_ref])

    coeff_R2Ts = np.zeros((nchannels-1, 2, warpdeg*warpdeg))
    coeff_Ts2R = np.zeros((nchannels-1, 2, warpdeg*warpdeg))
    for j in range(1, nchannels):
        indx_tar, indy_tar = im_peakfinder(imbead_focus[j], weights[j], cam_offset[j], cam_a2d[j], cam_gain, boxsz, PSFsigmax[j], PSFsigmay[j], kernel[j], Segmentation)
        isinside = (indx_tar > margin) & (indx_tar < chipszx//2-margin) & (indy_tar > margin) & (indy_tar < chipszy-margin)
        indx_tar = indx_tar[isinside]
        indy_tar = indy_tar[isinside]
        raw_stack_tar = im_seg2d(imbead_focus[j], boxsz, indx_tar, indy_tar, offset=cam_offset[j], a2d=cam_a2d[j]*cam_gain)
        var_stack_tar = im_seg2d(cam_var[j], boxsz, indx_tar, indy_tar) if not cam_var is None else None
        xvec_tar = cspline3d(raw_stack_tar, var_stack_tar, splineszx, splineszz, coeff_PSF[j], 'MLE', iterations, batsz)[0]
        coordx_tar = xvec_tar[:, 0] + np.float32(indx_tar - boxsz//2)
        coordy_tar = xvec_tar[:, 1] + np.float32(indy_tar - boxsz//2)
        coords_tar = np.vstack([coordx_tar, coordy_tar])
        coords_tar_nm = np.vstack([coordx_tar*cam_pxszx, coordy_tar*cam_pxszy])

        for _ in range(5):
            coordx_rint, coordy_rint = coor_polywarpB2A(2, coords_ref, tmp_coeff_R2Ts[j-1])
            coords_rint_nm = np.vstack([coordx_rint*cam_pxszx, coordy_rint*cam_pxszy])
            
            matched_ind_ref, matched_ind_tar, matched_nnd = nnd_match(coords_rint_nm, coords_tar_nm, tolr_nm)

            matched_coords_ref = np.vstack([coordx_ref[matched_ind_ref], coordy_ref[matched_ind_ref]])
            matched_coords_tar = np.vstack([coordx_tar[matched_ind_tar], coordy_tar[matched_ind_tar]])
            tmp_coeff_R2Ts[j-1] = get_coeff_B2A(2, warpdeg, matched_coords_ref, matched_coords_tar)
        
        coeff_R2Ts[j-1] = get_coeff_B2A(2, warpdeg, matched_coords_ref, matched_coords_tar)
        coeff_Ts2R[j-1] = get_coeff_B2A(2, warpdeg, matched_coords_tar, matched_coords_ref)
        
        # Validation
        fig = _validate(boxsz, coords_ref, coords_tar, matched_ind_ref, matched_ind_tar, coeff_R2Ts[j-1], coeff_Ts2R[j-1], cam_pxszx, cam_pxszy, tolr_nm)
        fig.savefig(os.path.join(bead_fpath, 'chnlwarp_errors_chnl{}.png'.format(j)), dpi=300)
    
    chnlwarp_coeff = {'warpdeg':warpdeg, 'R2Ts':coeff_R2Ts, 'Ts2R':coeff_Ts2R}
    with open(os.path.join(bead_fpath, 'chnlwarp_coeff.pkl'), 'wb') as fid:
        pickle.dump(chnlwarp_coeff, fid, pickle.HIGHEST_PROTOCOL)






if __name__ == '__main__':
    
    view_type = 'dualview'
    nbeads = 9
    margin = 5
    tolr_nm = 35.0
    warpdeg = 3

    imbead_fpath = 'C:/Users/Yando/Projects/smlm_analysis/beads2D_stck_EMCCD'
    imfname = 'spool_2_jmzeng_20230710_beads_gain4.tif'
    camfname = 'Camera_settings.pkl'
    PSFfname = 'PSFCalibration_focus_unlinked.pkl'

    imbead_fname = os.path.join(imbead_fpath, imfname)
    with open(os.path.join(imbead_fpath, camfname), 'rb') as fid:
        Camsetting = pickle.load(fid)
    with open(os.path.join(imbead_fpath, PSFfname), 'rb') as fid:
        PSF = pickle.load(fid)

    ims, = im_loader(Camsetting, imbead_fname, zPos=PSF['zFocus'])    
    Segmentation = {'boxsz':11, 'filtmethod':'DoG', 'segmethod':'nms', 'threshmod':'dynamic', 'threshval':1.7}
    chnlwarp(imbead_fpath, Camsetting, ims, nbeads, PSF, margin, tolr_nm, warpdeg, Segmentation, iterations=100, batsz=65535)