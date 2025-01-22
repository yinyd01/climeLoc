import os
import gc
import json
import pickle
import time
import numpy as np

from IMGProcess.ImageLoader import getROIs, getchipROI, im_loader
from IMGProcess.ImageSegmentation.fim_pfinder_basic import ims_peakfinder_DoK, ims_peakfinder_GPM, margin_refine
from IMGProcess.ImageSegmentation.fim_pseg import im_seg2d, ims_seg2d
from CHNProcess.Polywarp_Utils import coor_polywarpB2A
from PSFProcess.PSFfit_SINPA.PSFfit_SINPA_cspline import sinpa_cspline_1ch, sinpa_cspline_2ch
from PSFProcess.PSFfit_SINPA.PSFfit_SINPA_gauss import sinpa_gauss_1ch, sinpa_gauss_2ch
from PSFProcess.PSFfit_MULPA.PSFfit_MULPA_cspline import mulpa_cspline_1ch, mulpa_cspline_2ch
from PSFProcess.PSFfit_MULPA.PSFfit_MULPA_gauss import mulpa_gauss_1ch, mulpa_gauss_2ch
from PSFProcess.PSFfit_MULPA.mulpa_xvec_unpack import _mulpa_xvec_unpack2row
from LOCProcess.PostProcess.locqc import loc_campx2nm



def duplicate_chk(locs_tar, locs_src, radius):
    """
    Check and mask each of the loc in locs_tar that also appears in locs_src
    INPUTS:
        locs_tar:   (nspots_tar, 3) int ndarray, [[indx, indy, indf],..] each localization to check
        locs_src:   (nspots_src, 3) int ndarray, [[indx, indy, indf],...] source localizations for duplicate check
        radius:     int, radius to determine if duplicates
    RETURN:
        mask:       (nspots_tar) bool ndarray, Ture if the localization is not a duplicate
    """
    mask = np.ones(len(locs_tar), dtype=bool)
    indf_tar = locs_tar[:, 2]
    indf_src = locs_src[:, 2]
    for f in set(indf_src):
        indices_tar, = np.where(indf_tar == f)
        dumlocs_src = locs_src[indf_src == f]
        for idx in indices_tar:
            dltax = np.abs(dumlocs_src[:, 0] - locs_tar[idx, 0])
            dltay = np.abs(dumlocs_src[:, 1] - locs_tar[idx, 1])
            if np.any((dltax <= radius) & (dltay <= radius)):
                mask[idx] = False
    return mask



def parachk(config_loc):
    """
    check and unpack the key parameters
    see Appendix A for parameters of config_loc
    """
    # CAMERA PARAMETERS
    with open(config_loc['Camsetting'], 'rb') as fid:
        Camsetting = pickle.load(fid)
        if config_loc['cam_emgain'] != 'None':
            Camsetting['EMGain'] = float(config_loc['cam_emgain'])
    cam_pxszx = Camsetting['cam_pxszx']
    cam_pxszy = Camsetting['cam_pxszy']

    # PSF PARAMETERS
    with open(config_loc['PSF'], 'rb') as fid:
        PSF = pickle.load(fid)
    boxsz = PSF['boxsz']
    nchannels = PSF['nchannels']
    modality = PSF['modality']
    
    # CHANNEL PARAMETERS
    if config_loc['chnl_warp'] == 'None':
        coeff_R2T = None
        coeff_T2R = None 
    else:
        with open(config_loc['chnl_warp'], 'rb') as fid:
            chnl_warp = pickle.load(fid)
        coeff_R2T = chnl_warp['R2Ts']
        coeff_T2R = chnl_warp['Ts2R']
    
    # CHANNEL CHECK
    assert Camsetting['view_type'] in {'fullview', 'dualview'}, "unsupported view_type: supported are ['fullview', 'dualview'], input is {vt:s}".format(vt=Camsetting['view_type'])
    if Camsetting['view_type'] == 'fullview':
        coeff_R2T = None
        coeff_T2R = None
        assert nchannels == 1, "nchannels mismatch: Camsetting['view_type']={vt:s}, PSF['nchannels']={nch:d}".format(vt=Camsetting['view_type'], nch=nchannels)
    elif Camsetting['view_type'] == 'dualview':
        assert (not coeff_R2T is None) and (not coeff_T2R is None), "nchannels mismatch: Camsetting['view_type']={vt:s}, chnl_warp is None".format(vt=Camsetting['view_type'])
        assert nchannels == 2, "nchannels mismatch: Camsetting['view_type']={vt:s}, PSF['nchannels']={nch:d}".format(vt=Camsetting['view_type'], nch=nchannels)
        assert len(coeff_R2T) == nchannels - 1 and len(coeff_T2R) == nchannels - 1, "nchannel mismatch: len(coeff_R2T)={nch0:d}, len(coeff_T2R)={nch1:d}, nchannels={nch2:d}".format(nch0=len(chnl_warp['R2Ts']), nch1=len(chnl_warp['Ts2R']), nch2=nchannels)

    # MODALITY CHECK
    assert config_loc['PSFfit']['modality'] == modality, "modality mismatch: input.modality={}, PSF.modality={}".format(config_loc['PSFfit']['modality'], modality)
    assert modality in {'2Dfrm', '2D', 'AS', 'DH', 'BP'}, "unsupported modality: only '2Dfrm', '2D', 'AS', 'DH', and 'BP' are supported"
    if modality in {'2D', 'DH', 'BP'}:
        config_loc['PSFfit']['fitmethod'] = 'cspline'
    if modality == 'BP':
        assert Camsetting['view_type'] == 'dualview', "modality mismatch: PSF.modality={}, Camera.viewtype={}".format(modality, Camsetting['view_type'])
    if modality == 'DH':
        PSFsigmax = [None] * nchannels
        PSFsigmay = [None] * nchannels
        config_loc['Segmentation']['filtmethod'] = 'MIP'
    elif modality in {'2D', 'AS', 'BP'}:
        PSFsigmax = PSF['astigs'][:, 0]
        PSFsigmay = PSF['astigs'][:, 1]
    else:
        PSFsigmax = PSF['PSFsigmax']
        PSFsigmay = PSF['PSFsigmay']

    # BOXSZ CHECK
    assert boxsz % 2 != 0, "PSF Calibration Error: boxsz should be odd"
    if config_loc['PSFfit']['fitmethod'] == 'cspline':
        assert PSF['splineszx'] % 2 != 0, "PSF Calibration Error: splineszx should be odd"
        assert PSF['splineszx'] == 2 * boxsz + 1, "PSF Calibration Error: the splineszx should equal to 2 * boxsz + 1"
        coeffsz = PSF['splineszx']*PSF['splineszx']*16 if modality == '2Dfrm' else PSF['splineszx']*PSF['splineszx']*PSF['splineszz']*64
        assert len(PSF['coeff'][0]) == coeffsz, "size mismatch: coeffsz={}, splineszx={}, splineszz={}".format(coeffsz, PSF['splineszx'], PSF['splineszz'])
    
    # METHOD CHECK
    assert config_loc['Segmentation']['filtmethod'] in ('Gauss', 'DoG', 'DoA', 'PSF', 'MIP'), "filtmethod not supported: input filtmethod={}, supported='Gauss', 'DoG', 'DoA', 'PSF', 'MIP'".format(config_loc['Segmentation']['filtmethod'])
    assert config_loc['Segmentation']['threshmode'] in ('dynamic', 'pvalue'), "threshmode not supported: input threshmode={}, supported='dynamic', 'pvalue'".format(config_loc['Segmentation']['threshmode'])
    assert config_loc['PSFfit']['fitmethod'] in ('gauss', 'cspline'), "fitmethod not supported: input fitmethod={}, supported=('gauss', 'cspline')".format(config_loc['PSFfit']['fitmethod'])
    

    # KEY PARAMETER ADDED TO CONFIG (not neccessarily for smlm but for accessibility via notepad etc)
    config_loc['cam_pxszx']    = cam_pxszx
    config_loc['cam_pxszy']    = cam_pxszy
    if modality != 'DH':
        config_loc['PSFsigmax_nm'] = (PSFsigmax * cam_pxszx).tolist()
        config_loc['PSFsigmay_nm'] = (PSFsigmay * cam_pxszy).tolist()
    
    return Camsetting, PSF, coeff_R2T, coeff_T2R, PSFsigmax, PSFsigmay





def main_localization(config_loc, Camsetting, PSF, coeff_R2T, coeff_T2R, PSFsigmax, PSFsigmay): 
    """
    Main wrapper function for single molecule localization, sort-by-time, and inspection of the fit
    see parameter_list.md for configuration of parameters   
    img_fpath:  str, the absolute path saving the tiff images     
    """
    img_fpath       = config_loc['img_fpath']
    roi_fname       = config_loc['roi_fname']
    boxsz           = config_loc['Segmentation']['boxsz']
    mulpa           = config_loc['PSFfit']['isdense']
    fitmethod       = config_loc['PSFfit']['fitmethod']
    nmax            = config_loc['PSFfit']['nmax']
    p_val           = config_loc['PSFfit']['pval']
    optmethod       = config_loc['PSFfit']['optmethod']
    iterations      = config_loc['PSFfit']['iterations']
    batsz           = config_loc['PSFfit']['batsz']
    cam_pxszx       = Camsetting['cam_pxszx']
    cam_pxszy       = Camsetting['cam_pxszy']
    modality        = PSF['modality']
    nchannels       = PSF['nchannels']
    kernel_stack    = PSF['kernel']

    ## #################### RESULT PATH #################### ##
    spool_name = os.path.basename(img_fpath)
    result_fpath = os.path.join(img_fpath, 'smlm_result')
    if not os.path.exists(result_fpath):
        os.makedirs(result_fpath)
    jsonObj = json.dumps(config_loc, indent=4)
    with open(os.path.join(result_fpath, 'config_loc.json'), 'w') as jsfid:
        jsfid.write(jsonObj)
    
    ## #################### ROIS AND IMAGES ################### ##
    ## !!!!! NAMING OF THE TIFFS SHOULD BE TIMELY ORDERED !!!!! ##
    img_fname_list = [img_fname for img_fname in os.listdir(img_fpath) if img_fname.endswith('.tif')]
    if len(img_fname_list) == 0:
        raise ValueError("No tiff images detected")
    if roi_fname == 'None':
        rois = getchipROI(Camsetting, os.path.join(img_fpath, img_fname_list[0]))
    else:
        rois = getROIs(roi_fname, mode='rect')
    
    ## #################### INITIALIZE #################### ##
    ndim = 2 if modality == '2Dfrm' else 3
    VNUM = ndim + 1 + 1 if modality == 'BP' else ndim + nchannels + nchannels
    nroi = 1 if rois is None else len(rois)
    
    xvecs   = [np.zeros((0, VNUM), dtype=np.float32) for _ in range(nroi)]
    crlbs   = [np.zeros((0, VNUM), dtype=np.float32) for _ in range(nroi)]
    losss   = [np.array([], dtype=np.float32) for _ in range(nroi)]
    nnums   = [np.array([], dtype=np.int32) for _ in range(nroi)]
    indfs   = [np.array([], dtype=np.int32) for _ in range(nroi)]
    indls   = [np.zeros((nchannels, 0), dtype=np.int32) for _ in range(nroi)]
    indus   = [np.zeros((nchannels, 0), dtype=np.int32) for _ in range(nroi)]
    
    ## #################### SMLM TIFF BY TIFF #################### ##
    frm_offset = 0
    for img_fname in img_fname_list:
        
        # load a tif image
        img_fullfname = os.path.join(img_fpath, img_fname)
        print("loading {}...".format(img_fname), end='\r')
        tas = time.time()
        ims = im_loader(Camsetting, img_fullfname, zPos=None, rois=rois)
        nfrm = ims[0]['nfrm']
        print("{t:0.4f} secs elapsed for loading {fname:s} ({n:d} frames)".format(t=time.time()-tas, fname=img_fname, n=nfrm))
        
        # process roi by roi
        for (r, im) in enumerate(ims):
            
            print("Processing roi_{}...".format(r))
            roi = rois[r]
            im_stack = im['imstack']
            cam_offset = im['cam_offset']
            cam_a2d = im['cam_a2d']
            cam_var = im['cam_var']
            cam_gain = im['cam_gain']
            EMexcess = 2.0 if cam_gain > 1.0 else 1.0
            
            # SEGAMENTATION
            print("\tSegmenting PSF squares...", end='\r')
            tas = time.time()
            if not cam_var is None:
                dum_var = cam_var[0]
                weights = np.zeros_like(dum_var)
                weights[dum_var > 0] = 1.0 / dum_var[dum_var > 0]
            else:
                weights = [None] * nchannels
            
            ims_peakfinder = ims_peakfinder_GPM if config_loc['Segmentation']['filtmethod'] in {'Gauss', 'PSF', 'MIP'} else ims_peakfinder_DoK
            tmp_indxr, tmp_indyr, tmp_indf = ims_peakfinder(im_stack[0], weights[0], cam_offset[0], cam_a2d[0], cam_gain, boxsz, PSFsigmax[0], PSFsigmay[0], kernel_stack[0], config_loc['Segmentation'])
            for j in range(1, nchannels):
                dum_indxt, dum_indyt, dum_indft = ims_peakfinder(im_stack[j], weights[j], cam_offset[j], cam_a2d[j], cam_gain, boxsz, PSFsigmax[j], PSFsigmay[j], kernel_stack[j], config_loc['Segmentation'])
                dum_coor_cht = np.vstack((roi[1] + dum_indxt + 0.5, roi[0] + dum_indyt + 0.5))
                dum_coordx, dum_coordy = coor_polywarpB2A(2, dum_coor_cht, coeff_T2R[j-1])
                dum_indx_tinr = np.int32(dum_coordx) - roi[1]
                dum_indy_tinr = np.int32(dum_coordy) - roi[0]

                mask = duplicate_chk(np.vstack((dum_indx_tinr, dum_indy_tinr, dum_indft)).T, np.vstack((tmp_indxr, tmp_indyr, tmp_indf)).T, radius=2)
                tmp_indxr = np.hstack((tmp_indxr, dum_indx_tinr[mask]))
                tmp_indyr = np.hstack((tmp_indyr, dum_indy_tinr[mask]))
                tmp_indf = np.hstack((tmp_indf, dum_indft[mask]))
            margin_refine((roi[2]-roi[0], roi[3]-roi[1]), boxsz, tmp_indxr, tmp_indyr)
            NFits = len(tmp_indxr)
            if NFits == 0:
                continue

            ind_sort = np.argsort(tmp_indf)
            tmp_indxr[:] = tmp_indxr[ind_sort]
            tmp_indyr[:] = tmp_indyr[ind_sort]
            tmp_indf[:] = tmp_indf[ind_sort]
            
            lc = np.zeros((nchannels, NFits), dtype=np.int32)
            uc = np.zeros((nchannels, NFits), dtype=np.int32)
            h_data = np.zeros((nchannels, NFits, boxsz, boxsz), dtype=np.float32)
            h_varim = np.zeros((nchannels, NFits, boxsz, boxsz), dtype=np.float32) if not cam_var is None else None

            lc[0] = roi[1] + tmp_indxr - boxsz//2
            uc[0] = roi[0] + tmp_indyr - boxsz//2
            h_data[0] = ims_seg2d(im_stack[0], boxsz, tmp_indxr, tmp_indyr, tmp_indf, offset=cam_offset[0], a2d=cam_a2d[0]*cam_gain)
            if not cam_var is None:
                h_varim[0] = im_seg2d(cam_var[0], boxsz, tmp_indxr, tmp_indyr)

            tmp_coor_chr = np.vstack((roi[1] + tmp_indxr + 0.5, roi[0] + tmp_indyr + 0.5))
            for j in range(1, nchannels):
                tmp_coordx, tmp_coordy = coor_polywarpB2A(2, tmp_coor_chr, coeff_R2T[j-1])
                tmp_indxt = np.int32(tmp_coordx) - roi[1]
                tmp_indyt = np.int32(tmp_coordy) - roi[0]
                margin_refine((roi[2]-roi[0], roi[3]-roi[1]), boxsz, tmp_indxt, tmp_indyt)

                lc[j] = roi[1] + tmp_indxt - boxsz//2
                uc[j] = roi[0] + tmp_indyt - boxsz//2
                h_data[j] = ims_seg2d(im_stack[j], boxsz, tmp_indxt, tmp_indyt, tmp_indf, offset=cam_offset[j], a2d=cam_a2d[j]*cam_gain)
                if not cam_var is None:
                    h_varim[j] = im_seg2d(cam_var[j], boxsz, tmp_indxt, tmp_indyt)

            del im_stack
            gc.collect()
            print("\t{t:0.4f} secs elapsed for Segmentation (n={n:d})".format(t=time.time()-tas, n=NFits))
            
            # PSF FIT
            tas = time.time()
            if nchannels == 1:
                dum_varim = h_varim[0] if not cam_var is None else None
                if mulpa:
                    fitter = mulpa_cspline_1ch if fitmethod == 'cspline' else mulpa_gauss_1ch
                    dum_xvec, dum_crlb, tmp_loss, tmp_nnum = fitter(h_data[0]/np.sqrt(EMexcess), dum_varim, PSF, nmax, p_val, optmethod, iterations, batsz)
                else:
                    fitter = sinpa_cspline_1ch if fitmethod == 'cspline' else sinpa_gauss_1ch
                    dum_xvec, dum_crlb, tmp_loss = fitter(h_data[0]/np.sqrt(EMexcess), dum_varim, PSF, optmethod, iterations, batsz)
                    tmp_nnum = np.ones(NFits, dtype=np.int32)
            else:
                if mulpa:
                    fitter = mulpa_cspline_2ch if fitmethod == 'cspline' else mulpa_gauss_2ch
                    dum_xvec, dum_crlb, tmp_loss, tmp_nnum = fitter(h_data/np.sqrt(EMexcess), h_varim, PSF, lc, uc, coeff_R2T, nmax, p_val, optmethod, iterations, batsz)
                else:
                    fitter = sinpa_cspline_2ch if fitmethod == 'cspline' else sinpa_gauss_2ch
                    dum_xvec, dum_crlb, tmp_loss = fitter(h_data/np.sqrt(EMexcess), h_varim, PSF, lc, uc, coeff_R2T, optmethod, iterations, batsz)
                    tmp_nnum = np.ones(NFits, dtype=np.int32)

            # UNPACK   
            if mulpa:
                if modality == 'BP':
                    tmp_xvec, tmp_crlb = _mulpa_xvec_unpack2row(ndim, 1, tmp_nnum, dum_xvec, dum_crlb)
                else:
                    tmp_xvec, tmp_crlb = _mulpa_xvec_unpack2row(ndim, nchannels, tmp_nnum, dum_xvec, dum_crlb)
                tmp_xvec[:, 0] += np.float32(np.repeat(lc[0], tmp_nnum) - roi[1])
                tmp_xvec[:, 1] += np.float32(np.repeat(uc[0], tmp_nnum) - roi[0])
            else:
                tmp_xvec, tmp_crlb = dum_xvec, dum_crlb
                tmp_xvec[:, 0] += np.float32(lc[0] - roi[1])
                tmp_xvec[:, 1] += np.float32(uc[0] - roi[0])
            print("\t{t:0.4f} secs elapsed for {dense_mod:s} fit (n={n:d})".format(t=time.time()-tas, dense_mod='multi-emitter' if mulpa else 'single-emitter', n=NFits))

            # COLLECT
            xvecs[r]    = np.vstack((xvecs[r], tmp_xvec))
            crlbs[r]    = np.vstack((crlbs[r], tmp_crlb))
            losss[r]    = np.hstack((losss[r], tmp_loss))
            nnums[r]    = np.hstack((nnums[r], tmp_nnum))
            indfs[r]    = np.hstack((indfs[r], tmp_indf + frm_offset))
            indls[r]    = np.hstack((indls[r], lc - roi[1]))
            indus[r]    = np.hstack((indus[r], uc - roi[0]))
            
        frm_offset += nfrm
        del ims
        gc.collect()

    ## #################### CONVENTION #################### ##
    cam_pxszs = [cam_pxszx, cam_pxszy, PSF['zstepsz_nm']] if ndim == 3 else [cam_pxszx, cam_pxszy]
    for r, (roi, xvec, crlb, indf, loss, nnum, indl, indu) in enumerate(zip(rois, xvecs, crlbs, indfs, losss, nnums, indls, indus)):
        
        if np.any(np.diff(indf) < 0):
            print("!! WARNING: NOT SORTED BY 'indf' after smlmloc !!")

        # in-place convert from cam_pxsz to nanometers
        roi_nm = [roi[0]*cam_pxszy, roi[1]*cam_pxszx, roi[2]*cam_pxszy, roi[3]*cam_pxszx]
        loc_campx2nm(ndim, xvec, crlb, cam_pxszs) 
        
        # collect and save
        # save ndim, nchannels, and linkN for indexing the xvec and crlb
        # save boxsz for calculation of the degree of freedom
        smlm_result = {'ndim':ndim, 'nchannels':nchannels, 'boxsz':boxsz, 'roi':roi, 'indl':indl, 'indu':indu, 
                       'roi_nm':roi_nm, 'xvec':xvec, 'crlb':crlb, 'indf':indf, 'loss':loss, 'nnum':nnum}

        fname = '{spname:s}_roi_{roi:d}_rawlocsnm.pkl'.format(spname=spool_name, roi=r)
        with open(os.path.join(result_fpath, fname), 'wb') as fid:
            pickle.dump(smlm_result, fid, pickle.HIGHEST_PROTOCOL)

    return





if __name__ == '__main__':
    pass