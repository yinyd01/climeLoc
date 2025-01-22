import os
import pickle
import json
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.distributions import chi2

from IMGProcess.ImageLoader import _readROI, _readroizip
from LOCProcess.PostProcess.locqc import loc_filter
from LOCProcess.PostProcess.rcc import rcc2d, rcc3d
from LOCProcess.PostProcess.frc import frc_fft
from LOCProcess.PostProcess.locgroup import inframe_merge_dbscan, consecutive_merge_dbscan
from CHNProcess.Ratiometric_Utils import _ratio_assign, ratio_register, ratio_plot
from LOCProcess.Visualization.locrender import loc_render_mulfl, loc_render_sinfl
from LOCProcess.Visualization.locrender2 import finer_render_mulfl



def _textPos(ax, x_frac, y_frac):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_pos = xmin + x_frac * (xmax - xmin)
    y_pos = ymin + y_frac * (ymax - ymin)
    return x_pos, y_pos



def main_pp(config_pp, rawloc_fname):
    """
    Main wrapper function for localization drift correction (via rcc), consecutive merge, qc filtering, and channel registration       
    """
    ismerge             = config_pp['infrm_merge']['ismerge']
    maxRm_nm            = config_pp['infrm_merge']['maxR_nm']
    maxZm_nm            = config_pp['infrm_merge']['maxZ_nm']
    pval                = config_pp['qcfilter']['pval']
    frm_rng             = config_pp['qcfilter']['frm_rng'].copy()
    filt_zrange         = config_pp['qcfilter']['filt_zrange'].copy()
    filt_photon         = config_pp['qcfilter']['filt_photon']
    filt_precision_nm   = config_pp['qcfilter']['filt_precision_nm'].copy()
    isrcc               = config_pp['rcc']['isrcc']
    tar_pxsz            = config_pp['rcc']['tar_pxsz']
    frmbinsz            = config_pp['rcc']['frmbinsz']
    ccorrsz_nm          = config_pp['rcc']['ccorrsz_nm'].copy()
    fitsz_nm            = config_pp['rcc']['fitsz_nm'].copy()
    isgroup             = config_pp['consec_merge']['isgroup']
    maxRg_nm            = config_pp['consec_merge']['maxR_nm']
    maxZg_nm            = config_pp['consec_merge']['maxZ_nm']
    gap_tol             = config_pp['consec_merge']['gap_tol']

    result_fpath = os.path.dirname(rawloc_fname)
         
    ## #################### DATA PREPARATION #################### ##
    print("Processing {}".format(os.path.basename(rawloc_fname)[:-14]))
    with open(rawloc_fname, 'rb') as fid:
        rawloc_result = pickle.load(fid)
    ndim        = rawloc_result['ndim']
    nchannels   = rawloc_result['nchannels']
    boxsz       = rawloc_result['boxsz']
    roi_nm      = rawloc_result['roi_nm']
    xvec        = rawloc_result['xvec'].copy()
    crlb        = rawloc_result['crlb'].copy()
    indf        = rawloc_result['indf'].copy()
    loss        = rawloc_result['loss'].copy()
    
    nnum = rawloc_result['nnum']
    indf = np.repeat(indf, nnum)
    loss = np.repeat(loss, nnum)
    df = nchannels * boxsz * boxsz - nnum * (ndim + nchannels) - nchannels
    filt_loss = np.repeat(chi2.ppf(1-pval, df), nnum)

    frm_seq_ind = np.lexsort((xvec[:, 0], indf))
    indf[:] = indf[frm_seq_ind]
    xvec[:] = xvec[frm_seq_ind]
    crlb[:] = crlb[frm_seq_ind]
    loss[:] = loss[frm_seq_ind]
    filt_loss[:] = filt_loss[frm_seq_ind]

    if frm_rng[1] <= 0:
        frm_rng[1] = indf.max()
    frm_idx = (indf >= frm_rng[0]) & (indf < frm_rng[1])
    indf = indf[frm_idx]
    xvec = xvec[frm_idx]
    crlb = crlb[frm_idx]
    loss = loss[frm_idx]
    filt_loss = filt_loss[frm_idx]

    ## #################### INFRAME MERGE #################### ##
    if ismerge:
        print("\tmerging approximals within the same frame...", end='\r')
        tas = time.time()
        mask = inframe_merge_dbscan(ndim, indf, xvec, crlb, loss, [maxRm_nm]*2+[maxZm_nm], w='crlb')
        print("\t{t:0.4f} secs elapsed for in-frame merge of approximals".format(t=time.time()-tas)) 
        indf = indf[mask]
        xvec = xvec[mask]
        crlb = crlb[mask]
        loss = loss[mask]
        filt_loss = filt_loss[mask]

    ## #################### LABEL GOOD LOCS #################### ## 
    photon = xvec[:, ndim]
    locz = xvec[:, 2] if ndim == 3 else None
    ind_filt = loc_filter(ndim, loss, locz, photon, crlb, filt_loss, filt_zrange, filt_photon, filt_precision_nm)

    ## #################### RCC DRIFT CORRECTION #################### ## 
    if isrcc:
        print("\trcc drift correcting...", end='\r')
        tas = time.time()
        ccorrsz = np.array([np.ceil(sz_nm/tar_pxsz) for sz_nm in ccorrsz_nm], dtype=np.int32)
        fitsz = np.array([np.ceil(sz_nm/tar_pxsz) for sz_nm in fitsz_nm], dtype=np.int32)
        tmp_indf = indf[ind_filt]
        tmp_locs = xvec[ind_filt, :ndim] / tar_pxsz
        tmp_crlb = crlb[ind_filt, :ndim] / tar_pxsz / tar_pxsz
        tmp_photon = photon[ind_filt]
        func_rcc = rcc3d if ndim == 3 else rcc2d
        interp_shift, raw_shift = func_rcc(frmbinsz, ccorrsz[:ndim], tmp_indf, tmp_locs, tmp_crlb, tmp_photon, fitsz[:ndim], frm_rng[0], frm_rng[1], sorted=True, method='cc')
        for j in range(len(indf)):
            if indf[j] < len(interp_shift):
                xvec[j, :ndim] -= interp_shift[indf[j]] * tar_pxsz
        print("\t{t:0.4f} secs elapsed for RCC drift correction".format(t=time.time()-tas))

        # plot and save the rcc drifts
        rcc_fname = rawloc_fname[:-14] + '_rccshift.pkl'
        with open(os.path.join(result_fpath, rcc_fname), 'wb') as fid:
            pickle.dump(interp_shift, fid, pickle.HIGHEST_PROTOCOL)

        rcc_fname = rawloc_fname[:-14] + '_rccshift.png'
        fig_rcc, axs_rcc = plt.subplots(1, ndim, figsize=(19.2, 10.8))
        ax_label = ['x-axis', 'y-axis', 'z-axis']
        for i in range(ndim):
            axs_rcc[i].plot(raw_shift[:, 0], raw_shift[:, i+1]*tar_pxsz)
            axs_rcc[i].plot(np.arange(frm_rng[0], frm_rng[1]), interp_shift[:, i]*tar_pxsz)
            axs_rcc[i].set_ylabel('drift (nm)')
            axs_rcc[i].set_title('rcc drift at ' + ax_label[i])
        plt.savefig(os.path.join(result_fpath, rcc_fname), dpi=300)
        plt.close(fig_rcc)

    ## #################### FILTER BAD LOCS #################### ## 
    indf = indf[ind_filt]
    xvec = xvec[ind_filt]
    crlb = crlb[ind_filt]
    loss = loss[ind_filt]

    ## #################### FRC EVALUATION #################### ##
    print("\tfrc evaluating...", end='\r')
    tas = time.time()
    frc, frc_val = frc_fft(xvec[:, :2])
    print("\t{t:0.4f} secs elapsed for raw frc evaluation".format(t=time.time()-tas))
    
    fig_frc = plt.figure(figsize=(19.2, 10.8))
    ax_frc = fig_frc.add_subplot()
    ax_frc.plot(frc[0], frc[1], color='tab:blue', alpha=0.25)
    ax_frc.plot(frc[0], frc[2], color='tab:blue', label='frc')
    ax_frc.text(*_textPos(ax_frc, 0.1, 0.6), "res={:0.2f} nm".format(frc_val), color='tab:blue')
    ax_frc.set_xlabel("Spatial Frequency (1/nm)")
    ax_frc.set_ylabel("Fourier Ring Correlation")
    freqmin, freqmax = ax_frc.get_xlim()
    ax_frc.hlines(y = 1.0/7.0, xmin=freqmin, xmax=freqmax)
    ax_frc.set_xscale('log')
    ax_frc.legend()
    
    frc = {'frc':frc, 'frcres':frc_val}
    frc_fname = rawloc_fname[:-14] + '_frc.pkl'
    with open(os.path.join(result_fpath, frc_fname), 'wb') as fid:
        pickle.dump(frc, fid, pickle.HIGHEST_PROTOCOL)

    frc_fname = rawloc_fname[:-14] + '_frc.png'
    plt.savefig(os.path.join(result_fpath, frc_fname), dpi=300)
    plt.close(fig_frc)
    
    # ################ GROUP THE APPROXIMALS THROUGH CONSECUTIVE FRAMES #######################
    if isgroup:    
        print("\tconsecutive merging...", end='\r')
        tas = time.time()
        mask = consecutive_merge_dbscan(ndim, indf, xvec, crlb, loss, [maxRg_nm]*2+[maxZg_nm], gap_tol)
        print("\t{t:0.4f} secs elapsed for consecutive grouping".format(t=time.time()-tas))
        indf = indf[mask]
        xvec = xvec[mask]
        crlb = crlb[mask]
        loss = loss[mask]
    
    ## ################################ COLLECT RESULTS ####################################
    smlm_result = {'roi_nm':roi_nm, 'ndim':ndim, 'nchannels':nchannels, 'indf':indf, 'xvec':xvec, 'crlb':crlb, 'loss':loss, 'nnum':nnum}
    if nchannels == 1:
        smlm_result['creg'] = np.zeros(len(smlm_result['indf']), dtype=np.int32)
     
    fname = rawloc_fname[:-14] + '_locsnm.pkl'
    with open(os.path.join(result_fpath, fname), 'wb') as fid:
        pickle.dump(smlm_result, fid, pickle.HIGHEST_PROTOCOL)
    
    fname = rawloc_fname[:-14] + '_xvec.csv'
    if nchannels == 1:
        headlines = 'indf, locx, locy, locz, photon, bkg' if ndim == 3 else 'indf, locx, locy, photon, bkg'
        fmt = '%d, %.4e, %.4e, %.4e, %.4e, %.4e' if ndim == 3 else '%d, %.4e, %.4e, %.4e, %.4e'
    elif nchannels == 2:
        headlines = 'indf, locx, locy, locz, photon, fracN, bkg, bkg' if ndim == 3 else 'indf, locx, locy, photon, fracN, bkg, bkg'
        fmt = '%d, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e' if ndim == 3 else '%d, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e'
    np.savetxt(os.path.join(result_fpath, fname), np.hstack((indf[...,np.newaxis], xvec)), delimiter=',', header=headlines, fmt=fmt)
        
    return





def main_ratio_metric(config_RatioMetric, loc_fname):
    """
    Main wrapper function for ratiometric color assignment         
    """
    nfluorophores   = config_RatioMetric['nfluorophores']
    result_fpath = os.path.dirname(loc_fname)
    
    with open(loc_fname, 'rb') as fid:
        smlm_data = pickle.load(fid)

    if 'crng' in config_RatioMetric.keys():
        smlm_data['crng'] = np.array(config_RatioMetric['crng'])
        _ratio_assign(smlm_data)
    else:
        ratio_register(smlm_data, nfluorophores)
        config_RatioMetric['crng'] = smlm_data['crng'].tolist()
    
    jsonObj = json.dumps(config_RatioMetric, indent=4)
    with open(os.path.join(result_fpath, 'creg_mode.json'), "w") as jsfid:
        jsfid.write(jsonObj)

    with open(loc_fname, 'wb') as fid:
        pickle.dump(smlm_data, fid, pickle.HIGHEST_PROTOCOL)

    fig = ratio_plot(smlm_data)
    figfname = loc_fname[:-11] + '_ratiometric.png'
    plt.savefig(os.path.join(result_fpath, figfname), dpi=300)
    plt.close(fig)

    return





def main_render(config_render, loc_fname):
    """
    Main wrapper function for visualization of localizations        
    """
    
    tar_pxsz        = config_render['tar_pxsz']
    frm_range       = config_render['frm_range']
    nosinglet       = config_render['nosinglet']
    rlim_nm         = config_render['rlim_nm']
    isblur          = config_render['isblur']
    sig             = config_render['sig']
    intensity       = config_render['intensity']
    alpha           = config_render['alpha']
    isRGB           = config_render['isRGB']
    feature         = config_render['feature']
    zmin_nm         = config_render['zmin_nm']
    zmax_nm         = config_render['zmax_nm']
    nfluorophores   = config_render['nfluorophores']
    fluornames      = config_render['fluorname']
    clim            = config_render['clim'].copy()
    colormap        = config_render['colormap']
    clim = np.array(clim, dtype=np.float32)
    
    colorbits = 256
    assert len(fluornames) == nfluorophores, "nfluorophores mismatch: input nfluorophores={}, len(fluornames)={}".format(nfluorophores, len(fluornames))
    assert len(clim) == nfluorophores, "nfluorophores mismatch: input nfluorophores={}, len(clim)={}".format(nfluorophores, len(clim))
    assert len(colormap) == nfluorophores, "nfluorophores mismatch: input nfluorophores={}, len(colormap)={}".format(nfluorophores, len(colormap))
    
    result_fpath = os.path.dirname(loc_fname)
        
    print("Processing {}".format(os.path.basename(loc_fname)[:-11]))
    with open(loc_fname, 'rb') as fid:
        smlm_data = pickle.load(fid)

    # image render
    tas = time.time()
    SRim, Immax = loc_render_mulfl(smlm_data, tar_pxsz, isblur, intensity, clim, colormap, zmin_nm, zmax_nm, nosinglet, rlim_nm, sig, frm_range, feature, alpha, colorbits)
    fname = loc_fname[:-11] + '_SRim_comb.png' 
    ImObj = Image.fromarray(SRim, 'RGB')
    ImObj.save(os.path.join(result_fpath, fname))
    
    single_feature = 'locz' if smlm_data['ndim'] == 3 else 'photon'
    single_clim = np.array([0.01, 0.89]) if smlm_data['ndim'] == 3 else np.array([0.1, 0.9])
    for i in range(nfluorophores):
        SRim, Immax = loc_render_sinfl(smlm_data, i, tar_pxsz, isblur, intensity, True, single_clim, 'turbo', zmin_nm, zmax_nm, nosinglet, rlim_nm, sig, frm_range, single_feature, alpha, colorbits)
        if isRGB:
            fname = loc_fname[:-11] + '_SRim_' + fluornames[i] + '.png' 
            ImObj = Image.fromarray(SRim, 'RGB')
            ImObj.save(os.path.join(result_fpath, fname))
        else:
            fname = loc_fname[:-11] + '_SRim_' + fluornames[i] + '.tif'
            ImObj = Image.fromarray(SRim)
            ImObj.save(os.path.join(result_fpath, fname), format='TIFF', save_all=True)
    print("\t{t:0.4f} secs elapsed for render and saving the image".format(t=time.time()-tas))
    
    return





def main_finer_render(config_render, fpath_result):
    """
    Main wrapper function for visualization of localizations
    see parameter_list.md for configuration of parameters
    img_fpath:  str, the absolute path saving the tiff images
                may not be the same with config_pp['img_fpath'], depending on the batch level           
    """
    fname_smlm      = config_render['fname_smlm']
    fname_rois      = config_render['fname_roi']
    roi_orient      = config_render['roi_orient']
    thickness       = config_render['thickness']
    pxsz_i          = config_render['pxsz_i']
    pxsz_o          = config_render['pxsz_o']
    nosinglet       = config_render['nosinglet']
    rlim_nm         = config_render['rlim_nm']
    frmrange        = config_render['frm_range']
    isblur          = config_render['isblur']
    sig             = config_render['sig']
    intensity       = config_render['intensity']
    alpha           = config_render['alpha']
    feature         = config_render['feature']
    zmin_nm         = config_render['zmin_nm']
    zmax_nm         = config_render['zmax_nm']
    colorbits       = config_render['colorbits']
    flIDs           = config_render['flID']
    fluornames      = config_render['fluorname']
    clim            = config_render['clim'].copy()
    colormap        = config_render['colormap']
    clim = np.array(clim, dtype=np.float32)

    # smlm_data and the working place
    with open(fname_smlm, 'rb') as fid:
        smlm_data = pickle.load(fid)
    fname_result = os.path.basename(fname_smlm)[:-11]

    # extract the rois
    if fname_rois.endswith('.zip'):
        rois = _readroizip(fname_rois)
    else:
        rois = _readROI(fname_rois)
        rois = [rois]
    
    # render
    for i, roi in enumerate(rois):

        if len(flIDs) > 1:
            fname = fname_result + '_{idx:d}_comb.png'.format(idx=i)
        else:
            fname = fname_result + '_{idx:d}_{flname:s}.png'.format(idx=i, flname=fluornames[0])
        roi_i = np.float32(roi[1])*pxsz_i
        
        SRim, Immax = finer_render_mulfl(smlm_data, roi[0], roi_orient, roi_i, thickness, pxsz_o, isblur, intensity, flIDs, clim, colormap, zmin_nm, zmax_nm, nosinglet, rlim_nm, sig, frmrange, feature, alpha, colorbits)
        if not SRim is None:
            ImObj = Image.fromarray(SRim, 'RGB')
            ImObj.save(os.path.join(fpath_result, fname))




if __name__ == '__main__':
    pass