from win32api import GetSystemMetrics
import numpy as np
from numbers import Number
from scipy.stats.distributions import chi2
from matplotlib import pyplot as plt

"""
quality control of the fit result 
see parameter_list.md for controlling parameters/options
"""

scrWidth, scrHeight = GetSystemMetrics(0), GetSystemMetrics(1)


def _textPos(ax, x_frac, y_frac):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_pos = xmin + x_frac * (xmax - xmin)
    y_pos = ymin + y_frac * (ymax - ymin)
    return x_pos, y_pos



def loc_campx2nm(ndim, xvec, crlb, cam_pxszs):
    """
    In-place translate the localizations from the unit of CamPxsz to that of nm
    INPUT:
        ndim:               int, 2 for modality=2D or 3 for modality=3D/AS
        xvec:               (nspots, vnum) float ndarray, [[locx, locy(, locz)(, s_j)(, sx_j, sy_j), photons_j, ...]
        crlb:               (nspots, vnum) float ndarray, crlb corresponding to xvec
        cam_pxszs:          (ndim,) float ndarray, [cam_pxszx, cam_pxszy(, zstepsz_nm)], size of the camera pixel   
    """
    for i in range(ndim):
        xvec[:, i] *= cam_pxszs[i]
        crlb[:, i] *= cam_pxszs[i] * cam_pxszs[i]
    return



def loc_inspector2d(nchannels, xvec, crlb, loss, nnum, df, nbins=100):
    """
    plot the ... to inspect the sinpa results
    INPUT:
        nchannels:          int, number of channels
        xvec:               (nspots, vnum) float ndarray, [locx, locy(, s_j)(, sx_j, sy_j), photons_j, ...] in nm if applicable
        crlb:               (nspots, vnum) float ndarray, crlb corresponding to the xvec
        Loss:               (NFits,) float ndarray, the log-likelihood ratio (MLE) or the Loss (LSQ) of the fits
        nnum:               (NFits,) int ndarray, the number of emitters for each fit
        df:                 int, degree of freedom
    RETURN:     
        fig:                matplotlib figure object
    """
    
    # parameters
    ndim = 2
    precnm_markers = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 120.0])
    photon_markers = np.array([1000.0, 2000.0, 5000.0, 10000.0, 20000.0])
    pvalue_markers = np.array([0.05, 1e-3, 1e-6])
    loss_markers = np.array([chi2.ppf(1-pval, df) for pval in pvalue_markers])

    # data collection
    loss_ext = np.repeat(loss, nnum)
    ind_valid = (xvec[:, ndim] > 1.0)
    if nchannels == 2:
        ind_valid = ind_valid & (xvec[:, ndim+1] > 0.0) & (xvec[:, ndim+1] < 1.0)
    for i in range(ndim):
        ind_valid = ind_valid & (crlb[:, i] > 0.0)
    
    log_prec = np.log10(np.clip(np.sqrt(crlb[ind_valid, :ndim]), 0.01, 1000.0)).T
    log_phot = np.log10(np.clip(xvec[ind_valid, ndim], 10.0, 1e6))
    frac = xvec[ind_valid, ndim+1] if nchannels == 2 else None
    log_loss_ext = np.log10(np.clip(loss_ext[ind_valid], 0.01, 1e5))
    log_loss = np.log10(np.clip(loss, 0.01, 1e5))

    # bins
    bins_log_prec = [np.linspace(log_prec[i].min(), log_prec[i].max(), num=nbins+1, endpoint=True) for i in range(ndim)]
    bins_log_phot = np.linspace(log_phot.min(), log_phot.max(), num=nbins+1, endpoint=True)
    bins_frac = np.linspace(0.0, 1.0, num=nbins+1, endpoint=True) if nchannels == 2 else None
    bins_log_loss = np.linspace(log_loss.min(), log_loss.max(), num=nbins+1, endpoint=True)
    
    # lims
    lims_log_phot = (bins_log_phot[0], bins_log_phot[-1])
    lims_log_loss = (bins_log_loss[0], bins_log_loss[-1])
    lims_log_prec = [(bins_log_prec[i][0], bins_log_prec[i][-1]) for i in range(ndim)]

    # FIGURE
    if nchannels == 1:
        fig = plt.figure(figsize=(0.35*scrWidth/72, 0.5*scrHeight/72))
        gs = fig.add_gridspec(3, nchannels+2, left=.1, right=.9, bottom=.1, top=.95, height_ratios=(1, 2, 2), width_ratios=(*([2]*nchannels), 2, 1), hspace=0.025, wspace=0.025)
        axs0 = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
    elif nchannels == 2:
        fig = plt.figure(figsize=(0.7*scrWidth/72, 0.5*scrHeight/72))
        gs = fig.add_gridspec(1, 2, left=.05, right=.95, bottom=.1, top=.95, width_ratios=(2, 1), wspace=0.2)
        gs0 = gs[0].subgridspec(3, nchannels+2, height_ratios=(1, 2, 2), width_ratios=(*([2]*nchannels), 2, 1), hspace=0.025, wspace=0.025)
        axs0 = [[fig.add_subplot(gs0[i, j]) for j in range(nchannels+2)] for i in range(3)]
        gs1 = gs[1].subgridspec(2, 1, hspace=0.025)
        axs1 = [fig.add_subplot(gs1[i]) for i in range(2)]
    else:
        raise ValueError("unsupported nchannels: input nchannels={}, should be 1 or 2".format(nchannels))
    
    # Left part
    dimlabels = ['x', 'y']
    hist_log_phot = np.histogram(log_phot, bins=bins_log_phot)[0]
    axs0[0][0].bar(bins_log_phot[:-1], hist_log_phot, width=bins_log_phot[1]-bins_log_phot[0], align='edge')
    axs0[0][0].set_xlim(lims_log_phot)
    axs0[0][0].tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
    axs0[0][0].tick_params(left=False, labelleft=False)
    axs0[0][0].grid(True, 'major', 'y')
    axs0[0][0].vlines(np.log10(photon_markers), 0, 1.1*hist_log_phot.max(), color='y', linewidth=0.5)
    axs0[0][0].set_ylim(0, 1.1*hist_log_phot.max())

    if nchannels == 2:
        hist_frac = np.histogram(frac, bins=bins_frac)[0]
        axs0[0][1].bar(bins_frac[:-1], hist_frac, width=bins_frac[1]-bins_frac[0], align='edge')
        axs0[0][1].set_xlim(0.0, 1.0)
        axs0[0][1].tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
        axs0[0][1].tick_params(left=False, labelleft=False)
        axs0[0][1].grid(True, 'major', 'y')
        axs0[0][1].vlines(0.5, 0, 1.1*hist_frac.max(), color='y', linewidth=0.5)
        axs0[0][1].set_ylim(0, 1.1*hist_frac.max())
        
    hist_log_loss = np.histogram(log_loss, bins=bins_log_loss)[0]
    axs0[0][nchannels].bar(bins_log_loss[:-1], hist_log_loss, width=bins_log_loss[1]-bins_log_loss[0], align='edge')
    axs0[0][nchannels].vlines(np.log10(loss_markers[0]), 0, 1.1*hist_log_loss.max(), color='tab:orange', linewidth=0.5, label='pval={p:.2f}'.format(p=pvalue_markers[0]))
    axs0[0][nchannels].vlines(np.log10(loss_markers[1]), 0, 1.1*hist_log_loss.max(), color='tab:red', linewidth=0.5, label='pval={p:.0e}'.format(p=pvalue_markers[1]))
    axs0[0][nchannels].vlines(np.log10(loss_markers[2]), 0, 1.1*hist_log_loss.max(), color='tab:purple', linewidth=0.5, label='pval={p:.0e}'.format(p=pvalue_markers[2]))
    axs0[0][nchannels].set_ylim(0, 1.1*hist_log_loss.max())
    axs0[0][nchannels].set_xlim(lims_log_loss)
    axs0[0][nchannels].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axs0[0][nchannels].tick_params(top=True, labeltop=True)
    axs0[0][nchannels].legend()

    axs0[0][nchannels+1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axs0[0][nchannels+1].text(0.5, 0.5, "NFits={}\nnspots={}".format(len(nnum), nnum.sum()), va="center", ha="center")

    for i in range(ndim):
        
        hist_log_prec = np.histogram(log_prec[i], bins=bins_log_prec[i])[0]
        axs0[i+1][nchannels+1].barh(bins_log_prec[i][:-1], hist_log_prec, height=bins_log_prec[i][1]-bins_log_prec[i][0], align='edge', label='prec_{}'.format(dimlabels[i]))
        axs0[i+1][nchannels+1].set_ylim(lims_log_prec[i])
        axs0[i+1][nchannels+1].set_yticks(np.log10(precnm_markers), labels=precnm_markers)
        axs0[i+1][nchannels+1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axs0[i+1][nchannels+1].tick_params(right=True, labelright=True)        
        axs0[i+1][nchannels+1].hlines(np.log10(precnm_markers), 0.0, 1.1 * hist_log_prec.max(), color='k', linewidth=0.5)
        axs0[i+1][nchannels+1].legend()
        axs0[i+1][nchannels+1].set_xlim(0.0, 1.1 * hist_log_prec.max())

        hist_log_prec2 = np.histogram2d(log_prec[i], log_phot, bins=[bins_log_prec[i], bins_log_phot])[0]
        axs0[i+1][0].pcolormesh(bins_log_phot[:-1], bins_log_prec[i][:-1], hist_log_prec2, cmap='turbo', shading='auto')
        axs0[i+1][0].vlines(np.log10(photon_markers), *lims_log_prec[i], color='y', linewidth=0.5)
        axs0[i+1][0].hlines(np.log10(precnm_markers), *lims_log_phot, color='y', linewidth=0.5)
        axs0[i+1][0].set_ylim(lims_log_prec[i])
        axs0[i+1][0].set_xlim(lims_log_phot)
        axs0[i+1][0].set_ylabel('log_precision_{} (nm)'.format(dimlabels[i]))
        if i == 2:
            axs0[i+1][0].set_xlabel('log10(photon)')
        else:
            axs0[i+1][0].tick_params(bottom=False, labelbottom=False)
        
        if nchannels == 2:
            hist_log_prec2 = np.histogram2d(log_prec[i], frac, bins=[bins_log_prec[i], bins_frac])[0]
            axs0[i+1][1].pcolormesh(bins_frac[:-1], bins_log_prec[i][:-1], hist_log_prec2, cmap='turbo', shading='auto')
            axs0[i+1][1].vlines(0.5, *lims_log_prec[i], color='y', linewidth=0.5)
            axs0[i+1][1].hlines(np.log10(precnm_markers), 0.0, 1.0, color='y', linewidth=0.5)
            axs0[i+1][1].set_ylim(lims_log_prec[i])
            axs0[i+1][1].set_xlim(0.0, 1.0)
            axs0[i+1][1].tick_params(left=False, labelleft=False)
            if i == 2:
                axs0[i+1][1].set_xlabel('frac')
            else:
                axs0[i+1][1].tick_params(bottom=False, labelbottom=False)
         
        hist_log_prec2 = np.histogram2d(log_prec[i], log_loss_ext, bins=[bins_log_prec[i], bins_log_loss])[0]
        axs0[i+1][nchannels].pcolormesh(bins_log_loss[:-1], bins_log_prec[i][:-1], hist_log_prec2, cmap='turbo', shading='auto')
        axs0[i+1][nchannels].vlines(np.log10(loss_markers[0]), *lims_log_prec[i], color='tab:orange', linewidth=0.5)
        axs0[i+1][nchannels].vlines(np.log10(loss_markers[1]), *lims_log_prec[i], color='tab:red', linewidth=0.5)
        axs0[i+1][nchannels].vlines(np.log10(loss_markers[2]), *lims_log_prec[i], color='tab:purple', linewidth=0.5)
        axs0[i+1][nchannels].hlines(np.log10(precnm_markers), *lims_log_loss, color='y', linewidth=0.5)
        axs0[i+1][nchannels].set_ylim(lims_log_prec[i])
        axs0[i+1][nchannels].set_xlim(lims_log_loss)
        axs0[i+1][nchannels].tick_params(left=False, labelleft=False)
        if i == 2:
            axs0[i+1][nchannels].set_xlabel('log10(loss)')
        else:
            axs0[i+1][nchannels].tick_params(bottom=False, labelbottom=False)

    # Right part
    if nchannels == 2:
        
        hist_log_photon2 = np.histogram2d(log_phot, frac, bins=[bins_log_phot, bins_frac])[0]
        axs1[0].pcolormesh(bins_frac[:-1], bins_log_phot[:-1], hist_log_photon2, cmap='turbo', shading='auto')
        axs1[0].hlines(np.log10(photon_markers), 0.0, 1.0, color='y', linewidth=0.5)
        axs1[0].set_xlim(0.0, 1.0)
        axs1[0].set_ylim(lims_log_phot)
        axs1[0].set_xlabel('frac')
        axs1[0].set_ylabel('log10(photon)')

        log_phot_chn0 = log_phot + np.log10(frac)
        log_phot_chn1 = log_phot + np.log10(1.0 - frac)
        log_phot_min = max(min(log_phot_chn0.min(), log_phot_chn1.min()), 2.5)
        log_phot_max = min(max(log_phot_chn0.max(), log_phot_chn1.max()), 5.5)
        bins_log_phot = np.linspace(log_phot_min, log_phot_max, num=nbins+1, endpoint=True)
        
        hist_log_photon2 = np.histogram2d(log_phot_chn0, log_phot_chn1, bins=[bins_log_phot, bins_log_phot])[0]
        axs1[1].pcolormesh(bins_log_phot[:-1], bins_log_phot[:-1], hist_log_photon2, cmap='turbo', shading='auto')
        axs1[1].plot([log_phot_min, log_phot_max], [log_phot_min, log_phot_max], 'w--', linewidth=0.5)
        axs1[1].vlines(np.log10(photon_markers), log_phot_min, log_phot_max, color='y', linewidth=0.5)
        axs1[1].hlines(np.log10(photon_markers), log_phot_min, log_phot_max, color='y', linewidth=0.5)
        axs1[1].set_xlim(log_phot_min, log_phot_max)
        axs1[1].set_ylim(log_phot_min, log_phot_max)
        axs1[1].set_xlabel('log10(photon)_chnl1')
        axs1[1].set_ylabel('log10(photon)_chnl0')

    return fig



def loc_inspector3d(nchannels, xvec, crlb, loss, nnum, df, nbins=100):
    """
    plot the ... to inspect the sinpa results
    INPUT:
        nchannels:          int, number of channels
        xvec:               (nspots, vnum) float ndarray, [[locx, locy, locz, ...],...] in nm if applicable
        crlb:               (nspots, vnum) float ndarray, [[crlbx, crlby, crlbz, ...], ...] crlbs of the inspected localizations in nm if applicable
        Loss:               (NFits,) float ndarray, the log-likelihood ratio (MLE) or the Loss (LSQ) of the fits
        nnum:               (NFits,) int ndarray, the number of emitters for each fit
        df:                 (2,) or int, degree(s) of freedom
    RETURN:     
        fig:                matplotlib figure object
    """
    
    # parameters
    ndim = 3
    precnm_markers = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 120.0])
    photon_markers = np.array([1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0])
    pvalue_markers = np.array([0.05, 1e-3, 1e-6])
    loss_markers = np.array([chi2.ppf(1-pval, df) for pval in pvalue_markers])

    # data collection
    loss_ext = np.repeat(loss, nnum)
    ind_valid = (xvec[:, ndim] > 1.0)
    if nchannels == 2:
        ind_valid = ind_valid & (xvec[:, ndim+1] > 0.0) & (xvec[:, ndim+1] < 1.0)
    for i in range(ndim):
        ind_valid = ind_valid & (crlb[:, i] > 0.0)
    
    locz = xvec[ind_valid, 2]
    log_prec = np.log10(np.clip(np.sqrt(crlb[ind_valid, :ndim]), 0.01, 1000.0)).T
    log_phot = np.log10(np.clip(xvec[ind_valid, ndim], 10.0, 1e6))
    frac = xvec[ind_valid, ndim+1] if nchannels == 2 else None
    log_loss_ext = np.log10(np.clip(loss_ext[ind_valid], 0.01, 1e5))
    log_loss = np.log10(np.clip(loss, 0.01, 1e5))

    # bins
    bins_locz = np.linspace(locz.min(), locz.max(), num=nbins+1, endpoint=True)
    bins_log_prec = [np.linspace(log_prec[i].min(), log_prec[i].max(), num=nbins+1, endpoint=True) for i in range(ndim)]
    bins_log_phot = np.linspace(log_phot.min(), log_phot.max(), num=nbins+1, endpoint=True)
    bins_frac = np.linspace(0.0, 1.0, num=nbins+1, endpoint=True) if nchannels == 2 else None
    bins_log_loss = np.linspace(log_loss.min(), log_loss.max(), num=nbins+1, endpoint=True)
    
    # lims
    lims_log_phot = (bins_log_phot[0], bins_log_phot[-1])
    lims_log_loss = (bins_log_loss[0], bins_log_loss[-1])
    lims_locz = (bins_locz[0], bins_locz[-2])
    lims_log_prec = [(bins_log_prec[i][0], bins_log_prec[i][-1]) for i in range(ndim)]

    # FIGURE
    fig = plt.figure(figsize=(0.9*scrWidth/72, 0.85*scrHeight/72))
    if nchannels == 1:
        gs = fig.add_gridspec(1, 2, left=.05, right=.95, bottom=.1, top=.95, wspace=0.2)
        gs0 = gs[0].subgridspec(4, nchannels+3, height_ratios=(1, 2, 2, 2), width_ratios=(*([2]*nchannels), 2, 2, 1), hspace=0.025, wspace=0.025)
        axs0 = [[fig.add_subplot(gs0[i, j]) for j in range(nchannels+3)] for i in range(4)]
        gs1 = gs[1].subgridspec(3, nchannels+1, height_ratios=(2, 1, 1), hspace=0.025, wspace=0.025)
        axs1 = [[fig.add_subplot(gs1[i, j]) for j in range(nchannels+1)] for i in range(3)]
    elif nchannels == 2:
        gs = fig.add_gridspec(2, 2, left=.05, right=.95, bottom=.1, top=.95, wspace=0.2)
        gs0 = gs[:, 0].subgridspec(4, nchannels+3, height_ratios=(1, 2, 2, 2), width_ratios=(*([2]*nchannels), 2, 2, 1), hspace=0.025, wspace=0.025)
        axs0 = [[fig.add_subplot(gs0[i, j]) for j in range(nchannels+3)] for i in range(4)]
        gs1 = gs[0, 1].subgridspec(3, nchannels+1, height_ratios=(2, 1, 1), hspace=0.025, wspace=0.025)
        axs1 = [[fig.add_subplot(gs1[i, j]) for j in range(nchannels+1)] for i in range(3)]
        gs2 = gs[1, 1].subgridspec(1, 2, wspace=0.025)
        axs2 = [fig.add_subplot(gs2[i]) for i in range(2)]
    else:
        raise ValueError("unsupported nchannels: input nchannels={}, should be 1 or 2".format(nchannels))
    
    # Left part
    dimlabels = ['x', 'y', 'z']
    hist_log_phot = np.histogram(log_phot, bins=bins_log_phot)[0]
    axs0[0][0].bar(bins_log_phot[:-1], hist_log_phot, width=bins_log_phot[1]-bins_log_phot[0], align='edge')
    axs0[0][0].set_xlim(lims_log_phot)
    axs0[0][0].tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
    axs0[0][0].tick_params(left=False, labelleft=False)
    axs0[0][0].grid(True, 'major', 'y')
    axs0[0][0].vlines(np.log10(photon_markers), 0, 1.1*hist_log_phot.max(), color='y', linewidth=0.5)
    axs0[0][0].set_ylim(0, 1.1*hist_log_phot.max())

    if nchannels == 2:
        hist_frac = np.histogram(frac, bins=bins_frac)[0]
        axs0[0][1].bar(bins_frac[:-1], hist_frac, width=bins_frac[1]-bins_frac[0], align='edge')
        axs0[0][1].set_xlim(0.0, 1.0)
        axs0[0][1].tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
        axs0[0][1].tick_params(left=False, labelleft=False)
        axs0[0][1].grid(True, 'major', 'y')
        axs0[0][1].vlines(0.5, 0, 1.1*hist_frac.max(), color='y', linewidth=0.5)
        axs0[0][1].set_ylim(0, 1.1*hist_frac.max())
        
    hist_log_loss = np.histogram(log_loss, bins=bins_log_loss)[0]
    axs0[0][nchannels].bar(bins_log_loss[:-1], hist_log_loss, width=bins_log_loss[1]-bins_log_loss[0], align='edge')
    axs0[0][nchannels].vlines(np.log10(loss_markers[0]), 0, 1.1*hist_log_loss.max(), color='tab:orange', linewidth=0.5, label='pval={p:.2f}'.format(p=pvalue_markers[0]))
    axs0[0][nchannels].vlines(np.log10(loss_markers[1]), 0, 1.1*hist_log_loss.max(), color='tab:red', linewidth=0.5, label='pval={p:.0e}'.format(p=pvalue_markers[1]))
    axs0[0][nchannels].vlines(np.log10(loss_markers[2]), 0, 1.1*hist_log_loss.max(), color='tab:purple', linewidth=0.5, label='pval={p:.0e}'.format(p=pvalue_markers[2]))
    axs0[0][nchannels].set_ylim(0, 1.1*hist_log_loss.max())
    axs0[0][nchannels].set_xlim(lims_log_loss)
    axs0[0][nchannels].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axs0[0][nchannels].tick_params(top=True, labeltop=True)
    axs0[0][nchannels].legend()
    
    hist_locz = np.histogram(locz, bins=bins_locz)[0]
    axs0[0][nchannels+1].bar(bins_locz[:-1], hist_locz, width=bins_locz[1]-bins_locz[0], align='edge')
    axs0[0][nchannels+1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axs0[0][nchannels+1].tick_params(top=True, labeltop=True)
    axs0[0][nchannels+1].set_ylim(0, 1.1*hist_locz.max())
    axs0[0][nchannels+1].set_xlim(lims_locz)
    axs0[0][nchannels+1].grid(True, 'major', 'x')

    axs0[0][nchannels+2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axs0[0][nchannels+2].text(0.5, 0.5, "NFits={}\nnspots={}".format(len(nnum), nnum.sum()), va="center", ha="center")

    for i in range(ndim):
        
        hist_log_prec = np.histogram(log_prec[i], bins=bins_log_prec[i])[0]
        axs0[i+1][nchannels+2].barh(bins_log_prec[i][:-1], hist_log_prec, height=bins_log_prec[i][1]-bins_log_prec[i][0], align='edge', label='prec_{}'.format(dimlabels[i]))
        axs0[i+1][nchannels+2].set_ylim(lims_log_prec[i])
        axs0[i+1][nchannels+2].set_yticks(np.log10(precnm_markers), labels=precnm_markers)
        axs0[i+1][nchannels+2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axs0[i+1][nchannels+2].tick_params(right=True, labelright=True)        
        axs0[i+1][nchannels+2].hlines(np.log10(precnm_markers), 0.0, 1.1 * hist_log_prec.max(), color='k', linewidth=0.5)
        axs0[i+1][nchannels+2].legend()
        axs0[i+1][nchannels+2].set_xlim(0.0, 1.1 * hist_log_prec.max())

        hist_log_prec2 = np.histogram2d(log_prec[i], log_phot, bins=[bins_log_prec[i], bins_log_phot])[0]
        axs0[i+1][0].pcolormesh(bins_log_phot[:-1], bins_log_prec[i][:-1], hist_log_prec2, cmap='turbo', shading='auto')
        axs0[i+1][0].vlines(np.log10(photon_markers), *lims_log_prec[i], color='y', linewidth=0.5)
        axs0[i+1][0].hlines(np.log10(precnm_markers), *lims_log_phot, color='y', linewidth=0.5)
        axs0[i+1][0].set_ylim(lims_log_prec[i])
        axs0[i+1][0].set_xlim(lims_log_phot)
        axs0[i+1][0].set_ylabel('log_precision_{} (nm)'.format(dimlabels[i]))
        if i == 2:
            axs0[i+1][0].set_xlabel('log10(photon)')
        else:
            axs0[i+1][0].tick_params(bottom=False, labelbottom=False)
        
        if nchannels == 2:
            hist_log_prec2 = np.histogram2d(log_prec[i], frac, bins=[bins_log_prec[i], bins_frac])[0]
            axs0[i+1][1].pcolormesh(bins_frac[:-1], bins_log_prec[i][:-1], hist_log_prec2, cmap='turbo', shading='auto')
            axs0[i+1][1].vlines(0.5, *lims_log_prec[i], color='y', linewidth=0.5)
            axs0[i+1][1].hlines(np.log10(precnm_markers), 0.0, 1.0, color='y', linewidth=0.5)
            axs0[i+1][1].set_ylim(lims_log_prec[i])
            axs0[i+1][1].set_xlim(0.0, 1.0)
            axs0[i+1][1].tick_params(left=False, labelleft=False)
            if i == 2:
                axs0[i+1][1].set_xlabel('frac')
            else:
                axs0[i+1][1].tick_params(bottom=False, labelbottom=False)
         
        hist_log_prec2 = np.histogram2d(log_prec[i], log_loss_ext, bins=[bins_log_prec[i], bins_log_loss])[0]
        axs0[i+1][nchannels].pcolormesh(bins_log_loss[:-1], bins_log_prec[i][:-1], hist_log_prec2, cmap='turbo', shading='auto')
        axs0[i+1][nchannels].vlines(np.log10(loss_markers[0]), *lims_log_prec[i], color='tab:orange', linewidth=0.5)
        axs0[i+1][nchannels].vlines(np.log10(loss_markers[1]), *lims_log_prec[i], color='tab:red', linewidth=0.5)
        axs0[i+1][nchannels].vlines(np.log10(loss_markers[2]), *lims_log_prec[i], color='tab:purple', linewidth=0.5)
        axs0[i+1][nchannels].hlines(np.log10(precnm_markers), *lims_log_loss, color='y', linewidth=0.5)
        axs0[i+1][nchannels].set_ylim(lims_log_prec[i])
        axs0[i+1][nchannels].set_xlim(lims_log_loss)
        axs0[i+1][nchannels].tick_params(left=False, labelleft=False)
        if i == 2:
            axs0[i+1][nchannels].set_xlabel('log10(loss)')
        else:
            axs0[i+1][nchannels].tick_params(bottom=False, labelbottom=False)

        hist_log_prec2 = np.histogram2d(log_prec[i], locz, bins=[bins_log_prec[i], bins_locz])[0]
        axs0[i+1][nchannels+1].pcolormesh(bins_locz[:-1], bins_log_prec[i][:-1], hist_log_prec2, cmap='turbo', shading='auto')
        axs0[i+1][nchannels+1].hlines(np.log10(precnm_markers), *lims_locz, color='y', linewidth=0.5)
        axs0[i+1][nchannels+1].set_ylim(lims_log_prec[i])
        axs0[i+1][nchannels+1].set_xlim(lims_locz)
        axs0[i+1][nchannels+1].grid(True, 'major', 'x')
        axs0[i+1][nchannels+1].tick_params(left=False, labelleft=False)
        if i == 2:
            axs0[i+1][nchannels+1].set_xlabel('locz (nm)')
        else:
            axs0[i+1][nchannels+1].tick_params(bottom=False, labelbottom=False)

    # Right part
    hist_log_locz2 = np.histogram2d(locz, log_phot, bins=[bins_locz, bins_log_phot])[0]
    axs1[0][0].pcolormesh(bins_log_phot[:-1], bins_locz[:-1], hist_log_locz2, cmap='turbo', shading='auto')
    axs1[0][0].vlines(np.log10(photon_markers), *lims_locz, color='y', linewidth=0.5)
    axs1[0][0].set_xlim(lims_log_phot)
    axs1[0][0].set_ylim(lims_locz)
    axs1[0][0].grid(True, 'both', 'y')
    axs1[0][0].tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
    axs1[0][0].set_ylabel('locz (nm)')
    
    if nchannels == 2:
        hist_locz2 = np.histogram2d(locz, frac, bins=[bins_locz, bins_frac])[0]
        axs1[0][1].pcolormesh(bins_frac[:-1], bins_locz[:-1], hist_locz2, cmap='turbo', shading='auto')
        axs1[0][1].vlines(0.5, *lims_locz, color='y', linewidth=0.5)
        axs1[0][1].set_xlim(0.0, 1.0)
        axs1[0][1].set_ylim(lims_locz)
        axs1[0][1].grid(True, 'both', 'y')
        axs1[0][1].tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
        axs1[0][1].tick_params(left=False, labelleft=False)

    hist_log_locz2 = np.histogram2d(locz, log_loss_ext, bins=[bins_locz, bins_log_loss])[0]
    axs1[0][nchannels].pcolormesh(bins_log_loss[:-1], bins_locz[:-1], hist_log_locz2, cmap='turbo', shading='auto')
    axs1[0][nchannels].vlines(np.log10(loss_markers[0]), *lims_locz, color='tab:orange', linewidth=0.5)
    axs1[0][nchannels].vlines(np.log10(loss_markers[1]), *lims_locz, color='tab:red', linewidth=0.5)
    axs1[0][nchannels].vlines(np.log10(loss_markers[2]), *lims_locz, color='tab:purple', linewidth=0.5)
    axs1[0][nchannels].set_xlim(lims_log_loss)
    axs1[0][nchannels].set_ylim(lims_locz)
    axs1[0][nchannels].grid(True, 'both', 'y')
    axs1[0][nchannels].tick_params(left=False, labelleft=False, right=True, labelright=True)
    axs1[0][nchannels].tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)

    indz = (locz < 300) | (locz > 900)
    hist_log_phot = np.histogram(log_phot[indz], bins=bins_log_phot)[0]
    axs1[1][0].bar(bins_log_phot[:-1], hist_log_phot, width=bins_log_phot[1]-bins_log_phot[0], align='edge')
    axs1[1][0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    axs1[1][0].set_xlim(lims_log_phot)
    axs1[1][0].text(*_textPos(axs1[1][0], 0.05, 0.95), 'locz $ \\notin $ [300, 900] ({:.1%})'.format(indz.sum()/nnum.sum()))
    axs1[1][0].vlines(np.log10(photon_markers), 0, 1.1*hist_log_phot.max(), color='y', linewidth=0.5)
    axs1[1][0].set_ylim(0, 1.1*hist_log_phot.max())
    
    if nchannels == 2:
        hist_frac = np.histogram(frac[indz], bins=bins_frac)[0]
        axs1[1][1].bar(bins_frac[:-1], hist_frac, width=bins_frac[1]-bins_frac[0], align='edge')
        axs1[1][1].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        axs1[1][1].set_xlim(0.0, 1.0)
        axs1[1][1].vlines(0.5, 0, 1.1*hist_frac.max(), color='y', linewidth=0.5)
        axs1[1][1].set_ylim(0, 1.1*hist_frac.max())
        
    hist_log_loss = np.histogram(log_loss_ext, bins=bins_log_loss)[0]
    axs1[1][nchannels].bar(bins_log_loss[:-1], hist_log_loss, width=bins_log_loss[1]-bins_log_loss[0], align='edge')
    axs1[1][nchannels].vlines(np.log10(loss_markers[0]), 0, 1.1*hist_log_loss.max(), color='tab:orange', linewidth=0.5, label='pval={p:.2f}'.format(p=pvalue_markers[0]))
    axs1[1][nchannels].vlines(np.log10(loss_markers[1]), 0, 1.1*hist_log_loss.max(), color='tab:red', linewidth=0.5, label='pval={p:.0e}'.format(p=pvalue_markers[1]))
    axs1[1][nchannels].vlines(np.log10(loss_markers[2]), 0, 1.1*hist_log_loss.max(), color='tab:purple', linewidth=0.5, label='pval={p:.0e}'.format(p=pvalue_markers[2]))
    axs1[1][nchannels].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    axs1[1][nchannels].set_xlim(lims_log_loss)
    axs1[1][nchannels].set_ylim(0, 1.1*hist_log_loss.max())
    axs1[1][nchannels].legend()    

    indz = (locz >= 300) & (locz <= 900)
    hist_log_phot = np.histogram(log_phot[indz], bins=bins_log_phot)[0]
    axs1[2][0].bar(bins_log_phot[:-1], hist_log_phot, width=bins_log_phot[1]-bins_log_phot[0], align='edge')
    axs1[2][0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    axs1[2][0].set_xlim(lims_log_phot)
    axs1[2][0].text(*_textPos(axs1[2][0], 0.05, 0.95), 'locz $ \\in $ [300, 900] ({:.1%})'.format(indz.sum()/nnum.sum()))
    axs1[2][0].vlines(np.log10(photon_markers), 0, 1.1*hist_log_phot.max(), color='y', linewidth=0.5)
    axs1[2][0].set_ylim(0, 1.1*hist_log_phot.max())
    
    if nchannels == 2:
        hist_frac = np.histogram(frac[indz], bins=bins_frac)[0]
        axs1[2][1].bar(bins_frac[:-1], hist_frac, width=bins_frac[1]-bins_frac[0], align='edge')
        axs1[2][1].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        axs1[2][1].set_xlim(0.0, 1.0)
        axs1[2][1].vlines(0.5, 0, 1.1*hist_frac.max(), color='y', linewidth=0.5)
        axs1[2][1].set_ylim(0, 1.1*hist_frac.max())
        
    hist_log_loss = np.histogram(log_loss_ext, bins=bins_log_loss)[0]
    axs1[2][nchannels].bar(bins_log_loss[:-1], hist_log_loss, width=bins_log_loss[1]-bins_log_loss[0], align='edge')
    axs1[2][nchannels].vlines(np.log10(loss_markers[0]), 0, 1.1*hist_log_loss.max(), color='tab:orange', linewidth=0.5, label='pval={p:.2f}'.format(p=pvalue_markers[0]))
    axs1[2][nchannels].vlines(np.log10(loss_markers[1]), 0, 1.1*hist_log_loss.max(), color='tab:red', linewidth=0.5, label='pval={p:.0e}'.format(p=pvalue_markers[1]))
    axs1[2][nchannels].vlines(np.log10(loss_markers[2]), 0, 1.1*hist_log_loss.max(), color='tab:purple', linewidth=0.5, label='pval={p:.0e}'.format(p=pvalue_markers[2]))
    axs1[2][nchannels].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    axs1[2][nchannels].set_xlim(lims_log_loss)
    axs1[2][nchannels].set_ylim(0, 1.1*hist_log_loss.max())
    axs1[2][nchannels].legend()  

    if nchannels == 2:
        
        hist_log_photon2 = np.histogram2d(log_phot, frac, bins=[bins_log_phot, bins_frac])[0]
        axs2[0].pcolormesh(bins_frac[:-1], bins_log_phot[:-1], hist_log_photon2, cmap='turbo', shading='auto')
        axs2[0].hlines(np.log10(photon_markers), 0.0, 1.0, color='y', linewidth=0.5)
        axs2[0].set_xlim(0.0, 1.0)
        axs2[0].set_ylim(lims_log_phot)
        axs2[0].set_xlabel('frac')
        axs2[0].set_ylabel('log10(photon)')

        log_phot_chn0 = log_phot + np.log10(frac)
        log_phot_chn1 = log_phot + np.log10(1.0 - frac)
        log_phot_min = max(min(log_phot_chn0.min(), log_phot_chn1.min()), 2.5)
        log_phot_max = min(max(log_phot_chn0.max(), log_phot_chn1.max()), 5.5)
        bins_log_phot = np.linspace(log_phot_min, log_phot_max, num=nbins+1, endpoint=True)
        hist_log_photon2 = np.histogram2d(log_phot_chn0, log_phot_chn1, bins=[bins_log_phot, bins_log_phot])[0]
        axs2[1].pcolormesh(bins_log_phot[:-1], bins_log_phot[:-1], hist_log_photon2, cmap='turbo', shading='auto')
        axs2[1].plot([log_phot_min, log_phot_max], [log_phot_min, log_phot_max], 'w--', linewidth=0.5)
        axs2[1].vlines(np.log10(photon_markers), log_phot_min, log_phot_max, color='y', linewidth=0.5)
        axs2[1].hlines(np.log10(photon_markers), log_phot_min, log_phot_max, color='y', linewidth=0.5)
        axs2[1].set_xlim(log_phot_min, log_phot_max)
        axs2[1].set_ylim(log_phot_min, log_phot_max)
        axs2[1].tick_params(left=False, labelleft=False)
        axs2[1].set_ylabel('log10(photon)_chnl0')
        axs2[1].set_xlabel('log10(photon)_chnl1')

    return fig



def loc_filter(ndim, loss, locz, photon, crlb_nm, filt_loss, filt_zrange, filt_photon, filt_precision_nm):
    """
    filt the localizations with the given filt_photon and filt_precision
    INPUT:
        ndim:               int, 2 for modality=2D or 3 for modality=3D/AS
        loss:               (nspots,) float ndarray, the loss of each fit (extended from (NFits,))
        locz:               (nspots,) float ndarray, the locz of each fit, ignored if ndim == 3
        photon:             (nspots,) float ndarray, total photon number of the inspected localizations
        crlb_nm:            (nspots, vnum) float ndarray, [[crlbx, crlby(, crlbz)...],...] crlbs of the inspected localizations in nm
        filt_loss:          (nspots,) float, the threshold for the loss
        filt_zrange:        (2,) the zrange for filtering
        filt_photon:        float, the threshold for photon filter
        filt_precision_nm:  (ndim,) float ndarray, the thresholds for localization filter
    RETURN:     
        ind_filt:           (nspots,) blooen array, for localizations surviving the filters
    NOTE:
        indf and loss should be extended from (NFits,) to (nnum.sum(), )   
    """
    
    ind_filt_loss = loss < filt_loss
    ind_filt_photon = photon > filt_photon
    ind_filt = ind_filt_loss & ind_filt_photon
    if ndim == 3:
        ind_filt = ind_filt & (locz >= np.min(filt_zrange)) & (locz <= np.max(filt_zrange))

    for i in range(ndim):
        dum_filt = (crlb_nm[:, i] < filt_precision_nm[i]*filt_precision_nm[i]) & (crlb_nm[:, i] > 0)
        ind_filt = ind_filt & dum_filt
    
    return ind_filt