from win32api import GetSystemMetrics
import os
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

scrWidth, scrHeight = GetSystemMetrics(0), GetSystemMetrics(1)
TAB_COLORS = [item for item in mcolors.TABLEAU_COLORS.keys()]



def _draw_bounds(photon, fracN, nfluorophores, nbins):
    """
    Mannually draw the boundries for the log_diff between photon_chn0 and photon_ch1
    INPUT:
        photon:         (nspots,) float, the photons
        fracN:          (nspots,) float, the fracN
        nfluorophores:  int, number of the fluorophores to register
        nbins:          int, number of bins for histogram (10*nbins for log_diff)
    RETURN:
        bounds:         (nfluorophore, 2) float, user-defined [fracN_min, fracN_max] for each fluorophore
    """
    ########## my click function ##########
    def _click_bounds(event, lines, nfluorophores, fig, ax):
        if ax != event.inaxes:
            return
        ymin, ymax = ax.get_ylim()
        xc = np.float32(event.xdata)
        if len(lines) % 2:
            ax.vlines(xc, ymin, ymax, color=TAB_COLORS[len(lines)//2])
        else:
            ax.vlines(xc, ymin, ymax, color=TAB_COLORS[len(lines)//2], alpha=0.5)
        ax.figure.canvas.draw()
        lines.append(xc)
        if len(lines) >= 2 * nfluorophores:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
        return
    #######################################

    photon_markers = np.array([1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0])
    log_phot = np.log10(np.clip(photon, 100.0, 1e7))
    bins_log_phot = np.linspace(log_phot.min(), log_phot.max(), num=nbins+1, endpoint=True)
    lims_log_phot = (bins_log_phot[0], bins_log_phot[-1])

    bins_fracN = np.linspace(0.0, 1.0, num=nbins+1, endpoint=True)
    hist_log_phot_fracN = np.histogram2d(log_phot, fracN, bins=[bins_log_phot, bins_fracN])[0]

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot()
    ax.pcolormesh(bins_fracN[:-1], bins_log_phot[:-1], hist_log_phot_fracN, cmap='turbo', shading='auto')
    ax.hlines(np.log10(photon_markers), 0.0, 1.0, color='y', linewidth=0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(lims_log_phot)
    ax.set_xlabel('frac')
    ax.set_ylabel('log10(photon)')
    
    lines = []
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: _click_bounds(event, lines, nfluorophores, fig, ax))
    plt.show()
    plt.close(fig)
    bounds = np.reshape(lines, (nfluorophores, 2))
    lefts = np.array(lines[::2])
    rights = np.array(lines[1::2])
    if np.any(rights[:-1] - lefts[1:] >= 0.0):
        raise ValueError("mannually chosen range overlap detected!")
    
    return bounds



def _ratio_assign(smlm_data):
    """
    Mannually draw the boundries for the log_diff between photon_chn0 and photon_ch1
    INPUT:
        smlm_data[in]:  'roi_nm':       (4,) float ndarray, [top, left, bottom, right] in nm
                        'ndim':         int, number of dimention
                        'nchannels':    int, number of channels
                        'indf':         (nspots,) int ndarray, the frame index of each localization (extended from (NFits,))
                        'xvec':         (nspots, ndim + nchannels + nchannels) float ndarray, the localization information
                        'crlb':         (nspots, ndim + nchannels + nchannels) float ndarray, the crlb for the corresponding to the xvec
                        'loss':         (nspots,) float ndarray, the loss for each localization (extended from (NFits,))
                        'crng':         (nfluorophore, 3) float, user-defined [fracN_min, fracN_max, thresh_phot] for each fluorophore
        smlm_data[out]  +'creg':        (nspots,) int ndarray, the registration of each fit to different channels, -1 for failed registration, -1 for unclassfied
    """
    if smlm_data['nchannels'] == 1:
        smlm_data['creg'] = np.zeros(len(smlm_data['xvec']), dtype=np.int32)
        return
    
    ndim = smlm_data['ndim']
    xvec = smlm_data['xvec']
    crng = smlm_data['crng']

    fracN = xvec[:, ndim + 1]
    nspots = len(fracN)
    creg = np.zeros(nspots, dtype=np.int32) - 1
    for i, frac in enumerate(fracN):
        for dumind, dum_crng in enumerate(crng):
            if frac > dum_crng[0] and frac < dum_crng[1]:
                creg[i] = dumind
                break        
    smlm_data['creg'] = creg
    return



def ratio_register(smlm_data, nfluorophores, nbins=100):
    """
    register manually (register the fluorphore by mannually choose left and right boundary from the log_ratio profile)
    INPUT:
        smlm_data[in]:  'ndim':         int, number of dimention
                        'nchannels':    int, number of channels
                        'xvec':         (nspots, ndim + nchannels + nchannels) float ndarray, the localization information
        smlm_data[out]  +'crng':        (nfluorophore, 3) float, user-defined [fracN_min, fracN_max, thresh_phot] for each fluorophore  
                        +'creg':        (nspots,) int ndarray, the registration of each fit to different channels, -1 for failed registration, -1 for unclassfied               
        nfluorophores:  int, number of the fluorophores to register
        nbins:          int, number of bins for histogram (10*nbins for log_diff)
    """
    if smlm_data['nchannels'] == 1:
        return
    
    ndim = smlm_data['ndim']
    xvec = smlm_data['xvec']
    smlm_data['crng'] = _draw_bounds(xvec[:, ndim], xvec[:, ndim + 1], nfluorophores, nbins)
    _ratio_assign(smlm_data)
    
    return



def ratio_plot(smlm_data, phot_min=1e2, phot_max=1.5e5, frac_min=0.001, frac_max=0.999, nbins=1000):
    """
    register manually (register the fluorphore by mannually choose left and right boundary from the log_ratio profile)
    INPUT:
        smlm_data[in]:      'ndim':             int, number of dimention
                            'nchannels':        int, number of channels
                            'xvec':             (nspots, ndim + nchannels + nchannels) float ndarray, the localization information
                            'creg':             (nspots,) int ndarray, the registration of each fit to different channels, -1 for failed registration, -1 for failed registration
                            'crng':             (nfluorophores, 2) float ndarray, the left and right bound of the log_diff between the photons in the two channels for each fluorophore 
        phot_min/_max:      float, the minimum / maximum number of photons for analysis
        frac_min/_max:      float, ratios are clipped into frac_min and frac_max 
        nbins:              int, number of bins for histogram (10*nbins for log_diff)
    RETURN:
        fig:                matplotlib figure handles
    """    
    # Parse the inputs
    ndim        = smlm_data['ndim']
    nchannels   = smlm_data['nchannels']
    xvec        = smlm_data['xvec']
    crng        = smlm_data['crng']
    creg        = smlm_data['creg']
    
    phot = xvec[:, ndim]
    frac = xvec[:, ndim+1]
    phot_ch0 = phot * frac
    phot_ch1 = phot - phot_ch0

    phot_min = max(phot_min, 1e2)
    phot_max = min(phot_max, 2e5)
    frac_min = max(frac_min, 1e-3)
    frac_max = min(frac_max, 0.999)
    
    assert nchannels == 2, "nchannels mismatch: input.nchannels={}, required=2".format(nchannels)
    ind_all = creg > -1
    assert ind_all.sum() > 0, "ch_reg fails: None of the input smlm_data is registered to a fluorophore"
    
    log_phot = np.log10(np.clip(phot, phot_min, phot_max))
    log_phot_ch0 = np.log10(np.clip(phot_ch0, phot_min, phot_max))
    log_phot_ch1 = np.log10(np.clip(phot_ch1, phot_min, phot_max))
    frac = np.clip(frac, frac_min, frac_max)

    # Histogram
    logp_bins = np.linspace(np.log10(phot_min), np.log10(phot_max), num=nbins+1, endpoint=True)
    frac_bins = np.linspace(frac_min, frac_max, num=nbins+1, endpoint=True)
    hist_logp_frac = np.histogram2d(log_phot, frac, bins=[logp_bins, frac_bins])[0]
    hist_logp_logp = np.histogram2d(log_phot_ch0, log_phot_ch1, bins=[logp_bins, logp_bins])[0]
        
    # Plot
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(scrWidth/72, scrHeight/72))
    ax0.pcolormesh(frac_bins[:-1], logp_bins[:-1], hist_logp_frac, cmap='turbo', shading='auto')
    for i, dum_crng in enumerate(crng):
        ax0.vlines(dum_crng[0], logp_bins[0], logp_bins[-1], color=TAB_COLORS[i], linewidth=1)
        ax0.vlines(dum_crng[1], logp_bins[0], logp_bins[-1], color=TAB_COLORS[i], linewidth=0.5)
    ax0.set_xlim(0.0, 1.0)
    ax0.set_ylim(logp_bins[0], logp_bins[-1])
    ax0.set_xlabel('frac')
    ax0.set_ylabel('log10(photon)')

    ax1.pcolormesh(logp_bins[:-1], logp_bins[:-1], hist_logp_logp, cmap='turbo', shading='auto')
    for i, dum_crng in enumerate(crng):
        dumx = [logp_bins[0], logp_bins[-1]]
        dumy = [logp_bins[0] + np.log10(dum_crng[0]/(1.0-dum_crng[0])), logp_bins[-1] + np.log10(dum_crng[0]/(1.0-dum_crng[0]))]
        ax1.plot(dumx, dumy, color=TAB_COLORS[i], linewidth=1)
        dumy = [logp_bins[0] + np.log10(dum_crng[1]/(1.0-dum_crng[1])), logp_bins[-1] + np.log10(dum_crng[1]/(1.0-dum_crng[1]))]
        ax1.plot(dumx, dumy, color=TAB_COLORS[i], linewidth=0.5)
    ax1.set_xlim(logp_bins[0], logp_bins[-1])
    ax1.set_ylim(logp_bins[0], logp_bins[-1])
    ax1.set_xlabel('log10(photon_ch1)')
    ax1.set_ylabel('log10(photon_ch0)')
        
    return fig




if __name__ == '__main__':
    pass