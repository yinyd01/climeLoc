import os
import pickle
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline
from matplotlib import pyplot as plt
import time


################################ Correlation Computation ################################
def _sphe2cart(radius, phi, theta):
    """ (x, y, z) = _sphe2cart(r, phi, theta) conversion from spherial to cartesian """
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return x, y, z


def _pol2cart(radius, theta):
    """ (x, y) = _pol2cart(r, theta) conversion from polar to cartesian """
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def _get_norm2d(roisz, locs, redges, ntheta=180):
    """
    get the normalization term at the radius for each localization
    INPUTS:
        roi:        (2,) float, (roiszy, roiszx)
        locs:       (nspots, ndim) float, the [[x, y, z],...] localizations (ndim >= 2)
        redges:     (nbins + 1) float, the edges for radial bins, must be sorted
        ntheta:     int, number of theta values for linspace(0, 2*pi, num=ntheta, endpoint=False)
    RETURN:
        rnorm:      (nspots, nbins) float, the normalization term for each localization at each radius
    """
    nspots = len(locs)
    maxR = redges[-1]
    nbins = len(redges) - 1
    r_centers = 0.5 * (redges[:-1] + redges[1:])
    
    theta = np.linspace(-np.pi, np.pi, num=ntheta, endpoint=False)
    Rs, THETAs = np.meshgrid(r_centers, theta, indexing='ij')
    tmpXX, tmpYY = _pol2cart(Rs, THETAs)

    rvol = 2.0 * np.pi * r_centers * (redges[1:] - redges[:-1])
    rnorm = np.tile(rvol, nspots).reshape(nspots, nbins)
    dumind = np.where((locs[:,0]<maxR) | (locs[:,0]>roisz[0]-maxR) | (locs[:,1]<maxR) | (locs[:,1]>roisz[1]-maxR))[0]
    for i in dumind:
        XX = locs[i, 0] + tmpXX
        YY = locs[i, 1] + tmpYY
        ind_inCanvas = (XX > 0) & (XX < roisz[0]) & (YY > 0) & (YY < roisz[1])
        frac_inCanvas = np.sum(ind_inCanvas, axis=-1) / ntheta
        rnorm[i] *= frac_inCanvas    
    return rnorm

    
def _get_norm3d(roisz, locs, redges, nphi=90, ntheta=180):
    """
    get the normalization term at the radius for each localization
    INPUTS:
        roi:        (3,) float, (roiszz, roiszy, roiszx)
        locs:       (nspots, ndim) float, the [[x, y, z],...] localizations (ndim >= 3)
        redges:     (nbins + 1) float, the edges for radial bins, must be sorted
        nphi:       int, number of phi values for linspace(0, pi, num=nphi, endpoint=False)
        ntheta:     int, number of theta values for linspace(-pi, pi, num=ntheta, endpoint=False)
    RETURN:
        rnorm:      (nspots, nbins) float, the normalization term for each localization at each radius
    """
    nspots = len(locs)
    maxR = redges[-1]
    nbins = len(redges) - 1
    r_centers = 0.5 * (redges[:-1] + redges[1:])
    
    theta = np.linspace(-np.pi, np.pi, num=ntheta, endpoint=False)
    phi = np.linspace(0, np.pi, num=nphi, endpoint=False)
    Rs, PHIs, THETAs = np.meshgrid(r_centers, phi, theta, indexing='ij')
    tmpXX, tmpYY, tmpZZ = _sphe2cart(Rs, PHIs, THETAs)

    rvol = 4.0 / 3.0 * np.pi * (redges[1:]**3 - redges[:-1]**3)
    rnorm = np.tile(rvol, nspots).reshape(nspots, nbins)
    dumind = np.where((locs[:,0]<maxR) | (locs[:,0]>roisz[0]-maxR) | (locs[:,1]<maxR) | (locs[:,1]>roisz[1]-maxR) | (locs[:,2]<maxR) | (locs[:,2]>roisz[2]-maxR))[0]
    for i in dumind:
        XX = locs[i, 0] + tmpXX
        YY = locs[i, 1] + tmpYY
        ZZ = locs[i, 2] + tmpZZ
        ind_inCanvas = (XX > 0) & (XX < roisz[0]) & (YY > 0) & (YY < roisz[1]) & (ZZ > 0) & (ZZ < roisz[2])
        frac_inCanvas = np.sum(np.sum(ind_inCanvas, axis=-1), axis=-1) / ntheta / nphi
        rnorm[i] *= frac_inCanvas 
    return rnorm 


def pcorr_cc(ndim, locs_A, locs_B, redges):
    """
    calculate radial distribution function
    for each loc_A in locs_A, the radial distribution of locs_B from loc_A is calculated by rho_B(r) / <rho_B>
    the radial distribution of loc_A are then averaged and returned
    INPUTS:
        ndim:       int, number of dimensions, must be in {2, 3}
        locs_A:     (nspots, ndim) float, the [[x, y, z],...] localizations
        locs_B:     (nspots, ndim) float, the [[x, y, z],...] localizations
        redges:     (nbins + 1,) float, the edges for radial bins, must be sorted
    RETURN:
        pcorr:      (nbins,) float, the paircorrelation at each r_center
        r_center:   (nbins,) float, the r center defined by the redges
    """
    assert locs_A.shape[1] >= ndim, "ndim mismatch: locs_A.shape[1]={}, input.ndim={}".format(locs_A.shape[1], ndim)
    assert locs_B.shape[1] >= ndim, "ndim mismatch: locs_B.shape[1]={}, input.ndim={}".format(locs_B.shape[1], ndim)
    npoints_A = len(locs_A)
    npoints_B = len(locs_B)
    
    roisz = np.zeros(ndim)
    _locs_A = locs_A[:, :ndim].copy()
    _locs_B = locs_B[:, :ndim].copy()
    for dimID in range(ndim):
        loc_max = max(_locs_A[:, dimID].max(), _locs_B[:, dimID].max())
        loc_min = min(_locs_A[:, dimID].min(), _locs_B[:, dimID].min())
        roisz[dimID] = loc_max - loc_min
        _locs_A[:, dimID] -= loc_min
        _locs_B[:, dimID] -= loc_min
    
    rho_B = npoints_B / np.prod(roisz)
    nbins = len(redges) - 1

    kdt_A = cKDTree(_locs_A)
    kdt_B = cKDTree(_locs_B)
    fnorm = _get_norm2d if ndim == 2 else _get_norm3d
    rnorm_A = fnorm(roisz, _locs_A, redges)
    dumind = rnorm_A > 0.0

    rneighbors = np.zeros((npoints_A, nbins + 1))
    for i, r in enumerate(redges):
        indexes = kdt_A.query_ball_tree(kdt_B, r)
        rneighbors[:, i] = np.array([len(idx) for idx in indexes])
    pcorr = rneighbors[:, 1:] - rneighbors[:, :-1]
    pcorr[dumind] /= rnorm_A[dumind]
    pcorr[~dumind] = 0.0
    pcorr = np.mean(pcorr, axis=0) / rho_B
    return pcorr, npoints_A/np.prod(roisz), rho_B


def pcorr_ac(ndim, locs, redges):
    return pcorr_cc(ndim, locs, locs, redges)[0::2]



################################ Correlation Fit ################################
def _func_gauss_gauss(xdata, *p):
    """sum of two gaussian"""
    A1, s1, A2, ds = p
    s2 = s1 + ds
    norm1 = np.sqrt(2.0 * np.pi) * s1
    norm2 = np.sqrt(2.0 * np.pi) * s2 / np.sqrt(2)      # s2 = np.sqrt(2) * sigma
    return A1/norm1 * np.exp(-0.5*xdata*xdata/s1/s1) + A2/norm2 * np.exp(-xdata*xdata/s2/s2) + 1.0


def _func_gauss_exp(xdata, *p):
    """sum of a gaussian and an exponential"""
    A1, s1, A2, ds = p
    s2 = s1 + ds
    norm = np.sqrt(2.0 * np.pi) * s1
    return A1/norm * np.exp(-0.5*xdata*xdata/s1/s1) + A2/s2 * np.exp(-xdata/s2) + 1.0


def _func_gauss_stretchexp(xdata, *p):
    """sum of a gaussian and a stretched exponential"""
    A1, s1, A2, ds, beta = p
    s2 = s1 + ds
    norm1 = np.sqrt(2.0 * np.pi) * s1
    norm2 = s2 * gamma(1.0 + 1.0 / beta)
    return A1/norm1 * np.exp(-0.5*xdata*xdata/s1/s1) + A2/norm2 * np.exp(-(xdata/s2)**beta) + 1.0


def _func_gauss_sym(xdata, *p):
    """a gaussian"""
    A, c, s = p
    norm = np.sqrt(2.0 * np.pi) * s
    return A/norm * np.exp(-0.5*(xdata-c)*(xdata-c)/s/s) + 1.0


def _func_gauss_asym(xdata, *p):
    """a gaussian"""
    A, c, s1, s2 = p
    norm = np.sqrt(2.0 * np.pi) * 0.5 * (s1 + s2)
    ydata = np.zeros_like(xdata)
    ydata[xdata<=c] = A/norm * np.exp(-0.5*(xdata[xdata<=c]-c)*(xdata[xdata<=c]-c)/s1/s1) + 1.0
    ydata[xdata>c] = A/norm * np.exp(-0.5*(xdata[xdata>c]-c)*(xdata[xdata>c]-c)/s2/s2) + 1.0
    return ydata


def _get_fitfunc(functype, funcname):
    """get the fitting function"""
    if functype == 'ac':
        if funcname == 'gauss':
            return _func_gauss_gauss
        elif funcname == 'exp':
            return _func_gauss_exp
        elif funcname == 'strexp':
            return _func_gauss_stretchexp
        else:
            raise ValueError("unsupported funcname for functype='ac': supported are ('gauss', 'exp', 'strexp')")
    elif functype == 'cc':
        if funcname == 'gauss':
            return _func_gauss_sym
        elif funcname == 'agauss':
            return _func_gauss_asym
        else:
            raise ValueError("unsupported funcname for functype='cc': supported are ('gauss', 'agauss')")
    else:
        raise ValueError("unsupported functype: supported are ('ac', 'cc')")


def acorr_fit(r_center_nm, acorr, funcname):
    """
    fit the auto-pcorr
    INPUTS:
        r_center_nm:    (nbins,) float ndarray, xdata for fit, in nanometer
        acorr:          (nbins,) float ndarray, auto-correlation ydata for fit
        funcname:       str, {'gauss', 'exp', 'strexp'} functions for autocorrelation fit
    RETURN:
        popt:           (vnum,) float ndarray, optimized parameters fitting the acorr
        pstd:           (vnum,) float ndarray, std of the popt 
    """
    s1_guess = 20.0     # approximate localization uncertainty in nanometer
    ds_guess = max(np.sum(r_center_nm * acorr) / np.sum(acorr) - s1_guess, 0.0)
    A1_guess = A2_guess = max(0.5 * np.sum(acorr - 1.0), 0.0)
    pini = [A1_guess, s1_guess, A2_guess, ds_guess]
    p_lb = [0.0, 2.0, 0.0, 0.0]
    p_ub = [np.inf, np.inf, np.inf, np.inf]
    if funcname == 'strexp':
        pini.append(1.5)
        p_lb.append(0.5)
        p_ub.append(4.0)

    _func = _get_fitfunc('ac', funcname)
    popt, pcov = curve_fit(_func, r_center_nm, acorr, p0=np.array(pini), bounds=(p_lb, p_ub))
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def ccorr_fit(r_center, ccorr, funcname):
    """
    fit the auto-pcorr
    INPUTS:
        r_center_nm:    (nbins,) float ndarray, xdata for fit
        ccorr:          (nbins,) float ndarray, cross-correlation ydata for fit
        funcname:       str, {'gauss', 'agauss'} functions for autocorrelation fit
    RETURN:
        popt:           (vnum,) float ndarray, optimized parameters fitting the acorr
        pstd:           (vnum,) float ndarray, std of the popt 
    """
    w = (ccorr - 1.0) * (ccorr - 1.0)
    c_guess = np.sum(r_center * w) / np.sum(w)
    s_guess = np.sqrt(np.sum(r_center**2 * w) / np.sum(w) - c_guess**2)
    A_guess = np.sum(ccorr - 1.0)
    pini = [A_guess, c_guess, s_guess]
    p_lb = [-np.inf, 0.0, 0.0]
    p_ub = [np.inf, np.inf, np.inf]
    if funcname == 'agauss':
        pini.append(s_guess)
        p_lb.append(0.0)
        p_ub.append(np.inf)
        
    _func = _get_fitfunc('cc', funcname)
    popt, pcov = curve_fit(_func, r_center, ccorr, p0=np.array(pini), bounds=(p_lb, p_ub))
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def pcorr_calc(r_center_nm, p, functype, funcname):
    """draw an paircorrelation"""
    _func = _get_fitfunc(functype, funcname)
    return _func(r_center_nm, *p)




################################ Correlation Smoothing, Parameterization, Visualization ################################
def pcorr_smooth(r_center, pcorr_):
    """smooth the pcorr_ via spline representation"""
    tck = splrep(r_center, pcorr_, w=np.ones(len(r_center)), s=len(r_center)-np.sqrt(2.0*len(r_center)))
    return BSpline(*tck)(r_center)


def pcorr_par(redges, pcorr_):
    """
    get the descriptive parameters of pair correlation profile
    RETURN:
        Intensity:      float, integral of the correlation profile over the given redges
        radius:         float, average radius of the pcorr_ (average of the rcenters weighted by pcorr_) 
    """
    bins = redges[1:] - redges[:-1]
    rcenters = 0.5 * (redges[1:] + redges[:-1])
    Intensity = np.sum((pcorr_ - 1.0) * bins)
    w = (pcorr_ - 1.0) ** (pcorr_ - 1.0)
    radius = np.sum(rcenters * w) / np.sum(w)
    return Intensity, radius


def pcorr_visual(r_center_nm, pcorr_, pcorr_fit):
    """
    Visulization of the correlation profile
    RETURN:
        pcorr_fig:              matplotlib pyplot object ploting optics cluster analysis
    """
    fig = plt.figure(figsize=(11, 8.5), tight_layout=True)
    ax = fig.add_subplot()
    ax.plot(r_center_nm, pcorr_, color='tab:blue', ls='-', marker='o', mfc='None')
    ax.plot(r_center_nm, pcorr_fit, color='tab:red')
    ax.hlines(1.0, r_center_nm.min(), r_center_nm.max())
    ax.set_xscale('log')
    return fig



################################ Auto Correlation Kernels ################################
def acorr_kernel(locs_nm, redges_nm, fit=True, funcname='gauss'):
    """
    kernel function for runing the Auto-Correlation and related functions for a set of locs
    INPUT:
        locs_nm:    (nspots, ndim) float ndarray, localizations in nanometer
        redges_nm:  (nbins + 1) float ndarray, the edges for radial bins, must be sorted
    RETURN:
        integrated dictionary for autocorrelation info
    """
    ndim = len(locs_nm[0])
    rcenters = 0.5 * (redges_nm[1:] + redges_nm[:-1])

    acorr, rho = pcorr_ac(ndim, locs_nm, redges_nm)
    intensity, radius = pcorr_par(redges_nm, acorr)
    if fit:
        popt, perr = acorr_fit(rcenters, acorr, funcname)
        _acorr_fit = pcorr_calc(rcenters, popt, 'ac', funcname)
    else:
        _acorr_fit = pcorr_smooth(rcenters, acorr)
    fig_acorr = pcorr_visual(rcenters, acorr, _acorr_fit)
    
    AutoCorrelationData = {'acorr':acorr[...,np.newaxis], 'acorr_fit':_acorr_fit[...,np.newaxis]}
    AutoCorrelationSumm = {'rho':rho,  'intensity':intensity, 'radius':radius}
    if fit:
        AutoCorrelationFits = {'Amp1':popt[0], 'r1':popt[1], 'Amp2':popt[2], 'r2':popt[1]+popt[3]}
        AutoCorrelationFitsErr = {'Amp1_err':perr[0], 'r1_err':perr[1], 'Amp2_err':perr[3], 'r2_err':np.sqrt(perr[1]**2+perr[3]*2)}
        if funcname == 'strexp':
            AutoCorrelationFits['beta'] = popt[4]
            AutoCorrelationFitsErr['beta_err'] = perr[4]
    else:
        AutoCorrelationFits = None
        AutoCorrelationFitsErr = None
    return AutoCorrelationData, AutoCorrelationSumm, AutoCorrelationFits, AutoCorrelationFitsErr, fig_acorr



def acorr_batch(sample_fpath, ndim, redges_nm, fit=True, funcname='gauss', version='1.11'):
    """
    batch function for runing the Auto-Correlation kernel through all the rois in one sample folder
    INPUT:
        sample_fpath:       str, the sample path in wich the spool_fpath/smlm_results stores the _locsnm.pkl file of each of the rois
        ndim:               int, number of dimensions 
    RETURN:
        integrated dictionary for autocorrelation info
    """
    nbins = len(redges_nm) - 1
    kw_data     = ['acorr', 'acorr_fit']
    kw_summ     = ['rho', 'intensity', 'radius']
    kw_fits     = ['Amp1', 'r1', 'Amp2', 'r2', 'beta'] if funcname == 'strexp' else ['Amp1', 'r1', 'Amp2', 'r2']
    kw_fits_err = [kw + '_err' for kw in kw_fits]

    AutoCorrelationData     = [{}, {}, {}, {}]
    AutoCorrelationSumm     = [{}, {}, {}, {}]
    AutoCorrelationFits     = [{}, {}, {}, {}] if fit else None
    AutoCorrelationFitsErr  = [{}, {}, {}, {}] if fit else None
    for i in range(4):
        for key in kw_data:
            AutoCorrelationData[i][key] = np.zeros((nbins, 0), dtype=np.float64)

        AutoCorrelationSumm[i]['roiname'] = np.array([], dtype=str)
        for key in kw_summ:
            AutoCorrelationSumm[i][key] = np.array([], dtype=np.float64)
        
        if fit:
            AutoCorrelationFits[i]['roiname'] = np.array([], dtype=str)
            for key in kw_fits:
                AutoCorrelationFits[i][key] = np.array([], dtype=np.float64)
            
            AutoCorrelationFitsErr[i]['roiname'] = np.array([], dtype=str)
            for key in kw_fits_err:
                AutoCorrelationFitsErr[i][key] = np.array([], dtype=np.float64)

    # file name collection
    spool_fpaths = [spool_fpath for spool_fpath in os.listdir(sample_fpath) if spool_fpath.startswith('spool_') and os.path.isdir(os.path.join(sample_fpath, spool_fpath))]
    if len(spool_fpaths) == 0:
        print("no spool folders found in {}".format(sample_fpath))
        return AutoCorrelationData, AutoCorrelationSumm, AutoCorrelationFits, AutoCorrelationFitsErr
    
    for spool_fpath in spool_fpaths:
        
        spool_fpath = os.path.join(sample_fpath, spool_fpath)
        if not os.path.isdir(os.path.join(spool_fpath, 'smlm_result')):
            print("the smlm_result folder is not found in {}".format(spool_fpath))
            continue
        
        spool_fpath = os.path.join(spool_fpath, 'smlm_result')
        smlm_loc_fnames = [smlm_loc_fname for smlm_loc_fname in os.listdir(spool_fpath) if smlm_loc_fname.endswith('_locsnm.pkl')]
        if len(smlm_loc_fnames) == 0:
            print("no _locsnm.pkl files found in {}".format(spool_fpath))
            continue

        # acorr kernel
        for smlm_loc_fname in smlm_loc_fnames:
            
            print("Auto-Correlation on {:s}...".format(smlm_loc_fname))
            with open(os.path.join(spool_fpath, smlm_loc_fname), 'rb') as fid:
                smlm_data = pickle.load(fid)
                assert ndim <= smlm_data['ndim'], "ndim mismatch, the input.ndim={}, the smlm_data.ndim={}".format(ndim, smlm_data['ndim'])
                creg = smlm_data['creg'] 
            
            fluorophores = set(creg)
            if -1 in fluorophores:
                fluorophores.remove(-1)
            
            for fluor in fluorophores:
                
                tas = time.time()
                locs_nm = smlm_data['xvec'][creg==fluor, :ndim] if version=='1.11' else smlm_data['xvec'][:ndim, creg==fluor].T
                ACData, ACSumm, ACFits, ACFitsErr, fig_acorr = acorr_kernel(locs_nm, redges_nm, fit, funcname)
                print("\t{:0.4f} secs elapsed for auto-correlation for fl{}, n{:d}".format(time.time()-tas, fluor, len(locs_nm)))
                
                plt.savefig(os.path.join(spool_fpath, smlm_loc_fname[:-11]+'_acorr_fl{}.png'.format(fluor)), dpi=300)
                plt.close(fig_acorr)

                for key in ACData.keys():
                    AutoCorrelationData[fluor][key] = np.hstack((AutoCorrelationData[fluor][key], ACData[key]))

                AutoCorrelationSumm[fluor]['roiname'] = np.hstack((AutoCorrelationSumm[fluor]['roiname'], smlm_loc_fname[:-11]))
                for key in ACSumm.keys():
                    AutoCorrelationSumm[fluor][key] = np.hstack((AutoCorrelationSumm[fluor][key], ACSumm[key]))
                
                if fit:
                    AutoCorrelationFits[fluor]['roiname'] = np.hstack((AutoCorrelationFits[fluor]['roiname'], smlm_loc_fname[:-11]))
                    for key in ACFits.keys():
                        AutoCorrelationFits[fluor][key] = np.hstack((AutoCorrelationFits[fluor][key], ACFits[key]))
                    
                    AutoCorrelationFitsErr[fluor]['roiname'] = np.hstack((AutoCorrelationFitsErr[fluor]['roiname'], smlm_loc_fname[:-11]))
                    for key in ACFitsErr.keys():
                        AutoCorrelationFitsErr[fluor][key] = np.hstack((AutoCorrelationFitsErr[fluor][key], ACFitsErr[key]))
            
    return AutoCorrelationData, AutoCorrelationSumm, AutoCorrelationFits, AutoCorrelationFitsErr



################################ Cross Correlation Kernels ################################
def ccorr_kernel(locs_A, locs_B, redges, fit=True, funcname='gauss'):
    """
    kernel function for runing the Cross-Correlation and related functions for a set of locs
    INPUT:
        locs:       (nspots, ndim) float ndarray, localizations in nanometer
        redge:      (nbins + 1) float ndarray, the edges for radial bins, must be sorted
    RETURN:
        integrated dictionary for crosscorrelation info
    """
    assert locs_A.shape[1] == locs_B.shape[1], "ndim mismatch: locs_A.shape[1]={}, locs_B.shape[1]={}".format(locs_A.shape[1], locs_B.shape[1])
    ndim = locs_A.shape[1]
    rcenters = 0.5 * (redges[1:] + redges[:-1])

    ccorr, rho_A, rho_B = pcorr_cc(ndim, locs_A, locs_B, redges)
    intensity, radius = pcorr_par(redges, ccorr)
    if fit:
        popt, perr = ccorr_fit(rcenters, ccorr, funcname)
        _ccorr_fit = pcorr_calc(rcenters, popt, 'cc', funcname)
    else:
        _ccorr_fit = pcorr_smooth(rcenters, ccorr)
    fig_ccorr = pcorr_visual(rcenters, ccorr, _ccorr_fit)
    
    CrossCorrelationData = {'ccorr':ccorr[...,np.newaxis], 'ccorr_fit':_ccorr_fit[...,np.newaxis]}
    CrossCorrelationSumm = {'rho_A':rho_A, 'rho_B':rho_B, 'intensity':intensity, 'radius':radius}
    if fit:
        CrossCorrelationFits = {'Amp':popt[0], 'c':popt[1], 's1':popt[2]}
        CrossCorrelationFitsErr = {'Amp_err':perr[0], 'c_err':perr[1], 's1_err':perr[2]}
        if funcname == 'agauss':
            CrossCorrelationFits['s2'] = popt[3]
            CrossCorrelationFitsErr['s2_err'] = perr[3]
    else:
        CrossCorrelationFits = None
        CrossCorrelationFitsErr = None
        
    return CrossCorrelationData, CrossCorrelationSumm, CrossCorrelationFits, CrossCorrelationFitsErr, fig_ccorr



def ccorr_batch(sample_fpath, ndim, redges_nm, flID_A, flID_B, fit=True, funcname='gauss', version='1.11'):
    """
    batch function for runing the Cross-Correlation kernel through all the rois in one sample folder
    INPUT:
        sample_fpath:       str, the sample path in wich the spool_fpath/smlm_results stores the _locsnm.pkl file of each of the rois
        ndim:               int, number of dimensions
        flID_A:             int, the fluorophore ID for A
        flID_B:             int, the fluorophore ID for B 
    RETURN:
        integrated dictionary for crosscorrelation info
    """
    nbins = len(redges_nm) - 1
    kw_data     = ['ccorr', 'ccorr_fit']
    kw_summ     = ['rho_A', 'rho_B', 'intensity', 'radius']
    kw_fits     = ['Amp', 'c', 's1', 's2'] if funcname == 'agauss' else ['Amp', 'c', 's1']
    kw_fits_err = [kw + '_err' for kw in kw_fits]

    CrossCorrelationData     = dict()
    CrossCorrelationSumm     = dict()
    CrossCorrelationFits     = dict() if fit else None
    CrossCorrelationFitsErr  = dict() if fit else None
    for key in kw_data:
        CrossCorrelationData[key] = np.zeros((nbins, 0), dtype=np.float64)
    
    CrossCorrelationSumm['roiname'] = np.array([], dtype=str)
    for key in kw_summ:
        CrossCorrelationSumm[key] = np.array([], dtype=np.float64)
    
    if fit:
        CrossCorrelationFits['roiname'] = np.array([], dtype=str)
        for key in kw_fits:
            CrossCorrelationFits[key] = np.array([], dtype=np.float64)
        
        CrossCorrelationFitsErr['roiname'] = np.array([], dtype=str)
        for key in kw_fits_err:
            CrossCorrelationFitsErr[key] = np.array([], dtype=np.float64)

    # file name collection
    spool_fpaths = [spool_fpath for spool_fpath in os.listdir(sample_fpath) if spool_fpath.startswith('spool_') and os.path.isdir(os.path.join(sample_fpath, spool_fpath))]
    if len(spool_fpaths) == 0:
        print("no spool folders found in {}".format(sample_fpath))
        return CrossCorrelationData, CrossCorrelationSumm, CrossCorrelationFits, CrossCorrelationFitsErr

    for spool_fpath in spool_fpaths:

        spool_fpath = os.path.join(sample_fpath, spool_fpath)
        if not os.path.isdir(os.path.join(spool_fpath, 'smlm_result')):
            print("the smlm_result folder is not found in {}".format(spool_fpath))
            continue
        
        spool_fpath = os.path.join(spool_fpath, 'smlm_result')
        smlm_loc_fnames = [smlm_loc_fname for smlm_loc_fname in os.listdir(spool_fpath) if smlm_loc_fname.endswith('_locsnm.pkl')]
        if len(smlm_loc_fnames) == 0:
            print("no _locsnm.pkl files found in {}".format(spool_fpath))
            continue

        # ccorr kernel
        for smlm_loc_fname in smlm_loc_fnames:
            
            print("Cross-Correlation on {:s}...".format(smlm_loc_fname))
            with open(os.path.join(spool_fpath, smlm_loc_fname), 'rb') as fid:
                smlm_data = pickle.load(fid)
                assert ndim <= smlm_data['ndim'], "ndim mismatch, the input.ndim={}, the smlm_data.ndim={}".format(ndim, smlm_data['ndim'])
                creg = smlm_data['creg'] 
            
            fluorophores = set(creg)
            if -1 in fluorophores:
                fluorophores.remove(-1)
            assert flID_A in fluorophores, "flID_A={}, not in the registered fluorophores={}".format(flID_A, fluorophores)
            assert flID_B in fluorophores, "flID_B={}, not in the registered fluorophores={}".format(flID_B, fluorophores)
            
            tas = time.time()
            locs_A = smlm_data['xvec'][creg==flID_A, :ndim] if version=='1.11' else smlm_data['xvec'][:ndim, creg==flID_A].T
            locs_B = smlm_data['xvec'][creg==flID_B, :ndim] if version=='1.11' else smlm_data['xvec'][:ndim, creg==flID_B].T
            CCData, CCSumm, CCFits, CCFitsErr, fig_ccorr = ccorr_kernel(locs_A, locs_B, redges_nm, fit, funcname)
            print("\t{:0.4f} secs elapsed for cross-correlation for n{:d} x n{:d}".format(time.time()-tas, len(locs_A), len(locs_B)))
            
            plt.savefig(os.path.join(spool_fpath, smlm_loc_fname[:-11]+'_ccorr_fl{}_{}.png'.format(flID_A, flID_B)), dpi=300)
            plt.close(fig_ccorr)

            for key in CCData.keys():
                CrossCorrelationData[key] = np.hstack((CrossCorrelationData[key], CCData[key]))
                
            CrossCorrelationSumm['roiname'] = np.hstack((CrossCorrelationSumm['roiname'], smlm_loc_fname[:-11]))
            for key in CCSumm.keys():
                CrossCorrelationSumm[key] = np.hstack((CrossCorrelationSumm[key], CCSumm[key]))
            
            if fit:
                CrossCorrelationFits['roiname'] = np.hstack((CrossCorrelationFits['roiname'], smlm_loc_fname[:-11]))
                for key in CCFits.keys():
                    CrossCorrelationFits[key] = np.hstack((CrossCorrelationFits[key], CCFits[key]))
                
                CrossCorrelationFitsErr['roiname'] = np.hstack((CrossCorrelationFitsErr['roiname'], smlm_loc_fname[:-11]))
                for key in CCFitsErr.keys():
                    CrossCorrelationFitsErr[key] = np.hstack((CrossCorrelationFitsErr[key], CCFitsErr[key]))
                       
    return CrossCorrelationData, CrossCorrelationSumm, CrossCorrelationFits, CrossCorrelationFitsErr





if __name__ == '__main__':

    """
    import sys
    
    r = 10
    roiszx = 5000
    roiszy = 8000
    roiszz = 2000
    redges = np.arange(101)

    rng = np.random.default_rng()
    
    if sys.argv[1] == 'cc':
        npoints_A = 10000
        if sys.argv[2] == '3':
            locs_A = rng.random(size=(npoints_A, 3)) * np.array([roiszx, roiszy, roiszz])
            
            radius = rng.normal(loc=r, scale=1.0, size=npoints_A)
            theta = 2.0 * np.pi * rng.random(size=npoints_A) - np.pi
            phi = np.pi * rng.random(size=npoints_A)
            locs_B = np.vstack([radius*np.sin(phi)*np.cos(theta), radius*np.sin(phi)*np.sin(theta), radius*np.cos(phi)]).T
            locs_B += locs_A
            dum_ind = (locs_B[:,0]>0) & (locs_B[:,0]<roiszx) & (locs_B[:,1]>0) & (locs_B[:,1]<roiszy) & (locs_B[:,2]>0) & (locs_B[:,2]<roiszz)
            locs_B = locs_B[dum_ind]
            
            locs_B = rng.random(size=(npoints_A, 3)) * np.array([roiszx, roiszy, roiszz])
            import time
            tas = time.time()
            pcorr_, rcenters = pcorr_cc(3, locs_A, locs_B, redges)
            print("3d paircorrelation runtime for {}x{}: {} s.".format(npoints_A, len(locs_B), time.time() - tas))
            popt, perr = ccorr_fit(rcenters, pcorr_, 'agauss')
            print(popt)
            print(perr)
            pcorr_fit = pcorr_calc(rcenters, popt, 'cc', 'agauss')
        
        elif sys.argv[2] == '2':
            locs_A = rng.random(size=(npoints_A, 2)) * np.array([roiszx, roiszy])
            
            radius = rng.normal(loc=r, scale=1.0, size=npoints_A)
            theta = 2.0 * np.pi * rng.random(size=npoints_A) - np.pi
            locs_B = np.vstack([radius*np.cos(theta), radius*np.sin(theta)]).T
            locs_B += locs_A
            dum_ind = (locs_B[:,0]>0) & (locs_B[:,0]<roiszx) & (locs_B[:,1]>0) & (locs_B[:,1]<roiszy)
            locs_B = locs_B[dum_ind]
            
            #locs_B = rng.random(size=(npoints_A, 2)) * np.array([roiszx, roiszy])
            import time
            tas = time.time()
            pcorr_, rcenters = pcorr_cc(2, locs_A, locs_B, redges)
            print("2d paircorrelation runtime for {}x{}: {} s.".format(npoints_A, len(locs_B), time.time() - tas))

    elif sys.argv[1] == 'ac':
        ncluster = 100
        nspots_percluster = 100
        if sys.argv[2] == '3':
            cluster_centers = rng.random(size=(ncluster, 3)) * np.array([roiszx, roiszy, roiszz])
            sigmas = rng.normal(loc=r, scale=1.0, size=ncluster)
            locs = np.zeros((ncluster * nspots_percluster, 3))
            for i in range(ncluster):
                dumlocs = rng.normal(loc=0.0, scale=sigmas[i], size=(nspots_percluster, 3))
                locs[i*nspots_percluster:(i+1)*nspots_percluster] = dumlocs + cluster_centers[i]

            dum_ind = (locs[:,0]>0) & (locs[:,0]<roiszx) & (locs[:,1]>0) & (locs[:,1]<roiszy) & (locs[:,2]>0) & (locs[:,2]<roiszz)
            locs = locs[dum_ind]
            
            #locs = rng.random(size=(ncluster * nspots_percluster, 3)) * np.array([roiszx, roiszy, roiszz])
            import time
            tas = time.time()
            pcorr_, rcenters = pcorr_ac(3, locs, redges)
            print("3d paircorrelation runtime for {}x{}: {:.4f} s.".format(ncluster, nspots_percluster, time.time() - tas))
        
        elif sys.argv[2] == '2':
            cluster_centers = rng.random(size=(ncluster, 2)) * np.array([roiszx, roiszy])
            sigmas = rng.normal(loc=r, scale=1.0, size=ncluster)
            locs = np.zeros((ncluster * nspots_percluster, 2))
            for i in range(ncluster):
                dumlocs = rng.normal(loc=0.0, scale=sigmas[i], size=(nspots_percluster, 2))
                locs[i*nspots_percluster:(i+1)*nspots_percluster] = dumlocs + cluster_centers[i]

            dum_ind = (locs[:,0]>0) & (locs[:,0]<roiszx) & (locs[:,1]>0) & (locs[:,1]<roiszy)
            locs = locs[dum_ind]
            
            #locs = rng.random(size=(ncluster * nspots_percluster, 2)) * np.array([roiszx, roiszy])
            import time
            tas = time.time()
            pcorr_, rcenters = pcorr_ac(2, locs, redges)
            print("2d paircorrelation runtime for {}x{}: {:.4f} s.".format(ncluster, nspots_percluster, time.time() - tas))

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(rcenters, pcorr_, ls='-', marker='o', mfc='None')
    ax.plot(rcenters, pcorr_fit)
    plt.show()
    """

    import sys
    import os
    import pickle
    import matplotlib.pyplot as plt
    import time

    fname1 = 'spool_2D_30ms_500mw_2_2_1__roi_4_locsnm.pkl'
    fname2 = 'C:/Users/hades/Downloads/spool_2D_30ms_800mw_3_3_1_smlm_result/spool_2D_30ms_800mw_3_3_1__roi_9_locsnm.pkl'

    fname = fname1 if sys.argv[1] == '1' else fname2
    ndim = 3 if sys.argv[1] == '1' else 2

    redges = np.array(list(range(0,32,2)) + list(range(32,192,5)) + list(range(192,513,20)))
    rcenters = 0.5 * (redges[1:] + redges[:-1])
    cfluoID = np.array([0, 1])

    with open(fname, 'rb') as fid:
        smlm_data = pickle.load(fid)
        xvec = smlm_data['xvec']
        creg = smlm_data['creg']
        flIDs = set(creg)
        if -1 in flIDs:
            flIDs.remove(-1)

    fig, ax = plt.subplots(1, len(flIDs))
    for i, flID in enumerate(np.array(list(flIDs))):    
        locs = xvec[:ndim, creg==flID].T
        tas = time.time()
        acorr, rho = pcorr_ac(ndim, locs, redges)
        print('{:.4f}'.format(time.time()-tas))
        acorr_smooth = pcorr_smooth(rcenters, acorr)
        popt, perr = acorr_fit(rcenters, acorr, 'gauss')
        acorr_fit_ = pcorr_calc(rcenters, popt, 'ac', 'gauss')
        Amp, radius = pcorr_par(redges, acorr)
        ax[i].plot(rcenters, acorr, ls='-', marker='o', mfc='None')
        ax[i].plot(rcenters, acorr_smooth)
        ax[i].plot(rcenters, acorr_fit_)
        ax[i].text(rcenters.min()+0.8*(rcenters.max()-rcenters.min()), acorr.min()+0.55*(acorr.max()-acorr.min()), '{:.2e}'.format(rho))
        ax[i].text(rcenters.min()+0.8*(rcenters.max()-rcenters.min()), acorr.min()+0.5*(acorr.max()-acorr.min()), '{:.2f}'.format(Amp))
        ax[i].text(rcenters.min()+0.8*(rcenters.max()-rcenters.min()), acorr.min()+0.45*(acorr.max()-acorr.min()), '{:.2f}'.format(radius))
        ax[i].set_xscale('log')
    plt.show()

    locs_A = xvec[:ndim, creg==cfluoID[0]].T
    locs_B = xvec[:ndim, creg==cfluoID[1]].T
    tas = time.time()
    ccorr_A, rhoA, rhoB = pcorr_cc(ndim, locs_A, locs_B, redges)
    ccorr_B = pcorr_cc(ndim, locs_B, locs_A, redges)[0]
    ccorr_smooth = pcorr_smooth(rcenters, ccorr_A)
    print('{:.4f}'.format(time.time()-tas))

    fig, ax = plt.subplots(1, 2)
    for i, ccorr in enumerate([ccorr_A, ccorr_B]):
        Amp, radius = pcorr_par(redges, ccorr)
        ccorr_smooth = pcorr_smooth(rcenters, ccorr)
        popt, perr = ccorr_fit(rcenters, ccorr, 'gauss')
        ccorr_fit_ = pcorr_calc(rcenters, popt, 'cc', 'gauss')
        ax[i].plot(rcenters, ccorr, ls='-', marker='o', mfc='None')
        ax[i].plot(rcenters, ccorr_smooth)
        ax[i].plot(rcenters, ccorr_fit_)
        ax[i].text(rcenters.min()+0.8*(rcenters.max()-rcenters.min()), ccorr.min()+0.5*(ccorr.max()-ccorr.min()), '{:.2f}'.format(Amp))
        ax[i].text(rcenters.min()+0.8*(rcenters.max()-rcenters.min()), ccorr.min()+0.45*(ccorr.max()-ccorr.min()), '{:.2f}'.format(radius))
        ax[i].set_xscale('log')
    plt.show()