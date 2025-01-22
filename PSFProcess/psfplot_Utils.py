from win32api import GetSystemMetrics
import numpy as np
from scipy.special import erf
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import ImageGrid



scrWidth, scrHeight = GetSystemMetrics(0), GetSystemMetrics(1)



def _drawPSF_gauss2d(boxsz, xc, yc, PSFsigmax, PSFsigmay, photon, bkg):
    """
    Given the localizations and intensitys information of an emitter within a SQAURE box from different channels,
    render the expected Gauss intensity (mu) at each pixel within the SQAURE box in different channels
    INPUT:
        boxsz:          int, size of the box
        xc:             float, the x-centers in different channels
        yc:             float, the y-centers in different channels
        PSFsigmax:      float, the sigma (x-axis) in different channels
        PSFsigmay:      float, the sigma (y-axis) in different channels
        photon:         float, the photon number of the psf in different channels
        bkg:            float, the background of the psf in different channels
    RETURN:
        psf:            (boxsz, boxsz) ndarray, expected fluorescent intensity at each pixel
    """
    
    assert boxsz % 2 != 0, "boxsz not odd: boxsz={}".format(boxsz)
    YY, XX = np.meshgrid(np.arange(boxsz), np.arange(boxsz), indexing='ij')
    delta_xu = (XX + 1 - xc) / PSFsigmax
    delta_yu = (YY + 1 - yc) / PSFsigmay
    delta_xl = (XX - xc) / PSFsigmax
    delta_yl = (YY - yc) / PSFsigmay

    PSFx = 0.5 * (erf(delta_xu / np.sqrt(2)) - erf(delta_xl / np.sqrt(2)))
    PSFy = 0.5 * (erf(delta_yu / np.sqrt(2)) - erf(delta_yl / np.sqrt(2)))
    psf = photon * PSFx * PSFy + bkg
    
    return psf



def _drawPSF_cspline2d(splinesz, coeffs):
    """
    draw a 2d psf with the cubic spline coefficients
    INPUT:
        splinesz:   [splinesz_y, splinesz_x] ndarray, size of the 2D-spline
        coeffs:     (splinesz.prod()*16) ndarray, the coefficient of the spline. see _get_cspline2d for indexing
    RETURN:
        psf:        (splinesz_y, splinesz_x) ndarray, the psf
    """
    
    szy, szx = splinesz
    assert len(coeffs) == szy*szx*16, "coeff size mismatch: len(coeff)={}, splinesz={}".format(len(coeffs), splinesz)

    psf = np.zeros((szy, szx))
    for ind_y in range(szy):
        for ind_x in range(szx):
            
            spID = ind_y*szx + ind_x
            model = 0.0
            cy = 1.0
            for i in range(4):
                cx = 1.0
                for j in range(4):
                    model += coeffs[spID*16 + i*4+j] * cy*cx
                    cx *= 0.5
                cy *= 0.5
            
            psf[ind_y, ind_x] = model
    
    return psf



def _drawPSF_cspline3d(splinesz, coeff):
    """
    draw a 3d psf with the cubic spline coefficients
    INPUT:
        splinesz:   [splinesz_z, splinesz_y, splinesz_x] ndarray, size of the 3D-spline
        coeff:      (splinesz.prod()*64) ndarray, the coefficient of the spline. see _get_cspline3d for indexing
    RETURN:
        psf:        (splineszz, splineszy, splineszx) ndarray, the model psf
    """

    szz, szy, szx = splinesz
    assert len(coeff) == szz*szy*szx*64, "coeff size mismatch: len(coeff)={}, splinesz={}".format(len(coeff), splinesz)
        
    psf = np.zeros((szz, szy, szx))
    for ind_z in range(szz):
        for ind_y in range(szy):
            for ind_x in range(szx):
                
                spID = ind_z*szy*szx + ind_y*szx + ind_x
                model = 0.0
                cz = 1.0
                for i in range(4):
                    cy = 1.0
                    for j in range(4):
                        cx = 1.0
                        for k in range(4):
                            model += coeff[spID*64 + i*16+j*4+k] * cz*cy*cx
                            cx *= 0.5
                        cy *= 0.5
                    cz *= 0.5
                
                psf[ind_z, ind_y, ind_x] = model    
    return psf



def astigmatic(zdata, sigmax0, sigmay0, shiftx, shifty, dof, Ax, Bx, Ay, By):
    """
    To build the Astigmatic model
    INPUT:
        zdata:      (ndata,) ndarray, the user input z position array (sorted)
        sigmax0:    float, the minimum gauss-sigma (x-axis) of the psf
        sigmay0:    float, the minimum gauss-sigma (y-axis) of the psf
        shiftx:     float, the z-position where the gauss-sigma (x-axis) of the psf drops to minimum
        shifty:     float, the z-position where the gauss-sigma (y-axis) of the psf drops to minimum
        dof:        float, the depth of the field
        Ax:         float, the coefficient for the 3rd-order correction of the astigmatic (x-axis) profile
        Bx:         float, the coefficient for the 4th-order correction of the astigmatic (x-axis) profile
        Ay:         float, the coefficient for the 3rd-order correction of the astigmatic (y-axis) profile
        By:         float, the coefficient for the 4th-order correction of the astigmatic (y-axis) profile
    RETURN:
        model:      (2*ndata,) ndarray, the Astigmatic model for x and y, respectively
    """
    zx = (zdata - shiftx) / dof
    zy = (zdata - shifty) / dof
    model = np.zeros(2 * len(zdata))
    model[: len(zdata)] = sigmax0 * np.sqrt(np.maximum(1.0 + zx*zx + Ax*zx*zx*zx + Bx*zx*zx*zx*zx, 1.0))
    model[len(zdata) : 2*len(zdata)] = sigmay0 * np.sqrt(np.maximum(1.0 + zy*zy + Ay*zy*zy*zy + By*zy*zy*zy*zy, 1.0)) 
    return model



#################### PLOT FUNCTIONS ####################
def plot_psf_gauss2d(Calibration):
    """
    Plot the gauss form of the calibrated 2d-psf 
    INPUT:      Calibration:        dictionary, contains calibration information
    RETURN:     fig_gauss:          matplotlib figure handle, ploting gauss calibration result
    """

    ##### PARSE THE INPUT #####
    nchannels       = Calibration['nchannels']
    boxsz           = Calibration['boxsz']
    splineszx       = Calibration['splineszx']
    PSFsigmax       = Calibration['PSFsigmax']
    PSFsigmay       = Calibration['PSFsigmay']
    gauss_xvec      = Calibration['gauss_xvec']
    psf             = Calibration['kernel']
    psf_vali        = np.copy(psf [:, splineszx//2-boxsz//2:splineszx//2+boxsz//2+1, splineszx//2-boxsz//2:splineszx//2+boxsz//2+1])
    cmin = psf_vali.min()
    cmax = psf_vali.max()

    gauss_models = np.array([_drawPSF_gauss2d(boxsz, gauss_xvec[j,0], gauss_xvec[j,1], PSFsigmax[j], PSFsigmay[j], gauss_xvec[j,2], gauss_xvec[j,3]) for j in range(nchannels)])
    delta_PSF = psf_vali - gauss_models
    delta_PSF_min = delta_PSF.min()
    delta_PSF_max = delta_PSF.max() 

    ##### PLOT THE RESULTS #####
    fig_gauss = plt.figure(figsize=(scrWidth/72, scrHeight/72))
    gs0 = gridspec.GridSpec(2, 2, figure=fig_gauss)

    gaxs = ImageGrid(fig_gauss, 221, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, label_mode="L",
                        cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(psf_vali[j], cmap='turbo', vmin=cmin, vmax=cmax, aspect='equal')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('rawPSF_chnl{}'.format(j))
    fig_gauss.colorbar(imobj, cax=gaxs[nchannels-1].cax)
    
    gaxs = ImageGrid(fig_gauss, 222, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        nrm = mcolors.TwoSlopeNorm(vmin=delta_PSF_min, vmax=delta_PSF_max, vcenter=0)
        imobj = gaxs[j].imshow(delta_PSF[j], cmap='RdBu', norm=nrm, aspect='equal')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('deltaPSF_chnl{}'.format(j))
    fig_gauss.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    gs10 = gs0[1, 0].subgridspec(1, nchannels, wspace=0.0)
    for j in range(nchannels):
        ax = fig_gauss.add_subplot(gs10[j])
        ax.plot(np.arange(boxsz)+0.5, psf_vali[j][:, boxsz//2], 'o', color='tab:blue')
        ax.plot(np.arange(boxsz)+0.5, gauss_models[j][:, boxsz//2], color='tab:blue')
        ax.plot(np.arange(boxsz)+0.5, psf_vali[j][boxsz//2], 'o', color='tab:red')
        ax.plot(np.arange(boxsz)+0.5, gauss_models[j][boxsz//2], color='tab:red')
        ax.set_xlim(0.0, np.float32(boxsz))
        ax.set_ylim(cmin, 1.2*cmax)
        ax.text(1.0, cmin+0.8*cmax, PSFsigmax[j], color='tab:blue')
        ax.text(1.0, cmin+0.7*cmax, PSFsigmay[j], color='tab:red')
        ax.grid(which='major', axis='y', ls='--')
        ax.set_xlabel('pixels')
        if j == 0:
            ax.set_ylabel('Intensity (normalized)')
        if j > 0:
            ax.set_yticklabels([])

    gs11 = gs0[1, 1].subgridspec(1, nchannels, wspace=0.1)
    for j in range(nchannels):
        absres = abs(delta_PSF[j])
        absres[absres <= 0] = 1E-8
        logres = np.log10(absres)
        bins = np.linspace(logres.min(), logres.max(), 10+1)
        histcounts = np.histogram(logres, bins)[0]

        ax = fig_gauss.add_subplot(gs11[j])
        ax.bar(bins[:-1], histcounts, width=bins[1]-bins[0], align='edge')
        ax.set_xlabel('log10(rawPSF - modelPSF)')
        ax.grid(which='major', axis='x', ls='--')
        if j == 0:
            ax.set_ylabel('Counts')
    
    return fig_gauss



def plot_psf_cspline2d(Calibration):
    """
    Plot the cspline form of the calibrated 2d-psf
    INPUT:      Calibration:        dictionary, contains calibration information
    RETURN:     fig_cspline:        matplotlib figure handle, ploting cspline calibration result
    """

    ##### PARSE THE INPUT #####
    nchannels       = Calibration['nchannels']
    splineszx       = Calibration['splineszx']
    coeffs          = Calibration['coeff']
    psf             = Calibration['kernel']
    cmin = psf.min()
    cmax = psf.max()

    cspline_models = np.array([_drawPSF_cspline2d((splineszx, splineszx), coeff) for coeff in coeffs])
    delta_PSF = psf - cspline_models 
    delta_PSF_min = delta_PSF.min()
    delta_PSF_max = delta_PSF.max()

    ##### PLOT THE RESULTS #####
    fig_cspline = plt.figure(figsize=(scrWidth/72, scrHeight/72))
    gs0 = gridspec.GridSpec(2, 2, figure=fig_cspline)

    gaxs = ImageGrid(fig_cspline, 221, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(psf[j], cmap='turbo', vmin=cmin, vmax=cmax, aspect='equal')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('rawPSF_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)
    
    gaxs = ImageGrid(fig_cspline, 222, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        nrm = mcolors.TwoSlopeNorm(vmin=delta_PSF_min, vmax=delta_PSF_max, vcenter=0)
        imobj = gaxs[j].imshow(delta_PSF[j], cmap='RdBu', norm=nrm, aspect='equal')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('deltaPSF_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    gs10 = gs0[1, 0].subgridspec(1, nchannels, wspace=0.0)
    for j in range(nchannels):
        ax = fig_cspline.add_subplot(gs10[j])
        ax.plot(np.arange(splineszx)+0.5, psf[j][:, splineszx//2], 'o', color='tab:blue')
        ax.plot(np.arange(splineszx)+0.5, cspline_models[j][:, splineszx//2], color='tab:blue')
        ax.plot(np.arange(splineszx)+0.5, psf[j][splineszx//2], 'o', color='tab:red')
        ax.plot(np.arange(splineszx)+0.5, cspline_models[j][splineszx//2], color='tab:red')
        ax.set_ylim(cmin, 1.2*cmax)
        ax.grid(which='major', axis='y', ls='--')
        ax.set_xlabel('pixels')
        if j == 0:
            ax.set_ylabel('Intensity (normalized)')
        if j > 0:
            ax.set_yticklabels([])

    gs11 = gs0[1, 1].subgridspec(1, nchannels, wspace=0.1)
    for j in range(nchannels):
        absres = abs(delta_PSF[j])
        absres[absres <= 0] = 1E-8
        logres = np.log10(absres)
        bins = np.linspace(logres.min(), logres.max(), 20+1)
        histcounts = np.histogram(logres, bins)[0]

        ax = fig_cspline.add_subplot(gs11[j])
        ax.bar(bins[:-1], histcounts, width=bins[1]-bins[0], align='edge')
        ax.set_xlabel('log10(rawPSF - modelPSF)')
        ax.grid(which='major', axis='x', ls='--')
        if j == 0:
            ax.set_ylabel('Counts')
    
    return fig_cspline



def plot_psf_astigs(Astigmatics):
    """
    Plot the astigmatics of the psf
    INPUT:      Astigmatic:     (nchannels, npsfs) dict, Astigmatic profile of each individual psf data_stack
    RETURN:     fig_astigs:     matplotlib figure handle        
    """
    nchannels = len(Astigmatics)
    npsfs = len(Astigmatics[0]) 
    
    # determine the ylims for each subgraph
    individual_ymins = np.zeros((4, nchannels, npsfs))
    individual_ymaxs = np.zeros((4, nchannels, npsfs))
    for j in range(nchannels):
        for i in range(npsfs):
            astig = Astigmatics[j][i]
            
            individual_ymins[0, j, i] = np.min([astig['locx'][0].min(), astig['locx'][1].min(), astig['locy'][0].min(), astig['locy'][1].min()])
            individual_ymins[1, j, i] = np.min([astig['sigmax'][0].min(), astig['sigmax'][1].min(), astig['sigmay'][0].min(), astig['sigmay'][1].min()])
            individual_ymins[2, j, i] = np.min([astig['sigma2'][0].min(), astig['sigma2'][1].min()])
            individual_ymins[3, j, i] = np.min(astig['llr'])

            individual_ymaxs[0, j, i] = np.max([astig['locx'][0].max(), astig['locx'][1].max(), astig['locy'][0].max(), astig['locy'][1].max()])
            individual_ymaxs[1, j, i] = np.max([astig['sigmax'][0].max(), astig['sigmax'][1].max(), astig['sigmay'][0].max(), astig['sigmay'][1].max()])
            individual_ymaxs[2, j, i] = np.min([astig['sigma2'][0].max(), astig['sigma2'][1].max()])
            individual_ymaxs[3, j, i] = np.max(astig['llr'])
    ymins = np.min(np.min(individual_ymins, axis=-1), axis=-1)
    ymaxs = np.max(np.max(individual_ymaxs, axis=-1), axis=-1)

    # plot astigmatics
    fig_astigs = plt.figure(figsize=(scrWidth/72, scrHeight/72))
    gs0 = gridspec.GridSpec(2, 2, figure=fig_astigs)
    gs00 = gs0[0, 0].subgridspec(1, nchannels, wspace=0.0)
    gs01 = gs0[0, 1].subgridspec(1, nchannels, wspace=0.0)
    gs10 = gs0[1, 0].subgridspec(1, nchannels, wspace=0.0)
    gs11 = gs0[1, 1].subgridspec(1, nchannels, wspace=0.0)
    
    for j in range(nchannels):
        ax00 = fig_astigs.add_subplot(gs00[j])
        ax01 = fig_astigs.add_subplot(gs01[j])
        ax10 = fig_astigs.add_subplot(gs10[j])
        ax11 = fig_astigs.add_subplot(gs11[j])
        for i in range(npsfs):
            astig = Astigmatics[j][i]
            dumz = astig['locz']
            
            ax00.plot(dumz, astig['locx'][0], '.', color='tab:blue')
            ax00.plot(dumz, astig['locx'][1], color='tab:blue')
            ax00.plot(dumz, astig['locy'][0], '.', color='tab:red')
            ax00.plot(dumz, astig['locy'][1], color='tab:red')
                 
            ax01.plot(dumz, astig['sigmax'][0], '.', color='tab:blue')              
            ax01.plot(dumz, astig['sigmax'][1], color='tab:blue')
            ax01.plot(dumz, astig['sigmay'][0], '.', color='tab:red')
            ax01.plot(dumz, astig['sigmay'][1], color='tab:red')

            ax10.plot(dumz, astig['sigma2'][0], '.', color='tab:blue')
            ax10.plot(dumz, astig['sigma2'][1], color='tab:blue')
            
            ax11.plot(dumz, astig['llr'])
        
        ax00.set_ylim(ymins[0], ymaxs[0])
        ax00.grid(which='both', axis='both')
        ax00.set_title('Wobble_chnl{}'.format(j))
        ax00.set_xlabel("z")
        if j == 0:
            ax00.set_ylabel("xy-position (x:blue, y:red)")
        if j > 0:
            ax00.set_yticklabels([])

        ax01.set_ylim(ymins[1], ymaxs[1])
        ax01.grid(which='both', axis='both')
        ax01.set_title('PSFsigma_chnl{}'.format(j))
        ax01.set_xlabel("z")
        if j == 0:
            ax01.set_ylabel("PSFsigmas (x:blue, y:red)")
        if j > 0:
            ax01.set_yticklabels([])
        
        ax10.set_ylim(ymins[2], ymaxs[2])
        ax10.grid(which='both', axis='both')
        ax10.set_title('s_profile_chnl{}'.format(j))
        ax10.set_xlabel("z")
        if j == 0:
            ax10.set_ylabel("sigmax**2-sigmay**2")
        if j > 0:
            ax10.set_yticklabels([])
        
        ax11.set_ylim(ymins[3], ymaxs[3])
        ax11.grid(which='both', axis='both')
        ax11.set_title('LLR_chnl{}'.format(j))
        ax11.set_xlabel("z")
        if j == 0:
            ax11.set_ylabel("Individual LLR")
        if j > 0:
            ax11.set_yticklabels([])
        
    return fig_astigs



def plot_psf_gauss3d(Calibration, astigmatic2d_result, gauss_validation):
    """
    Plot the gauss form of the calibrated 3d-psf
    INPUT:
        Calibration:            dictionary, contains calibration information
        astigmatic2d_result:    (nchannels, splineszz, 7) ndarray, the gauss2d fit of each slice of the psf
        gauss_validation:       (nchannels, splineszz, 6) ndarray, the validation of the guass fit
    RETURN:
        fig_gauss:              matplotlib figure handle, ploting gauss calibration result
    """

    ##### PARSE THE INPUTS #####
    nchannels   = Calibration['nchannels']
    zrange      = Calibration['zrange']
    
    zdata = np.arange(zrange) + 0.5
    gauss_loc = gauss_validation[:, :, :3]
    gauss_loc_fit = np.zeros((nchannels, zrange, 3))
    for j in range(nchannels):
        for dimID, spl_wobble in enumerate([Calibration['spl_wobblex'], Calibration['spl_wobbley'], Calibration['spl_wobblez']]):
            gauss_loc_fit[j, :, dimID] = spl_wobble[j](zdata)
        gauss_loc_fit[j, :, -1] += zdata
    gauss_loc_res = gauss_loc - gauss_loc_fit
        
    gauss_sig = astigmatic2d_result[:, :, 4:6]
    gauss_sig2 = astigmatic2d_result[:, :, 4] * astigmatic2d_result[:, :, 4] - astigmatic2d_result[:, :, 5] * astigmatic2d_result[:, :, 5]
    gauss_sig_fit = np.zeros((nchannels, zrange, 2))
    gauss_sig2_fit = np.zeros((nchannels, zrange))
    for j in range(nchannels):
        gauss_sig_fit[j] = astigmatic(zdata, *Calibration['astigs'][j]).reshape((2, zrange)).T
        gauss_sig2_fit[j] = Calibration['spl_sxy2z'][j](gauss_sig2[j])
    
    gauss_llr = gauss_validation[:, :, -1]

    ylims = np.zeros((9, 2))
    ylims[0] = min(gauss_loc[:,:,0].min(), gauss_loc_fit[:,:,0].min()), max(gauss_loc[:,:,0].max(), gauss_loc_fit[:,:,0].max())
    ylims[1] = min(gauss_loc[:,:,1].min(), gauss_loc_fit[:,:,1].min()), max(gauss_loc[:,:,1].max(), gauss_loc_fit[:,:,1].max())
    ylims[2] = min(gauss_loc[:,:,2].min(), gauss_loc_fit[:,:,2].min()), max(gauss_loc[:,:,2].max(), gauss_loc_fit[:,:,2].max())
    ylims[3] = gauss_loc_res[:,:,0].min(), gauss_loc_res[:,:,0].max()
    ylims[4] = gauss_loc_res[:,:,1].min(), gauss_loc_res[:,:,1].max()
    ylims[5] = gauss_loc_res[:,:,2].min(), gauss_loc_res[:,:,2].max()
    ylims[6] = np.min([gauss_sig[:,:,0].min(), gauss_sig_fit[:,:,0].min(), gauss_sig[:,:,1].min(), gauss_sig_fit[:,:,1].min()]), np.max([gauss_sig[:,:,0].max(), gauss_sig_fit[:,:,0].max(), gauss_sig[:,:,1].max(), gauss_sig_fit[:,:,1].max()])
    ylims[7] = np.min([gauss_sig2_fit.min(), gauss_sig2.min(), 0]), np.max([gauss_sig2_fit.max(), gauss_sig2.max(), zrange])
    ylims[8] = gauss_llr.min(), gauss_llr.max()

    ##### PLOT THE GAUSS RESULTS #####
    fig_gauss = plt.figure(figsize=(scrWidth/72, scrHeight/72))
    grid_outer = gridspec.GridSpec(3, 3, figure=fig_gauss)
    grid_inners = [[grid_outer[i, j].subgridspec(1, nchannels, wspace=0.0) for j in range(3)] for i in range(3)]
    for j in range(nchannels):
        for dimID in range(3):
            ax = fig_gauss.add_subplot(grid_inners[0][dimID][j])
            ax.plot(zdata, gauss_loc[j, :, dimID], 'o', label='uncorrected')
            ax.plot(zdata, gauss_loc_fit[j, :, dimID], label='wobble') 
            ax.set_ylim(*ylims[dimID])
            ax.grid(which='both', axis='both')   
            ax.set_xlabel("z-pos (zstepsz)") 
            ax.set_title("{}-wobble chnl{}".format('x' if dimID==0 else 'y' if dimID==1 else 'z', j))
            if j == 0:
                ax.set_ylabel("{}-center (px)".format('x' if dimID==0 else 'y' if dimID==1 else 'z'))
            if j > 0:
                ax.set_yticklabels([])
            if j == nchannels - 1:
                ax.legend()           

            ax = fig_gauss.add_subplot(grid_inners[1][dimID][j])
            ax.plot(zdata, gauss_loc_res[j, :, dimID], 'o-', label='corrected')
            ax.set_ylim(*ylims[3 + dimID])
            ax.grid(which='both', axis='both')
            ax.set_xlabel("z-pos (zstepsz)")
            ax.set_title("{}-err chnl{}".format('x' if dimID==0 else 'y' if dimID==1 else 'z', j))
            if j == 0:
                ax.set_ylabel("{}-err (px)".format('x' if dimID==0 else 'y' if dimID==1 else 'z'))
            if j > 0:
                ax.set_yticklabels([])
            if j == nchannels - 1:
                ax.legend()    

        ax = fig_gauss.add_subplot(grid_inners[2][0][j])
        ax.plot(zdata, gauss_sig[j, :, 0], 'o', markerfacecolor='None', markeredgecolor='tab:blue')
        ax.plot(zdata, gauss_sig_fit[j, :, 0], color='tab:blue', label='PSFsigmax')
        ax.plot(zdata, gauss_sig[j, :, 1], 'o', markerfacecolor='None', markeredgecolor='tab:red')
        ax.plot(zdata, gauss_sig_fit[j, :, 1], color='tab:red', label='PSFsigmay')   
        ax.set_ylim(*ylims[6])
        ax.grid(which='both', axis='both')
        ax.set_xlabel("z-pos (zstepsz)")
        ax.set_title("astig chnl{}".format(j))
        if j == 0:
            ax.set_ylabel("sigma (px)")
        if j > 0:
            ax.set_yticklabels([])
        if j == nchannels - 1:
            ax.legend()

        ax = fig_gauss.add_subplot(grid_inners[2][1][j])
        ax.plot(gauss_sig2[j], zdata, 'o-', color='tab:blue')
        ax.plot(gauss_sig2[j], gauss_sig2_fit[j], color='tab:orange')
        ax.set_ylim(*ylims[7])
        ax.grid(which='both', axis='both')
        ax.set_xlabel("sigmax**2-sigmay**2")
        ax.set_title("s2z profile chnl{}".format(j))
        if j == 0:
            ax.set_ylabel("z-pos (zstepz)")
        if j > 0:
            ax.set_yticklabels([])

        ax = fig_gauss.add_subplot(grid_inners[2][2][j])
        ax.plot(zdata, gauss_llr[j])
        ax.set_ylim(*ylims[8])
        ax.grid(which='both', axis='x')
        ax.set_xlabel("z-pos (zstepsz)")
        ax.set_title("LLR chnl{}".format(j))
        if j == 0:
            ax.set_ylabel("llr")
        if j > 0:
            ax.set_yticklabels([])
        
    return fig_gauss



def plot_psf_cspline3d(Calibration):
    """
    Plot the cspline form of the calibrated 3d-psf
    INPUT:      Calibration:            dictionary, contains calibration information
    RETURN:     fig_cspline:            matplotlib figure handle, ploting cspline calibration result
    """

    ##### PARSE THE INPUTS #####
    nchannels       = Calibration['nchannels']
    splineszx       = Calibration['splineszx']
    splineszz       = Calibration['splineszz']
    coeffs          = Calibration['coeff']
    psf             = Calibration['kernel']
    
    zPos = np.arange(splineszz) + 0.5
    
    rawPSF_photon = np.array([np.sum(np.sum(psf[j], axis=-1), axis=-1) for j in range(nchannels)])
    rawPSF_Focus = np.array([psf[j][splineszz//2] for j in range(nchannels)])
    rawPSF_xzslice = np.array([psf[j][:, splineszx//2, :] for j in range(nchannels)])
    rawPSF_yzslice = np.array([psf[j][:, :, splineszx//2] for j in range(nchannels)])

    cspline_models = np.array([_drawPSF_cspline3d((splineszz, splineszx, splineszx), coeffs[j]) for j in range(nchannels)])
    modelPSF_photon = np.array([np.sum(np.sum(cspline_models[j], axis=-1), axis=-1) for j in range(nchannels)])
    modelPSF_Focus = np.array([cspline_models[j][splineszz//2] for j in range(nchannels)])
    modelPSF_xzslice = np.array([cspline_models[j][:, splineszx//2, :] for j in range(nchannels)])
    modelPSF_yzslice = np.array([cspline_models[j][:, :, splineszx//2] for j in range(nchannels)])
    
    Photon_min = min(rawPSF_photon.min(), modelPSF_photon.min())
    Focus_min = min(rawPSF_Focus.min(), modelPSF_Focus.min())
    xzslice_min = min(rawPSF_xzslice.min(), modelPSF_xzslice.min())
    yzslice_min = min(rawPSF_yzslice.min(), modelPSF_yzslice.min())

    Photon_max = max(rawPSF_photon.max(), modelPSF_photon.max())
    Focus_max = max(rawPSF_Focus.max(), modelPSF_Focus.max())
    xzslice_max = max(rawPSF_xzslice.max(), modelPSF_xzslice.max())
    yzslice_max = min(rawPSF_yzslice.max(), modelPSF_yzslice.max())

    delta_xzslice = rawPSF_xzslice - modelPSF_xzslice
    delta_xzslice_cmin = delta_xzslice.min()
    delta_xzslice_cmax = delta_xzslice.max()
    nrm_xzslice = mcolors.TwoSlopeNorm(vmin=delta_xzslice_cmin, vmax=delta_xzslice_cmax, vcenter=0)

    delta_yzslice = rawPSF_yzslice - modelPSF_yzslice
    delta_yzslice_cmin = delta_xzslice.min()
    delta_yzslice_cmax = delta_xzslice.max()
    nrm_yzslice = mcolors.TwoSlopeNorm(vmin=delta_yzslice_cmin, vmax=delta_yzslice_cmax, vcenter=0)

    delta_Focus = rawPSF_Focus - modelPSF_Focus
    delta_Focus_cmin = delta_Focus.min()
    delta_Focus_cmax = delta_Focus.max()
    nrm_Focus = mcolors.TwoSlopeNorm(vmin=delta_Focus_cmin, vmax=delta_Focus_cmax, vcenter=0)

    ##### PLOT THE CSPLINE RESULTS #####
    fig_cspline = plt.figure(figsize=(scrWidth/72, scrHeight/72))
    grid_outer = gridspec.GridSpec(3, 4, figure=fig_cspline)
    grid_inner0 = grid_outer[0, 3].subgridspec(1, nchannels, wspace=0.0)
    grid_inner1 = grid_outer[1, 3].subgridspec(1, nchannels, wspace=0.1)
    grid_inner2 = grid_outer[2, 3].subgridspec(1, nchannels, wspace=0.1)
    
    gaxs = ImageGrid(fig_cspline, 341, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, aspect=False, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(rawPSF_xzslice[j], cmap='turbo', vmin=xzslice_min, vmax=xzslice_max, aspect='auto')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('rawPSF_xzslice_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    gaxs = ImageGrid(fig_cspline, 342, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, aspect=False, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(modelPSF_xzslice[j], cmap='turbo', vmin=xzslice_min, vmax=xzslice_max, aspect='auto')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('modelPSF_xzslice_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    gaxs = ImageGrid(fig_cspline, 343, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, aspect=False, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(delta_xzslice[j], cmap='RdBu', norm=nrm_xzslice, aspect='auto')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('delta_xzslice_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    gaxs = ImageGrid(fig_cspline, 345, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, aspect=False, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(rawPSF_yzslice[j], cmap='turbo', vmin=yzslice_min, vmax=yzslice_max, aspect='auto')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('rawPSF_yzslice_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    gaxs = ImageGrid(fig_cspline, 346, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, aspect=False, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(modelPSF_yzslice[j], cmap='turbo', vmin=yzslice_min, vmax=yzslice_max, aspect='auto')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('modelPSF_yzslice_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    gaxs = ImageGrid(fig_cspline, 347, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, aspect=False, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(delta_yzslice[j], cmap='RdBu', norm=nrm_yzslice, aspect='auto')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('delta_yzslice_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    gaxs = ImageGrid(fig_cspline, 349, nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(rawPSF_Focus[j], cmap='turbo', vmin=Focus_min, vmax=Focus_max, aspect='equal')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('rawPSF_Focus_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    gaxs = ImageGrid(fig_cspline, (3,4,10), nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(modelPSF_Focus[j], cmap='turbo', vmin=Focus_min, vmax=Focus_max, aspect='equal')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('modelPSF_Focus_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    gaxs = ImageGrid(fig_cspline, (3,4,11), nrows_ncols=(1, nchannels), direction="row", axes_pad=0.05, share_all=True, label_mode="L",
                    cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.05)
    for j in range(nchannels):
        imobj = gaxs[j].imshow(delta_Focus[j], cmap='RdBu', norm=nrm_Focus, aspect='equal')
        gaxs[j].set_xticklabels([])
        gaxs[j].set_yticklabels([])
        gaxs[j].set_title('delta_Focus_chnl{}'.format(j))
    fig_cspline.colorbar(imobj, cax=gaxs[nchannels-1].cax)

    for j in range(nchannels):

        ax = fig_cspline.add_subplot(grid_inner0[j])        
        ax.plot(zPos, rawPSF_photon[j], '.', label='rawPSF_Photon')
        ax.plot(zPos, modelPSF_photon[j], label='modelPSF_Photon')
        ax.set_ylim(Photon_min, Photon_max)
        ax.grid(which='major', axis='both', ls='--')
        ax.set_title('Photon_chnl{}'.format(j))
        ax.set_xlabel('z position')
        if j == 0:
            ax.set_ylabel("photon (norm)")
        if j > 0:
            ax.set_yticklabels([])
        if j == nchannels - 1:
            ax.legend()
        
        absres = abs(psf[j]- cspline_models[j])
        absres[absres == 0] = 1E-8
        logres = np.log10(absres)
        bins = np.linspace(logres.min(), logres.max(), 200+1)
        histcounts = np.histogram(logres, bins)[0]
        ax = fig_cspline.add_subplot(grid_inner1[j])
        ax.bar(bins[:-1], histcounts, width=bins[1]-bins[0], align='edge')
        ax.set_xlabel('log10(rawPSF - modelPSF)')
        ax.grid(which='major', axis='x', ls='--')

        absres = abs(rawPSF_Focus[j] - modelPSF_Focus[j])
        absres[absres == 0] = 1E-8
        logres = np.log10(absres)
        bins = np.linspace(logres.min(), logres.max(), 20+1)
        histcounts = np.histogram(logres, bins)[0]
        ax = fig_cspline.add_subplot(grid_inner2[j])
        ax.bar(bins[:-1], histcounts, width=bins[1]-bins[0], align='edge')
        ax.set_xlabel('log10(rawPSF_Focus - modelPSF_Focus)')
        ax.grid(which='major', axis='x', ls='--')

    return fig_cspline